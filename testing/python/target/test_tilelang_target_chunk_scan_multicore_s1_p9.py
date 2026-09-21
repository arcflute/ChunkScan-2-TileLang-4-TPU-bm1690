"""Structural and compact-ABI tests for the P9.2 multi-core S1 kernel."""

import contextlib
import io

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_abi_p9 import (
    make_inputs,
    make_physical_output,
    pack_inputs,
    unpack_output,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    make_chunk_scan_multicore_s1_kernel,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_s1_p9 import (
    SHAPES,
    static_local_memory_end,
)


def test_multicore_s1_lowering_and_compact_abi():
    for index, (name, config) in enumerate(SHAPES.items()):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            artifact = tilelang.lower(
                make_chunk_scan_multicore_s1_kernel(config),
                target="tpu",
            )
        source = str(artifact.kernel_source)
        assert (
            f"for (int task = 0; task < {config.task_count}; ++task)"
            in source
        )
        assert "tpu_workitem_index()" in source
        assert "tpu_workitem_num()" in source
        assert "ppl.workitem_index" not in source
        assert "ppl.workitem_num" not in source
        assert "tpu_parallel_start(" not in source
        assert "tpu_parallel_end(" not in source
        local_end = static_local_memory_end(source)
        assert local_end is not None
        assert local_end <= 256 * 1024

        inputs = make_inputs(config, seed=20260922 + index)
        pack_inputs(config, inputs)
        physical = make_physical_output(config)
        logical = (
            torch.arange(
                config.batch
                * config.seqlen
                * config.nheads
                * config.headdim,
                dtype=torch.int32,
            )
            .remainder(1000)
            .to(torch.float16)
            .view(
                config.batch,
                config.seqlen,
                config.nheads,
                config.headdim,
            )
        )
        task_major = (
            logical.view(
                config.batch,
                config.nchunks,
                config.chunk_size,
                config.nheads,
                config.headdim,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(-1, config.headdim)
        )
        physical[1:-1].copy_(task_major)
        assert torch.equal(unpack_output(config, physical), logical)
        assert name == config.name


if __name__ == "__main__":
    test_multicore_s1_lowering_and_compact_abi()
    print("P9.2 S1 lowering and compact ABI PASS")
