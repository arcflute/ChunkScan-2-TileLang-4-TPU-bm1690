"""Structural tests for the P9.3 multi-core S3/P6 kernels."""

import contextlib
import io

import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_p6_p9 import (
    make_chunk_scan_multicore_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s3_p9 import (
    make_chunk_scan_multicore_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import SCHEDULE_NAME
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_s1_p9 import (
    SHAPES,
    static_local_memory_end,
)


CANDIDATES = {
    "s3": make_chunk_scan_multicore_s3_kernel,
    "p6": make_chunk_scan_multicore_p6_kernel,
}


def test_multicore_s3_p6_lowering():
    for candidate, builder in CANDIDATES.items():
        for config in SHAPES.values():
            program = builder(config)
            schedule = program.attrs.get("p6_pipeline_schedule")
            if candidate == "p6":
                assert str(schedule) == SCHEDULE_NAME
            else:
                assert schedule is None

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                artifact = tilelang.lower(program, target="tpu")
            source = str(artifact.kernel_source)
            assert (
                f"for (int task = 0; task < {config.task_count}; ++task)"
                in source
            )
            assert source.count("tpu_workitem_index()") == 1
            assert source.count("tpu_workitem_num()") == 1
            assert "ppl.workitem_index" not in source
            assert "ppl.workitem_num" not in source
            assert source.count("tpu_parallel_start(") == 1
            assert source.count("tpu_parallel_end(") == 1
            local_end = static_local_memory_end(source)
            assert local_end is not None
            assert local_end <= 256 * 1024


if __name__ == "__main__":
    test_multicore_s3_p6_lowering()
    print("P9.3 S3/P6 lowering PASS")
