"""Structural tests for BM1690 PPL work-item intrinsics."""

import contextlib
import io

import tilelang

from tpu_demo.mamba2_chunk_scan.multicore_workitem_probe_p9 import (
    make_multicore_workitem_probe_p9_kernel,
)


def test_ppl_workitem_intrinsics_lower_to_runtime_calls():
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        artifact = tilelang.lower(
            make_multicore_workitem_probe_p9_kernel(),
            target="tpu",
        )
    source = str(artifact.kernel_source)

    assert "tpu_workitem_index()" in source
    assert "tpu_workitem_num()" in source
    assert "ppl.workitem_index" not in source
    assert "ppl.workitem_num" not in source


if __name__ == "__main__":
    test_ppl_workitem_intrinsics_lower_to_runtime_calls()
    print("PPL work-item codegen PASS")
