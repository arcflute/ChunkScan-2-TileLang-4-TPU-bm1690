"""R0 cmodel regression for the independent P9.2 multi-core S1 route."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
from pathlib import Path

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_abi_p9 import (
    guards_unchanged,
    make_inputs,
    make_physical_output,
    pack_inputs,
    payload_has_no_sentinel,
    poison_upper_triangle,
    unpack_output,
    variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ChunkScanMulticoreConfig,
    make_chunk_scan_multicore_s1_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)


HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts" / "multicore_s1_p9_cmodel"
RUNTIME_DIR = ARTIFACT_DIR / "runtime"
CONFIG = ChunkScanMulticoreConfig("r0", 1, 2, 1)


def prepare_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    template_dir = REPOSITORY_ROOT / "src/tl_templates/tpu"
    for name in (
        "kernel_template.cpp",
        "kernel_template.h",
        "main_template.cpp",
    ):
        shutil.copy2(template_dir / name, RUNTIME_DIR / name)
    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = lambda path=str(RUNTIME_DIR): path
    os.environ["TPU_KERNEL_PATH"] = str(RUNTIME_DIR)
    os.environ["PPL_KERNEL_PATH"] = str(RUNTIME_DIR / "libkernel.so")


def run_case(kernel, inputs, name: str):
    expected = chunk_scan_reference(*inputs)
    physical = make_physical_output(CONFIG)
    returned = kernel(*pack_inputs(CONFIG, inputs), physical)
    actual = unpack_output(CONFIG, physical)
    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(
            actual.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    result = {
        "case": name,
        "close": close,
        "guard_rows_unchanged": guards_unchanged(physical),
        "payload_fully_covered": payload_has_no_sentinel(physical),
        "returned": None if returned is None else int(returned),
        **stats,
    }
    if not (
        close
        and result["guard_rows_unchanged"]
        and result["payload_fully_covered"]
        and returned in (None, 0)
        and stats["nan_count"] == 0
        and stats["inf_count"] == 0
    ):
        raise AssertionError(result)
    return actual, result


def main() -> None:
    toolchain = assert_chunkscan_toolchain()
    prepare_runtime()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        program = make_chunk_scan_multicore_s1_kernel(CONFIG)
        lowered = tilelang.lower(program, target="tpu")
        kernel = tilelang.compile(
            program,
            out_idx=[7],
            target="tpu",
            mode="cmodel",
        )
    (ARTIFACT_DIR / "compile_stdout.log").write_text(
        captured.getvalue(), encoding="utf-8"
    )
    source = str(lowered.kernel_source)
    (ARTIFACT_DIR / "kernel_raw.c").write_text(
        source, encoding="utf-8"
    )
    if "tpu_workitem_index()" not in source:
        raise AssertionError("work-item index was not emitted")
    if "tpu_workitem_num()" not in source:
        raise AssertionError("work-item count was not emitted")

    base = make_inputs(CONFIG, seed=20260922)
    all_output, all_result = run_case(
        kernel,
        variant_inputs(base, "all_terms"),
        "all_terms",
    )
    del all_output
    clean_inputs = variant_inputs(base, "scan_only")
    clean_output, clean_result = run_case(
        kernel,
        clean_inputs,
        "scan_only",
    )
    poison_output, poison_result = run_case(
        kernel,
        poison_upper_triangle(clean_inputs),
        "causal_upper_triangle_poison",
    )
    poison_result["bitwise_equal_to_clean_scan"] = bool(
        torch.equal(clean_output, poison_output)
    )
    if not poison_result["bitwise_equal_to_clean_scan"]:
        raise AssertionError("causal poison changed cmodel output")

    report = {
        "status": "PASS",
        "stage": "P9.2",
        "candidate": "S1",
        "mode": "cmodel single work-item regression",
        "logical_shape": CONFIG.logical_shape(),
        "cases": {
            "all_terms": all_result,
            "scan_only": clean_result,
            "causal_upper_triangle_poison": poison_result,
        },
        "toolchain": toolchain,
        "physical_multicore_claim": False,
        "performance_claim": "none",
    }
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("P9.2 S1 R0 CMODEL PASS")
    print(ARTIFACT_DIR / "result.json")


if __name__ == "__main__":
    main()
