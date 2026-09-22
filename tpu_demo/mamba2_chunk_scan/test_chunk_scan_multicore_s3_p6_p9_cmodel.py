"""R0 cmodel regression for the P9.3 multi-core S3/P6 route."""

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
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_p6_p9 import (
    make_chunk_scan_multicore_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ChunkScanMulticoreConfig,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s3_p9 import (
    make_chunk_scan_multicore_s3_kernel,
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
ARTIFACT_DIR = HERE / "artifacts" / "multicore_s3_p6_p9_cmodel"
CONFIG = ChunkScanMulticoreConfig("r0", 1, 2, 1)
CANDIDATES = {
    "s3": make_chunk_scan_multicore_s3_kernel,
    "p6": make_chunk_scan_multicore_p6_kernel,
}


def prepare_runtime(candidate: str) -> Path:
    runtime_dir = ARTIFACT_DIR / candidate / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    template_dir = REPOSITORY_ROOT / "src/tl_templates/tpu"
    for name in (
        "kernel_template.cpp",
        "kernel_template.h",
        "main_template.cpp",
    ):
        shutil.copy2(template_dir / name, runtime_dir / name)
    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = lambda path=str(runtime_dir): path
    os.environ["TPU_KERNEL_PATH"] = str(runtime_dir)
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")
    return runtime_dir


def run_case(kernel, runtime_dir: Path, inputs, name: str):
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")
    expected = chunk_scan_reference(*inputs)
    physical = make_physical_output(CONFIG)
    returned = kernel(*pack_inputs(CONFIG, inputs), physical)
    actual = unpack_output(CONFIG, physical)
    stats = comparison_stats(actual, expected)
    result = {
        "case": name,
        "close": bool(
            torch.allclose(
                actual.float(), expected.float(), atol=1e-2, rtol=1e-2
            )
        ),
        "guard_rows_unchanged": guards_unchanged(physical),
        "payload_fully_covered": payload_has_no_sentinel(physical),
        "returned": None if returned is None else int(returned),
        **stats,
    }
    if not (
        result["close"]
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
    base = make_inputs(CONFIG, seed=20260923)
    cases = {
        "all_terms": variant_inputs(base, "all_terms"),
        "scan_only": variant_inputs(base, "scan_only"),
    }
    cases["causal_upper_triangle_poison"] = poison_upper_triangle(
        cases["scan_only"]
    )

    reports = {}
    outputs = {}
    for candidate, builder in CANDIDATES.items():
        runtime_dir = prepare_runtime(candidate)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            program = builder(CONFIG)
            lowered = tilelang.lower(program, target="tpu")
            kernel = tilelang.compile(
                program, out_idx=[7], target="tpu", mode="cmodel"
            )
        candidate_dir = ARTIFACT_DIR / candidate
        (candidate_dir / "compile_stdout.log").write_text(
            captured.getvalue(), encoding="utf-8"
        )
        source = str(lowered.kernel_source)
        (candidate_dir / "kernel_raw.c").write_text(
            source, encoding="utf-8"
        )
        if source.count("tpu_workitem_index()") != 1:
            raise AssertionError(f"{candidate}: missing work-item index")
        if source.count("tpu_workitem_num()") != 1:
            raise AssertionError(f"{candidate}: missing work-item count")
        if source.count("tpu_parallel_start(") != 1:
            raise AssertionError(f"{candidate}: missing pipeline start")
        if source.count("tpu_parallel_end(") != 1:
            raise AssertionError(f"{candidate}: missing pipeline end")

        candidate_outputs = {}
        candidate_cases = {}
        for case_name, inputs in cases.items():
            output, result = run_case(
                kernel, runtime_dir, inputs, case_name
            )
            candidate_outputs[case_name] = output
            candidate_cases[case_name] = result
        poison_equal = torch.equal(
            candidate_outputs["scan_only"],
            candidate_outputs["causal_upper_triangle_poison"],
        )
        if not poison_equal:
            raise AssertionError(f"{candidate}: causal poison changed output")
        candidate_cases["causal_upper_triangle_poison"][
            "bitwise_equal_to_clean_scan"
        ] = True
        outputs[candidate] = candidate_outputs
        reports[candidate] = {
            "status": "PASS",
            "cases": candidate_cases,
        }

    cross_candidate = all(
        torch.equal(outputs["s3"][name], outputs["p6"][name])
        for name in cases
    )
    if not cross_candidate:
        raise AssertionError("S3 and P6 cmodel outputs differ")

    report = {
        "status": "PASS",
        "stage": "P9.3",
        "candidates": reports,
        "s3_p6_bitwise_equal": True,
        "mode": "cmodel single work-item regression",
        "logical_shape": CONFIG.logical_shape(),
        "toolchain": toolchain,
        "physical_multicore_claim": False,
        "performance_claim": "none",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("P9.3 S3/P6 R0 CMODEL PASS")
    print(ARTIFACT_DIR / "result.json")


if __name__ == "__main__":
    main()
