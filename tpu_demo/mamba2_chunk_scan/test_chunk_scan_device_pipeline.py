"""BM1690 single-core hardware checks for ChunkScan S1, S2, S3 and P6."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_reduction_serial import (
    make_chunk_scan_reduction_serial_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s2 import (
    make_chunk_scan_pipeline_s2_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s3 import (
    make_chunk_scan_pipeline_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import (
    make_chunk_scan_pipeline_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_serial import CHUNK_SIZE
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_cmodel import (
    _make_inputs,
    _physical_args,
    _variant_inputs,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT_ROOT = HERE / "artifacts" / "device_pipeline"

BUILDERS = {
    "s1": make_chunk_scan_reduction_serial_kernel,
    "s2": make_chunk_scan_pipeline_s2_kernel,
    "s3": make_chunk_scan_pipeline_s3_kernel,
    "p6": make_chunk_scan_pipeline_p6_kernel,
}


def prepare_runtime(stage: str) -> tuple[Path, Path]:
    stage_dir = ARTIFACT_ROOT / stage
    runtime_dir = stage_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    template_dir = ROOT / "src" / "tl_templates" / "tpu"
    for name in ("kernel_template.cpp", "kernel_template.h"):
        shutil.copy2(template_dir / name, runtime_dir / name)
    shutil.copy2(
        HERE / "main_template_device.cpp",
        runtime_dir / "main_template.cpp",
    )

    import importlib

    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = (
            lambda path=str(runtime_dir): path
        )

    os.environ["TPU_KERNEL_PATH"] = str(runtime_dir)
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")
    return stage_dir, runtime_dir


def compile_stage(stage: str):
    stage_dir, runtime_dir = prepare_runtime(stage)
    captured = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured):
            kernel = tilelang.compile(
                BUILDERS[stage](),
                out_idx=[7],
                target="tpu",
                mode="pcie",
            )
    finally:
        (stage_dir / "compile_stdout.log").write_text(
            captured.getvalue(), encoding="utf-8"
        )

    source_path = runtime_dir / "kernel.c"
    device_lib = runtime_dir / "libkernel.so"
    host_lib = runtime_dir / "main.so"
    for path in (source_path, device_lib, host_lib):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = source_path.read_text(encoding="utf-8")
    starts = source.count("tpu_parallel_start(")
    ends = source.count("tpu_parallel_end(")
    expected_markers = 0 if stage == "s1" else 1
    if (starts, ends) != (expected_markers, expected_markers):
        raise AssertionError(
            f"{stage}: unexpected pipeline markers "
            f"start={starts}, end={ends}; expected={expected_markers}"
        )

    device_file = subprocess.run(
        ["file", str(device_lib)],
        check=True, capture_output=True, text=True,
    ).stdout
    host_file = subprocess.run(
        ["file", str(host_lib)],
        check=True, capture_output=True, text=True,
    ).stdout
    linked = subprocess.run(
        ["ldd", str(host_lib)],
        check=True, capture_output=True, text=True,
    ).stdout

    runtime_root = os.environ["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]
    expected_runtime = str(
        Path(runtime_root) / "lib" / "libtpuv7_rt.so"
    )
    if "RISC-V" not in device_file or "x86-64" not in host_file:
        raise AssertionError(
            f"{stage}: wrong binary architecture:\n"
            f"{device_file}{host_file}"
        )
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(
            f"{stage}: host library is not linked to real runtime:\n"
            f"{linked}"
        )

    manifest = {
        "stage": stage,
        "mode": "BM1690 PCIe single core",
        "kernel_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "pipeline_start_count": starts,
        "pipeline_end_count": ends,
        "device_library": device_file.strip(),
        "host_library": host_file.strip(),
        "runtime_library": expected_runtime,
        "performance_claim": "none",
    }
    (stage_dir / "compile_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{stage.upper()} COMPILE PASS", flush=True)
    return kernel, stage_dir, manifest


def run_case(kernel, name: str, inputs):
    expected = chunk_scan_reference(*inputs)
    actual = torch.full_like(expected, float("nan"))
    ret = kernel(*_physical_args(inputs, actual))
    if ret != 0:
        raise RuntimeError(f"{name}: TPU host call returned {ret}")

    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(
            actual.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    result = {"case": name, "close": close, **stats}
    print(json.dumps(result, sort_keys=True), flush=True)
    if not close:
        raise AssertionError(f"{name}: BM1690 result mismatch")
    return actual, result


def run_stage(stage: str, kernel, stage_dir: Path, manifest: dict) -> None:
    base = _make_inputs(seed=20260910)
    results = {}
    outputs = {}

    for name in ("residual_only", "state_only", "scan_only", "all_terms"):
        inputs = _variant_inputs(base, name)
        outputs[name], results[name] = run_case(kernel, name, inputs)

    negative = list(_variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    _, results["residual_only_negative_D"] = run_case(
        kernel, "residual_only_negative_D", tuple(negative)
    )

    scan_inputs = _variant_inputs(base, "scan_only")
    upper = torch.triu(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool),
        diagonal=1,
    ).view(1, 1, 1, CHUNK_SIZE, CHUNK_SIZE)
    poisoned_cb = torch.where(
        upper,
        torch.full_like(scan_inputs[0], 8.0),
        scan_inputs[0],
    ).contiguous()
    poisoned = (poisoned_cb,) + scan_inputs[1:]
    poisoned_output, poison_result = run_case(
        kernel, "causal_upper_triangle_poison", poisoned
    )
    poison_result["bitwise_equal_to_clean_scan"] = bool(
        torch.equal(poisoned_output, outputs["scan_only"])
    )
    if not poison_result["bitwise_equal_to_clean_scan"]:
        raise AssertionError(
            f"{stage}: poisoned upper triangle changed device output"
        )
    results["causal_upper_triangle_poison"] = poison_result

    report = {
        "status": "PASS",
        "stage": stage,
        "compile": manifest,
        "cases": results,
        "tolerance": {"atol": 1e-2, "rtol": 1e-2},
        "performance_claim": "none",
    }
    (stage_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{stage.upper()} HARDWARE PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=BUILDERS)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    kernel, stage_dir, manifest = compile_stage(args.stage)
    if not args.run:
        print("Compile-only: no TPU kernel was launched.", flush=True)
        return
    run_stage(args.stage, kernel, stage_dir, manifest)


if __name__ == "__main__":
    main()
