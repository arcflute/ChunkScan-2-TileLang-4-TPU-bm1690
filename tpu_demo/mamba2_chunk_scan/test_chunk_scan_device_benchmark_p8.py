"""P8 synchronized-call benchmark for single-core BM1690 ChunkScan.

The benchmark excludes compilation, module setup, device allocation, and
host/device copies from every recorded sample.  Each recorded sample is one
synchronous host wrapper call, so it includes launch, device execution, and
stream synchronization.  This is a latency benchmark, not an async throughput
benchmark and not proof of physical GDMA/BDC overlap.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import (
    make_chunk_scan_pipeline_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s2 import (
    make_chunk_scan_pipeline_s2_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s3 import (
    make_chunk_scan_pipeline_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_reduction_serial import (
    make_chunk_scan_reduction_serial_kernel,
)
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
ARTIFACT_ROOT = HERE / "artifacts" / "device_benchmark_p8"
STAGE_NAMES = ("s1", "s2", "s3", "p6")
BUILDERS = {
    "s1": make_chunk_scan_reduction_serial_kernel,
    "s2": make_chunk_scan_pipeline_s2_kernel,
    "s3": make_chunk_scan_pipeline_s3_kernel,
    "p6": make_chunk_scan_pipeline_p6_kernel,
}
EVENT_SYMBOLS = (
    "tpuRtEventCreate",
    "tpuRtEventRecord",
    "tpuRtEventSynchronize",
    "tpuRtEventElapsedTime",
)
ATOL = 1e-2
RTOL = 1e-2


@dataclass
class CompiledStage:
    name: str
    kernel: object
    stage_dir: Path
    runtime_dir: Path
    manifest: Dict[str, object]


def utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}"


def run_text(command: Iterable[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        list(command),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_text(*args: str) -> str:
    return run_text(("git", "-C", str(ROOT), *args))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_environment() -> Dict[str, Path]:
    required = {
        "PPL_PROJECT_ROOT": "PPL project root",
        "CHUNKSCAN_DEVICE_RUNTIME_ROOT": "BM1690 runtime root",
        "CHUNKSCAN_RISCV_TOOLCHAIN_ROOT": "RISC-V toolchain root",
    }
    values: Dict[str, Path] = {}
    for variable, description in required.items():
        text = os.environ.get(variable)
        if not text:
            raise EnvironmentError(f"{variable} is not set ({description})")
        values[variable] = Path(text).expanduser().resolve()

    checks = {
        values["PPL_PROJECT_ROOT"]
        / "runtime/bm1690/lib/libbm1690.a": "libbm1690.a",
        values["PPL_PROJECT_ROOT"]
        / "runtime/customize/src/ppl_helper.c": "ppl_helper.c",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]
        / "include/tpuv7_rt.h": "tpuv7_rt.h",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]
        / "lib/libtpuv7_rt.so": "libtpuv7_rt.so",
        values["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        / "bin/riscv64-unknown-linux-gnu-gcc": "RISC-V gcc",
        ROOT / "build/libtilelang_module.so": "libtilelang_module.so",
        ROOT / "build/tvm/libtvm.so": "libtvm.so",
    }
    for path, description in checks.items():
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    return values


def runtime_event_capability(runtime_library: Path) -> Dict[str, object]:
    completed = subprocess.run(
        ["nm", "-D", str(runtime_library)],
        capture_output=True,
        text=True,
    )
    combined = completed.stdout + completed.stderr
    symbols = {name: name in combined for name in EVENT_SYMBOLS}
    return {
        "nm_exit_code": completed.returncode,
        "symbols": symbols,
        "all_required_symbols_present": all(symbols.values()),
        "used_by_p8": False,
        "note": (
            "P8 records synchronized host-call latency. Event/async timing is "
            "deferred to the combined multi-core performance stage."
        ),
    }


def collect_environment(paths: Mapping[str, Path], args: argparse.Namespace) -> dict:
    runtime_library = (
        paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"] / "lib/libtpuv7_rt.so"
    )
    lspci = subprocess.run(
        ["lspci", "-nnk"], capture_output=True, text=True
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_text("rev-parse", "HEAD"),
        "repository_status": git_text("status", "--short").splitlines(),
        "tvm_commit": run_text(
            ("git", "-C", str(ROOT / "3rdparty/tvm"), "rev-parse", "HEAD")
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device_id": os.environ.get("CHUNKSCAN_DEVICE_ID", "0"),
        "ppl_project_root": str(paths["PPL_PROJECT_ROOT"]),
        "runtime_root": str(paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]),
        "runtime_library": str(runtime_library),
        "runtime_library_sha256": sha256_file(runtime_library),
        "riscv_toolchain_root": str(paths["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]),
        "event_capability": runtime_event_capability(runtime_library),
        "pci_1690_lines": [
            line
            for line in lspci.stdout.splitlines()
            if "1f1c:1690" in line.lower()
            or "sg-host-drv" in line.lower()
        ],
        "benchmark_arguments": {
            "stages": list(args.resolved_stages),
            "rounds": args.rounds,
            "warmup": args.warmup,
            "min_seconds": args.min_seconds,
            "min_runs": args.min_runs,
            "quick": args.quick,
            "compile_only": args.compile_only,
        },
        "measurement_scope": (
            "one synchronous kernel wrapper call; includes launch, device "
            "execution, and stream synchronization; excludes compile, module "
            "load, allocation, H2D, and D2H"
        ),
        "performance_claim_scope": "preliminary fixed-shape single-core P8",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextlib.contextmanager
def temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def prepare_runtime(run_dir: Path, stage: str) -> tuple[Path, Path]:
    stage_dir = run_dir / stage
    runtime_dir = stage_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    base_template_dir = ROOT / "src/tl_templates/tpu"
    for name in ("kernel_template.cpp", "kernel_template.h"):
        shutil.copy2(base_template_dir / name, runtime_dir / name)
    shutil.copy2(
        HERE / "main_template_device_bench.cpp",
        runtime_dir / "main_template.cpp",
    )

    import importlib

    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = lambda path=str(runtime_dir): path

    os.environ["TPU_KERNEL_PATH"] = str(runtime_dir)
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")
    return stage_dir, runtime_dir


def compile_stage(run_dir: Path, stage: str) -> CompiledStage:
    stage_dir, runtime_dir = prepare_runtime(run_dir, stage)
    captured_stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured_stdout):
            kernel = tilelang.compile(
                BUILDERS[stage](),
                out_idx=[7],
                target="tpu",
                mode="pcie",
            )
    finally:
        (stage_dir / "compile_stdout.log").write_text(
            captured_stdout.getvalue(), encoding="utf-8"
        )

    source_path = runtime_dir / "kernel.c"
    main_path = runtime_dir / "main.cpp"
    device_library = runtime_dir / "libkernel.so"
    host_library = runtime_dir / "main.so"
    for path in (source_path, main_path, device_library, host_library):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = source_path.read_text(encoding="utf-8")
    main_source = main_path.read_text(encoding="utf-8")
    starts = source.count("tpu_parallel_start(")
    ends = source.count("tpu_parallel_end(")
    expected_markers = 0 if stage == "s1" else 1
    if (starts, ends) != (expected_markers, expected_markers):
        raise AssertionError(
            f"{stage}: pipeline markers are {(starts, ends)}, "
            f"expected {(expected_markers, expected_markers)}"
        )
    if "steady_clock_sync_kernel_call" not in main_source:
        raise AssertionError(f"{stage}: benchmark template was not installed")

    device_file = run_text(("file", str(device_library)))
    host_file = run_text(("file", str(host_library)))
    linked = run_text(("ldd", str(host_library)))
    runtime_root = Path(os.environ["CHUNKSCAN_DEVICE_RUNTIME_ROOT"])
    expected_runtime = str(runtime_root / "lib/libtpuv7_rt.so")
    if "RISC-V" not in device_file or "x86-64" not in host_file:
        raise AssertionError(
            f"{stage}: wrong ELF architecture:\n{device_file}\n{host_file}"
        )
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(
            f"{stage}: host library is not linked to the real runtime:\n{linked}"
        )

    manifest: Dict[str, object] = {
        "stage": stage,
        "mode": "BM1690 PCIe single-core synchronized-call benchmark",
        "kernel_source_sha256": sha256_file(source_path),
        "main_source_sha256": sha256_file(main_path),
        "device_library_sha256": sha256_file(device_library),
        "host_library_sha256": sha256_file(host_library),
        "pipeline_start_count": starts,
        "pipeline_end_count": ends,
        "device_library_file": device_file,
        "host_library_file": host_file,
        "linked_libraries": linked.splitlines(),
        "runtime_library": expected_runtime,
        "core_num": 1,
        "shape": {
            "B": 1,
            "S": 128,
            "Ck": 2,
            "L": 64,
            "G": 1,
            "H": 1,
            "P": 64,
            "N": 128,
        },
        "performance_claim_scope": "preliminary fixed-shape P8",
    }
    write_json(stage_dir / "compile_manifest.json", manifest)
    print(f"{stage.upper()} COMPILE PASS", flush=True)
    return CompiledStage(stage, kernel, stage_dir, runtime_dir, manifest)


def call_kernel(
    compiled: CompiledStage,
    inputs,
    output: torch.Tensor,
    mode: str,
    extra_environment: Mapping[str, str] | None = None,
) -> object:
    updates = {
        "PPL_KERNEL_PATH": str(compiled.runtime_dir / "libkernel.so"),
        "CHUNKSCAN_BENCH_MODE": mode,
        "CHUNKSCAN_BENCH_STAGE": compiled.name,
    }
    if extra_environment:
        updates.update(extra_environment)
    with temporary_environment(updates):
        return compiled.kernel(*_physical_args(inputs, output))


def correctness_check(
    compiled: CompiledStage,
    inputs,
    expected: torch.Tensor,
    label: str,
) -> dict:
    actual = torch.full_like(expected, float("nan"))
    returned = call_kernel(compiled, inputs, actual, "correctness")
    if returned not in (None, 0):
        raise RuntimeError(
            f"{compiled.name} {label}: host call returned {returned}"
        )
    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    result = {
        "stage": compiled.name,
        "label": label,
        "close": close,
        "atol": ATOL,
        "rtol": RTOL,
        "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        **stats,
    }
    write_json(compiled.stage_dir / f"correctness_{label}.json", result)
    if (
        not close
        or stats["nan_count"] != 0
        or stats["inf_count"] != 0
        or result["expected_nonzero_count"] == 0
    ):
        raise AssertionError(f"correctness failure: {result}")
    print(
        f"{compiled.name.upper()} CORRECTNESS {label.upper()} PASS "
        f"max_abs={stats['max_abs_diff']}",
        flush=True,
    )
    return result


def benchmark_round(
    compiled: CompiledStage,
    inputs,
    expected: torch.Tensor,
    round_index: int,
    args: argparse.Namespace,
) -> dict:
    raw_path = compiled.stage_dir / f"round_{round_index:02d}.json"
    actual = torch.full_like(expected, float("nan"))
    returned = call_kernel(
        compiled,
        inputs,
        actual,
        "benchmark",
        {
            "CHUNKSCAN_BENCH_RESULT_PATH": str(raw_path),
            "CHUNKSCAN_BENCH_WARMUP": str(args.warmup),
            "CHUNKSCAN_BENCH_MIN_SECONDS": str(args.min_seconds),
            "CHUNKSCAN_BENCH_MIN_RUNS": str(args.min_runs),
        },
    )
    if returned not in (None, 0):
        raise RuntimeError(
            f"{compiled.name} round {round_index}: host call returned {returned}"
        )
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    result = json.loads(raw_path.read_text(encoding="utf-8"))

    required_keys = {
        "status",
        "sample_count",
        "measurement_wall_seconds",
        "p50_us",
        "p95_us",
        "cv",
        "samples_us",
    }
    missing = sorted(required_keys - result.keys())
    if missing:
        raise AssertionError(f"{raw_path}: missing keys {missing}")
    if result["status"] != "PASS":
        raise AssertionError(f"{raw_path}: status is not PASS")
    if int(result["sample_count"]) != len(result["samples_us"]):
        raise AssertionError(f"{raw_path}: sample count mismatch")
    if int(result["sample_count"]) < args.min_runs:
        raise AssertionError(f"{raw_path}: too few samples")
    if float(result["measurement_wall_seconds"]) + 0.01 < args.min_seconds:
        raise AssertionError(f"{raw_path}: measurement duration is too short")

    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    result["round"] = round_index
    result["post_round_close"] = close
    result["post_round_stats"] = stats
    write_json(raw_path, result)
    if not close or stats["nan_count"] or stats["inf_count"]:
        raise AssertionError(
            f"{compiled.name} round {round_index}: post-round mismatch {stats}"
        )
    print(
        f"{compiled.name.upper()} ROUND {round_index} PASS "
        f"p50_us={result['p50_us']:.3f} "
        f"p95_us={result['p95_us']:.3f} cv={result['cv']:.6f}",
        flush=True,
    )
    return result


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0.0:
        return 0.0
    return statistics.pstdev(values) / mean


def summarize_stage(rounds: list[dict]) -> dict:
    p50_values = [float(item["p50_us"]) for item in rounds]
    p95_values = [float(item["p95_us"]) for item in rounds]
    median_p50 = statistics.median(p50_values)
    round_cv = coefficient_of_variation(p50_values)
    if round_cv > 0.10:
        stability = "INVALID_CV_GT_10_PERCENT"
    elif round_cv > 0.05:
        stability = "WARN_CV_GT_5_PERCENT"
    else:
        stability = "STABLE"
    return {
        "round_count": len(rounds),
        "median_of_round_p50_us": median_p50,
        "median_of_round_p95_us": statistics.median(p95_values),
        "min_round_p50_us": min(p50_values),
        "max_round_p50_us": max(p50_values),
        "round_p50_cv": round_cv,
        "stability": stability,
        "total_sample_count": sum(int(item["sample_count"]) for item in rounds),
        "total_measurement_seconds": sum(
            float(item["measurement_wall_seconds"]) for item in rounds
        ),
    }


def write_samples_jsonl(
    path: Path,
    round_results: Mapping[str, list[dict]],
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for stage, rounds in round_results.items():
            for round_result in rounds:
                round_index = int(round_result["round"])
                for sample_index, value in enumerate(round_result["samples_us"]):
                    output.write(
                        json.dumps(
                            {
                                "stage": stage,
                                "round": round_index,
                                "sample": sample_index,
                                "latency_us": value,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )


def write_summary_csv(path: Path, summaries: Mapping[str, dict]) -> None:
    fields = (
        "stage",
        "round_count",
        "median_of_round_p50_us",
        "median_of_round_p95_us",
        "min_round_p50_us",
        "max_round_p50_us",
        "round_p50_cv",
        "stability",
        "pipeline_speedup_vs_s1",
        "total_sample_count",
        "total_measurement_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for stage in STAGE_NAMES:
            if stage not in summaries:
                continue
            writer.writerow({"stage": stage, **summaries[stage]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P8 single-core BM1690 ChunkScan benchmark"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("all",) + STAGE_NAMES,
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--min-runs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.rounds = 1
        args.warmup = 5
        args.min_seconds = 0.5
        args.min_runs = 20
    if args.rounds <= 0 or args.warmup <= 0 or args.min_seconds <= 0:
        parser.error("rounds, warmup, and min-seconds must be positive")
    if args.min_runs <= 0:
        parser.error("min-runs must be positive")
    args.resolved_stages = STAGE_NAMES if args.stage == "all" else (args.stage,)
    return args


def main() -> None:
    args = parse_args()
    paths = require_environment()
    run_dir = ARTIFACT_ROOT / "runs" / utc_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "environment.json", collect_environment(paths, args))

    compiled = {
        stage: compile_stage(run_dir, stage) for stage in args.resolved_stages
    }
    if args.compile_only:
        result = {
            "status": "COMPILE_PASS",
            "run_dir": str(run_dir),
            "stages": list(compiled),
            "performance_claim": "none; no TPU kernel was launched",
        }
        write_json(run_dir / "result.json", result)
        write_json(ARTIFACT_ROOT / "result.json", result)
        print(f"P8 COMPILE PASS: {run_dir}", flush=True)
        return

    inputs = _variant_inputs(_make_inputs(seed=20260910), "all_terms")
    expected = chunk_scan_reference(*inputs)
    correctness_before = {
        stage: correctness_check(item, inputs, expected, "before")
        for stage, item in compiled.items()
    }

    round_results: Dict[str, list[dict]] = {stage: [] for stage in compiled}
    for round_index in range(1, args.rounds + 1):
        order = list(compiled)
        random.Random(20260921 + round_index).shuffle(order)
        print(f"ROUND {round_index} ORDER: {','.join(order)}", flush=True)
        for stage in order:
            round_results[stage].append(
                benchmark_round(
                    compiled[stage], inputs, expected, round_index, args
                )
            )

    correctness_after = {
        stage: correctness_check(item, inputs, expected, "after")
        for stage, item in compiled.items()
    }

    summaries = {
        stage: summarize_stage(rounds)
        for stage, rounds in round_results.items()
    }
    if "s1" in summaries:
        baseline = float(summaries["s1"]["median_of_round_p50_us"])
        for stage, summary in summaries.items():
            summary["pipeline_speedup_vs_s1"] = (
                baseline / float(summary["median_of_round_p50_us"])
            )
    else:
        for summary in summaries.values():
            summary["pipeline_speedup_vs_s1"] = None

    unstable = [
        stage
        for stage, summary in summaries.items()
        if summary["stability"] == "INVALID_CV_GT_10_PERCENT"
    ]
    status = "PASS" if not unstable else "FAIL_UNSTABLE"
    result = {
        "status": status,
        "run_dir": str(run_dir),
        "shape": {
            "B": 1,
            "S": 128,
            "Ck": 2,
            "L": 64,
            "G": 1,
            "H": 1,
            "P": 64,
            "N": 128,
        },
        "timer": "steady_clock_sync_kernel_call",
        "measurement_scope": (
            "includes launch, device execution, and stream sync; excludes "
            "compile, module load, allocation, H2D, and D2H"
        ),
        "rounds": args.rounds,
        "warmup_per_round": args.warmup,
        "minimum_seconds_per_round": args.min_seconds,
        "correctness_before": correctness_before,
        "correctness_after": correctness_after,
        "summaries": summaries,
        "unstable_stages": unstable,
        "performance_claim_scope": (
            "preliminary fixed-shape single-core comparison; not multi-core, "
            "not async throughput, and not proof of physical engine overlap"
        ),
    }
    write_samples_jsonl(run_dir / "samples.jsonl", round_results)
    write_json(run_dir / "summary.json", summaries)
    write_summary_csv(run_dir / "summary.csv", summaries)
    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_ROOT / "result.json", result)

    if unstable:
        raise SystemExit(
            "P8 FAIL_UNSTABLE: round p50 CV exceeds 10% for "
            + ", ".join(unstable)
            + f"; evidence: {run_dir}"
        )
    print(f"P8 PASS: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
