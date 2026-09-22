"""P10 synchronized-call performance matrix for multi-core ChunkScan.

The formal matrix covers S1/S3/P6, 1/2/4/8 cores, and the R1/R2/R3 project
shapes.  Compilation, module loading, allocation, and transfers are outside
each recorded sample.  Every sample is one synchronous launch plus stream
synchronization; it is not a pure device-event time or proof of GDMA/BDC
overlap.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_abi_p9 import (
    guards_unchanged,
    logical_task_max_abs,
    make_inputs,
    make_physical_output,
    pack_inputs,
    payload_has_no_sentinel,
    unpack_output,
    variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_p6_p9 import (
    make_chunk_scan_multicore_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ChunkScanMulticoreConfig,
    make_chunk_scan_multicore_s1_kernel,
    physical_shapes,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s3_p9 import (
    make_chunk_scan_multicore_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import SCHEDULE_NAME
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_device_benchmark_p8 import (
    runtime_event_capability,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_s1_p9 import (
    ALLOWED_CORE_COUNTS,
    ROOT,
    parse_csv_subset,
    prepare_runtime,
    require_environment,
    run_text,
    static_local_memory_end,
    write_json,
)


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE / "artifacts" / "device_multicore_benchmark_p10"
STAGES = {
    "s1": make_chunk_scan_multicore_s1_kernel,
    "s3": make_chunk_scan_multicore_s3_kernel,
    "p6": make_chunk_scan_multicore_p6_kernel,
}
SHAPES = {
    "r1": ChunkScanMulticoreConfig("r1", 1, 16, 8),
    "r2": ChunkScanMulticoreConfig("r2", 1, 64, 8),
    "r3": ChunkScanMulticoreConfig("r3", 8, 32, 8),
}
ATOL = 1e-2
RTOL = 1e-2
MODULE_LOAD_FAILURE = -5
MODULE_LOAD_RETRIES = 2


@dataclass
class CompiledPoint:
    stage: str
    config: ChunkScanMulticoreConfig
    kernel: object
    point_dir: Path
    runtime_dir: Path
    manifest: Dict[str, object]

    @property
    def name(self) -> str:
        return f"{self.config.name}:{self.stage}"


@dataclass
class ShapeData:
    packed: Tuple[torch.Tensor, ...]
    expected: torch.Tensor


def utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def temporary_environment(
    updates: Mapping[str, str],
) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def collect_environment(
    paths: Mapping[str, Path], args: argparse.Namespace
) -> Dict[str, object]:
    runtime_library = (
        paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"] / "lib/libtpuv7_rt.so"
    )
    lspci = subprocess.run(
        ["lspci", "-nnk"], capture_output=True, text=True
    )
    tool_names = (
        "bm-smi",
        "tpu-smi",
        "tpuv7-profiler",
        "perf",
    )
    tools = {name: shutil.which(name) for name in tool_names}
    event_capability = runtime_event_capability(runtime_library)
    event_capability.pop("used_by_p8", None)
    event_capability["used_by_p10"] = False
    event_capability["note"] = (
        "P10 uses synchronized host-call latency. Event timing is not used "
        "and no physical engine-overlap claim is made."
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": run_text(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD")
        ),
        "repository_status": run_text(
            ("git", "-C", str(ROOT), "status", "--short")
        ).splitlines(),
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
        "riscv_toolchain_root": str(
            paths["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        ),
        "event_capability": event_capability,
        "profiler_tool_inventory": tools,
        "hardware_trace_captured": False,
        "pci_1690_lines": [
            line
            for line in lspci.stdout.splitlines()
            if "1f1c:1690" in line.lower()
            or "sg-host-drv" in line.lower()
        ],
        "benchmark_arguments": {
            "stages": list(args.resolved_stages),
            "shapes": list(args.resolved_shapes),
            "core_counts": list(args.resolved_core_counts),
            "rounds": args.rounds,
            "warmup": args.warmup,
            "min_seconds": args.min_seconds,
            "min_runs": args.min_runs,
            "quick": args.quick,
            "compile_only": args.compile_only,
        },
        "measurement_scope": (
            "one synchronous kernel call; includes launch, device execution "
            "and stream synchronization; excludes compilation, module load, "
            "allocation, H2D and D2H"
        ),
        "physical_overlap_claim": False,
    }


def validate_generated_host_sources(
    label: str,
    kernel_host_source: str,
    benchmark_main_source: str,
) -> None:
    for marker in (
        "CHUNKSCAN_CORE_NUM",
        "block_num = static_cast<uint64_t>(core_num)",
        "apis.data()",
    ):
        if marker not in kernel_host_source:
            raise AssertionError(f"{label}: kernel.cpp lacks {marker!r}")
    if "steady_clock_sync_kernel_call" not in benchmark_main_source:
        raise AssertionError(
            f"{label}: main.cpp lacks 'steady_clock_sync_kernel_call'"
        )


def compile_point(
    run_dir: Path,
    stage: str,
    config: ChunkScanMulticoreConfig,
) -> CompiledPoint:
    stage_dir = run_dir / stage
    point_dir, runtime_dir = prepare_runtime(stage_dir, config.name)
    program = STAGES[stage](config)
    schedule = program.attrs.get("p6_pipeline_schedule")
    if stage == "p6":
        if str(schedule) != SCHEDULE_NAME:
            raise AssertionError("P6 explicit schedule attribute is absent")
    elif schedule is not None:
        raise AssertionError(f"{stage} unexpectedly has a P6 schedule")

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            kernel = tilelang.compile(
                program, out_idx=[7], target="tpu", mode="pcie"
            )
    finally:
        (point_dir / "compile_stdout.log").write_text(
            captured.getvalue(), encoding="utf-8"
        )

    source_path = runtime_dir / "kernel.c"
    kernel_host_path = runtime_dir / "kernel.cpp"
    benchmark_main_path = runtime_dir / "main.cpp"
    header_path = runtime_dir / "kernel.h"
    device_library = runtime_dir / "libkernel.so"
    host_library = runtime_dir / "main.so"
    for path in (
        source_path,
        kernel_host_path,
        benchmark_main_path,
        header_path,
        device_library,
        host_library,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = source_path.read_text(encoding="utf-8")
    kernel_host_source = kernel_host_path.read_text(encoding="utf-8")
    benchmark_main_source = benchmark_main_path.read_text(encoding="utf-8")
    header = header_path.read_text(encoding="utf-8")
    for marker in (
        "tpu_workitem_index()",
        "tpu_workitem_num()",
        f"for (int task = 0; task < {config.task_count}; ++task)",
    ):
        if marker not in source:
            raise AssertionError(
                f"{config.name}/{stage}: source lacks {marker!r}"
            )
    expected_markers = 0 if stage == "s1" else 1
    starts = source.count("tpu_parallel_start(")
    ends = source.count("tpu_parallel_end(")
    if (starts, ends) != (expected_markers, expected_markers):
        raise AssertionError(
            f"{config.name}/{stage}: pipeline markers {starts}/{ends}"
        )
    validate_generated_host_sources(
        f"{config.name}/{stage}",
        kernel_host_source,
        benchmark_main_source,
    )
    for index in range(1, 9):
        if f"ptr_v{index};" not in header:
            raise AssertionError(
                f"{config.name}/{stage}: unexpected kernel ABI"
            )

    device_file = run_text(("file", str(device_library)))
    host_file = run_text(("file", str(host_library)))
    linked = run_text(("ldd", str(host_library)))
    expected_runtime = str(
        Path(os.environ["CHUNKSCAN_DEVICE_RUNTIME_ROOT"])
        / "lib/libtpuv7_rt.so"
    )
    if "RISC-V" not in device_file or "x86-64" not in host_file:
        raise AssertionError(f"{config.name}/{stage}: wrong ELF architecture")
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(
            f"{config.name}/{stage}: wrong runtime:\n{linked}"
        )

    local_end = static_local_memory_end(source)
    if local_end is None or local_end > 256 * 1024:
        raise AssertionError(
            f"{config.name}/{stage}: invalid LMEM {local_end}"
        )
    manifest: Dict[str, object] = {
        "status": "COMPILE_PASS",
        "shape_name": config.name,
        "stage": stage,
        "logical_shape": config.logical_shape(),
        "task_count": config.task_count,
        "physical_shapes": {
            name: list(shape)
            for name, shape in physical_shapes(config).items()
        },
        "kernel_source_sha256": sha256_file(source_path),
        "kernel_host_source_sha256": sha256_file(kernel_host_path),
        "benchmark_main_source_sha256": sha256_file(benchmark_main_path),
        "device_library_sha256": sha256_file(device_library),
        "host_library_sha256": sha256_file(host_library),
        "device_file": device_file,
        "host_file": host_file,
        "linked_libraries": linked.splitlines(),
        "pipeline_start_count": starts,
        "pipeline_end_count": ends,
        "explicit_schedule": SCHEDULE_NAME if stage == "p6" else None,
        "static_local_memory_end_bytes": local_end,
        "bm1690_local_memory_limit_bytes": 256 * 1024,
    }
    write_json(point_dir / "compile_manifest.json", manifest)
    print(
        f"{config.name.upper()} {stage.upper()} COMPILE PASS "
        f"tasks={config.task_count}",
        flush=True,
    )
    return CompiledPoint(
        stage, config, kernel, point_dir, runtime_dir, manifest
    )


def make_shape_data(config: ChunkScanMulticoreConfig, seed: int) -> ShapeData:
    logical = variant_inputs(make_inputs(config, seed), "all_terms")
    expected = chunk_scan_reference(*logical)
    if not torch.all(logical_task_max_abs(config, expected) > 0):
        raise AssertionError(f"{config.name}: reference has a zero task")
    return ShapeData(pack_inputs(config, logical), expected)


def call_host(
    compiled: CompiledPoint,
    core_num: int,
    packed: Tuple[torch.Tensor, ...],
    output: torch.Tensor,
    mode: str,
    extra_environment: Mapping[str, str],
    retry_events: list[Dict[str, object]],
) -> object:
    updates = {
        "PPL_KERNEL_PATH": str(compiled.runtime_dir / "libkernel.so"),
        "CHUNKSCAN_BENCH_MODE": mode,
        "CHUNKSCAN_BENCH_STAGE": (
            f"{compiled.config.name}:{compiled.stage}:c{core_num}"
        ),
        "CHUNKSCAN_CORE_NUM": str(core_num),
        **extra_environment,
    }
    for attempt in range(MODULE_LOAD_RETRIES + 1):
        with temporary_environment(updates):
            returned = compiled.kernel(*packed, output)
        if returned != MODULE_LOAD_FAILURE:
            return returned
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "shape": compiled.config.name,
            "stage": compiled.stage,
            "core_num": core_num,
            "mode": mode,
            "attempt": attempt + 1,
            "return_code": int(returned),
        }
        retry_events.append(event)
        if attempt == MODULE_LOAD_RETRIES:
            return returned
        time.sleep(0.25)
    raise AssertionError("unreachable")


def validate_output(
    compiled: CompiledPoint,
    core_num: int,
    output: torch.Tensor,
    expected: torch.Tensor,
    returned: object,
    label: str,
) -> Dict[str, object]:
    actual = unpack_output(compiled.config, output)
    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    result: Dict[str, object] = {
        "shape": compiled.config.name,
        "stage": compiled.stage,
        "core_num": core_num,
        "label": label,
        "close": close,
        "guard_rows_unchanged": guards_unchanged(output),
        "payload_fully_covered": payload_has_no_sentinel(output),
        "all_expected_tasks_nonzero": bool(
            torch.all(
                logical_task_max_abs(compiled.config, expected) > 0
            ).item()
        ),
        "output_sha256": tensor_sha256(actual),
        "returned": None if returned is None else int(returned),
        **stats,
    }
    if not (
        close
        and result["guard_rows_unchanged"]
        and result["payload_fully_covered"]
        and result["all_expected_tasks_nonzero"]
        and returned in (None, 0)
        and stats["nan_count"] == 0
        and stats["inf_count"] == 0
    ):
        raise AssertionError(result)
    return result


def correctness_check(
    compiled: CompiledPoint,
    core_num: int,
    data: ShapeData,
    label: str,
    retry_events: list[Dict[str, object]],
) -> Dict[str, object]:
    output = make_physical_output(compiled.config)
    returned = call_host(
        compiled,
        core_num,
        data.packed,
        output,
        "correctness",
        {},
        retry_events,
    )
    result = validate_output(
        compiled, core_num, output, data.expected, returned, label
    )
    write_json(
        compiled.point_dir / f"correctness_c{core_num}_{label}.json",
        result,
    )
    print(
        f"{compiled.config.name.upper()} {compiled.stage.upper()} "
        f"{core_num}-CORE CORRECTNESS {label.upper()} PASS "
        f"max_abs={result['max_abs_diff']}",
        flush=True,
    )
    return result


def benchmark_round(
    compiled: CompiledPoint,
    core_num: int,
    data: ShapeData,
    round_index: int,
    args: argparse.Namespace,
    retry_events: list[Dict[str, object]],
) -> Dict[str, object]:
    raw_path = (
        compiled.point_dir
        / f"c{core_num}_round_{round_index:02d}.json"
    )
    if raw_path.exists():
        raw_path.unlink()
    output = make_physical_output(compiled.config)
    returned = call_host(
        compiled,
        core_num,
        data.packed,
        output,
        "benchmark",
        {
            "CHUNKSCAN_BENCH_RESULT_PATH": str(raw_path),
            "CHUNKSCAN_BENCH_WARMUP": str(args.warmup),
            "CHUNKSCAN_BENCH_MIN_SECONDS": str(args.min_seconds),
            "CHUNKSCAN_BENCH_MIN_RUNS": str(args.min_runs),
        },
        retry_events,
    )
    if returned not in (None, 0):
        raise RuntimeError(
            f"{compiled.name}/c{core_num}/round{round_index}: "
            f"host returned {returned}"
        )
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "sample_count",
        "measurement_wall_seconds",
        "p50_us",
        "p95_us",
        "cv",
        "samples_us",
    ):
        if key not in result:
            raise AssertionError(f"{raw_path}: missing {key}")
    if result["status"] != "PASS":
        raise AssertionError(f"{raw_path}: status is not PASS")
    if int(result["sample_count"]) != len(result["samples_us"]):
        raise AssertionError(f"{raw_path}: sample count mismatch")
    if int(result["sample_count"]) < args.min_runs:
        raise AssertionError(f"{raw_path}: too few samples")
    if float(result["measurement_wall_seconds"]) + 0.01 < args.min_seconds:
        raise AssertionError(f"{raw_path}: duration is too short")

    validation = validate_output(
        compiled,
        core_num,
        output,
        data.expected,
        returned,
        f"round_{round_index}",
    )
    result.update({
        "shape": compiled.config.name,
        "stage": compiled.stage,
        "core_num": core_num,
        "round": round_index,
        "post_round_correctness": validation,
    })
    write_json(raw_path, result)
    print(
        f"{compiled.config.name.upper()} {compiled.stage.upper()} "
        f"{core_num}-CORE ROUND {round_index} PASS "
        f"p50_us={result['p50_us']:.3f} "
        f"p95_us={result['p95_us']:.3f} cv={result['cv']:.6f}",
        flush=True,
    )
    return result


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return 0.0 if mean == 0.0 else statistics.pstdev(values) / mean


def summarize_point(
    shape: str,
    stage: str,
    core_num: int,
    rounds: list[Dict[str, object]],
    config: ChunkScanMulticoreConfig,
) -> Dict[str, object]:
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
    output_elements = (
        config.batch * config.seqlen * config.nheads * config.headdim
    )
    return {
        "shape": shape,
        "stage": stage,
        "core_num": core_num,
        "task_count": config.task_count,
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
        "output_elements": output_elements,
        "output_elements_per_second": (
            output_elements / (median_p50 * 1e-6)
        ),
    }


def add_comparative_metrics(
    summaries: Dict[str, Dict[str, object]],
    shape_names: Tuple[str, ...],
    stages: Tuple[str, ...],
    core_counts: Tuple[int, ...],
) -> Dict[str, Dict[str, object]]:
    best: Dict[str, Dict[str, object]] = {}
    for shape in shape_names:
        available = [
            summaries[f"{shape}:{stage}:c{core}"]
            for stage in stages
            for core in core_counts
        ]
        best_point = min(
            available,
            key=lambda item: float(item["median_of_round_p50_us"]),
        )
        best[shape] = {
            "stage": best_point["stage"],
            "core_num": best_point["core_num"],
            "median_p50_us": best_point["median_of_round_p50_us"],
        }
        for stage in stages:
            one_key = f"{shape}:{stage}:c1"
            one_latency = float(
                summaries[one_key]["median_of_round_p50_us"]
            )
            s1_one = float(
                summaries[f"{shape}:s1:c1"]["median_of_round_p50_us"]
            ) if "s1" in stages else None
            for core in core_counts:
                key = f"{shape}:{stage}:c{core}"
                latency = float(summaries[key]["median_of_round_p50_us"])
                speedup = one_latency / latency
                summaries[key]["multicore_speedup_vs_same_stage_c1"] = speedup
                summaries[key]["parallel_efficiency"] = speedup / core
                summaries[key]["combined_gain_vs_s1_c1"] = (
                    None if s1_one is None else s1_one / latency
                )
                s1_key = f"{shape}:s1:c{core}"
                summaries[key]["pipeline_gain_vs_s1_same_core"] = (
                    None
                    if s1_key not in summaries
                    else float(summaries[s1_key]["median_of_round_p50_us"])
                    / latency
                )
    return best


def write_samples_jsonl(
    path: Path,
    round_results: Mapping[str, list[Dict[str, object]]],
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for point, rounds in round_results.items():
            for round_result in rounds:
                for sample_index, value in enumerate(
                    round_result["samples_us"]
                ):
                    output.write(
                        json.dumps({
                            "point": point,
                            "shape": round_result["shape"],
                            "stage": round_result["stage"],
                            "core_num": round_result["core_num"],
                            "round": round_result["round"],
                            "sample": sample_index,
                            "latency_us": value,
                        }, sort_keys=True)
                        + "\n"
                    )


def write_summary_csv(
    path: Path, summaries: Mapping[str, Dict[str, object]]
) -> None:
    fields = (
        "shape",
        "stage",
        "core_num",
        "task_count",
        "round_count",
        "median_of_round_p50_us",
        "median_of_round_p95_us",
        "round_p50_cv",
        "stability",
        "multicore_speedup_vs_same_stage_c1",
        "parallel_efficiency",
        "pipeline_gain_vs_s1_same_core",
        "combined_gain_vs_s1_c1",
        "output_elements_per_second",
        "total_sample_count",
        "total_measurement_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key in sorted(summaries):
            writer.writerow({name: summaries[key].get(name) for name in fields})


def write_report_markdown(
    path: Path,
    status: str,
    summaries: Mapping[str, Dict[str, object]],
    best: Mapping[str, Dict[str, object]],
    retry_count: int,
    invalid: list[str],
    warnings: list[str],
) -> None:
    lines = [
        "# P10 BM1690 Multi-core ChunkScan Performance Report",
        "",
        f"Status: `{status}`",
        "",
        "Timing is synchronized host-call latency. It includes launch, device ",
        "execution, and stream synchronization, but excludes compilation, ",
        "module loading, allocation, H2D, and D2H. No physical GDMA/BDC ",
        "overlap is claimed because no hardware trace was captured.",
        "",
        f"Runtime module-load retries: `{retry_count}`",
        "",
        "## Best configuration by shape",
        "",
        "| Shape | Stage | Cores | Median p50 (us) |",
        "| --- | --- | ---: | ---: |",
    ]
    for shape, item in best.items():
        lines.append(
            f"| {shape} | {item['stage']} | {item['core_num']} | "
            f"{float(item['median_p50_us']):.3f} |"
        )
    lines.extend([
        "",
        "## Complete matrix",
        "",
        "| Shape | Stage | Cores | p50 us | CV | Pipeline gain | "
        "Multi-core speedup | Efficiency | Combined gain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in sorted(summaries):
        item = summaries[key]
        pipeline = item.get("pipeline_gain_vs_s1_same_core")
        combined = item.get("combined_gain_vs_s1_c1")
        lines.append(
            f"| {item['shape']} | {item['stage']} | {item['core_num']} | "
            f"{float(item['median_of_round_p50_us']):.3f} | "
            f"{float(item['round_p50_cv']):.4f} | "
            f"{'N/A' if pipeline is None else f'{float(pipeline):.4f}'} | "
            f"{float(item['multicore_speedup_vs_same_stage_c1']):.4f} | "
            f"{float(item['parallel_efficiency']):.4f} | "
            f"{'N/A' if combined is None else f'{float(combined):.4f}'} |"
        )
    lines.extend([
        "",
        "## Stability",
        "",
        f"Invalid points (round-p50 CV > 10%): `{invalid}`",
        "",
        f"Warning points (5% < round-p50 CV <= 10%): `{warnings}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P10 BM1690 multi-core ChunkScan performance matrix"
    )
    parser.add_argument("--stages", default="s1,s3,p6")
    parser.add_argument("--shapes", default="r1,r2,r3")
    parser.add_argument("--core-counts", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--min-runs", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.resolved_stages = parse_csv_subset(
        parser, args.stages, tuple(STAGES), "--stages"
    )
    args.resolved_shapes = parse_csv_subset(
        parser, args.shapes, tuple(SHAPES), "--shapes"
    )
    core_text = parse_csv_subset(
        parser,
        args.core_counts,
        tuple(str(value) for value in ALLOWED_CORE_COUNTS),
        "--core-counts",
    )
    args.resolved_core_counts = tuple(int(value) for value in core_text)
    if args.quick:
        args.rounds = 1
        args.warmup = 5
        args.min_seconds = 0.5
        args.min_runs = 5
    if (
        args.rounds <= 0
        or args.warmup <= 0
        or args.min_seconds <= 0
        or args.min_runs <= 0
    ):
        parser.error("rounds, warmup, min-seconds, and min-runs must be positive")
    if not args.compile_only and args.resolved_core_counts[0] != 1:
        parser.error("benchmark execution requires core count 1 first")
    return args


def main() -> None:
    args = parse_args()
    paths = require_environment()
    run_dir = ARTIFACT_ROOT / "runs" / utc_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "environment.json", collect_environment(paths, args))

    compiled = {
        (shape_name, stage): compile_point(
            run_dir, stage, SHAPES[shape_name]
        )
        for stage in args.resolved_stages
        for shape_name in args.resolved_shapes
    }
    if args.compile_only:
        result = {
            "status": "COMPILE_PASS",
            "stage": "P10",
            "run_dir": str(run_dir),
            "compiled_points": {
                f"{shape}:{stage}": item.manifest
                for (shape, stage), item in compiled.items()
            },
            "performance_validated": False,
        }
        write_json(run_dir / "result.json", result)
        write_json(ARTIFACT_ROOT / "result.json", result)
        print(f"P10 COMPILE-ONLY PASS: {run_dir}", flush=True)
        return

    shape_data = {
        name: make_shape_data(SHAPES[name], 20260924 + index)
        for index, name in enumerate(args.resolved_shapes)
    }
    retry_events: list[Dict[str, object]] = []
    correctness_before: Dict[str, Dict[str, object]] = {}
    for shape in args.resolved_shapes:
        for stage in args.resolved_stages:
            for core in args.resolved_core_counts:
                key = f"{shape}:{stage}:c{core}"
                correctness_before[key] = correctness_check(
                    compiled[(shape, stage)],
                    core,
                    shape_data[shape],
                    "before",
                    retry_events,
                )

    for shape in args.resolved_shapes:
        for core in args.resolved_core_counts:
            s3_key = f"{shape}:s3:c{core}"
            p6_key = f"{shape}:p6:c{core}"
            if s3_key in correctness_before and p6_key in correctness_before:
                if (
                    correctness_before[s3_key]["output_sha256"]
                    != correctness_before[p6_key]["output_sha256"]
                ):
                    raise AssertionError(f"S3/P6 differ before: {shape}/c{core}")

    round_results: Dict[str, list[Dict[str, object]]] = {
        f"{shape}:{stage}:c{core}": []
        for shape in args.resolved_shapes
        for stage in args.resolved_stages
        for core in args.resolved_core_counts
    }
    points = list(round_results)
    for round_index in range(1, args.rounds + 1):
        order = list(points)
        random.Random(20260924 + round_index).shuffle(order)
        print(f"ROUND {round_index} ORDER: {','.join(order)}", flush=True)
        for key in order:
            shape, stage, core_text = key.split(":")
            core = int(core_text[1:])
            round_results[key].append(benchmark_round(
                compiled[(shape, stage)],
                core,
                shape_data[shape],
                round_index,
                args,
                retry_events,
            ))

    correctness_after: Dict[str, Dict[str, object]] = {}
    for shape in args.resolved_shapes:
        for stage in args.resolved_stages:
            for core in args.resolved_core_counts:
                key = f"{shape}:{stage}:c{core}"
                correctness_after[key] = correctness_check(
                    compiled[(shape, stage)],
                    core,
                    shape_data[shape],
                    "after",
                    retry_events,
                )
                if (
                    correctness_after[key]["output_sha256"]
                    != correctness_before[key]["output_sha256"]
                ):
                    raise AssertionError(f"before/after differ: {key}")

    summaries = {
        key: summarize_point(
            key.split(":")[0],
            key.split(":")[1],
            int(key.split(":")[2][1:]),
            rounds,
            SHAPES[key.split(":")[0]],
        )
        for key, rounds in round_results.items()
    }
    best = add_comparative_metrics(
        summaries,
        args.resolved_shapes,
        args.resolved_stages,
        args.resolved_core_counts,
    )
    invalid = [
        key
        for key, summary in summaries.items()
        if summary["stability"] == "INVALID_CV_GT_10_PERCENT"
    ]
    warnings = [
        key
        for key, summary in summaries.items()
        if summary["stability"] == "WARN_CV_GT_5_PERCENT"
    ]
    full_protocol = (
        tuple(args.resolved_stages) == tuple(STAGES)
        and tuple(args.resolved_shapes) == tuple(SHAPES)
        and tuple(args.resolved_core_counts) == ALLOWED_CORE_COUNTS
        and args.rounds >= 7
        and args.warmup >= 100
        and args.min_seconds >= 5.0
        and args.min_runs >= 20
    )
    status = (
        "FAIL_UNSTABLE"
        if invalid
        else "PASS" if full_protocol else "SUBSET_PASS"
    )
    result = {
        "status": status,
        "stage": "P10",
        "run_dir": str(run_dir),
        "full_protocol": full_protocol,
        "timer": "steady_clock_sync_kernel_call",
        "measurement_scope": (
            "includes launch, device execution and stream sync; excludes "
            "compile, module load, allocation, H2D and D2H"
        ),
        "correctness_before": correctness_before,
        "correctness_after": correctness_after,
        "summaries": summaries,
        "best_configuration_by_shape": best,
        "invalid_points": invalid,
        "warning_points": warnings,
        "runtime_module_load_retry_count": len(retry_events),
        "runtime_module_load_retries": retry_events,
        "physical_gdma_bdc_overlap_validated": False,
        "trace_status": "NOT_CAPTURED",
        "performance_validated": status == "PASS",
        "claim_boundary": (
            "BM1690 synchronized-call latency and scaling only; no pure "
            "device-event timing and no physical GDMA/BDC overlap claim"
        ),
    }
    write_samples_jsonl(run_dir / "samples.jsonl", round_results)
    write_json(run_dir / "summary.json", summaries)
    write_summary_csv(run_dir / "summary.csv", summaries)
    write_report_markdown(
        run_dir / "report.md",
        status,
        summaries,
        best,
        len(retry_events),
        invalid,
        warnings,
    )
    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_ROOT / "result.json", result)

    if invalid:
        raise SystemExit(
            "P10 FAIL_UNSTABLE: round p50 CV exceeds 10% for "
            + ", ".join(invalid)
            + f"; evidence: {run_dir}"
        )
    if status == "PASS":
        print(f"P10 PASS: {run_dir}", flush=True)
    else:
        print(f"P10 SUBSET PASS: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
