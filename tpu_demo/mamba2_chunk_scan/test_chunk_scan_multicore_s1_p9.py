"""Compile and validate task-major S1 on BM1690 with 1/2/4/8 cores."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_abi_p9 import (
    Inputs,
    guards_unchanged,
    logical_task_max_abs,
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
    physical_shapes,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT_ROOT = HERE / "artifacts" / "device_multicore_s1_p9"
ALLOWED_CORE_COUNTS = (1, 2, 4, 8)
ATOL = 1e-2
RTOL = 1e-2

SHAPES = {
    "r0": ChunkScanMulticoreConfig("r0", batch=1, nchunks=2, nheads=1),
    "r1": ChunkScanMulticoreConfig("r1", batch=1, nchunks=16, nheads=8),
    "edge3": ChunkScanMulticoreConfig(
        "edge3", batch=1, nchunks=1, nheads=3
    ),
    "edge10": ChunkScanMulticoreConfig(
        "edge10", batch=1, nchunks=2, nheads=5
    ),
}
CASE_NAMES = (
    "residual_only",
    "state_only",
    "scan_only",
    "all_terms",
    "residual_only_negative_D",
    "causal_upper_triangle_poison",
)


@dataclass
class CompiledShape:
    config: ChunkScanMulticoreConfig
    kernel: object
    shape_dir: Path
    runtime_dir: Path
    manifest: Dict[str, object]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run_text(command: Tuple[str, ...]) -> str:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_environment() -> Dict[str, Path]:
    variables = (
        "PPL_PROJECT_ROOT",
        "CHUNKSCAN_DEVICE_RUNTIME_ROOT",
        "CHUNKSCAN_RISCV_TOOLCHAIN_ROOT",
    )
    values: Dict[str, Path] = {}
    for variable in variables:
        value = os.environ.get(variable)
        if not value:
            raise EnvironmentError(f"{variable} is not set")
        values[variable] = Path(value).expanduser().resolve()

    required = (
        values["PPL_PROJECT_ROOT"] / "runtime/bm1690/lib/libbm1690.a",
        values["PPL_PROJECT_ROOT"]
        / "runtime/customize/src/ppl_helper.c",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]
        / "include/tpuv7_rt.h",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]
        / "lib/libtpuv7_rt.so",
        values["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        / "bin/riscv64-unknown-linux-gnu-gcc",
        ROOT / "build/libtilelang_module.so",
        ROOT / "build/tvm/libtvm.so",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return values


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


def static_local_memory_end(source: str) -> int | None:
    dtype_bytes = {
        "DT_FP16": 2,
        "DT_BFP16": 2,
        "DT_FP32": 4,
        "DT_INT32": 4,
        "DT_INT8": 1,
        "DT_FP8E4M3": 1,
    }
    ends = []
    for shape_text, address, dtype in re.findall(
        r"\.shape\s*=\s*\{([^}]+)\}.*?\.addr\s*=\s*(\d+)"
        r".*?\.dtype\s*=\s*(DT_[A-Z0-9]+)",
        source,
    ):
        elements = 1
        for dimension in re.findall(r"\d+", shape_text):
            elements *= int(dimension)
        if dtype in dtype_bytes:
            ends.append(
                int(address) + max(64, elements * dtype_bytes[dtype])
            )
    return max(ends, default=None)


def prepare_runtime(run_dir: Path, shape_name: str) -> Tuple[Path, Path]:
    shape_dir = run_dir / shape_name
    runtime_dir = shape_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        ROOT / "src/tl_templates/tpu/kernel_template.h",
        runtime_dir / "kernel_template.h",
    )
    shutil.copy2(
        HERE / "kernel_template_device_multicore.cpp",
        runtime_dir / "kernel_template.cpp",
    )
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
    return shape_dir, runtime_dir


def compile_shape(
    run_dir: Path,
    config: ChunkScanMulticoreConfig,
) -> CompiledShape:
    shape_dir, runtime_dir = prepare_runtime(run_dir, config.name)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            kernel = tilelang.compile(
                make_chunk_scan_multicore_s1_kernel(config),
                out_idx=[7],
                target="tpu",
                mode="pcie",
            )
    finally:
        (shape_dir / "compile_stdout.log").write_text(
            captured.getvalue(), encoding="utf-8"
        )

    source_path = runtime_dir / "kernel.c"
    host_path = runtime_dir / "kernel.cpp"
    header_path = runtime_dir / "kernel.h"
    device_library = runtime_dir / "libkernel.so"
    host_library = runtime_dir / "main.so"
    for path in (
        source_path,
        host_path,
        header_path,
        device_library,
        host_library,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = source_path.read_text(encoding="utf-8")
    host_source = host_path.read_text(encoding="utf-8")
    header = header_path.read_text(encoding="utf-8")
    required_source = (
        "tpu_workitem_index()",
        "tpu_workitem_num()",
        f"for (int task = 0; task < {config.task_count}; ++task)",
    )
    for marker in required_source:
        if marker not in source:
            raise AssertionError(
                f"{config.name}: device source lacks {marker!r}"
            )
    for abstract_name in ("ppl.workitem_index", "ppl.workitem_num"):
        if abstract_name in source:
            raise AssertionError(
                f"{config.name}: abstract call remains: {abstract_name}"
            )
    if source.count("tpu_parallel_start(") or source.count(
        "tpu_parallel_end("
    ):
        raise AssertionError(f"{config.name}: S1 contains pipeline markers")
    for marker in (
        "CHUNKSCAN_CORE_NUM",
        "block_num = static_cast<uint64_t>(core_num)",
        "apis.data()",
    ):
        if marker not in host_source:
            raise AssertionError(
                f"{config.name}: host source lacks {marker!r}"
            )
    for index in range(1, 9):
        if f"ptr_v{index};" not in header:
            raise AssertionError(
                f"{config.name}: unexpected kernel ABI in header"
            )

    device_file = run_text(("file", str(device_library)))
    host_file = run_text(("file", str(host_library)))
    linked = run_text(("ldd", str(host_library)))
    expected_runtime = str(
        Path(os.environ["CHUNKSCAN_DEVICE_RUNTIME_ROOT"])
        / "lib/libtpuv7_rt.so"
    )
    if "RISC-V" not in device_file or "x86-64" not in host_file:
        raise AssertionError(
            f"{config.name}: wrong ELF architecture:\n"
            f"{device_file}\n{host_file}"
        )
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(
            f"{config.name}: wrong host runtime:\n{linked}"
        )

    local_end = static_local_memory_end(source)
    if local_end is None or local_end > 256 * 1024:
        raise AssertionError(
            f"{config.name}: invalid LMEM high-water mark {local_end}"
        )

    manifest: Dict[str, object] = {
        "status": "COMPILE_PASS",
        "stage": "s1",
        "shape_name": config.name,
        "logical_shape": config.logical_shape(),
        "task_count": config.task_count,
        "task_formula": "task=(batch*Ck+chunk)*H+head",
        "task_owner_formula": "task%workitem_num==workitem_index",
        "physical_shapes": {
            name: list(shape)
            for name, shape in physical_shapes(config).items()
        },
        "task_major_shared_input_policy": "replicate cb and C per task",
        "output_guard_rows_each_side": 1,
        "kernel_source_sha256": sha256_file(source_path),
        "host_source_sha256": sha256_file(host_path),
        "device_library_sha256": sha256_file(device_library),
        "host_library_sha256": sha256_file(host_library),
        "device_file": device_file,
        "host_file": host_file,
        "linked_libraries": linked.splitlines(),
        "pipeline_start_count": 0,
        "pipeline_end_count": 0,
        "workitem_index_call_count": source.count(
            "tpu_workitem_index()"
        ),
        "workitem_num_call_count": source.count("tpu_workitem_num()"),
        "static_local_memory_end_bytes": local_end,
        "bm1690_local_memory_limit_bytes": 256 * 1024,
        "performance_claim": "none; P9.2 correctness only",
    }
    write_json(shape_dir / "compile_manifest.json", manifest)
    print(
        f"{config.name.upper()} S1 COMPILE PASS "
        f"tasks={config.task_count}",
        flush=True,
    )
    return CompiledShape(
        config=config,
        kernel=kernel,
        shape_dir=shape_dir,
        runtime_dir=runtime_dir,
        manifest=manifest,
    )


def build_cases(base: Inputs) -> Dict[str, Inputs]:
    cases = {
        name: variant_inputs(base, name)
        for name in (
            "residual_only",
            "state_only",
            "scan_only",
            "all_terms",
        )
    }
    negative = list(variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    cases["residual_only_negative_D"] = tuple(negative)  # type: ignore[assignment]
    cases["causal_upper_triangle_poison"] = poison_upper_triangle(
        cases["scan_only"]
    )
    return cases


def call_kernel(
    compiled: CompiledShape,
    core_num: int,
    inputs: Inputs,
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    packed = pack_inputs(compiled.config, inputs)
    physical_output = make_physical_output(compiled.config)
    with temporary_environment(
        {
            "PPL_KERNEL_PATH": str(
                compiled.runtime_dir / "libkernel.so"
            ),
            "CHUNKSCAN_BENCH_MODE": "correctness",
            "CHUNKSCAN_CORE_NUM": str(core_num),
        }
    ):
        returned = compiled.kernel(*packed, physical_output)
    logical = unpack_output(compiled.config, physical_output)
    return logical, physical_output, returned


def validate_output(
    compiled: CompiledShape,
    core_num: int,
    case_name: str,
    actual: torch.Tensor,
    physical: torch.Tensor,
    returned: object,
    expected: torch.Tensor,
    single_core: torch.Tensor | None,
) -> Dict[str, object]:
    stats = comparison_stats(actual, expected)
    close = bool(
        torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    guard_ok = guards_unchanged(physical)
    coverage_ok = payload_has_no_sentinel(physical)
    task_signal = logical_task_max_abs(compiled.config, expected)
    all_tasks_nonzero = bool(torch.all(task_signal > 0).item())
    bitwise_single = (
        True if single_core is None else bool(torch.equal(actual, single_core))
    )
    returned_ok = returned in (None, 0)
    result: Dict[str, object] = {
        "case": case_name,
        "core_num": core_num,
        "close": close,
        "guard_rows_unchanged": guard_ok,
        "payload_fully_covered": coverage_ok,
        "all_expected_tasks_nonzero": all_tasks_nonzero,
        "bitwise_equal_to_single_core": bitwise_single,
        "output_sha256": tensor_sha256(actual),
        "returned": None if returned is None else int(returned),
        **stats,
    }
    if not (
        close
        and guard_ok
        and coverage_ok
        and all_tasks_nonzero
        and bitwise_single
        and returned_ok
        and stats["nan_count"] == 0
        and stats["inf_count"] == 0
    ):
        raise AssertionError(
            f"{compiled.config.name}/{core_num}/{case_name}: {result}"
        )
    return result


def run_shape(
    compiled: CompiledShape,
    core_counts: Tuple[int, ...],
    repeat: int,
    seed: int,
) -> Dict[str, object]:
    base = make_inputs(compiled.config, seed)
    cases = build_cases(base)
    expected = {
        name: chunk_scan_reference(*inputs)
        for name, inputs in cases.items()
    }
    if not torch.equal(
        expected["scan_only"],
        expected["causal_upper_triangle_poison"],
    ):
        raise AssertionError(
            f"{compiled.config.name}: CPU causal-poison oracle changed"
        )

    single_core_outputs: Dict[str, torch.Tensor] = {}
    core_results: Dict[str, object] = {}
    for core_num in core_counts:
        case_results: Dict[str, object] = {}
        outputs: Dict[str, torch.Tensor] = {}
        for case_name in CASE_NAMES:
            actual, physical, returned = call_kernel(
                compiled,
                core_num,
                cases[case_name],
            )
            baseline = (
                None
                if core_num == 1
                else single_core_outputs[case_name]
            )
            case_results[case_name] = validate_output(
                compiled,
                core_num,
                case_name,
                actual,
                physical,
                returned,
                expected[case_name],
                baseline,
            )
            outputs[case_name] = actual

        poison_equal = bool(
            torch.equal(
                outputs["scan_only"],
                outputs["causal_upper_triangle_poison"],
            )
        )
        if not poison_equal:
            raise AssertionError(
                f"{compiled.config.name}/{core_num}: poison changed output"
            )
        case_results["causal_upper_triangle_poison"][
            "bitwise_equal_to_clean_scan"
        ] = True

        reference_all_terms = outputs["all_terms"]
        repetitions = [
            {
                "repetition": 1,
                "bitwise_deterministic": True,
                "guard_rows_unchanged": True,
                "payload_fully_covered": True,
                "output_sha256": tensor_sha256(reference_all_terms),
            }
        ]
        for repetition in range(2, repeat + 1):
            actual, physical, returned = call_kernel(
                compiled,
                core_num,
                cases["all_terms"],
            )
            deterministic = bool(torch.equal(actual, reference_all_terms))
            guard_ok = guards_unchanged(physical)
            coverage_ok = payload_has_no_sentinel(physical)
            returned_ok = returned in (None, 0)
            repetitions.append(
                {
                    "repetition": repetition,
                    "bitwise_deterministic": deterministic,
                    "guard_rows_unchanged": guard_ok,
                    "payload_fully_covered": coverage_ok,
                    "output_sha256": tensor_sha256(actual),
                }
            )
            if not (deterministic and guard_ok and coverage_ok and returned_ok):
                raise AssertionError(
                    f"{compiled.config.name}/{core_num}: "
                    f"determinism repetition {repetition} failed"
                )

        assignment_counts = [
            sum(
                1
                for task in range(compiled.config.task_count)
                if task % core_num == workitem
            )
            for workitem in range(core_num)
        ]
        core_results[str(core_num)] = {
            "status": "PASS",
            "core_num": core_num,
            "task_count": compiled.config.task_count,
            "assignment_counts": assignment_counts,
            "idle_workitem_count": assignment_counts.count(0),
            "cases": case_results,
            "repeat": repeat,
            "all_terms_repetitions": repetitions,
        }
        if core_num == 1:
            single_core_outputs = {
                name: output.clone() for name, output in outputs.items()
            }
        print(
            f"{compiled.config.name.upper()} S1 {core_num}-CORE PASS "
            f"tasks={compiled.config.task_count} repeat={repeat}",
            flush=True,
        )

    return {
        "status": "PASS",
        "shape_name": compiled.config.name,
        "logical_shape": compiled.config.logical_shape(),
        "task_count": compiled.config.task_count,
        "compile": compiled.manifest,
        "core_results": core_results,
    }


def parse_csv_subset(
    parser: argparse.ArgumentParser,
    value: str,
    allowed: Tuple[str, ...],
    option: str,
) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or any(item not in allowed for item in items):
        parser.error(f"{option} must be a subset of {','.join(allowed)}")
    if len(items) != len(set(items)):
        parser.error(f"{option} contains duplicates")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P9.2 BM1690 multi-core ChunkScan S1 correctness"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--core-counts", default="1,2,4,8")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    args.resolved_shapes = parse_csv_subset(
        parser,
        args.shapes,
        tuple(SHAPES),
        "--shapes",
    )
    core_text = parse_csv_subset(
        parser,
        args.core_counts,
        tuple(str(value) for value in ALLOWED_CORE_COUNTS),
        "--core-counts",
    )
    args.resolved_core_counts = tuple(int(value) for value in core_text)
    if args.run and args.resolved_core_counts[0] != 1:
        parser.error("--run requires core count 1 first as the bitwise baseline")
    return args


def main() -> None:
    args = parse_args()
    paths = require_environment()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ARTIFACT_ROOT / "runs" / f"{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    repository_commit = run_text(("git", "-C", str(ROOT), "rev-parse", "HEAD"))
    tvm_commit = run_text(
        ("git", "-C", str(ROOT / "3rdparty/tvm"), "rev-parse", "HEAD")
    )
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit,
        "tvm_commit": tvm_commit,
        "device_id": os.environ.get("CHUNKSCAN_DEVICE_ID", "0"),
        "ppl_project_root": str(paths["PPL_PROJECT_ROOT"]),
        "runtime_root": str(paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]),
        "riscv_toolchain_root": str(
            paths["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        ),
        "shapes": list(args.resolved_shapes),
        "core_counts": list(args.resolved_core_counts),
        "repeat": args.repeat,
        "stage": "P9.2",
        "candidate": "S1",
        "performance_claim": "none",
    }
    write_json(run_dir / "environment.json", environment)

    compiled = {
        name: compile_shape(run_dir, SHAPES[name])
        for name in args.resolved_shapes
    }
    if not args.run:
        result = {
            "status": "COMPILE_PASS",
            "run_dir": str(run_dir),
            "compiled_shapes": {
                name: item.manifest for name, item in compiled.items()
            },
            "chunk_scan_multicore_s1_validated": False,
            "performance_claim": "none; no TPU kernel was launched",
        }
        write_json(run_dir / "result.json", result)
        write_json(ARTIFACT_ROOT / "result.json", result)
        print(f"P9.2 S1 COMPILE-ONLY PASS: {run_dir}", flush=True)
        return

    shape_results = {
        name: run_shape(
            item,
            args.resolved_core_counts,
            args.repeat,
            seed=20260922 + index,
        )
        for index, (name, item) in enumerate(compiled.items())
    }
    full_matrix = (
        tuple(args.resolved_shapes) == tuple(SHAPES)
        and tuple(args.resolved_core_counts) == ALLOWED_CORE_COUNTS
        and args.repeat >= 20
    )
    result = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "stage": "P9.2",
        "candidate": "S1",
        "shape_results": shape_results,
        "physical_multicore_dispatch_prerequisite": "P9.1 PASS",
        "chunk_scan_multicore_s1_validated": full_matrix,
        "chunk_scan_multicore_s3_p6_validated": False,
        "performance_claim": "none; correctness matrix only",
    }
    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_ROOT / "result.json", result)
    if not full_matrix:
        print(
            "P9.2 S1 SUBSET PASS (not the full acceptance matrix): "
            f"{run_dir}",
            flush=True,
        )
    else:
        print(f"P9.2 S1 PASS: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
