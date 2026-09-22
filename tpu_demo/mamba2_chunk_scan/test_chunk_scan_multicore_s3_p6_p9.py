"""Compile and validate task-major S3/P6 on BM1690 with 1/2/4/8 cores."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_abi_p9 import (
    Inputs,
    guards_unchanged,
    make_inputs,
    payload_has_no_sentinel,
    poison_upper_triangle,
    variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_p6_p9 import (
    make_chunk_scan_multicore_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ChunkScanMulticoreConfig,
    physical_shapes,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s3_p9 import (
    make_chunk_scan_multicore_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import SCHEDULE_NAME
from tpu_demo.mamba2_chunk_scan.reference import chunk_scan_reference
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_s1_p9 import (
    ALLOWED_CORE_COUNTS,
    CASE_NAMES,
    ROOT,
    SHAPES,
    CompiledShape,
    call_kernel,
    parse_csv_subset,
    prepare_runtime,
    require_environment,
    run_text,
    sha256_file,
    static_local_memory_end,
    tensor_sha256,
    validate_output,
    write_json,
)


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE / "artifacts" / "device_multicore_s3_p6_p9"
CANDIDATES = {
    "s3": make_chunk_scan_multicore_s3_kernel,
    "p6": make_chunk_scan_multicore_p6_kernel,
}


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


def compile_candidate(
    run_dir: Path,
    candidate: str,
    config: ChunkScanMulticoreConfig,
) -> CompiledShape:
    candidate_dir = run_dir / candidate
    shape_dir, runtime_dir = prepare_runtime(candidate_dir, config.name)
    program = CANDIDATES[candidate](config)
    schedule = program.attrs.get("p6_pipeline_schedule")
    if candidate == "p6":
        if str(schedule) != SCHEDULE_NAME:
            raise AssertionError("P6 explicit schedule attribute is absent")
    elif schedule is not None:
        raise AssertionError("S3 unexpectedly has the P6 schedule attribute")

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            kernel = tilelang.compile(
                program,
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
    for marker in (
        "tpu_workitem_index()",
        "tpu_workitem_num()",
        f"for (int task = 0; task < {config.task_count}; ++task)",
    ):
        if marker not in source:
            raise AssertionError(
                f"{candidate}/{config.name}: source lacks {marker!r}"
            )
    for abstract_name in ("ppl.workitem_index", "ppl.workitem_num"):
        if abstract_name in source:
            raise AssertionError(
                f"{candidate}/{config.name}: abstract call remains"
            )
    pipeline_start_count = source.count("tpu_parallel_start(")
    pipeline_end_count = source.count("tpu_parallel_end(")
    if pipeline_start_count != 1 or pipeline_end_count != 1:
        raise AssertionError(
            f"{candidate}/{config.name}: invalid pipeline markers "
            f"{pipeline_start_count}/{pipeline_end_count}"
        )
    for marker in (
        "CHUNKSCAN_CORE_NUM",
        "block_num = static_cast<uint64_t>(core_num)",
        "apis.data()",
    ):
        if marker not in host_source:
            raise AssertionError(
                f"{candidate}/{config.name}: host lacks {marker!r}"
            )
    for index in range(1, 9):
        if f"ptr_v{index};" not in header:
            raise AssertionError(
                f"{candidate}/{config.name}: unexpected kernel ABI"
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
            f"{candidate}/{config.name}: wrong ELF architecture"
        )
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(
            f"{candidate}/{config.name}: wrong host runtime:\n{linked}"
        )

    local_end = static_local_memory_end(source)
    if local_end is None or local_end > 256 * 1024:
        raise AssertionError(
            f"{candidate}/{config.name}: invalid LMEM {local_end}"
        )

    manifest: Dict[str, object] = {
        "status": "COMPILE_PASS",
        "stage": candidate,
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
        "pipeline_start_count": pipeline_start_count,
        "pipeline_end_count": pipeline_end_count,
        "pipeline_num_stages": 2,
        "explicit_schedule": (
            SCHEDULE_NAME if candidate == "p6" else None
        ),
        "workitem_index_call_count": source.count(
            "tpu_workitem_index()"
        ),
        "workitem_num_call_count": source.count("tpu_workitem_num()"),
        "static_local_memory_end_bytes": local_end,
        "bm1690_local_memory_limit_bytes": 256 * 1024,
        "performance_claim": "none; P9.3 correctness only",
    }
    write_json(shape_dir / "compile_manifest.json", manifest)
    print(
        f"{config.name.upper()} {candidate.upper()} COMPILE PASS "
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


def run_candidate_shape(
    compiled: CompiledShape,
    candidate: str,
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
                compiled, core_num, cases[case_name]
            )
            baseline = (
                None if core_num == 1 else single_core_outputs[case_name]
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

        poison_equal = torch.equal(
            outputs["scan_only"],
            outputs["causal_upper_triangle_poison"],
        )
        if not poison_equal:
            raise AssertionError(
                f"{candidate}/{compiled.config.name}/{core_num}: "
                "causal poison changed output"
            )
        case_results["causal_upper_triangle_poison"][
            "bitwise_equal_to_clean_scan"
        ] = True

        reference_all_terms = outputs["all_terms"]
        repetitions = [{
            "repetition": 1,
            "bitwise_deterministic": True,
            "guard_rows_unchanged": True,
            "payload_fully_covered": True,
            "output_sha256": tensor_sha256(reference_all_terms),
        }]
        for repetition in range(2, repeat + 1):
            actual, physical, returned = call_kernel(
                compiled, core_num, cases["all_terms"]
            )
            deterministic = torch.equal(actual, reference_all_terms)
            guard_ok = guards_unchanged(physical)
            coverage_ok = payload_has_no_sentinel(physical)
            returned_ok = returned in (None, 0)
            repetitions.append({
                "repetition": repetition,
                "bitwise_deterministic": bool(deterministic),
                "guard_rows_unchanged": guard_ok,
                "payload_fully_covered": coverage_ok,
                "output_sha256": tensor_sha256(actual),
            })
            if not (
                deterministic and guard_ok and coverage_ok and returned_ok
            ):
                raise AssertionError(
                    f"{candidate}/{compiled.config.name}/{core_num}: "
                    f"repetition {repetition} failed"
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
            f"{compiled.config.name.upper()} {candidate.upper()} "
            f"{core_num}-CORE PASS tasks={compiled.config.task_count} "
            f"repeat={repeat}",
            flush=True,
        )

    return {
        "status": "PASS",
        "candidate": candidate,
        "shape_name": compiled.config.name,
        "logical_shape": compiled.config.logical_shape(),
        "task_count": compiled.config.task_count,
        "compile": compiled.manifest,
        "core_results": core_results,
    }


def compare_candidates(
    candidate_results: Dict[str, Dict[str, object]],
    shape_names: Tuple[str, ...],
    core_counts: Tuple[int, ...],
) -> Dict[str, object]:
    if set(candidate_results) != {"s3", "p6"}:
        return {
            "checked": False,
            "bitwise_equal": None,
            "reason": "both s3 and p6 were not selected",
        }
    comparisons = 0
    for shape_name in shape_names:
        s3_shape = candidate_results["s3"][shape_name]
        p6_shape = candidate_results["p6"][shape_name]
        for core_num in core_counts:
            s3_core = s3_shape["core_results"][str(core_num)]
            p6_core = p6_shape["core_results"][str(core_num)]
            for case_name in CASE_NAMES:
                s3_hash = s3_core["cases"][case_name]["output_sha256"]
                p6_hash = p6_core["cases"][case_name]["output_sha256"]
                if s3_hash != p6_hash:
                    raise AssertionError(
                        f"S3/P6 differ: {shape_name}/{core_num}/{case_name}"
                    )
                comparisons += 1
    return {
        "checked": True,
        "bitwise_equal": True,
        "comparison_count": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P9.3 BM1690 multi-core ChunkScan S3/P6 correctness"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--candidates", default="s3,p6")
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--core-counts", default="1,2,4,8")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    args.resolved_candidates = parse_csv_subset(
        parser,
        args.candidates,
        tuple(CANDIDATES),
        "--candidates",
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
    if args.run and args.resolved_core_counts[0] != 1:
        parser.error("--run requires core count 1 first")
    return args


def main() -> None:
    args = parse_args()
    paths = require_environment()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ARTIFACT_ROOT / "runs" / f"{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": run_text(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD")
        ),
        "tvm_commit": run_text(
            ("git", "-C", str(ROOT / "3rdparty/tvm"), "rev-parse", "HEAD")
        ),
        "device_id": os.environ.get("CHUNKSCAN_DEVICE_ID", "0"),
        "ppl_project_root": str(paths["PPL_PROJECT_ROOT"]),
        "runtime_root": str(paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]),
        "riscv_toolchain_root": str(
            paths["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        ),
        "candidates": list(args.resolved_candidates),
        "shapes": list(args.resolved_shapes),
        "core_counts": list(args.resolved_core_counts),
        "repeat": args.repeat,
        "stage": "P9.3",
        "performance_claim": "none",
    }
    write_json(run_dir / "environment.json", environment)

    compiled = {
        candidate: {
            shape_name: compile_candidate(
                run_dir, candidate, SHAPES[shape_name]
            )
            for shape_name in args.resolved_shapes
        }
        for candidate in args.resolved_candidates
    }
    if not args.run:
        result = {
            "status": "COMPILE_PASS",
            "stage": "P9.3",
            "run_dir": str(run_dir),
            "compiled_candidates": {
                candidate: {
                    name: item.manifest for name, item in shapes.items()
                }
                for candidate, shapes in compiled.items()
            },
            "chunk_scan_multicore_s3_p6_validated": False,
            "performance_claim": "none; no TPU kernel was launched",
        }
        write_json(run_dir / "result.json", result)
        write_json(ARTIFACT_ROOT / "result.json", result)
        print(f"P9.3 S3/P6 COMPILE-ONLY PASS: {run_dir}", flush=True)
        return

    candidate_results = {
        candidate: {
            shape_name: run_candidate_shape(
                item,
                candidate,
                args.resolved_core_counts,
                args.repeat,
                seed=20260923 + index,
            )
            for index, (shape_name, item) in enumerate(shapes.items())
        }
        for candidate, shapes in compiled.items()
    }
    cross_candidate = compare_candidates(
        candidate_results,
        args.resolved_shapes,
        args.resolved_core_counts,
    )
    full_matrix = (
        tuple(args.resolved_candidates) == tuple(CANDIDATES)
        and tuple(args.resolved_shapes) == tuple(SHAPES)
        and tuple(args.resolved_core_counts) == ALLOWED_CORE_COUNTS
        and args.repeat >= 20
        and cross_candidate["bitwise_equal"] is True
    )
    result = {
        "status": "PASS",
        "stage": "P9.3",
        "run_dir": str(run_dir),
        "candidate_results": candidate_results,
        "s3_p6_comparison": cross_candidate,
        "physical_multicore_dispatch_prerequisite": "P9.1 PASS",
        "chunk_scan_multicore_s1_prerequisite": "P9.2 PASS",
        "chunk_scan_multicore_s3_validated": full_matrix,
        "chunk_scan_multicore_p6_validated": full_matrix,
        "chunk_scan_multicore_s3_p6_validated": full_matrix,
        "multicore_performance_validated": False,
        "performance_claim": "none; correctness matrix only",
    }
    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_ROOT / "result.json", result)
    if full_matrix:
        print(f"P9.3 S3/P6 PASS: {run_dir}", flush=True)
    else:
        print(
            "P9.3 S3/P6 SUBSET PASS (not full acceptance): "
            f"{run_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
