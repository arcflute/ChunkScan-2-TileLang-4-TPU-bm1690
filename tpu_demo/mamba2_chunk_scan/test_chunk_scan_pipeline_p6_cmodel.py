"""Validate P6 explicit-sProgram lowering for BM1690 ChunkScan."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict

import tilelang
import tilelang.language as T
import torch
from tilelang import tvm

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import (
    LOWERED_STATEMENT_COUNT,
    PIPELINE_ORDER,
    PIPELINE_STAGE,
    SPROG_B_CONTRACT,
    make_chunk_scan_pipeline_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s3 import (
    make_chunk_scan_pipeline_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_cmodel import (
    _make_inputs,
    _source_metrics,
    _variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_pipeline_matrix_p5_cmodel import (
    _pipeline_region_evidence,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_pipeline_s3_cmodel import (
    EXPECTED_DOUBLE_BUFFER_DECLARATIONS,
    EXPECTED_REDUCTION_LOOP,
    EXPECTED_SOURCE_SITE_COUNTS,
    _compile_cmodel,
    _worker_outputs,
)
from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)
from tilelang.engine.phase import LowerAndLegalize


tir = tvm.tir
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_p6_explicit"
P6_RUNTIME_DIR = ARTIFACT_DIR / "runtime_p6"
S3_RUNTIME_DIR = ARTIFACT_DIR / "runtime_s3_control"
P5_RESULT_PATH = (
    ROOT / "artifacts" / "pipeline_matrix_p5" / "result.json"
)
S3_RESULT_PATH = ROOT / "artifacts" / "pipeline_s3" / "result.json"
ATOL = 1e-2
RTOL = 1e-2


@T.prim_func
def _insufficient_depth_probe(
    A: T.Tensor((2, 1), "float32"),
    C: T.Tensor((2, 1), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        A_shared = T.alloc_shared((1, 1), "float32")
        for i in T.Pipelined(
            2,
            num_stages=2,
            order=[0, 1],
            stage=[0, 2],
        ):
            A_shared[0, 0] = A[i, 0]
            C[i, 0] = A_shared[0, 0]


@T.prim_func
def _dependency_violation_probe(
    A: T.Tensor((4, 1), "float32"),
    C: T.Tensor((4, 1), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        A_shared = T.alloc_shared((1, 1), "float32")
        for i in T.Pipelined(
            4,
            num_stages=2,
            order=[1, 0],
            stage=[2, 0],
        ):
            A_shared[0, 0] = A[i, 0]
            C[i, 0] = A_shared[0, 0]


@T.prim_func
def _lmem_overflow_probe(
    C: T.Tensor((1088, 4096), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        too_large = T.alloc_shared((1088, 4096), "float32")
        T.ppl_fill(too_large, T.float32(0.0))
        T.ppl_copy(too_large, C[0, 0])


def _load_pass_result(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} prerequisite: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise AssertionError(f"{label} prerequisite is not PASS: {result}")
    return result


def _bound_tpu_module(program):
    func = program.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    return tir.transform.BindTarget(tvm.target.Target("tpu"))(mod)


def _expect_compiler_error(
    gate: str,
    action: Callable[[], object],
    expected_text: str,
) -> Dict[str, object]:
    try:
        action()
    except (tvm.TVMError, ValueError) as error:
        message = str(error)
        if expected_text not in message:
            raise AssertionError(
                f"{gate} failed for the wrong reason: {message}"
            ) from error
        reason_line = next(
            line.strip()
            for line in message.splitlines()
            if expected_text in line
        )
        return {
            "status": "REJECTED_AS_EXPECTED",
            "reason_contains": expected_text,
            "reason_line": reason_line,
        }
    raise AssertionError(f"{gate} was unexpectedly accepted")


def _negative_compiler_gates() -> Dict[str, object]:
    depth_mod = _bound_tpu_module(_insufficient_depth_probe)
    depth = _expect_compiler_error(
        "insufficient_loop_depth",
        lambda: tilelang.transform.PipelinePlanning()(depth_mod),
        "nonempty steady state",
    )

    dependency_mod = tilelang.transform.PipelinePlanning()(
        _bound_tpu_module(_dependency_violation_probe)
    )
    dependency = _expect_compiler_error(
        "dependency_violation",
        lambda: tilelang.transform.InjectSoftwarePipeline()(dependency_mod),
        "in a later stage",
    )

    lmem = _expect_compiler_error(
        "lmem_overflow",
        lambda: tilelang.lower(_lmem_overflow_probe, target="tpu"),
        "BM1690 local memory allocation failed",
    )
    return {
        "insufficient_loop_depth": depth,
        "dependency_violation": dependency,
        "lmem_overflow": lmem,
    }


def _archive_planned_schedule(program) -> Dict[str, object]:
    target = tvm.target.Target("tpu", host="llvm")
    symbol = str(program.attrs["global_symbol"])
    mod = tvm.IRModule({symbol: program})
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)

    loops = []

    def visitor(node):
        if (
            isinstance(node, tir.For)
            and "software_pipeline_order" in node.annotations
        ):
            loops.append(node)

    func = next(iter(mod.functions.values()))
    tir.stmt_functor.post_order_visit(func.body, visitor)
    if len(loops) != 1:
        raise AssertionError(f"expected one planned loop, found {len(loops)}")

    loop = loops[0]
    order = [int(value) for value in loop.annotations[
        "software_pipeline_order"
    ]]
    stage = [int(value) for value in loop.annotations[
        "software_pipeline_stage"
    ]]
    marker = int(loop.annotations.get("tl_pipeline_explicit_schedule", 0))
    if order != list(PIPELINE_ORDER):
        raise AssertionError(f"P6 planner changed explicit order: {order}")
    if stage != list(PIPELINE_STAGE):
        raise AssertionError(f"P6 planner changed explicit stage: {stage}")
    if marker != 1:
        raise AssertionError("P6 planner did not record the explicit path")
    if "tl_pipeline_order" in loop.annotations:
        raise AssertionError("P6 planner did not consume tl_pipeline_order")
    if "tl_pipeline_stage" in loop.annotations:
        raise AssertionError("P6 planner did not consume tl_pipeline_stage")
    if len(order) != LOWERED_STATEMENT_COUNT:
        raise AssertionError("unexpected P6 lowered statement count")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "planned_device.tir").write_text(
        mod.script(), encoding="utf-8"
    )
    evidence = {
        "explicit_schedule_marker": marker,
        "lowered_statement_count": len(order),
        "software_pipeline_order": order,
        "software_pipeline_stage": stage,
        "frontend_order_consumed": True,
        "frontend_stage_consumed": True,
    }
    (ARTIFACT_DIR / "planner_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _archive_lowering(program) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = tilelang.lower(program, target="tpu")
    raw_source = str(artifact.kernel_source)
    metrics = _source_metrics(raw_source)
    (ARTIFACT_DIR / "lowered_host.tir").write_text(
        artifact.host_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "lowered_device.tir").write_text(
        artifact.device_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "kernel_raw.c").write_text(
        raw_source, encoding="utf-8"
    )
    (ARTIFACT_DIR / "raw_source_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"raw_source": raw_source, "source_metrics": metrics}


def _archive_p6_cmodel_sources() -> None:
    for source_name, archive_name in (
        ("kernel.c", "kernel_cmodel.c"),
        ("kernel.cpp", "kernel.cpp"),
        ("kernel.h", "kernel.h"),
        ("main.cpp", "main.cpp"),
    ):
        source = P6_RUNTIME_DIR / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ARTIFACT_DIR / archive_name)


def _worker_main(variant: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "s3":
        program = make_chunk_scan_pipeline_s3_kernel()
        runtime_dir = S3_RUNTIME_DIR
    elif variant == "p6":
        program = make_chunk_scan_pipeline_p6_kernel()
        runtime_dir = P6_RUNTIME_DIR
        _archive_lowering(program)
    else:
        raise ValueError(f"unknown P6 worker variant: {variant}")

    kernel = _compile_cmodel(program, runtime_dir)
    if variant == "p6":
        _archive_p6_cmodel_sources()
    torch.save(
        _worker_outputs(kernel),
        ARTIFACT_DIR / f"worker_{variant}_outputs.pt",
    )


def _run_worker(variant: str) -> Dict[str, torch.Tensor]:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", variant],
        check=True,
    )
    output_path = ARTIFACT_DIR / f"worker_{variant}_outputs.pt"
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    return torch.load(output_path, map_location="cpu", weights_only=True)


def _inputs_for_case(name: str):
    base = _make_inputs(seed=20260910)
    if name != "residual_only_negative_D":
        return _variant_inputs(base, name)
    negative = list(_variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    return tuple(negative)


def _numerical_results(
    s3_outputs: Dict[str, torch.Tensor],
    p6_outputs: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for name in (
        "state_only",
        "scan_only",
        "residual_only",
        "residual_only_negative_D",
        "all_terms",
    ):
        expected = chunk_scan_reference(*_inputs_for_case(name))
        s3 = s3_outputs[name]
        p6 = p6_outputs[name]
        oracle_stats = comparison_stats(p6, expected)
        control_stats = comparison_stats(p6, s3)
        result = {
            "p6_close_to_oracle": bool(
                torch.allclose(
                    p6.float(), expected.float(), atol=ATOL, rtol=RTOL
                )
            ),
            "p6_bitwise_equal_to_s3": bool(torch.equal(p6, s3)),
            "p6_vs_oracle": oracle_stats,
            "p6_vs_s3": control_stats,
            "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        }
        if (
            not result["p6_close_to_oracle"]
            or not result["p6_bitwise_equal_to_s3"]
            or result["expected_nonzero_count"] == 0
            or oracle_stats["nan_count"]
            or oracle_stats["inf_count"]
            or control_stats["max_abs_diff"] != 0.0
        ):
            raise AssertionError(f"P6 numerical mismatch: {name}: {result}")
        results[name] = result

    expected = chunk_scan_reference(
        *_variant_inputs(_make_inputs(seed=20260910), "scan_only")
    )
    p6_clean = p6_outputs["causal_clean"]
    p6_poisoned = p6_outputs["causal_poisoned"]
    s3_poisoned = s3_outputs["causal_poisoned"]
    causal = {
        "p6_clean_equals_poisoned": bool(torch.equal(p6_clean, p6_poisoned)),
        "p6_poisoned_equals_s3": bool(torch.equal(p6_poisoned, s3_poisoned)),
        "p6_poisoned_close_to_oracle": bool(
            torch.allclose(
                p6_poisoned.float(),
                expected.float(),
                atol=ATOL,
                rtol=RTOL,
            )
        ),
        "p6_poisoned_vs_oracle": comparison_stats(p6_poisoned, expected),
        "p6_poisoned_vs_s3": comparison_stats(p6_poisoned, s3_poisoned),
    }
    if not all(
        causal[key]
        for key in (
            "p6_clean_equals_poisoned",
            "p6_poisoned_equals_s3",
            "p6_poisoned_close_to_oracle",
        )
    ):
        raise AssertionError(f"P6 causal gate failed: {causal}")
    results["causal_upper_triangle_poison"] = causal
    return results


def _raw_matches_manual_reference(
    raw_source: str,
    metrics: Dict[str, object],
    s3_result: Dict[str, object],
) -> Dict[str, object]:
    actual_counts = {
        name: metrics[name] for name in EXPECTED_SOURCE_SITE_COUNTS
    }
    if actual_counts != EXPECTED_SOURCE_SITE_COUNTS:
        raise AssertionError(f"unexpected P6 source metrics: {actual_counts}")
    if raw_source.count(EXPECTED_REDUCTION_LOOP) != 1:
        raise AssertionError("P6 must contain one two-iteration steady loop")
    for declaration in EXPECTED_DOUBLE_BUFFER_DECLARATIONS:
        if raw_source.count(declaration) != 1:
            raise AssertionError(
                f"P6 missing double-buffer declaration: {declaration}"
            )

    p6_region = _pipeline_region_evidence(raw_source, 2)
    manual_region = s3_result["pipeline_structure"]["raw_region_evidence"]
    for key in (
        "producer_sites",
        "scan_gemm_sites",
        "steady_future_load_order",
    ):
        if p6_region[key] != manual_region[key]:
            raise AssertionError(
                f"P6 differs from manual S3 for {key}: {p6_region[key]}"
            )

    manual_metrics = s3_result["source_metrics"]
    compared_metric_names = list(EXPECTED_SOURCE_SITE_COUNTS) + [
        "static_local_memory_end_bytes",
        "bm1690_local_memory_limit_bytes",
    ]
    for name in compared_metric_names:
        if metrics[name] != manual_metrics[name]:
            raise AssertionError(
                f"P6 differs from manual S3 metric {name}: "
                f"{metrics[name]} != {manual_metrics[name]}"
            )
    if metrics["static_local_memory_end_bytes"] > metrics[
        "bm1690_local_memory_limit_bytes"
    ]:
        raise AssertionError(f"P6 exceeds LMEM: {metrics}")
    return {
        "matched_region_fields": [
            "producer_sites",
            "scan_gemm_sites",
            "steady_future_load_order",
        ],
        "matched_source_metrics": compared_metric_names,
        "raw_region_evidence": p6_region,
        "manual_reference": str(S3_RESULT_PATH),
    }


def _strip_parallel_markers(raw_source: str) -> str:
    return raw_source.replace(
        "      tpu_parallel_start(); \n", ""
    ).replace(
        "      tpu_parallel_end(); \n", ""
    ).replace(
        "tpu_parallel_start(); \n", ""
    ).replace(
        "tpu_parallel_end(); \n", ""
    )


def test_pipeline_p6_explicit_schedule_cmodel() -> None:
    toolchain = assert_chunkscan_toolchain()
    print(json.dumps(toolchain, indent=2, sort_keys=True), flush=True)
    p5_result = _load_pass_result(P5_RESULT_PATH, "P5 matrix")
    s3_result = _load_pass_result(S3_RESULT_PATH, "P4.2 S3")

    p6_program = make_chunk_scan_pipeline_p6_kernel()
    planner_evidence = _archive_planned_schedule(p6_program)
    negative_gates = _negative_compiler_gates()

    s3_outputs = _run_worker("s3")
    p6_outputs = _run_worker("p6")

    raw_source = (ARTIFACT_DIR / "kernel_raw.c").read_text(encoding="utf-8")
    metrics = _source_metrics(raw_source)
    manual_comparison = _raw_matches_manual_reference(
        raw_source, metrics, s3_result
    )
    numerical = _numerical_results(s3_outputs, p6_outputs)

    cmodel_source = (ARTIFACT_DIR / "kernel_cmodel.c").read_text(
        encoding="utf-8"
    )
    cmodel_equals_raw = cmodel_source == _strip_parallel_markers(raw_source)
    if not cmodel_equals_raw:
        raise AssertionError(
            "P6 cmodel source differs by more than parallel-marker removal"
        )

    result = {
        "status": "PASS",
        "stage": "P6",
        "variant": "compiler_explicit_sprog_b",
        "objective": "compiler-generated designated ChunkScan pipeline",
        "schedule_contract": SPROG_B_CONTRACT.as_dict(),
        "planner_evidence": planner_evidence,
        "negative_compiler_gates": negative_gates,
        "manual_s3_structural_comparison": manual_comparison,
        "numerical_results": numerical,
        "source_metrics": metrics,
        "cmodel_equals_raw_after_marker_strip": cmodel_equals_raw,
        "prerequisites": {
            "pipeline_s3": {
                "path": str(S3_RESULT_PATH),
                "status": s3_result["status"],
            },
            "pipeline_matrix_p5": {
                "path": str(P5_RESULT_PATH),
                "status": p5_result["status"],
            },
        },
        "toolchain": toolchain,
        "selection_policy": (
            "fixed paper-designated sProg-B contract; no timing search"
        ),
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (ARTIFACT_DIR / "manifest.json").write_text(
        serialized, encoding="utf-8"
    )
    (ARTIFACT_DIR / "result.json").write_text(
        serialized, encoding="utf-8"
    )
    print(serialized, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
    elif len(sys.argv) == 1:
        test_pipeline_p6_explicit_schedule_cmodel()
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [--worker s3|p6]")