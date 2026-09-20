"""Validate the bounded P5 ChunkScan pipeline-configuration matrix."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import tilelang
import torch

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p5 import (
    CHUNK_SIZE,
    REDUCE_TILE,
    REDUCE_TILES,
    SPROG_A,
    SPROG_B,
    make_chunk_scan_pipeline_p5_kernel,
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
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_pipeline_s3_cmodel import (
    _compile_cmodel,
    _worker_outputs,
)
from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_matrix_p5"
SPROG_A_DIR = ARTIFACT_DIR / "sprog_a_k16_stage2"
STAGE3_DIR = ARTIFACT_DIR / "sprog_b_k16_stage3"
S3_RUNTIME_DIR = ARTIFACT_DIR / "runtime_s3_control"
STAGE3_RUNTIME_DIR = ARTIFACT_DIR / "runtime_sprog_b_stage3"

DEFAULT_EVIDENCE_ROOT = REPOSITORY_ROOT / "tpu_demo" / "mamba2_chunk_scan"
S2_RESULT_PATH = Path(
    os.environ.get(
        "P5_S2_RESULT_PATH",
        DEFAULT_EVIDENCE_ROOT / "artifacts" / "pipeline_s2" / "result.json",
    )
)
S3_RESULT_PATH = Path(
    os.environ.get(
        "P5_S3_RESULT_PATH",
        DEFAULT_EVIDENCE_ROOT / "artifacts" / "pipeline_s3" / "result.json",
    )
)
S3_RAW_SOURCE_PATH = S3_RESULT_PATH.with_name("kernel_raw.c")

ATOL = 1e-2
RTOL = 1e-2
LOADY_FIRST_ORDER = ["LoadY.cb", "LoadY.dA", "LoadY.dt", "LoadX"]
LOADX_FIRST_ORDER = ["LoadX", "LoadY.cb", "LoadY.dA", "LoadY.dt"]

EXPECTED_STAGE3_METRICS = {
    "gdma_s2l_site_count": 85,
    "gdma_l2s_site_count": 1,
    "bdc_copy_site_count": 8,
    "bdc_cast_site_count": 21,
    "fp16_to_fp32_cast_site_count": 16,
    "fp32_to_fp16_cast_site_count": 5,
    "fill_site_count": 12,
    "bdc_add_site_count": 11,
    "bdc_sub_site_count": 8,
    "bdc_mul_site_count": 10,
    "fp_mm_site_count": 4,
    "fp_mm_r_trans_site_count": 1,
    "fp32_exp_site_count": 5,
    "npu_bcast_site_count": 9,
    "stride_c_zero_count": 0,
    "stride_w_zero_count": 6,
    "chunk_loop_count": 1,
    "dA_scatter_loop_count": 1,
    "parallel_start_count": 1,
    "parallel_end_count": 1,
    "bytes": 156844,
    "static_local_memory_end_bytes": 69632,
    "bm1690_local_memory_limit_bytes": 262144,
}

PRODUCER_ANCHORS = {
    "LoadY.cb": "tpu_gdma_cpy_S2L(cb_loaded_shared",
    "LoadY.dA": "tpu_gdma_cpy_S2L(dA_reduce_row_fp16_shared",
    "LoadY.dt": "tpu_gdma_cpy_S2L(dt_reduce_row_fp16_shared",
    "LoadX": "tpu_gdma_cpy_S2L(x_reduce_shared",
}
PRODUCER_BUFFERS = (
    "cb_loaded_shared",
    "dA_reduce_row_fp16_shared",
    "dt_reduce_row_fp16_shared",
    "x_reduce_shared",
)


def _archive_lowering(program, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = tilelang.lower(program, target="tpu")
    raw_source = str(artifact.kernel_source)
    metrics = _source_metrics(raw_source)
    (output_dir / "lowered_host.tir").write_text(
        artifact.host_mod.script(), encoding="utf-8"
    )
    (output_dir / "lowered_device.tir").write_text(
        artifact.device_mod.script(), encoding="utf-8"
    )
    (output_dir / "kernel_raw.c").write_text(raw_source, encoding="utf-8")
    (output_dir / "raw_source_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"raw_source": raw_source, "source_metrics": metrics}


def _archive_stage3_cmodel_sources() -> None:
    for source_name, archive_name in (
        ("kernel.c", "kernel_cmodel.c"),
        ("kernel.cpp", "kernel.cpp"),
        ("kernel.h", "kernel.h"),
        ("main.cpp", "main.cpp"),
    ):
        source = STAGE3_RUNTIME_DIR / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, STAGE3_DIR / archive_name)


def _pipeline_region_evidence(
    raw_source: str,
    num_stages: int,
) -> Dict[str, object]:
    start_marker = "tpu_parallel_start();"
    end_marker = "tpu_parallel_end();"
    if raw_source.count(start_marker) != 1 or raw_source.count(end_marker) != 1:
        raise AssertionError("candidate must contain exactly one raw parallel region")

    start = raw_source.index(start_marker)
    end = raw_source.index(end_marker, start)
    parts = {
        "prologue": raw_source[:start],
        "steady": raw_source[start:end],
        "epilogue": raw_source[end:],
    }
    producer_sites = {
        part: {
            name: source.count(anchor)
            for name, anchor in PRODUCER_ANCHORS.items()
        }
        for part, source in parts.items()
    }
    expected_producer_sites = {
        "prologue": {name: num_stages for name in PRODUCER_ANCHORS},
        "steady": {name: 1 for name in PRODUCER_ANCHORS},
        "epilogue": {name: 0 for name in PRODUCER_ANCHORS},
    }
    if producer_sites != expected_producer_sites:
        raise AssertionError(
            f"producer placement mismatch: {producer_sites}"
        )

    steady = parts["steady"]
    offsets = {
        name: steady.index(anchor)
        for name, anchor in PRODUCER_ANCHORS.items()
    }
    actual_order = sorted(offsets, key=offsets.get)

    gemm_anchor = "tpu_bdc_fp_mm(gemm_temp.addr"
    gemm_sites = {
        part: source.count(gemm_anchor) for part, source in parts.items()
    }
    expected_gemm_sites = {
        "prologue": 0,
        "steady": 1,
        "epilogue": num_stages,
    }
    if gemm_sites != expected_gemm_sites:
        raise AssertionError(f"scan GEMM placement mismatch: {gemm_sites}")

    for buffer_name in PRODUCER_BUFFERS:
        for version in range(num_stages):
            declaration = f"__ppl_tensor_info {buffer_name}_{version} ="
            if raw_source.count(declaration) != 1:
                raise AssertionError(
                    f"missing {num_stages}-stage buffer declaration: {declaration}"
                )

    return {
        "producer_sites": producer_sites,
        "scan_gemm_sites": gemm_sites,
        "steady_future_load_offsets": offsets,
        "steady_future_load_order": actual_order,
        "buffer_versions": num_stages,
    }


def _loop_depth_rejection(
    reduction_tile: int,
    num_stages: int,
) -> Dict[str, object]:
    if CHUNK_SIZE % reduction_tile != 0:
        reason = "reduction tile does not divide L=64"
        iterations = None
    else:
        iterations = CHUNK_SIZE // reduction_tile
        if iterations > num_stages:
            raise AssertionError(
                "configuration is not a loop-depth rejection: "
                f"tile={reduction_tile}, stages={num_stages}"
            )
        reason = (
            "nonempty steady state requires reduction_iterations > "
            f"num_stages, but {iterations} <= {num_stages}"
        )
    return {
        "status": "REJECTED",
        "reason_code": "insufficient_loop_depth",
        "reason": reason,
        "reduction_tile": reduction_tile,
        "reduction_iterations": iterations,
        "num_stages": num_stages,
        "lowering_attempted": False,
        "cmodel_attempted": False,
    }


def _worker_main(variant: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "s3":
        program = make_chunk_scan_pipeline_s3_kernel()
        runtime_dir = S3_RUNTIME_DIR
    elif variant == "stage3":
        program = make_chunk_scan_pipeline_p5_kernel(SPROG_B, 3)
        runtime_dir = STAGE3_RUNTIME_DIR
        _archive_lowering(program, STAGE3_DIR)
    else:
        raise ValueError(f"unknown P5 worker variant: {variant}")

    kernel = _compile_cmodel(program, runtime_dir)
    if variant == "stage3":
        _archive_stage3_cmodel_sources()
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


def _compare_accepted_outputs(
    s3_outputs: Dict[str, torch.Tensor],
    stage3_outputs: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for name in (
        "state_only",
        "scan_only",
        "residual_only",
        "residual_only_negative_D",
        "all_terms",
    ):
        inputs = _inputs_for_case(name)
        expected = chunk_scan_reference(*inputs)
        s3 = s3_outputs[name]
        stage3 = stage3_outputs[name]
        result = {
            "s3_close_to_oracle": bool(
                torch.allclose(s3.float(), expected.float(), atol=ATOL, rtol=RTOL)
            ),
            "stage3_close_to_oracle": bool(
                torch.allclose(
                    stage3.float(), expected.float(), atol=ATOL, rtol=RTOL
                )
            ),
            "stage3_close_to_s3": bool(
                torch.allclose(stage3.float(), s3.float(), atol=ATOL, rtol=RTOL)
            ),
            "s3_vs_oracle": comparison_stats(s3, expected),
            "stage3_vs_oracle": comparison_stats(stage3, expected),
            "stage3_vs_s3": comparison_stats(stage3, s3),
            "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        }
        if (
            not result["s3_close_to_oracle"]
            or not result["stage3_close_to_oracle"]
            or not result["stage3_close_to_s3"]
            or result["expected_nonzero_count"] == 0
            or result["s3_vs_oracle"]["nan_count"]
            or result["stage3_vs_oracle"]["nan_count"]
        ):
            raise AssertionError(f"P5 accepted-candidate mismatch: {name}: {result}")
        results[name] = result

    clean_s3 = s3_outputs["causal_clean"]
    poison_s3 = s3_outputs["causal_poisoned"]
    clean_stage3 = stage3_outputs["causal_clean"]
    poison_stage3 = stage3_outputs["causal_poisoned"]
    expected = chunk_scan_reference(*_variant_inputs(_make_inputs(20260910), "scan_only"))
    causal = {
        "s3_clean_equals_poisoned": bool(torch.equal(clean_s3, poison_s3)),
        "stage3_clean_equals_poisoned": bool(
            torch.equal(clean_stage3, poison_stage3)
        ),
        "stage3_poisoned_close_to_oracle": bool(
            torch.allclose(
                poison_stage3.float(), expected.float(), atol=ATOL, rtol=RTOL
            )
        ),
        "stage3_poisoned_close_to_s3": bool(
            torch.allclose(
                poison_stage3.float(), poison_s3.float(), atol=ATOL, rtol=RTOL
            )
        ),
        "stage3_poisoned_vs_oracle": comparison_stats(poison_stage3, expected),
        "stage3_poisoned_vs_s3": comparison_stats(poison_stage3, poison_s3),
    }
    if not all(
        causal[key]
        for key in (
            "s3_clean_equals_poisoned",
            "stage3_clean_equals_poisoned",
            "stage3_poisoned_close_to_oracle",
            "stage3_poisoned_close_to_s3",
        )
    ):
        raise AssertionError(f"P5 causal gate failed: {causal}")
    results["causal_upper_triangle_poison"] = causal
    return results


def _load_pass_result(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} prerequisite: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise AssertionError(f"{label} prerequisite is not PASS: {result}")
    return result


def test_pipeline_configuration_matrix_p5_cmodel() -> None:
    toolchain = assert_chunkscan_toolchain()
    print(json.dumps(toolchain, indent=2, sort_keys=True), flush=True)
    s2_result = _load_pass_result(S2_RESULT_PATH, "P4.1 S2")
    s3_result = _load_pass_result(S3_RESULT_PATH, "P4.2 S3")
    if not S3_RAW_SOURCE_PATH.is_file():
        raise FileNotFoundError(S3_RAW_SOURCE_PATH)

    configuration_matrix: Dict[str, Dict[str, object]] = {
        "s2_k16_stage2": {
            "status": "ACCEPTED",
            "source": "closed P4.1 control",
            "reduction_tile": 16,
            "reduction_iterations": 4,
            "num_stages": 2,
            "effective_cb_pipeline_load": False,
            "evidence": str(S2_RESULT_PATH),
        },
        "sprog_b_k16_stage2": {
            "status": "ACCEPTED",
            "source": "closed P4.2 manual reference",
            "schedule": SPROG_B,
            "reduction_tile": 16,
            "reduction_iterations": 4,
            "num_stages": 2,
            "effective_cb_pipeline_load": True,
            "evidence": str(S3_RESULT_PATH),
        },
    }

    s3_structure = s3_result["pipeline_structure"]
    if (
        s3_structure["num_stages"] != 2
        or s3_structure["reduction_iterations"] != 4
        or s3_structure["raw_region_evidence"]["steady_future_load_order"]
        != LOADY_FIRST_ORDER
    ):
        raise AssertionError(f"unexpected P4.2 S3 structure: {s3_structure}")

    # Lower the requested sProg-A source and reject it when the TPU planner
    # rewrites the raw target schedule back to LoadY-first.
    sprog_a_lowering = _archive_lowering(
        make_chunk_scan_pipeline_p5_kernel(SPROG_A, 2),
        SPROG_A_DIR,
    )
    sprog_a_evidence = _pipeline_region_evidence(
        sprog_a_lowering["raw_source"], 2
    )
    actual_a_order = sprog_a_evidence["steady_future_load_order"]
    if actual_a_order == LOADX_FIRST_ORDER:
        raise AssertionError(
            "sProg-A unexpectedly became representable; update the P5 matrix"
        )
    if actual_a_order != LOADY_FIRST_ORDER:
        raise AssertionError(f"unrecognized rewritten sProg-A order: {actual_a_order}")
    configuration_matrix["sprog_a_k16_stage2"] = {
        "status": "REJECTED",
        "reason_code": "schedule_order_not_preserved",
        "reason": (
            "requested LoadX-first source lowers to LoadY-first raw target order"
        ),
        "requested_order": LOADX_FIRST_ORDER,
        "actual_raw_order": actual_a_order,
        "reduction_tile": 16,
        "reduction_iterations": 4,
        "num_stages": 2,
        "lowering_attempted": True,
        "cmodel_attempted": False,
        "raw_evidence": sprog_a_evidence,
        "source_metrics": sprog_a_lowering["source_metrics"],
    }

    configuration_matrix["sprog_b_k32_stage2"] = _loop_depth_rejection(32, 2)
    configuration_matrix["sprog_b_k16_stage4"] = _loop_depth_rejection(16, 4)

    s3_outputs = _run_worker("s3")
    stage3_outputs = _run_worker("stage3")

    stage3_raw = (STAGE3_DIR / "kernel_raw.c").read_text(encoding="utf-8")
    stage3_metrics = _source_metrics(stage3_raw)
    if stage3_metrics != EXPECTED_STAGE3_METRICS:
        raise AssertionError(f"unexpected stage3 source metrics: {stage3_metrics}")
    steady_extent = REDUCE_TILES - 3
    if steady_extent != 1 or "for (int k_blk" in stage3_raw:
        raise AssertionError(
            "stage3 must contain one compiler-unrolled steady iteration"
        )
    stage3_structure = _pipeline_region_evidence(stage3_raw, 3)
    if stage3_structure["steady_future_load_order"] != LOADY_FIRST_ORDER:
        raise AssertionError(f"stage3 violates sProg-B: {stage3_structure}")
    if stage3_metrics["static_local_memory_end_bytes"] > stage3_metrics[
        "bm1690_local_memory_limit_bytes"
    ]:
        raise AssertionError(f"stage3 exceeds LMEM: {stage3_metrics}")

    cmodel_source = (STAGE3_DIR / "kernel_cmodel.c").read_text(encoding="utf-8")
    sanitized_raw = stage3_raw.replace("      tpu_parallel_start(); \n", "").replace(
        "      tpu_parallel_end(); \n", ""
    )
    sanitized_raw = sanitized_raw.replace("tpu_parallel_start(); \n", "").replace(
        "tpu_parallel_end(); \n", ""
    )
    if cmodel_source != sanitized_raw:
        raise AssertionError(
            "stage3 cmodel source differs by more than parallel-marker removal"
        )

    numerical_results = _compare_accepted_outputs(s3_outputs, stage3_outputs)
    configuration_matrix["sprog_b_k16_stage3"] = {
        "status": "ACCEPTED",
        "schedule": SPROG_B,
        "reduction_tile": REDUCE_TILE,
        "reduction_iterations": REDUCE_TILES,
        "num_stages": 3,
        "steady_loop_form": "compiler-unrolled single iteration",
        "steady_loop_extent": steady_extent,
        "effective_cb_pipeline_load": True,
        "raw_structure": stage3_structure,
        "source_metrics": stage3_metrics,
        "cmodel_equals_raw_after_marker_strip": True,
    }

    accepted = sorted(
        name
        for name, result in configuration_matrix.items()
        if result["status"] == "ACCEPTED"
    )
    rejected = sorted(
        name
        for name, result in configuration_matrix.items()
        if result["status"] == "REJECTED"
    )
    if len(accepted) != 3 or len(rejected) != 3:
        raise AssertionError(
            f"unexpected P5 coverage: accepted={accepted}, rejected={rejected}"
        )

    summary = {
        "status": "PASS",
        "stage": "P5",
        "objective": "dependency- and LMEM-aware configuration coverage",
        "configuration_matrix": configuration_matrix,
        "accepted_configurations": accepted,
        "rejected_configurations": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "numerical_results": numerical_results,
        "prerequisites": {
            "pipeline_s2": {"path": str(S2_RESULT_PATH), "status": s2_result["status"]},
            "pipeline_s3": {"path": str(S3_RESULT_PATH), "status": s3_result["status"]},
        },
        "toolchain": toolchain,
        "selection_policy": (
            "paper fidelity, dependency legality, raw structural coverage, "
            "cmodel correctness, and reproducibility; no timing ranking"
        ),
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (ARTIFACT_DIR / "configuration_matrix.json").write_text(
        serialized, encoding="utf-8"
    )
    (ARTIFACT_DIR / "result.json").write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
    elif len(sys.argv) == 1:
        test_pipeline_configuration_matrix_p5_cmodel()
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [--worker s3|stage3]")
