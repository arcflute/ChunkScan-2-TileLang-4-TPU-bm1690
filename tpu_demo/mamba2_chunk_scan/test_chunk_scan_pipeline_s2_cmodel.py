"""Validate the P4 two-stage ChunkScan S2 pipeline with BM1690 cmodel."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import tilelang
import torch

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s2 import (
    BATCH,
    CHUNK_SIZE,
    CORE_NUM,
    DSTATE,
    HEADDIM,
    NCHUNKS,
    NGROUPS,
    NHEADS,
    NUM_STAGES,
    PHYSICAL_SHAPES,
    REDUCE_TILE,
    REDUCE_TILES,
    SEQLEN,
    make_chunk_scan_pipeline_s2_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_reduction_serial import (
    make_chunk_scan_reduction_serial_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_cmodel import (
    Inputs,
    _make_inputs,
    _physical_args,
    _source_metrics,
    _variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_s2"
S1_RUNTIME_DIR = ARTIFACT_DIR / "runtime_s1_control"
S2_RUNTIME_DIR = ARTIFACT_DIR / "runtime_s2"
S1_RESULT_PATH = (
    REPOSITORY_ROOT
    / "tpu_demo"
    / "mamba2_chunk_scan"
    / "artifacts"
    / "reduction_serial"
    / "result.json"
)
ATOL = 1e-2
RTOL = 1e-2

EXPECTED_SOURCE_SITE_COUNTS = {
    "gdma_s2l_site_count": 175,
    "gdma_l2s_site_count": 1,
    "bdc_copy_site_count": 4,
    "bdc_cast_site_count": 17,
    "fp16_to_fp32_cast_site_count": 13,
    "fp32_to_fp16_cast_site_count": 4,
    "fill_site_count": 13,
    "bdc_add_site_count": 9,
    "bdc_sub_site_count": 3,
    "bdc_mul_site_count": 8,
    "fp_mm_site_count": 3,
    "fp_mm_r_trans_site_count": 1,
    "fp32_exp_site_count": 4,
    "npu_bcast_site_count": 7,
    "stride_c_zero_count": 0,
    "stride_w_zero_count": 5,
    "chunk_loop_count": 1,
    "dA_scatter_loop_count": 1,
    "parallel_start_count": 1,
    "parallel_end_count": 1,
}

EXPECTED_REDUCTION_LOOP = "for (int k_blk = 0; k_blk < 2; ++k_blk)"

EXPECTED_DOUBLE_BUFFER_DECLARATIONS = (
    "__ppl_tensor_info dA_reduce_row_fp16_shared_0 =",
    "__ppl_tensor_info dA_reduce_row_fp16_shared_1 =",
    "__ppl_tensor_info dt_reduce_row_fp16_shared_0 =",
    "__ppl_tensor_info dt_reduce_row_fp16_shared_1 =",
    "__ppl_tensor_info x_reduce_shared_0 =",
    "__ppl_tensor_info x_reduce_shared_1 =",
)

STASK_MAPPING = {
    "pipeline_axis": "causal_scan_reduction_s",
    "reduction_tile": REDUCE_TILE,
    "reduction_iterations": REDUCE_TILES,
    "tpu_engine_classes": {
        "GDMA": ["CausalPack.cb", "LoadY.dA", "LoadY.dt", "LoadX"],
        "BDC": [
            "DecayScale.exp",
            "DecayScale.broadcast",
            "DecayScale.mul",
            "MMA.gemm",
            "MMA.fp32_add",
        ],
    },
    "dependency_edges": [
        ["CausalPack.cb", "DecayScale.mul_cb"],
        ["LoadY.dA", "DecayScale.exp"],
        ["LoadY.dt", "DecayScale.mul_dt"],
        ["DecayScale.exp", "DecayScale.mul_cb"],
        ["DecayScale.mul_cb", "DecayScale.mul_dt"],
        ["DecayScale.mul_dt", "MMA.gemm"],
        ["LoadX", "MMA.gemm"],
        ["MMA.gemm", "MMA.fp32_add"],
        ["MMA.fp32_add[k]", "MMA.fp32_add[k+1]"],
    ],
    "pipeline_order_per_iteration": [
        "LoadY.dA+dt",
        "DecayScale",
        "LoadX",
        "MMA",
    ],
    "outside_pipeline": ["CausalPack.cb"],
    "pipeline_enabled": True,
    "buffer_versions": 2,
    "performance_claim": "NONE: cmodel timing is not TPU performance",
}


def _prepare_isolated_codegen_dir(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    template_dir = REPOSITORY_ROOT / "src" / "tl_templates" / "tpu"
    for name in ("kernel_template.cpp", "kernel_template.h", "main_template.cpp"):
        source = template_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, runtime_dir / name)

    adapter_utils = importlib.import_module("tilelang.jit.adapter.utils")
    libgen = importlib.import_module("tilelang.jit.adapter.libgen")
    wrapper = importlib.import_module("tilelang.jit.adapter.wrapper")
    redirected = lambda path=str(runtime_dir): path
    for module in (adapter_utils, libgen, wrapper):
        module.get_tpu_template_dir = redirected

    os.environ["TPU_KERNEL_PATH"] = str(runtime_dir)
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")


def _compile_cmodel(program, runtime_dir: Path):
    _prepare_isolated_codegen_dir(runtime_dir)
    return tilelang.compile(
        program,
        out_idx=[7],
        target="tpu",
        mode="cmodel",
    )


def _archive_lowering(program, toolchain: Dict[str, object]) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = tilelang.lower(program, target="tpu")
    raw_source = str(artifact.kernel_source)
    (ARTIFACT_DIR / "lowered_host.tir").write_text(
        artifact.host_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "lowered_device.tir").write_text(
        artifact.device_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "kernel_raw.c").write_text(raw_source, encoding="utf-8")

    metrics = _source_metrics(raw_source)
    (ARTIFACT_DIR / "raw_source_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ARTIFACT_DIR / "stask_mapping.json").write_text(
        json.dumps(STASK_MAPPING, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "variant": "reduction_pipeline_s2",
        "serial_control": "reduction_tiled_serial_s1",
        "logical_abi": [
            "cb",
            "x",
            "dt",
            "dA_cumsum",
            "C",
            "prev_states",
            "D",
            "out",
        ],
        "logical_shape": {
            "B": BATCH,
            "S": SEQLEN,
            "Ck": NCHUNKS,
            "L": CHUNK_SIZE,
            "G": NGROUPS,
            "H": NHEADS,
            "P": HEADDIM,
            "N": DSTATE,
        },
        "physical_2d_shapes": {
            name: list(shape) for name, shape in PHYSICAL_SHAPES.items()
        },
        "core_num": CORE_NUM,
        "num_stages": NUM_STAGES,
        "buffer_versions": 2,
        "reduction_axis": "causal scan source-position s",
        "reduction_tile": REDUCE_TILE,
        "reduction_iterations": REDUCE_TILES,
        "math_terms": ["historical_state", "causal_chunk_scan", "D*x"],
        "stask_mapping": STASK_MAPPING,
        "toolchain": toolchain,
        "source_metrics": metrics,
        "expected_source_site_counts": EXPECTED_SOURCE_SITE_COUNTS,
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "raw_source": raw_source}


def _archive_s2_cmodel_sources() -> None:
    for source_name, archive_name in (
        ("kernel.c", "kernel_cmodel.c"),
        ("kernel.cpp", "kernel.cpp"),
        ("kernel.h", "kernel.h"),
        ("main.cpp", "main.cpp"),
    ):
        source = S2_RUNTIME_DIR / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ARTIFACT_DIR / archive_name)


def _execute(kernel, inputs: Inputs) -> torch.Tensor:
    expected_shape = (BATCH, SEQLEN, NHEADS, HEADDIM)
    output = torch.full(expected_shape, float("nan"), dtype=torch.float16)
    kernel(*_physical_args(inputs, output))
    return output


def _worker_outputs(kernel) -> Dict[str, torch.Tensor]:
    base = _make_inputs(seed=20260910)
    outputs = {
        name: _execute(kernel, _variant_inputs(base, name))
        for name in ("state_only", "scan_only", "residual_only", "all_terms")
    }

    negative = list(_variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    outputs["residual_only_negative_D"] = _execute(kernel, tuple(negative))

    causal_base = _variant_inputs(base, "scan_only")
    upper = torch.triu(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool), diagonal=1
    ).view(1, 1, 1, CHUNK_SIZE, CHUNK_SIZE)
    poisoned_cb = torch.where(
        upper,
        torch.full_like(causal_base[0], 8.0),
        causal_base[0],
    )
    poisoned = (poisoned_cb.contiguous(),) + causal_base[1:]
    outputs["causal_clean"] = _execute(kernel, causal_base)
    outputs["causal_poisoned"] = _execute(kernel, poisoned)
    return outputs


def _worker_main(variant: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "s1":
        program = make_chunk_scan_reduction_serial_kernel()
        runtime_dir = S1_RUNTIME_DIR
    elif variant == "s2":
        program = make_chunk_scan_pipeline_s2_kernel()
        runtime_dir = S2_RUNTIME_DIR
    else:
        raise ValueError(f"unknown worker variant: {variant}")

    if variant == "s2":
        # Archive the raw source in the same fresh process that compiles the
        # cmodel adapter.  Descriptor declarations may be emitted in a
        # different harmless order across independent Python processes.
        _archive_lowering(program, assert_chunkscan_toolchain())
    kernel = _compile_cmodel(program, runtime_dir)
    if variant == "s2":
        _archive_s2_cmodel_sources()
    output_path = ARTIFACT_DIR / f"worker_{variant}_outputs.pt"
    torch.save(_worker_outputs(kernel), output_path)


def _run_worker(variant: str) -> Dict[str, torch.Tensor]:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", variant],
        check=True,
    )
    output_path = ARTIFACT_DIR / f"worker_{variant}_outputs.pt"
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    return torch.load(output_path, map_location="cpu", weights_only=True)


def _case_result(
    actual_s1: torch.Tensor,
    actual_s2: torch.Tensor,
    inputs: Inputs,
    name: str,
) -> Dict[str, object]:
    expected = chunk_scan_reference(*inputs)
    s1_stats = comparison_stats(actual_s1, expected)
    s2_stats = comparison_stats(actual_s2, expected)
    cross_stats = comparison_stats(actual_s2, actual_s1)
    s1_close = bool(
        torch.allclose(actual_s1.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    s2_close = bool(
        torch.allclose(actual_s2.float(), expected.float(), atol=ATOL, rtol=RTOL)
    )
    cross_close = bool(
        torch.allclose(actual_s2.float(), actual_s1.float(), atol=ATOL, rtol=RTOL)
    )
    expected_by_chunk = expected.view(
        BATCH, NCHUNKS, CHUNK_SIZE, NHEADS, HEADDIM
    )
    chunk_max_abs = [
        float(expected_by_chunk[:, chunk].float().abs().max().item())
        for chunk in range(NCHUNKS)
    ]
    result = {
        "case": name,
        "s1_close_to_oracle": s1_close,
        "s2_close_to_oracle": s2_close,
        "s2_close_to_s1": cross_close,
        "s1_vs_oracle": s1_stats,
        "s2_vs_oracle": s2_stats,
        "s2_vs_s1": cross_stats,
        "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        "expected_chunk_max_abs": chunk_max_abs,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if (
        not s1_close
        or not s2_close
        or not cross_close
        or s1_stats["nan_count"]
        or s2_stats["nan_count"]
        or result["expected_nonzero_count"] == 0
        or any(value == 0.0 for value in chunk_max_abs)
    ):
        raise AssertionError(f"P4 S2 mismatch for {name}: {result}")
    return result


def test_pipeline_s2_chunk_scan_cmodel() -> None:
    toolchain = assert_chunkscan_toolchain()
    print(json.dumps(toolchain, indent=2, sort_keys=True), flush=True)

    if not S1_RESULT_PATH.is_file():
        raise FileNotFoundError(
            f"missing closed P3 S1 evidence: {S1_RESULT_PATH}"
        )
    s1_result = json.loads(S1_RESULT_PATH.read_text(encoding="utf-8"))
    if s1_result.get("status") != "PASS":
        raise AssertionError(f"P3 S1 control is not PASS: {s1_result}")

    s2_program = make_chunk_scan_pipeline_s2_kernel()
    lowering = _archive_lowering(s2_program, toolchain)
    manifest = lowering["manifest"]
    raw_source = lowering["raw_source"]
    metrics = manifest["source_metrics"]
    actual_counts = {
        name: metrics[name] for name in EXPECTED_SOURCE_SITE_COUNTS
    }
    if actual_counts != EXPECTED_SOURCE_SITE_COUNTS:
        raise AssertionError(f"unexpected P4 S2 source structure: {actual_counts}")
    if raw_source.count(EXPECTED_REDUCTION_LOOP) != 1:
        raise AssertionError("P4 S2 does not contain one two-iteration steady loop")
    for declaration in EXPECTED_DOUBLE_BUFFER_DECLARATIONS:
        if raw_source.count(declaration) != 1:
            raise AssertionError(
                f"P4 S2 missing double-buffer declaration: {declaration}"
            )
    if REDUCE_TILES < 4:
        raise AssertionError("P4 S2 requires at least four reduction iterations")
    if metrics["static_local_memory_end_bytes"] > metrics["bm1690_local_memory_limit_bytes"]:
        raise AssertionError(f"P4 S2 exceeds LMEM: {metrics}")

    s1_outputs = _run_worker("s1")
    s2_outputs = _run_worker("s2")

    base = _make_inputs(seed=20260910)
    results = {
        name: _case_result(
            s1_outputs[name],
            s2_outputs[name],
            _variant_inputs(base, name),
            name,
        )
        for name in ("state_only", "scan_only", "residual_only", "all_terms")
    }

    negative = list(_variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    results["residual_only_negative_D"] = _case_result(
        s1_outputs["residual_only_negative_D"],
        s2_outputs["residual_only_negative_D"],
        tuple(negative),
        "residual_only_negative_D",
    )

    causal_base = _variant_inputs(base, "scan_only")
    clean_s1 = s1_outputs["causal_clean"]
    poisoned_s1 = s1_outputs["causal_poisoned"]
    clean_s2 = s2_outputs["causal_clean"]
    poisoned_s2 = s2_outputs["causal_poisoned"]
    expected = chunk_scan_reference(*causal_base)
    causal_result = {
        "bitwise_equal_clean_vs_poisoned_s1": bool(
            torch.equal(clean_s1, poisoned_s1)
        ),
        "bitwise_equal_clean_vs_poisoned_s2": bool(
            torch.equal(clean_s2, poisoned_s2)
        ),
        "poisoned_s2_close_to_oracle": bool(
            torch.allclose(
                poisoned_s2.float(), expected.float(), atol=ATOL, rtol=RTOL
            )
        ),
        "poisoned_s2_close_to_s1": bool(
            torch.allclose(
                poisoned_s2.float(), poisoned_s1.float(), atol=ATOL, rtol=RTOL
            )
        ),
        "poisoned_s2_vs_oracle": comparison_stats(poisoned_s2, expected),
        "poisoned_s2_vs_s1": comparison_stats(poisoned_s2, poisoned_s1),
    }
    if not all(
        (
            causal_result["bitwise_equal_clean_vs_poisoned_s1"],
            causal_result["bitwise_equal_clean_vs_poisoned_s2"],
            causal_result["poisoned_s2_close_to_oracle"],
            causal_result["poisoned_s2_close_to_s1"],
        )
    ):
        raise AssertionError(f"P4 S2 causal poisoning failed: {causal_result}")
    results["causal_upper_triangle_poison"] = causal_result

    archived_raw_source = (ARTIFACT_DIR / "kernel_raw.c").read_text(
        encoding="utf-8"
    )
    cmodel_source = (ARTIFACT_DIR / "kernel_cmodel.c").read_text(
        encoding="utf-8"
    )
    sanitized_raw_source = archived_raw_source.replace(
        "      tpu_parallel_start(); \n", ""
    ).replace("      tpu_parallel_end(); \n", "")
    sanitized_raw_source = sanitized_raw_source.replace(
        "tpu_parallel_start(); \n", ""
    ).replace("tpu_parallel_end(); \n", "")
    if cmodel_source != sanitized_raw_source:
        raise AssertionError(
            "P4 S2 cmodel source differs by more than parallel-marker removal"
        )

    summary = {
        "status": "PASS",
        "variant": "reduction_pipeline_s2",
        "pipeline_structure": {
            "pipeline_axis": "causal_scan_reduction_s",
            "reduction_iterations": REDUCE_TILES,
            "num_stages": NUM_STAGES,
            "steady_loop": EXPECTED_REDUCTION_LOOP,
            "steady_loop_extent": REDUCE_TILES - NUM_STAGES,
            "raw_parallel_start_count": raw_source.count(
                "tpu_parallel_start();"
            ),
            "raw_parallel_end_count": raw_source.count(
                "tpu_parallel_end();"
            ),
            "cmodel_parallel_start_count": cmodel_source.count(
                "tpu_parallel_start();"
            ),
            "cmodel_parallel_end_count": cmodel_source.count(
                "tpu_parallel_end();"
            ),
            "double_buffered_producers": [
                "dA_reduce_row_fp16_shared",
                "dt_reduce_row_fp16_shared",
                "x_reduce_shared",
            ],
            "causal_cb_strategy": (
                "four immutable lower-triangular K16 tiles materialized "
                "before the pipeline"
            ),
        },
        "reduction_serial_control_evidence": {
            "path": str(S1_RESULT_PATH),
            "status": s1_result["status"],
            "comparison_method": (
                "S1 and S2 run in separate cmodel worker processes, then the "
                "saved outputs are compared directly and against the same "
                "CPU oracle"
            ),
        },
        "cases": results,
        "source_metrics": metrics,
        "raw_equals_cmodel_source": False,
        "cmodel_equals_raw_after_marker_strip": True,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
    elif len(sys.argv) == 1:
        test_pipeline_s2_chunk_scan_cmodel()
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [--worker s1|s2]")
