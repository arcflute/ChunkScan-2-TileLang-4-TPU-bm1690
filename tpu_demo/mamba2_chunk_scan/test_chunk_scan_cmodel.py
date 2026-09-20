"""Compile and validate serial standalone ChunkScan with the BM1690 cmodel."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import tilelang
from tilelang import tvm

from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)

from tpu_demo.mamba2_chunk_scan.chunk_scan_serial import (
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
    SEQLEN,
    make_chunk_scan_serial_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "serial"
RUNTIME_DIR = ARTIFACT_DIR / "runtime"
ATOL = 1e-2
RTOL = 1e-2

EXPECTED_SOURCE_SITE_COUNTS = {
    "gdma_s2l_site_count": 71,
    "gdma_l2s_site_count": 1,
    "bdc_copy_site_count": 0,
    "bdc_cast_site_count": 9,
    "fp16_to_fp32_cast_site_count": 7,
    "fp32_to_fp16_cast_site_count": 2,
    "fill_site_count": 5,
    "bdc_add_site_count": 4,
    "bdc_sub_site_count": 1,
    "bdc_mul_site_count": 4,
    "fp_mm_site_count": 1,
    "fp_mm_r_trans_site_count": 1,
    "fp32_exp_site_count": 2,
    "npu_bcast_site_count": 3,
    "stride_c_zero_count": 0,
    "stride_w_zero_count": 3,
    "chunk_loop_count": 1,
    "dA_scatter_loop_count": 1,
    "parallel_start_count": 0,
    "parallel_end_count": 0,
}


Inputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]



def _prepare_isolated_codegen_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    template_dir = REPOSITORY_ROOT / "src" / "tl_templates" / "tpu"
    for name in ("kernel_template.cpp", "kernel_template.h", "main_template.cpp"):
        source = template_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RUNTIME_DIR / name)

    # The TPU backend resolves the template directory at several import sites.
    # Redirect them to the per-run artifact directory so generated files never
    # overwrite source-controlled templates.
    adapter_utils = importlib.import_module("tilelang.jit.adapter.utils")
    libgen = importlib.import_module("tilelang.jit.adapter.libgen")
    wrapper = importlib.import_module("tilelang.jit.adapter.wrapper")
    redirected = lambda path=str(RUNTIME_DIR): path
    for module in (adapter_utils, libgen, wrapper):
        module.get_tpu_template_dir = redirected

    os.environ["TPU_KERNEL_PATH"] = str(RUNTIME_DIR)
    os.environ["PPL_KERNEL_PATH"] = str(RUNTIME_DIR / "libkernel.so")


def _source_metrics(source: str) -> Dict[str, object]:
    dtype_bytes = {
        "DT_FP16": 2,
        "DT_BFP16": 2,
        "DT_FP32": 4,
        "DT_INT32": 4,
        "DT_INT8": 1,
        "DT_FP8E4M3": 1,
    }
    allocation_ends = []
    for shape_text, address, dtype in re.findall(
        r"\.shape\s*=\s*\{([^}]+)\}.*?\.addr\s*=\s*(\d+).*?\.dtype\s*=\s*(DT_[A-Z0-9]+)",
        source,
    ):
        dimensions = [int(value) for value in re.findall(r"\d+", shape_text)]
        elements = 1
        for dimension in dimensions:
            elements *= dimension
        if dtype in dtype_bytes:
            allocation_ends.append(
                int(address) + max(64, elements * dtype_bytes[dtype])
            )
    return {
        "bytes": len(source.encode("utf-8")),
        "gdma_s2l_site_count": source.count("tpu_gdma_cpy_S2L"),
        "gdma_l2s_site_count": source.count("tpu_gdma_cpy_L2S"),
        "bdc_copy_site_count": source.count("tpu_bdc_cpy"),
        "bdc_cast_site_count": source.count("tpu_bdc_cast("),
        "fp16_to_fp32_cast_site_count": source.count(
            "DT_FP32, DT_FP16, RM_HALF_TO_EVEN"
        ),
        "fp32_to_fp16_cast_site_count": source.count(
            "DT_FP16, DT_FP32, RM_HALF_TO_EVEN"
        ),
        "fill_site_count": source.count("tpu_bdc_set_C"),
        "bdc_add_site_count": source.count("tpu_bdc_fp_add"),
        "bdc_sub_site_count": source.count("tpu_bdc_fp_sub"),
        "bdc_mul_site_count": source.count("tpu_bdc_fp_mul"),
        "fp_mm_site_count": source.count("tpu_bdc_fp_mm("),
        "fp_mm_r_trans_site_count": source.count(
            "tpu_bdc_fp_mm_R_trans("
        ),
        "fp32_exp_site_count": source.count("tpu_bdc_fp32_exp"),
        "npu_bcast_site_count": source.count("tpu_bdc_npu_bcast"),
        "stride_c_zero_count": source.count(".c = 0;"),
        "stride_w_zero_count": source.count(".w = 0;"),
        "chunk_loop_count": source.count(
            "for (int chunk = 0; chunk < 2; ++chunk)"
        ),
        "dA_scatter_loop_count": source.count(
            "for (int index = 0; index < 64; ++index)"
        ),
        "parallel_start_count": source.count("tpu_parallel_start"),
        "parallel_end_count": source.count("tpu_parallel_end"),
        "static_local_memory_end_bytes": max(allocation_ends, default=None),
        "bm1690_local_memory_limit_bytes": 256 * 1024,
    }


def _archive_lowering(program, toolchain_paths: Dict[str, str]) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = tilelang.lower(program, target="tpu")
    source = str(artifact.kernel_source)
    (ARTIFACT_DIR / "lowered_host.tir").write_text(
        artifact.host_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "lowered_device.tir").write_text(
        artifact.device_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "kernel_raw.c").write_text(source, encoding="utf-8")
    metrics = _source_metrics(source)
    (ARTIFACT_DIR / "raw_source_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "variant": "serial_s0",
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
        "dtype": "float16",
        "accumulation_dtype": "float32",
        "core_num": CORE_NUM,
        "num_stages": NUM_STAGES,
        "math_terms": ["historical_state", "causal_chunk_scan", "D*x"],
        "toolchain": toolchain_paths,
        "source_metrics": metrics,
        "expected_source_site_counts": EXPECTED_SOURCE_SITE_COUNTS,
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _archive_cmodel_sources() -> None:
    for source_name, archive_name in (
        ("kernel.c", "kernel_cmodel.c"),
        ("kernel.cpp", "kernel.cpp"),
        ("kernel.h", "kernel.h"),
        ("main.cpp", "main.cpp"),
    ):
        source = RUNTIME_DIR / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ARTIFACT_DIR / archive_name)


def _random_tensor(
    shape: Iterable[int], generator: torch.Generator, scale: float
) -> torch.Tensor:
    value = (torch.rand(tuple(shape), generator=generator) * 2.0 - 1.0) * scale
    return value.to(torch.float16).contiguous()


def _make_inputs(seed: int = 20260910) -> Inputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cb = _random_tensor(
        (BATCH, NCHUNKS, NGROUPS, CHUNK_SIZE, CHUNK_SIZE), generator, 0.20
    )
    x = _random_tensor((BATCH, SEQLEN, NHEADS, HEADDIM), generator, 0.20)
    dt32 = torch.rand(
        (BATCH, NHEADS, NCHUNKS, CHUNK_SIZE), generator=generator
    ) * 0.10 + 0.05
    dt = dt32.to(torch.float16).contiguous()
    A = -(torch.rand((NHEADS,), generator=generator) * 0.15 + 0.05)
    dA = torch.cumsum(dt.float() * A.view(1, NHEADS, 1, 1), dim=-1)
    dA = dA.to(torch.float16).contiguous()
    C = _random_tensor((BATCH, SEQLEN, NGROUPS, DSTATE), generator, 0.10)
    states = _random_tensor(
        (BATCH, NCHUNKS, NHEADS, HEADDIM, DSTATE), generator, 0.10
    )
    D = torch.full((NHEADS,), 0.1875, dtype=torch.float16)
    return cb, x, dt, dA, C, states, D


def _variant_inputs(base: Inputs, variant: str) -> Inputs:
    cb, x, dt, dA, C, states, D = (tensor.clone() for tensor in base)
    if variant == "state_only":
        cb.zero_()
        x.zero_()
        D.zero_()
    elif variant == "scan_only":
        states.zero_()
        D.zero_()
    elif variant == "residual_only":
        cb.zero_()
        states.zero_()
    elif variant != "all_terms":
        raise ValueError(variant)
    return cb, x, dt, dA, C, states, D


def _physical_args(inputs: Inputs, output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    cb, x, dt, dA, C, states, D = inputs
    return (
        cb.view(PHYSICAL_SHAPES["cb"]),
        x.view(PHYSICAL_SHAPES["x"]),
        dt.view(PHYSICAL_SHAPES["dt"]),
        dA.view(PHYSICAL_SHAPES["dA_cumsum"]),
        C.view(PHYSICAL_SHAPES["C"]),
        states.view(PHYSICAL_SHAPES["prev_states"]),
        D.view(PHYSICAL_SHAPES["D"]),
        output.view(PHYSICAL_SHAPES["out"]),
    )


def _run_case(kernel, inputs: Inputs, name: str) -> Dict[str, object]:
    expected = chunk_scan_reference(*inputs)
    actual = torch.full_like(expected, float("nan"))
    returned = kernel(*_physical_args(inputs, actual))
    stats = comparison_stats(actual, expected)
    close = bool(torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL))
    expected_by_chunk = expected.view(
        BATCH,
        NCHUNKS,
        CHUNK_SIZE,
        NHEADS,
        HEADDIM,
    )
    expected_chunk_max_abs = [
        float(expected_by_chunk[:, chunk].float().abs().max().item())
        for chunk in range(NCHUNKS)
    ]
    result = {
        "case": name,
        "returned_type": type(returned).__name__,
        "close_atol_1e-2_rtol_1e-2": close,
        "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        "expected_max_abs": float(expected.float().abs().max().item()),
        "expected_chunk_max_abs": expected_chunk_max_abs,
        **stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if (
        not close
        or stats["nan_count"]
        or stats["inf_count"]
        or result["expected_nonzero_count"] == 0
        or any(value == 0.0 for value in expected_chunk_max_abs)
    ):
        raise AssertionError(f"cmodel mismatch for {name}: {result}")
    return result


def test_serial_chunk_scan_cmodel() -> None:
    toolchain_paths = assert_chunkscan_toolchain()
    print(json.dumps(toolchain_paths, indent=2, sort_keys=True), flush=True)
    _prepare_isolated_codegen_dir()
    program = make_chunk_scan_serial_kernel()
    manifest = _archive_lowering(program, toolchain_paths)
    metrics = manifest["source_metrics"]
    actual_source_site_counts = {
        name: metrics[name]
        for name in EXPECTED_SOURCE_SITE_COUNTS
    }
    if actual_source_site_counts != EXPECTED_SOURCE_SITE_COUNTS:
        raise AssertionError(
            "unexpected serial S0 source structure: "
            f"{actual_source_site_counts}"
        )
    if (
        metrics["static_local_memory_end_bytes"]
        > metrics["bm1690_local_memory_limit_bytes"]
    ):
        raise AssertionError(f"serial kernel exceeds LMEM: {metrics}")

    kernel = tilelang.compile(
        program,
        out_idx=[7],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    base = _make_inputs()
    results = {
        name: _run_case(kernel, _variant_inputs(base, name), name)
        for name in ("state_only", "scan_only", "residual_only", "all_terms")
    }
    residual_negative = list(_variant_inputs(base, "residual_only"))
    residual_negative[-1].fill_(-0.28125)
    results["residual_only_negative_D"] = _run_case(
        kernel,
        tuple(residual_negative),
        "residual_only_negative_D",
    )

    # Poisoning CB's upper triangle must not affect the device output.
    causal_base = _variant_inputs(base, "scan_only")
    upper = torch.triu(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool), diagonal=1
    ).view(1, 1, 1, CHUNK_SIZE, CHUNK_SIZE)
    poisoned_cb = torch.where(upper, torch.full_like(causal_base[0], 8.0), causal_base[0])
    poisoned = (poisoned_cb.contiguous(),) + causal_base[1:]
    expected = chunk_scan_reference(*causal_base)
    actual_clean = torch.full_like(expected, float("nan"))
    actual_poisoned = torch.full_like(expected, float("nan"))
    kernel(*_physical_args(causal_base, actual_clean))
    kernel(*_physical_args(poisoned, actual_poisoned))
    causal_equal = bool(torch.equal(actual_poisoned, actual_clean))
    causal_stats = comparison_stats(actual_poisoned, expected)
    causal_close = bool(
        torch.allclose(
            actual_poisoned.float(),
            expected.float(),
            atol=ATOL,
            rtol=RTOL,
        )
    )
    results["causal_upper_triangle_poison"] = {
        "bitwise_equal_to_clean_device_output": causal_equal,
        "close_to_oracle_atol_1e-2_rtol_1e-2": causal_close,
        **causal_stats,
    }
    if not causal_equal or not causal_close:
        raise AssertionError(f"upper-triangle CB affected output: {causal_stats}")

    summary = {
        "status": "PASS",
        "cases": results,
        "source_metrics": metrics,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    test_serial_chunk_scan_cmodel()
