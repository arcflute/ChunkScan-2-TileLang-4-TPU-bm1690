"""P1.5 cmodel probe for ChunkScan casts and the dynamic D*x residual."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
import tilelang
import tilelang.language as T

from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)


SIZE = 64
INPUT_DTYPE = "float16"
ACCUM_DTYPE = "float32"

THIS_DIR = Path(__file__).resolve().parent
CHUNKSCAN_DIR = THIS_DIR.parent
ARTIFACT_DIR = CHUNKSCAN_DIR / "artifacts" / "p1_residual_cast"
RUNTIME_DIR = ARTIFACT_DIR / "runtime"


def _prepare_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    template_dir = REPOSITORY_ROOT / "src" / "tl_templates" / "tpu"
    for name in (
        "kernel_template.cpp",
        "kernel_template.h",
        "main_template.cpp",
    ):
        source = template_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RUNTIME_DIR / name)

    redirected = lambda path=str(RUNTIME_DIR): path
    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = redirected

    os.environ["TPU_KERNEL_PATH"] = str(RUNTIME_DIR)
    os.environ["PPL_KERNEL_PATH"] = str(RUNTIME_DIR / "libkernel.so")


def make_residual_cast_kernel():
    @T.prim_func
    def residual_cast_kernel(
        x: T.Tensor((SIZE, SIZE), INPUT_DTYPE),
        D: T.Tensor((1, 1), INPUT_DTYPE),
        d_matrix_out: T.Tensor((SIZE, SIZE), ACCUM_DTYPE),
        residual_fp32_out: T.Tensor((SIZE, SIZE), ACCUM_DTYPE),
        residual_fp16_out: T.Tensor((SIZE, SIZE), INPUT_DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            x_fp16 = T.alloc_shared((SIZE, SIZE), INPUT_DTYPE)
            x_fp32 = T.alloc_shared((SIZE, SIZE), ACCUM_DTYPE)
            d_fp16 = T.alloc_shared((1, 1), INPUT_DTYPE)
            d_fp32 = T.alloc_shared((1, 1), ACCUM_DTYPE)
            d_row = T.alloc_shared((1, SIZE), ACCUM_DTYPE)
            d_matrix = T.alloc_shared((SIZE, SIZE), ACCUM_DTYPE)
            residual_fp32 = T.alloc_shared(
                (SIZE, SIZE),
                ACCUM_DTYPE,
            )
            residual_fp16 = T.alloc_shared(
                (SIZE, SIZE),
                INPUT_DTYPE,
            )

            T.ppl_copy(x[0, 0], x_fp16)
            T.ppl_copy(D[0, 0], d_fp16)

            T.ppl_copy(x_fp16, x_fp32)
            T.ppl_copy(d_fp16, d_fp32)

            T.ppl_fill(d_row, T.float32(0.0))
            T.ppl_add(d_row, d_row, d_fp32)
            T.ppl_npu_bcast(d_matrix, d_row)

            T.ppl_mul(residual_fp32, x_fp32, d_matrix)
            T.ppl_copy(residual_fp32, residual_fp16)

            T.ppl_copy(d_matrix, d_matrix_out[0, 0])
            T.ppl_copy(residual_fp32, residual_fp32_out[0, 0])
            T.ppl_copy(residual_fp16, residual_fp16_out[0, 0])

    return residual_cast_kernel


def _tensor_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Dict[str, object]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    finite = torch.isfinite(actual_fp32)
    absolute = torch.abs(actual_fp32 - expected_fp32)
    safe_absolute = torch.where(
        finite,
        absolute,
        torch.full_like(absolute, float("inf")),
    )

    return {
        "exact": bool(torch.equal(actual, expected)),
        "mismatch_count": int((actual != expected).sum().item()),
        "max_abs": float(safe_absolute.max().item()),
        "mean_abs": float(safe_absolute.mean().item()),
        "nan_count": int(torch.isnan(actual_fp32).sum().item()),
        "inf_count": int(torch.isinf(actual_fp32).sum().item()),
    }


def _archive_cmodel_sources() -> None:
    mappings = {
        "kernel.c": "kernel_cmodel.c",
        "kernel.cpp": "kernel.cpp",
        "kernel.h": "kernel.h",
        "main.cpp": "main.cpp",
    }

    for source_name, archive_name in mappings.items():
        source = RUNTIME_DIR / source_name
        if source.is_file():
            shutil.copy2(source, ARTIFACT_DIR / archive_name)


def main() -> None:
    toolchain = assert_chunkscan_toolchain()
    _prepare_runtime()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    program = make_residual_cast_kernel()
    lowered = tilelang.lower(program, target="tpu")
    kernel_source = str(lowered.kernel_source)

    (ARTIFACT_DIR / "lowered_host.tir").write_text(
        lowered.host_mod.script(),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "lowered_device.tir").write_text(
        lowered.device_mod.script(),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "kernel_raw.c").write_text(
        kernel_source,
        encoding="utf-8",
    )

    source_metrics = {
        "bdc_cast_site_count": kernel_source.count("tpu_bdc_cast("),
        "fp16_to_fp32_cast_site_count": kernel_source.count(
            "DT_FP32, DT_FP16, RM_HALF_TO_EVEN"
        ),
        "fp32_to_fp16_cast_site_count": kernel_source.count(
            "DT_FP16, DT_FP32, RM_HALF_TO_EVEN"
        ),
        "fill_site_count": kernel_source.count("tpu_bdc_set_C"),
        "bdc_add_site_count": kernel_source.count("tpu_bdc_fp_add"),
        "bdc_mul_site_count": kernel_source.count("tpu_bdc_fp_mul"),
        "npu_bcast_site_count": kernel_source.count(
            "tpu_bdc_npu_bcast"
        ),
        "gdma_s2l_site_count": kernel_source.count(
            "tpu_gdma_cpy_S2L"
        ),
        "gdma_l2s_site_count": kernel_source.count(
            "tpu_gdma_cpy_L2S"
        ),
        "stride_c_zero_count": kernel_source.count(".c = 0;"),
        "stride_w_zero_count": kernel_source.count(".w = 0;"),
    }

    manifest = {
        "stage": "P1.5",
        "probe": "fp16_fp32_cast_and_dynamic_scalar_residual",
        "size": SIZE,
        "input_dtype": INPUT_DTYPE,
        "accumulation_dtype": ACCUM_DTYPE,
        "scalar_cases": {
            "positive": 0.703125,
            "negative": -1.375,
        },
        "dataflow": [
            "x_fp16 -> x_fp32",
            "D_fp16 -> D_fp32",
            "D[1,1] -> D_row[1,64] by W-axis broadcast",
            "D_row[1,64] -> D_matrix[64,64] by npu_bcast",
            "residual_fp32 = x_fp32 * D_matrix",
            "residual_fp32 -> residual_fp16",
        ],
        "toolchain": toolchain,
        "source_metrics": source_metrics,
        "performance_claim": "NONE: cmodel correctness probe only",
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected_source_metrics = {
        "bdc_cast_site_count": 3,
        "fp16_to_fp32_cast_site_count": 2,
        "fp32_to_fp16_cast_site_count": 1,
        "fill_site_count": 1,
        "bdc_add_site_count": 1,
        "bdc_mul_site_count": 1,
        "npu_bcast_site_count": 1,
        "gdma_s2l_site_count": 2,
        "gdma_l2s_site_count": 3,
        "stride_c_zero_count": 0,
        "stride_w_zero_count": 1,
    }
    if source_metrics != expected_source_metrics:
        raise AssertionError(
            "Unexpected generated residual/cast structure: "
            f"{source_metrics}"
        )

    kernel = tilelang.compile(
        program,
        out_idx=[2, 3, 4],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    x = (
        (
            torch.arange(
                SIZE * SIZE,
                dtype=torch.int32,
            )
            % 257
            - 128
        )
        .to(torch.float32)
        .mul_(1.0 / 64.0)
        .to(torch.float16)
        .reshape(SIZE, SIZE)
        .contiguous()
    )

    cases: Dict[str, object] = {}
    for name, scalar in (
        ("positive", 0.703125),
        ("negative", -1.375),
    ):
        d = torch.tensor([[scalar]], dtype=torch.float16)
        actual_d = torch.full(
            (SIZE, SIZE),
            float("nan"),
            dtype=torch.float32,
        )
        actual_fp32 = torch.full_like(actual_d, float("nan"))
        actual_fp16 = torch.full(
            (SIZE, SIZE),
            float("nan"),
            dtype=torch.float16,
        )

        kernel(x, d, actual_d, actual_fp32, actual_fp16)

        expected_d = d.float().expand(SIZE, SIZE).clone()
        expected_fp32 = x.float() * d.float()
        expected_fp16 = expected_fp32.to(torch.float16)
        cases[name] = {
            "d_matrix_fp32": _tensor_stats(
                actual_d,
                expected_d,
            ),
            "residual_fp32": _tensor_stats(
                actual_fp32,
                expected_fp32,
            ),
            "residual_fp16": _tensor_stats(
                actual_fp16,
                expected_fp16,
            ),
        }

    passed = all(
        tensor_case["exact"]
        for scalar_case in cases.values()
        for tensor_case in scalar_case.values()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "cases": cases,
        "source_metrics": source_metrics,
        "performance_claim": "NONE: cmodel correctness probe only",
    }
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if not passed:
        raise AssertionError(f"P1.5 residual/cast probe failed: {result}")


if __name__ == "__main__":
    main()