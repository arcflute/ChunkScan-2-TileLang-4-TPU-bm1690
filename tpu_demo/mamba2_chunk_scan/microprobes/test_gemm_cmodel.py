"""P1.4 cmodel probe for the two GEMMs used by ChunkScan."""

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


CHUNK = 64
STATE = 128
INPUT_DTYPE = "float16"
ACCUM_DTYPE = "float32"
RTOL = 1.0e-4
ATOL = 1.0e-4

THIS_DIR = Path(__file__).resolve().parent
CHUNKSCAN_DIR = THIS_DIR.parent
ARTIFACT_DIR = CHUNKSCAN_DIR / "artifacts" / "p1_gemm"
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
        importlib.import_module(module_name).get_tpu_template_dir = redirected

    os.environ["TPU_KERNEL_PATH"] = str(RUNTIME_DIR)
    os.environ["PPL_KERNEL_PATH"] = str(RUNTIME_DIR / "libkernel.so")


def make_gemm_kernel():
    @T.prim_func
    def gemm_kernel(
        historical_left: T.Tensor((CHUNK, STATE), INPUT_DTYPE),
        historical_right: T.Tensor((CHUNK, STATE), INPUT_DTYPE),
        scan_left: T.Tensor((CHUNK, CHUNK), INPUT_DTYPE),
        scan_right: T.Tensor((CHUNK, CHUNK), INPUT_DTYPE),
        historical_out: T.Tensor((CHUNK, CHUNK), ACCUM_DTYPE),
        scan_out: T.Tensor((CHUNK, CHUNK), ACCUM_DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            historical_left_local = T.alloc_shared(
                (CHUNK, STATE),
                INPUT_DTYPE,
            )
            historical_right_local = T.alloc_shared(
                (CHUNK, STATE),
                INPUT_DTYPE,
            )
            scan_left_local = T.alloc_shared(
                (CHUNK, CHUNK),
                INPUT_DTYPE,
            )
            scan_right_local = T.alloc_shared(
                (CHUNK, CHUNK),
                INPUT_DTYPE,
            )
            historical_accum = T.alloc_shared(
                (CHUNK, CHUNK),
                ACCUM_DTYPE,
            )
            scan_accum = T.alloc_shared(
                (CHUNK, CHUNK),
                ACCUM_DTYPE,
            )

            T.ppl_copy(
                historical_left[0, 0],
                historical_left_local,
            )
            T.ppl_copy(
                historical_right[0, 0],
                historical_right_local,
            )
            T.ppl_copy(
                scan_left[0, 0],
                scan_left_local,
            )
            T.ppl_copy(
                scan_right[0, 0],
                scan_right_local,
            )

            T.ppl_fill(
                historical_accum,
                T.float32(0.0),
            )
            T.ppl_gemm(
                historical_left_local,
                historical_right_local,
                historical_accum,
                transpose_B=True,
            )

            T.ppl_fill(
                scan_accum,
                T.float32(0.0),
            )
            T.ppl_gemm(
                scan_left_local,
                scan_right_local,
                scan_accum,
            )

            T.ppl_copy(
                historical_accum,
                historical_out[0, 0],
            )
            T.ppl_copy(
                scan_accum,
                scan_out[0, 0],
            )

    return gemm_kernel


def _stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Dict[str, object]:
    finite = torch.isfinite(actual)
    absolute = torch.abs(actual - expected)
    safe_absolute = torch.where(
        finite,
        absolute,
        torch.full_like(absolute, float("inf")),
    )
    relative = safe_absolute / torch.clamp(
        torch.abs(expected),
        min=1.0e-30,
    )

    return {
        "close": bool(
            torch.allclose(
                actual,
                expected,
                rtol=RTOL,
                atol=ATOL,
            )
        ),
        "max_abs": float(safe_absolute.max().item()),
        "mean_abs": float(safe_absolute.mean().item()),
        "max_rel": float(relative.max().item()),
        "nan_count": int(torch.isnan(actual).sum().item()),
        "inf_count": int(torch.isinf(actual).sum().item()),
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
            shutil.copy2(
                source,
                ARTIFACT_DIR / archive_name,
            )


def main() -> None:
    toolchain = assert_chunkscan_toolchain()
    _prepare_runtime()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    program = make_gemm_kernel()
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
        "fp_mm_site_count": kernel_source.count(
            "tpu_bdc_fp_mm("
        ),
        "fp_mm_r_trans_site_count": kernel_source.count(
            "tpu_bdc_fp_mm_R_trans("
        ),
        "fill_site_count": kernel_source.count(
            "tpu_bdc_set_C"
        ),
        "gdma_s2l_site_count": kernel_source.count(
            "tpu_gdma_cpy_S2L"
        ),
        "gdma_l2s_site_count": kernel_source.count(
            "tpu_gdma_cpy_L2S"
        ),
        "fp16_input_fp32_output_site_count": kernel_source.count(
            "DT_FP32, DT_FP16"
        ),
    }

    manifest = {
        "stage": "P1.4",
        "probe": (
            "chunkscan_fp16_input_fp32_accumulation_gemms"
        ),
        "cases": {
            "historical": {
                "operation": "left @ right.T",
                "left_shape": [CHUNK, STATE],
                "right_shape": [CHUNK, STATE],
                "output_shape": [CHUNK, CHUNK],
            },
            "scan": {
                "operation": "left @ right",
                "left_shape": [CHUNK, CHUNK],
                "right_shape": [CHUNK, CHUNK],
                "output_shape": [CHUNK, CHUNK],
            },
        },
        "input_dtype": INPUT_DTYPE,
        "accumulation_and_output_dtype": ACCUM_DTYPE,
        "tolerance": {
            "rtol": RTOL,
            "atol": ATOL,
        },
        "toolchain": toolchain,
        "source_metrics": source_metrics,
        "performance_claim": (
            "NONE: cmodel correctness probe only"
        ),
    }

    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    expected_source_metrics = {
        "fp_mm_site_count": 1,
        "fp_mm_r_trans_site_count": 1,
        "fill_site_count": 2,
        "gdma_s2l_site_count": 4,
        "gdma_l2s_site_count": 2,
        "fp16_input_fp32_output_site_count": 2,
    }
    if source_metrics != expected_source_metrics:
        raise AssertionError(
            "Unexpected generated GEMM structure: "
            f"{source_metrics}"
        )

    kernel = tilelang.compile(
        program,
        out_idx=[4, 5],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    historical_left = (
        (
            torch.arange(
                CHUNK * STATE,
                dtype=torch.int32,
            )
            % 37
            - 18
        )
        .to(torch.float32)
        .mul_(1.0 / 64.0)
        .to(torch.float16)
        .reshape(CHUNK, STATE)
        .contiguous()
    )
    historical_right = (
        (
            (
                torch.arange(
                    CHUNK * STATE,
                    dtype=torch.int32,
                )
                * 3
            )
            % 41
            - 20
        )
        .to(torch.float32)
        .mul_(1.0 / 64.0)
        .to(torch.float16)
        .reshape(CHUNK, STATE)
        .contiguous()
    )
    scan_left = (
        (
            (
                torch.arange(
                    CHUNK * CHUNK,
                    dtype=torch.int32,
                )
                * 5
            )
            % 29
            - 14
        )
        .to(torch.float32)
        .mul_(1.0 / 64.0)
        .to(torch.float16)
        .reshape(CHUNK, CHUNK)
        .contiguous()
    )
    scan_right = (
        (
            (
                torch.arange(
                    CHUNK * CHUNK,
                    dtype=torch.int32,
                )
                * 7
            )
            % 31
            - 15
        )
        .to(torch.float32)
        .mul_(1.0 / 64.0)
        .to(torch.float16)
        .reshape(CHUNK, CHUNK)
        .contiguous()
    )

    actual_historical = torch.full(
        (CHUNK, CHUNK),
        float("nan"),
        dtype=torch.float32,
    )
    actual_scan = torch.full_like(
        actual_historical,
        float("nan"),
    )

    kernel(
        historical_left,
        historical_right,
        scan_left,
        scan_right,
        actual_historical,
        actual_scan,
    )

    cases = {
        "historical": _stats(
            actual_historical,
            historical_left.float()
            @ historical_right.float().T,
        ),
        "scan": _stats(
            actual_scan,
            scan_left.float()
            @ scan_right.float(),
        ),
    }

    result = {
        "status": (
            "PASS"
            if all(
                case["close"]
                for case in cases.values()
            )
            else "FAIL"
        ),
        "cases": cases,
        "source_metrics": source_metrics,
        "performance_claim": (
            "NONE: cmodel correctness probe only"
        ),
    }

    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    if result["status"] != "PASS":
        raise AssertionError(
            f"P1.4 GEMM probe failed: {result}"
        )


if __name__ == "__main__":
    main()