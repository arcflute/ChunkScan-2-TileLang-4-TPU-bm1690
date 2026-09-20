"""P1.2 cmodel probe for the FP32 natural-exponential primitive."""

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
DTYPE = "float32"
RTOL = 1.0e-5
ATOL = 1.0e-7

THIS_DIR = Path(__file__).resolve().parent
CHUNKSCAN_DIR = THIS_DIR.parent
ARTIFACT_DIR = CHUNKSCAN_DIR / "artifacts" / "p1_exp"
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


def make_exp_kernel():
    @T.prim_func
    def exp_2d_kernel(
        vector_input: T.Tensor((SIZE, 1), DTYPE),
        matrix_input: T.Tensor((SIZE, SIZE), DTYPE),
        vector_output: T.Tensor((SIZE, 1), DTYPE),
        matrix_output: T.Tensor((SIZE, SIZE), DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            vector_local = T.alloc_shared((SIZE, 1), DTYPE)
            vector_work0 = T.alloc_shared((SIZE, 1), DTYPE)
            vector_work1 = T.alloc_shared((SIZE, 1), DTYPE)

            matrix_local = T.alloc_shared((SIZE, SIZE), DTYPE)
            matrix_work0 = T.alloc_shared((SIZE, SIZE), DTYPE)
            matrix_work1 = T.alloc_shared((SIZE, SIZE), DTYPE)

            exp_coeff = T.alloc_shared((64, 32), DTYPE)
            exp_table = T.alloc_shared((64, 192), DTYPE)

            T.ppl_copy(vector_input[0, 0], vector_local)
            T.ppl_copy(matrix_input[0, 0], matrix_local)

            T.ppl_exp2(
                vector_local,
                vector_work0,
                vector_work1,
                exp_coeff,
                exp_table,
            )
            T.ppl_exp2(
                matrix_local,
                matrix_work0,
                matrix_work1,
                exp_coeff,
                exp_table,
            )

            T.ppl_copy(vector_local, vector_output[0, 0])
            T.ppl_copy(matrix_local, matrix_output[0, 0])

    return exp_2d_kernel


def _case_stats(
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
            shutil.copy2(source, ARTIFACT_DIR / archive_name)


def main() -> None:
    toolchain = assert_chunkscan_toolchain()
    _prepare_runtime()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    program = make_exp_kernel()
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
        "load_coeff_site_count": kernel_source.count(
            "tpu_bdc_load_fp32_exp_coeff"
        ),
        "load_table_site_count": kernel_source.count(
            "tpu_bdc_load_fp32_exp_table"
        ),
        "fp32_exp_site_count": kernel_source.count(
            "tpu_bdc_fp32_exp"
        ),
        "gdma_s2l_site_count": kernel_source.count(
            "tpu_gdma_cpy_S2L"
        ),
        "gdma_l2s_site_count": kernel_source.count(
            "tpu_gdma_cpy_L2S"
        ),
    }

    manifest = {
        "stage": "P1.2",
        "probe": "fp32_natural_exponential",
        "size": SIZE,
        "dtype": DTYPE,
        "cases": {
            "vector": {
                "shape": [SIZE, 1],
                "domain": [-8.0, 0.0],
            },
            "matrix": {
                "shape": [SIZE, SIZE],
                "construction": "dA[q] - dA[k]",
                "domain": [-8.0, 8.0],
            },
        },
        "tolerance": {
            "rtol": RTOL,
            "atol": ATOL,
        },
        "toolchain": toolchain,
        "source_metrics": source_metrics,
        "performance_claim": "NONE: cmodel correctness probe only",
    }

    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected_source_metrics = {
        "load_coeff_site_count": 2,
        "load_table_site_count": 2,
        "fp32_exp_site_count": 2,
        "gdma_s2l_site_count": 2,
        "gdma_l2s_site_count": 2,
    }
    if source_metrics != expected_source_metrics:
        raise AssertionError(
            "Unexpected generated exponential structure: "
            f"{source_metrics}"
        )

    kernel = tilelang.compile(
        program,
        out_idx=[2, 3],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    dA = -torch.linspace(
        0.0,
        8.0,
        SIZE,
        dtype=torch.float32,
    )
    vector_input = dA.reshape(SIZE, 1).contiguous()
    matrix_input = (
        dA[:, None] - dA[None, :]
    ).contiguous()

    actual_vector = torch.full_like(
        vector_input,
        float("nan"),
    )
    actual_matrix = torch.full_like(
        matrix_input,
        float("nan"),
    )

    kernel(
        vector_input,
        matrix_input,
        actual_vector,
        actual_matrix,
    )

    cases = {
        "vector": _case_stats(
            actual_vector,
            torch.exp(vector_input),
        ),
        "matrix": _case_stats(
            actual_matrix,
            torch.exp(matrix_input),
        ),
    }
    passed = all(case["close"] for case in cases.values())

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
        raise AssertionError(
            f"P1.2 exponential probe failed: {result}"
        )


if __name__ == "__main__":
    main()