"""P1.1 cmodel probe for rank-2 TPU elementwise broadcasting."""

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

THIS_DIR = Path(__file__).resolve().parent
CHUNKSCAN_DIR = THIS_DIR.parent
ARTIFACT_DIR = CHUNKSCAN_DIR / "artifacts" / "p1_broadcast_2d"
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


def make_broadcast_2d_kernel():
    @T.prim_func
    def broadcast_2d_kernel(
        column: T.Tensor((SIZE, 1), DTYPE),
        row: T.Tensor((1, SIZE), DTYPE),
        scalar: T.Tensor((1, 1), DTYPE),
        column_out: T.Tensor((SIZE, SIZE), DTYPE),
        row_out: T.Tensor((SIZE, SIZE), DTYPE),
        scalar_out: T.Tensor((SIZE, SIZE), DTYPE),
        difference_out: T.Tensor((SIZE, SIZE), DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            column_local = T.alloc_shared((SIZE, 1), DTYPE)
            row_local = T.alloc_shared((1, SIZE), DTYPE)
            scalar_local = T.alloc_shared((1, 1), DTYPE)
            scalar_row = T.alloc_shared((1, SIZE), DTYPE)

            column_matrix = T.alloc_shared((SIZE, SIZE), DTYPE)
            row_matrix = T.alloc_shared((SIZE, SIZE), DTYPE)
            scalar_matrix = T.alloc_shared((SIZE, SIZE), DTYPE)
            difference_matrix = T.alloc_shared((SIZE, SIZE), DTYPE)

            T.ppl_copy(column[0, 0], column_local)
            T.ppl_copy(row[0, 0], row_local)
            T.ppl_copy(scalar[0, 0], scalar_local)

            T.ppl_fill(column_matrix, T.float32(0.0))
            T.ppl_add(column_matrix, column_matrix, column_local)

            T.ppl_npu_bcast(row_matrix, row_local)

            T.ppl_fill(scalar_row, T.float32(0.0))
            T.ppl_add(scalar_row, scalar_row, scalar_local)
            T.ppl_npu_bcast(scalar_matrix, scalar_row)

            T.ppl_subtract(
                difference_matrix,
                column_matrix,
                row_matrix,
            )

            T.ppl_copy(column_matrix, column_out[0, 0])
            T.ppl_copy(row_matrix, row_out[0, 0])
            T.ppl_copy(scalar_matrix, scalar_out[0, 0])
            T.ppl_copy(difference_matrix, difference_out[0, 0])

    return broadcast_2d_kernel


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

    return {
        "close": bool(
            torch.allclose(
                actual,
                expected,
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "max_abs": float(safe_absolute.max().item()),
        "mean_abs": float(safe_absolute.mean().item()),
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

    program = make_broadcast_2d_kernel()
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
        "stride_c_zero_count": kernel_source.count(".c = 0;"),
        "stride_w_zero_count": kernel_source.count(".w = 0;"),
        "npu_bcast_site_count": kernel_source.count(
            "tpu_bdc_npu_bcast"
        ),
        "bdc_add_site_count": kernel_source.count("tpu_bdc_fp_add"),
        "bdc_sub_site_count": kernel_source.count("tpu_bdc_fp_sub"),
        "gdma_s2l_site_count": kernel_source.count("tpu_gdma_cpy_S2L"),
        "gdma_l2s_site_count": kernel_source.count("tpu_gdma_cpy_L2S"),
    }

    manifest = {
        "stage": "P1.1",
        "probe": "rank_2_rhs_broadcast",
        "size": SIZE,
        "dtype": DTYPE,
        "cases": {
            "column": "[64,1] -> [64,64]",
            "row": "[1,64] -> [64,64]",
            "scalar": "[1,1] -> [64,64]",
            "difference": "column_matrix - row_matrix",
        },
        "toolchain": toolchain,
        "source_metrics": source_metrics,
        "performance_claim": "NONE: cmodel correctness probe only",
    }

    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if source_metrics["stride_c_zero_count"] != 0:
        raise AssertionError(
            "Generated source still contains unsafe C-axis zero strides: "
            f"{source_metrics}"
        )

    if source_metrics["stride_w_zero_count"] < 2:
        raise AssertionError(
            "Generated source is missing column/scalar W-axis broadcast: "
            f"{source_metrics}"
        )

    if source_metrics["npu_bcast_site_count"] < 2:
        raise AssertionError(
            "Generated source is missing explicit row/scalar NPU broadcasts: "
            f"{source_metrics}"
        )

    kernel = tilelang.compile(
        program,
        out_idx=[3, 4, 5, 6],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    values = (
        torch.arange(SIZE, dtype=torch.float32) * 0.03125 - 0.75
    )
    column = values.reshape(SIZE, 1).contiguous()
    row = values.reshape(1, SIZE).contiguous()
    scalar = torch.tensor([[0.375]], dtype=torch.float32)

    actual_column = torch.full(
        (SIZE, SIZE),
        float("nan"),
        dtype=torch.float32,
    )
    actual_row = torch.full_like(actual_column, float("nan"))
    actual_scalar = torch.full_like(actual_column, float("nan"))
    actual_difference = torch.full_like(actual_column, float("nan"))

    kernel(
        column,
        row,
        scalar,
        actual_column,
        actual_row,
        actual_scalar,
        actual_difference,
    )

    expected_column = column.expand(SIZE, SIZE).clone()
    expected_row = row.expand(SIZE, SIZE).clone()
    expected_scalar = scalar.expand(SIZE, SIZE).clone()
    expected_difference = expected_column - expected_row

    cases = {
        "column": _case_stats(actual_column, expected_column),
        "row": _case_stats(actual_row, expected_row),
        "scalar": _case_stats(actual_scalar, expected_scalar),
        "difference": _case_stats(
            actual_difference,
            expected_difference,
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
        raise AssertionError(f"P1.1 broadcast probe failed: {result}")


if __name__ == "__main__":
    main()