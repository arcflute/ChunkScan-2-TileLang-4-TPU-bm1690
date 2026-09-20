"""P1.3 cmodel probe for exact FP16 lower-triangular materialization."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang.language.copy import region

from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)


SIZE = 64
DTYPE = "float16"

THIS_DIR = Path(__file__).resolve().parent
CHUNKSCAN_DIR = THIS_DIR.parent
ARTIFACT_DIR = CHUNKSCAN_DIR / "artifacts" / "p1_causal_mask"
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
    os.environ["PPL_KERNEL_PATH"] = str(
        RUNTIME_DIR / "libkernel.so"
    )


def make_causal_mask_kernel():
    @T.prim_func
    def causal_mask_kernel(
        cb_input: T.Tensor((SIZE, SIZE), DTYPE),
        cb_output: T.Tensor((SIZE, SIZE), DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            cb_local = T.alloc_shared((SIZE, SIZE), DTYPE)

            T.ppl_fill(cb_local, T.float16(0.0))

            for row in T.unroll(SIZE):
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(
                        cb_input[row, 0],
                        "r",
                        1,
                        row + 1,
                    ),
                    region(
                        cb_local[row, 0],
                        "w",
                        1,
                        row + 1,
                    ),
                )

            T.ppl_copy(cb_local, cb_output[0, 0])

    return causal_mask_kernel


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

    program = make_causal_mask_kernel()
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
        "fill_site_count": kernel_source.count(
            "tpu_bdc_set_C"
        ),
        "gdma_s2l_site_count": kernel_source.count(
            "tpu_gdma_cpy_S2L"
        ),
        "gdma_l2s_site_count": kernel_source.count(
            "tpu_gdma_cpy_L2S"
        ),
        "runtime_for_site_count": kernel_source.count(
            "for ("
        ),
        "local_c_lane_term_count": kernel_source.count(
            "% NPU_NUM) * LOCAL_MEM_SIZE"
        ),
        "local_c_group_term_count": kernel_source.count(
            "/ NPU_NUM) * cb_local.stride.c"
        ),
    }

    manifest = {
        "stage": "P1.3",
        "probe": "fp16_lower_triangular_materialization",
        "shape": [SIZE, SIZE],
        "dtype": DTYPE,
        "copied_lower_triangle_elements": (
            SIZE * (SIZE + 1) // 2
        ),
        "zeroed_upper_triangle_elements": (
            SIZE * (SIZE - 1) // 2
        ),
        "implementation": (
            "zero local tile, then T.unroll(64) "
            "static row-prefix S2L copies"
        ),
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
        "fill_site_count": 1,
        "gdma_s2l_site_count": SIZE,
        "gdma_l2s_site_count": 1,
        "runtime_for_site_count": 0,
        "local_c_lane_term_count": 130,
        "local_c_group_term_count": 130,
    }
    if source_metrics != expected_source_metrics:
        raise AssertionError(
            "Unexpected generated causal-mask structure: "
            f"{source_metrics}"
        )

    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="tpu",
        mode="cmodel",
    )
    _archive_cmodel_sources()

    cb_input = (
        torch.arange(
            SIZE * SIZE,
            dtype=torch.int32,
        )
        .remainder(127)
        .add(1)
        .to(torch.float16)
        .reshape(SIZE, SIZE)
        .contiguous()
    )
    actual = torch.full_like(
        cb_input,
        float("nan"),
    )

    kernel(cb_input, actual)

    expected = torch.tril(cb_input)
    lower_mask = torch.tril(
        torch.ones(
            (SIZE, SIZE),
            dtype=torch.bool,
        )
    )
    upper_mask = ~lower_mask
    mismatch = actual != expected

    exact = torch.equal(actual, expected)
    result = {
        "status": "PASS" if exact else "FAIL",
        "exact": bool(exact),
        "mismatch_count": int(
            mismatch.sum().item()
        ),
        "lower_mismatch_count": int(
            mismatch[lower_mask].sum().item()
        ),
        "upper_nonzero_count": int(
            torch.count_nonzero(
                actual[upper_mask]
            ).item()
        ),
        "input_upper_nonzero_count": int(
            torch.count_nonzero(
                cb_input[upper_mask]
            ).item()
        ),
        "nan_count": int(
            torch.isnan(actual).sum().item()
        ),
        "inf_count": int(
            torch.isinf(actual).sum().item()
        ),
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
            f"P1.3 causal-mask probe failed: {result}"
        )


if __name__ == "__main__":
    main()