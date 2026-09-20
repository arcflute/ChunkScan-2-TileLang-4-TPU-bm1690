"""BM1690 PCIe compile check and opt-in one-core ChunkScan S0 smoke test."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_serial import (
    make_chunk_scan_serial_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_cmodel import (
    _make_inputs,
    _physical_args,
    _variant_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME_DIR = HERE / "artifacts" / "device_s0" / "runtime"


def prepare_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    template_dir = ROOT / "src" / "tl_templates" / "tpu"

    for name in ("kernel_template.cpp", "kernel_template.h"):
        shutil.copy2(template_dir / name, RUNTIME_DIR / name)

    shutil.copy2(
        HERE / "main_template_device.cpp",
        RUNTIME_DIR / "main_template.cpp",
    )

    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = (
            lambda path=str(RUNTIME_DIR): path
        )

    os.environ["TPU_KERNEL_PATH"] = str(RUNTIME_DIR)
    os.environ["PPL_KERNEL_PATH"] = str(RUNTIME_DIR / "libkernel.so")


def main() -> None:
    if sys.argv[1:] not in ([], ["--run"]):
        raise SystemExit("Usage: test_chunk_scan_device_s0.py [--run]")

    prepare_runtime()
    kernel = tilelang.compile(
        make_chunk_scan_serial_kernel(),
        out_idx=[7],
        target="tpu",
        mode="pcie",
    )
    print(f"BM1690 PCIe compile/load OK: {RUNTIME_DIR}", flush=True)

    if not sys.argv[1:]:
        print("Compile-only mode: no TPU kernel was launched.", flush=True)
        return

    base = _make_inputs()
    for name in ("residual_only", "all_terms"):
        inputs = _variant_inputs(base, name)
        expected = chunk_scan_reference(*inputs)
        actual = torch.full_like(expected, float("nan"))

        ret = kernel(*_physical_args(inputs, actual))
        if ret != 0:
            raise RuntimeError(f"{name}: TPU host call returned {ret}")

        stats = comparison_stats(actual, expected)
        close = bool(
            torch.allclose(
                actual.float(),
                expected.float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )
        print(f"{name}: close={close}, stats={stats}", flush=True)
        if not close:
            raise AssertionError(f"{name}: BM1690 result mismatch")

    print("BM1690 S0 smoke PASS", flush=True)


if __name__ == "__main__":
    main()
