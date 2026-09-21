"""Compile and validate BM1690 1/2/4/8-core work-item dispatch."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import torch
import tilelang

from tpu_demo.mamba2_chunk_scan.multicore_workitem_probe_p9 import (
    MAX_CORES,
    OUTPUT_SHAPE,
    make_multicore_workitem_probe_p9_kernel,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT_ROOT = HERE / "artifacts" / "device_multicore_probe_p9"
ALLOWED_CORE_COUNTS = (1, 2, 4, 8)
SENTINEL = -777.0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_environment() -> Dict[str, Path]:
    variables = (
        "PPL_PROJECT_ROOT",
        "CHUNKSCAN_DEVICE_RUNTIME_ROOT",
        "CHUNKSCAN_RISCV_TOOLCHAIN_ROOT",
    )
    values: Dict[str, Path] = {}
    for variable in variables:
        value = os.environ.get(variable)
        if not value:
            raise EnvironmentError(f"{variable} is not set")
        values[variable] = Path(value).expanduser().resolve()

    required = (
        values["PPL_PROJECT_ROOT"] / "runtime/bm1690/lib/libbm1690.a",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"] / "include/tpuv7_rt.h",
        values["CHUNKSCAN_DEVICE_RUNTIME_ROOT"] / "lib/libtpuv7_rt.so",
        values["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        / "bin/riscv64-unknown-linux-gnu-gcc",
        ROOT / "build/libtilelang_module.so",
        ROOT / "build/tvm/libtvm.so",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return values


@contextlib.contextmanager
def temporary_environment(updates: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepare_runtime(run_dir: Path) -> Path:
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=False)

    base_template_dir = ROOT / "src/tl_templates/tpu"
    shutil.copy2(
        base_template_dir / "kernel_template.h",
        runtime_dir / "kernel_template.h",
    )
    shutil.copy2(
        HERE / "kernel_template_device_multicore.cpp",
        runtime_dir / "kernel_template.cpp",
    )
    shutil.copy2(
        HERE / "main_template_device_bench.cpp",
        runtime_dir / "main_template.cpp",
    )

    import importlib

    for module_name in (
        "tilelang.jit.adapter.utils",
        "tilelang.jit.adapter.libgen",
        "tilelang.jit.adapter.wrapper",
    ):
        module = importlib.import_module(module_name)
        module.get_tpu_template_dir = lambda path=str(runtime_dir): path

    os.environ["TPU_KERNEL_PATH"] = str(runtime_dir)
    os.environ["PPL_KERNEL_PATH"] = str(runtime_dir / "libkernel.so")
    return runtime_dir


def compile_probe(run_dir: Path):
    runtime_dir = prepare_runtime(run_dir)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            kernel = tilelang.compile(
                make_multicore_workitem_probe_p9_kernel(),
                out_idx=[0],
                target="tpu",
                mode="pcie",
            )
    finally:
        (run_dir / "compile_stdout.log").write_text(
            captured.getvalue(), encoding="utf-8"
        )

    source_path = runtime_dir / "kernel.c"
    host_path = runtime_dir / "kernel.cpp"
    device_library = runtime_dir / "libkernel.so"
    host_library = runtime_dir / "main.so"
    for path in (source_path, host_path, device_library, host_library):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = source_path.read_text(encoding="utf-8")
    host_source = host_path.read_text(encoding="utf-8")
    if "tpu_workitem_index()" not in source:
        raise AssertionError("device source lacks tpu_workitem_index()")
    if "tpu_workitem_num()" not in source:
        raise AssertionError("device source lacks tpu_workitem_num()")
    for abstract_name in ("ppl.workitem_index", "ppl.workitem_num"):
        if abstract_name in source:
            raise AssertionError(
                f"device source still contains abstract call {abstract_name!r}"
            )
    for marker in (
        "CHUNKSCAN_CORE_NUM",
        "block_num = static_cast<uint64_t>(core_num)",
        "apis.data()",
    ):
        if marker not in host_source:
            raise AssertionError(f"host source lacks {marker!r}")

    device_file = subprocess.run(
        ["file", str(device_library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    host_file = subprocess.run(
        ["file", str(host_library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    linked = subprocess.run(
        ["ldd", str(host_library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    expected_runtime = str(
        Path(os.environ["CHUNKSCAN_DEVICE_RUNTIME_ROOT"])
        / "lib/libtpuv7_rt.so"
    )
    if "RISC-V" not in device_file or "x86-64" not in host_file:
        raise AssertionError(f"wrong ELF architecture:\n{device_file}\n{host_file}")
    if expected_runtime not in linked or "emulator" in linked:
        raise AssertionError(f"host library uses wrong runtime:\n{linked}")

    manifest = {
        "status": "COMPILE_PASS",
        "device_source_sha256": sha256_file(source_path),
        "host_source_sha256": sha256_file(host_path),
        "device_library_sha256": sha256_file(device_library),
        "host_library_sha256": sha256_file(host_library),
        "device_file": device_file,
        "host_file": host_file,
        "linked_libraries": linked.splitlines(),
        "workitem_index_call_count": source.count("tpu_workitem_index()"),
        "workitem_num_call_count": source.count("tpu_workitem_num()"),
        "allowed_core_counts": list(ALLOWED_CORE_COUNTS),
        "block_num_source": "CHUNKSCAN_CORE_NUM",
        "performance_claim": "none; dispatch correctness probe only",
    }
    write_json(run_dir / "compile_manifest.json", manifest)
    print("P9 WORKITEM COMPILE PASS", flush=True)
    return kernel, runtime_dir, manifest


def expected_output(core_num: int) -> torch.Tensor:
    expected = torch.full(OUTPUT_SHAPE, SENTINEL, dtype=torch.float32)
    expected[:core_num, 0] = torch.arange(core_num, dtype=torch.float32)
    expected[:core_num, 1] = float(core_num)
    return expected


def run_core_count(kernel, runtime_dir: Path, core_num: int, repeat: int) -> dict:
    expected = expected_output(core_num)
    first_output = None
    repetitions = []

    for repetition in range(1, repeat + 1):
        actual = torch.full(OUTPUT_SHAPE, SENTINEL, dtype=torch.float32)
        with temporary_environment(
            {
                "PPL_KERNEL_PATH": str(runtime_dir / "libkernel.so"),
                "CHUNKSCAN_BENCH_MODE": "correctness",
                "CHUNKSCAN_CORE_NUM": str(core_num),
            }
        ):
            returned = kernel(actual)
        if returned not in (None, 0):
            raise RuntimeError(
                f"core_num={core_num} repetition={repetition}: "
                f"host call returned {returned}"
            )

        exact = bool(torch.equal(actual, expected))
        deterministic = first_output is None or bool(
            torch.equal(actual, first_output)
        )
        active_indices = actual[:core_num, 0].to(torch.int32).tolist()
        active_counts = actual[:core_num, 1].to(torch.int32).tolist()
        inactive_unchanged = bool(
            torch.all(actual[core_num:] == SENTINEL).item()
        )
        repetition_result = {
            "repetition": repetition,
            "exact": exact,
            "deterministic": deterministic,
            "active_indices": active_indices,
            "active_counts": active_counts,
            "inactive_sentinel_unchanged": inactive_unchanged,
        }
        repetitions.append(repetition_result)
        if not exact or not deterministic or not inactive_unchanged:
            raise AssertionError(
                f"work-item probe failed: core_num={core_num} "
                f"repetition={repetition} actual={actual.tolist()}"
            )
        if first_output is None:
            first_output = actual.clone()

    result = {
        "status": "PASS",
        "core_num": core_num,
        "repeat": repeat,
        "expected_indices": list(range(core_num)),
        "expected_workitem_num": core_num,
        "inactive_sentinel": SENTINEL,
        "repetitions": repetitions,
    }
    print(f"P9 WORKITEM {core_num}-CORE PASS repeat={repeat}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P9 BM1690 physical work-item dispatch probe"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--core-counts",
        default="1,2,4,8",
        help="comma-separated subset of 1,2,4,8",
    )
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    try:
        args.resolved_core_counts = tuple(
            int(item) for item in args.core_counts.split(",")
        )
    except ValueError as error:
        parser.error(f"invalid --core-counts: {error}")
    if not args.resolved_core_counts or any(
        item not in ALLOWED_CORE_COUNTS
        for item in args.resolved_core_counts
    ):
        parser.error("--core-counts must be a subset of 1,2,4,8")
    if len(set(args.resolved_core_counts)) != len(args.resolved_core_counts):
        parser.error("--core-counts contains duplicates")
    return args


def main() -> None:
    args = parse_args()
    paths = require_environment()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ARTIFACT_ROOT / "runs" / f"{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tvm_commit": subprocess.run(
            ["git", "-C", str(ROOT / "3rdparty/tvm"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device_id": os.environ.get("CHUNKSCAN_DEVICE_ID", "0"),
        "ppl_project_root": str(paths["PPL_PROJECT_ROOT"]),
        "runtime_root": str(paths["CHUNKSCAN_DEVICE_RUNTIME_ROOT"]),
        "riscv_toolchain_root": str(
            paths["CHUNKSCAN_RISCV_TOOLCHAIN_ROOT"]
        ),
        "requested_core_counts": list(args.resolved_core_counts),
        "repeat": args.repeat,
    }
    write_json(run_dir / "environment.json", environment)
    kernel, runtime_dir, manifest = compile_probe(run_dir)

    if not args.run:
        result = {
            "status": "COMPILE_PASS",
            "run_dir": str(run_dir),
            "compile": manifest,
            "performance_claim": "none; no TPU kernel was launched",
        }
        write_json(run_dir / "result.json", result)
        write_json(ARTIFACT_ROOT / "result.json", result)
        print(f"P9 WORKITEM COMPILE-ONLY PASS: {run_dir}", flush=True)
        return

    core_results = {
        str(core_num): run_core_count(
            kernel, runtime_dir, core_num, args.repeat
        )
        for core_num in args.resolved_core_counts
    }
    result = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "compile": manifest,
        "core_results": core_results,
        "physical_multicore_dispatch_validated": True,
        "chunk_scan_multicore_validated": False,
        "performance_claim": "none; work-item dispatch correctness only",
    }
    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_ROOT / "result.json", result)
    print(f"P9 WORKITEM PASS: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
