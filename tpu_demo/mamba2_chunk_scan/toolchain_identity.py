"""Validate and record the independent ChunkScan TileLang/TVM toolchain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import tilelang
from tilelang import tvm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOLCHAIN_ROOT = Path(
    os.environ.get("CHUNKSCAN_TOOLCHAIN_ROOT", str(REPOSITORY_ROOT))
).resolve()


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _require_local_path(name: str, path: Path) -> None:
    if not path.is_relative_to(EXPECTED_TOOLCHAIN_ROOT):
        raise RuntimeError(
            f"{name} was loaded outside the independent ChunkScan repository: "
            f"{path}; expected root: {EXPECTED_TOOLCHAIN_ROOT}"
        )


def assert_chunkscan_toolchain() -> Dict[str, object]:
    cython_executable = shutil.which("cython")
    if cython_executable is None:
        raise RuntimeError(
            "The Cython executable is not visible in PATH. "
            "run_cmodel.sh must prepend the selected Python environment's bin directory."
        )

    tilelang_source = Path(tilelang.__file__).resolve()
    tilelang_library = Path(tilelang._LIB_PATH).resolve()
    tvm_source = Path(tvm.__file__).resolve()
    tvm_library = Path(tvm._ffi.base._LIB._name).resolve()

    _require_local_path("TileLang Python", tilelang_source)
    _require_local_path("TileLang library", tilelang_library)
    _require_local_path("TVM Python", tvm_source)
    _require_local_path("TVM library", tvm_library)

    tvm_repository = EXPECTED_TOOLCHAIN_ROOT / "3rdparty" / "tvm"

    identity: Dict[str, object] = {
        "cython_executable": str(Path(cython_executable).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "repository_root": str(EXPECTED_TOOLCHAIN_ROOT),
        "repository_commit": _git_output(
            EXPECTED_TOOLCHAIN_ROOT, "rev-parse", "HEAD"
        ),
        "repository_dirty": bool(
            _git_output(
                EXPECTED_TOOLCHAIN_ROOT,
                "status",
                "--porcelain=v1",
            )
        ),
        "tvm_commit": _git_output(tvm_repository, "rev-parse", "HEAD"),
        "tvm_dirty": bool(
            _git_output(tvm_repository, "status", "--porcelain=v1")
        ),
        "tilelang_source": str(tilelang_source),
        "tilelang_library": str(tilelang_library),
        "tvm_source": str(tvm_source),
        "tvm_library": str(tvm_library),
    }
    return identity


def main() -> None:
    identity = assert_chunkscan_toolchain()
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()