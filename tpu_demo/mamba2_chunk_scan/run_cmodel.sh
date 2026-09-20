#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
CHUNKSCAN_TOOLCHAIN_ROOT="$(
    cd -- "${SCRIPT_DIR}/../.."
    pwd
)"

PPL_PROJECT_ROOT="${PPL_PROJECT_ROOT:-/root/autodl-tmp/ppl_v1.4.195-geb2acdd0-20250220}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/tilelang-tpu/.venv/bin/python}"
RUNTIME_DIR="${SCRIPT_DIR}/artifacts/serial/runtime"
TMP_ROOT="${CHUNKSCAN_TMP_ROOT:-/root/autodl-tmp/tmp}"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required file is missing: %s\n' "$1" >&2
        exit 1
    fi
}

require_directory() {
    if [[ ! -d "$1" ]]; then
        printf 'Required directory is missing: %s\n' "$1" >&2
        exit 1
    fi
}

require_file "${PYTHON_BIN}"
require_file "${CHUNKSCAN_TOOLCHAIN_ROOT}/build/libtilelang.so"
require_file "${CHUNKSCAN_TOOLCHAIN_ROOT}/build/libtilelang_module.so"
require_file "${CHUNKSCAN_TOOLCHAIN_ROOT}/build/tvm/libtvm.so"
require_file "${CHUNKSCAN_TOOLCHAIN_ROOT}/build/tvm/libtvm_runtime.so"
require_directory "${PPL_PROJECT_ROOT}/runtime/bm1690/lib"
require_directory \
    "${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator_1.1.3/lib"

mkdir -p "${TMP_ROOT}" "${RUNTIME_DIR}"

unset W8_TOOLCHAIN_ROOT

export CHUNKSCAN_TOOLCHAIN_ROOT
export PPL_PROJECT_ROOT
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TPU_KERNEL_PATH="${RUNTIME_DIR}"
export PPL_KERNEL_PATH="${RUNTIME_DIR}/libkernel.so"
export PYTHONPATH="${CHUNKSCAN_TOOLCHAIN_ROOT}"
export TVM_LIBRARY_PATH="${CHUNKSCAN_TOOLCHAIN_ROOT}/build/tvm"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CHUNKSCAN_TOOLCHAIN_ROOT}/build:${CHUNKSCAN_TOOLCHAIN_ROOT}/build/tvm:${PPL_PROJECT_ROOT}/runtime/bm1690/lib:${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator_1.1.3/lib"
export TMPDIR="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TMP="${TMP_ROOT}"

cd "${CHUNKSCAN_TOOLCHAIN_ROOT}"

if [[ "${1:-}" == "--identity-only" ]]; then
    exec "${PYTHON_BIN}" \
        -m tpu_demo.mamba2_chunk_scan.toolchain_identity
fi

if [[ "${1:-}" == "--broadcast-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_broadcast_2d_cmodel.py"
fi

if [[ "${1:-}" == "--exp-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_exp_cmodel.py"
fi

if [[ "${1:-}" == "--causal-mask-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_causal_mask_cmodel.py"
fi

if [[ "${1:-}" == "--gemm-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_gemm_cmodel.py"
fi

if [[ "${1:-}" == "--residual-cast-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_residual_cast_cmodel.py"
fi

if [[ "${1:-}" == "--reduction-serial" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_reduction_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-s2" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s2_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-s3" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s3_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-matrix-p5" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_matrix_p5_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-planner-p6-unit" ]]; then
    exec "${PYTHON_BIN}" -m pytest -q \
        "${CHUNKSCAN_TOOLCHAIN_ROOT}/testing/python/transform/test_tilelang_transform_pipeline_planning.py"
fi

if [[ "${1:-}" == "--pipeline-p6" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_p6_cmodel.py"
fi

if [[ "${1:-}" == "--p7" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/close_p7_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2|"\
"--pipeline-s3|--pipeline-matrix-p5|"\
"--pipeline-planner-p6-unit|--pipeline-p6|--p7]" >&2
    exit 2
fi

exec "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/test_chunk_scan_cmodel.py"