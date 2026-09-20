#!/usr/bin/env bash
set -euo pipefail

readonly W8_ROOT="${W8_ROOT:-/root/autodl-tmp/tilelang-tpu-mm_w8a16_dq_forward}"
readonly EXPECTED_BRANCH="pipe-a341-b2-k512-cmodel"
readonly EXPECTED_COMMIT="e653943ce3f3c0482da8f8e9289b8ab1b8522d1d"
readonly TVM_ROOT="${W8_ROOT}/3rdparty/tvm"
readonly EXPERIMENT_ROOT_REL="tpu_demo/ppl/fp8_codegen_tests"
readonly SNAPSHOT_PARENT="${W8_SNAPSHOT_ROOT:-/root/autodl-tmp/pipethreader-baselines}"
readonly SNAPSHOT_DIR="${SNAPSHOT_PARENT}/w8a16-a341-e653943"

if [[ ! -d "${W8_ROOT}/.git" ]]; then
    printf 'W8A16 repository not found: %s\n' "${W8_ROOT}" >&2
    exit 1
fi

actual_branch="$(git -C "${W8_ROOT}" branch --show-current)"
actual_commit="$(git -C "${W8_ROOT}" rev-parse HEAD)"

if [[ "${actual_branch}" != "${EXPECTED_BRANCH}" ]]; then
    printf 'Unexpected W8A16 branch: %s\n' "${actual_branch}" >&2
    exit 1
fi

if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    printf 'Unexpected W8A16 commit: %s\n' "${actual_commit}" >&2
    exit 1
fi

if [[ ! -d "${W8_ROOT}/${EXPERIMENT_ROOT_REL}" ]]; then
    printf 'W8A16 experiment directory not found: %s\n' \
        "${W8_ROOT}/${EXPERIMENT_ROOT_REL}" >&2
    exit 1
fi

if [[ -e "${SNAPSHOT_DIR}" ]]; then
    printf 'Refusing to overwrite existing snapshot: %s\n' \
        "${SNAPSHOT_DIR}" >&2
    exit 1
fi

mkdir -p "${SNAPSHOT_DIR}"

{
    printf 'snapshot_time_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'repository=%s\n' "${W8_ROOT}"
    printf 'branch=%s\n' "${actual_branch}"
    printf 'commit=%s\n' "${actual_commit}"
    printf '\n[root status]\n'
    git -C "${W8_ROOT}" status --short --branch
    printf '\n[submodules]\n'
    git -C "${W8_ROOT}" submodule status --recursive
    printf '\n[tvm status]\n'
    git -C "${TVM_ROOT}" status --short --branch
} > "${SNAPSHOT_DIR}/identity.txt"

git -C "${W8_ROOT}" diff \
    --binary \
    --no-ext-diff \
    > "${SNAPSHOT_DIR}/root_worktree.patch"

git -C "${TVM_ROOT}" diff \
    --binary \
    --no-ext-diff \
    > "${SNAPSHOT_DIR}/tvm_worktree.patch"

(
    cd "${W8_ROOT}"
    find "${EXPERIMENT_ROOT_REL}" -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum
) > "${SNAPSHOT_DIR}/experiment_files.sha256"

tar \
    -C "${W8_ROOT}" \
    -czf "${SNAPSHOT_DIR}/fp8_codegen_tests.tar.gz" \
    "${EXPERIMENT_ROOT_REL}"

(
    cd "${SNAPSHOT_DIR}"
    sha256sum \
        identity.txt \
        root_worktree.patch \
        tvm_worktree.patch \
        experiment_files.sha256 \
        fp8_codegen_tests.tar.gz \
        > SHA256SUMS
)

printf 'W8A16 baseline snapshot created:\n%s\n' "${SNAPSHOT_DIR}"
printf 'Verify with:\n'
printf 'cd %s && sha256sum -c SHA256SUMS\n' "${SNAPSHOT_DIR}"