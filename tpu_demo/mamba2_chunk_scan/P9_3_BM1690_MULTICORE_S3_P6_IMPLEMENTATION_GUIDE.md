# P9.3：BM1690 多核 ChunkScan S3/P6 实现与验收

P9.1 已验证 1/2/4/8 work-item runtime，P9.2 已验证 task-major S1。本阶段
保持相同 ABI 和任务映射，把两级流水 S3 与显式 Figure-6 sProg-B P6 一次性
迁移到多核，并执行相同的正确性矩阵。P9.3 不测量性能。

## 1. 实现边界

- 使用 `task = (batch * Ck + chunk) * H + head`；
- work-item 负责 `task % workitem_num == workitem_index` 的任务；
- 每个任务独占完整 `[L=64,P=64]` 输出，不进行跨核归约；
- S3 的 K16 reduction 使用 `num_stages=2` 自动流水；
- P6 在同一 S3 循环附加 `figure6_sprog_b_explicit` 契约；
- 两者源码都必须有一个 pipeline start/end；
- 对 CPU oracle、同源码单核以及 S3/P6 逐位结果进行三重检查；
- 保留输出 guard、六个语义用例和 20 次确定性测试。

形状仍为 R0、R1、edge3、edge10；核数仍为 1、2、4、8。

## 2. 获取提交

先在 AutoDL 推送 P9.3，然后在 BM1690 电脑执行：

```bash
cd ~/ChunkScan-bm1690-test-86add2f
git status --short
git fetch origin main
git cherry-pick <P9_3_COMMIT>
```

操作前后 `git status --short` 应只显示：

```text
 m 3rdparty/tvm
```

不要清理 TVM 子模块，也不要再次应用 TVM 补丁。

## 3. 恢复环境

```bash
cd ~/ChunkScan-bm1690-test-86add2f

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"
export TVM_LIBRARY_PATH="$PWD/build/tvm"
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/tvm:/opt/tpuv7/tpuv7-current/lib:${LD_LIBRARY_PATH:-}"

export PPL_PROJECT_ROOT="$HOME/ChunkScan-bm1690-deps/ppl_v1.4.195-geb2acdd0-20250220"
export CHUNKSCAN_DEVICE_RUNTIME_ROOT="/opt/tpuv7/tpuv7-current"
export CHUNKSCAN_RISCV_TOOLCHAIN_ROOT="/host-tools/gcc-riscv/gcc-riscv64-unknown-linux-gnu"
export CHUNKSCAN_DEVICE_ID=0

set -o pipefail
```

P9.3 没有修改 TileLang C++，不需要重新执行 `cmake --build`。

## 4. 静态检查

```bash
./.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_multicore_s3_p9.py \
  tpu_demo/mamba2_chunk_scan/chunk_scan_multicore_p6_p9.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s3_p6_p9.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s3_p6_p9_cmodel.py \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_s3_p6_p9.py

git diff --check
```

## 5. Lowering、流水线、LMEM 单元测试

```bash
./.venv/bin/python \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_s3_p6_p9.py \
  2>&1 | tee /tmp/chunkscan_p9_3_s3_p6_unit.log

unit_exit=${PIPESTATUS[0]}
echo "unit_exit=$unit_exit"
```

通过标志：

```text
P9.3 S3/P6 lowering PASS
unit_exit=0
```

该测试覆盖四种形状和两个候选，检查 task loop、work-item、一个流水线区域、
P6 显式契约和 LMEM 上限，不启动 TPU。

## 6. 全部候选与形状只编译

```bash
timeout 7200s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s3_p6_p9.py \
  2>&1 | tee /tmp/chunkscan_p9_3_s3_p6_compile.log

compile_exit=${PIPESTATUS[0]}
echo "compile_exit=$compile_exit"
```

S3 和 P6 应分别打印四种形状的 `COMPILE PASS`，最后为：

```text
P9.3 S3/P6 COMPILE-ONLY PASS: ...
compile_exit=0
```

## 7. 小矩阵真机预检

```bash
timeout 3600s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s3_p6_p9.py \
  --run --repeat 2 \
  --candidates s3,p6 \
  --shapes r0,edge3 \
  --core-counts 1,2,4,8 \
  2>&1 | tee /tmp/chunkscan_p9_3_s3_p6_quick.log

quick_exit=${PIPESTATUS[0]}
echo "quick_exit=$quick_exit"
```

通过标志：

```text
P9.3 S3/P6 SUBSET PASS (not full acceptance): ...
quick_exit=0
```

预检结果不能标记 P9.3 完成。

## 8. 正式多核正确性矩阵

该矩阵约为 P9.2 工作量的两倍。一个核数完成前可能长时间无输出，不要因此
中断。

```bash
timeout 14400s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s3_p6_p9.py \
  --run --repeat 20 \
  --candidates s3,p6 \
  --shapes r0,r1,edge3,edge10 \
  --core-counts 1,2,4,8 \
  2>&1 | tee /tmp/chunkscan_p9_3_s3_p6_full.log

full_exit=${PIPESTATUS[0]}
echo "full_exit=$full_exit"
```

最后必须为：

```text
P9.3 S3/P6 PASS: ...
full_exit=0
```

## 9. 自动验收 JSON

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_multicore_s3_p6_p9/result.json"
)
result = json.loads(path.read_text())

assert result["status"] == "PASS"
assert result["stage"] == "P9.3"
assert result["chunk_scan_multicore_s3_validated"] is True
assert result["chunk_scan_multicore_p6_validated"] is True
assert result["chunk_scan_multicore_s3_p6_validated"] is True
assert result["multicore_performance_validated"] is False
assert result["s3_p6_comparison"]["checked"] is True
assert result["s3_p6_comparison"]["bitwise_equal"] is True
assert set(result["candidate_results"]) == {"s3", "p6"}

case_names = {
    "residual_only",
    "state_only",
    "scan_only",
    "all_terms",
    "residual_only_negative_D",
    "causal_upper_triangle_poison",
}

for candidate_result in result["candidate_results"].values():
    assert set(candidate_result) == {"r0", "r1", "edge3", "edge10"}
    for shape_result in candidate_result.values():
        assert shape_result["status"] == "PASS"
        assert set(shape_result["core_results"]) == {"1", "2", "4", "8"}
        for core_result in shape_result["core_results"].values():
            assert core_result["status"] == "PASS"
            assert core_result["repeat"] == 20
            assert set(core_result["cases"]) == case_names
            assert sum(core_result["assignment_counts"]) == shape_result["task_count"]
            for case in core_result["cases"].values():
                assert case["close"]
                assert case["nan_count"] == 0
                assert case["inf_count"] == 0
                assert case["guard_rows_unchanged"]
                assert case["payload_fully_covered"]
                assert case["all_expected_tasks_nonzero"]
                assert case["bitwise_equal_to_single_core"]
            repeats = core_result["all_terms_repetitions"]
            assert len(repeats) == 20
            assert all(item["bitwise_deterministic"] for item in repeats)

print("P9.3 MULTICORE S3/P6 RESULT PASS")
print("run_dir =", result["run_dir"])
PY
```

## 10. 通过后的边界

P9.3 通过后可以声明：S1、S3、P6 均已在 BM1690 上完成 1/2/4/8 核正确性
验证，且 S3/P6 在完整矩阵中逐位一致。此时仍没有多核性能或物理 GDMA/BDC
重叠结论。下一阶段 P10 将融合多核性能基线、流水线收益和 scaling 测量。
