# P10：BM1690 多核 ChunkScan 性能、流水线收益与收口

P9 已完成 S1/S3/P6 在 R0、R1、edge3、edge10 上的 1/2/4/8 核正确性矩阵。
P10 不改变算子数学或任务映射，使用正式性能形状 R1–R3，测量流水线收益、
多核加速比、并行效率和联合收益。

## 1. 正式矩阵

```text
shape    = R1, R2, R3
stage    = S1, S3, P6
core_num = 1, 2, 4, 8
```

形状为：

| 形状 | B | S | Ck | H | 任务数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | 1 | 1024 | 16 | 8 | 128 |
| R2 | 1 | 4096 | 64 | 8 | 512 |
| R3 | 8 | 2048 | 32 | 8 | 2048 |

固定 `L=64,G=1,P=64,N=128`，FP16 输入输出、FP32 中间累加、K16 reduction。
R1–R3 是本项目的 TPU 性能形状，不冒充论文完整配置。

正式协议：

- 每个点 100 次 warm-up；
- 每轮至少 5 秒且至少 20 个样本；
- 7 个独立轮次，每轮确定性随机化 36 个点的顺序；
- 编译、module load、内存分配、H2D/D2H 不进入样本；
- 样本包含 host launch、设备执行和 stream synchronize；
- 测量前、每轮后和全部测量后均执行 all-terms 正确性检查；
- `round-p50 CV > 10%` 判为无效，`5%–10%` 记录警告；
- runtime module-load 偶发失败最多重试两次，所有重试必须写入结果。

## 2. 获取提交

在 AutoDL 推送 P10 后，在 BM1690 电脑执行：

```bash
cd ~/ChunkScan-bm1690-test-86add2f
git status --short
git fetch origin main
git cherry-pick <P10_COMMIT>
```

正常情况下 `git status --short` 只显示：

```text
 m 3rdparty/tvm
```

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

P10 没有修改 TileLang C++，不需要重新构建。

## 4. 静态与 lowering 检查

```bash
./.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_benchmark_p10.py \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_benchmark_p10.py

git diff --check

./.venv/bin/python \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_benchmark_p10.py \
  2>&1 | tee /tmp/chunkscan_p10_unit.log

unit_exit=${PIPESTATUS[0]}
echo "unit_exit=$unit_exit"
```

通过标志：

```text
P10 R1/R2/R3 S1/S3/P6 lowering PASS
unit_exit=0
```

## 5. 九个设备 kernel 只编译

```bash
timeout 7200s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_benchmark_p10.py \
  --compile-only \
  2>&1 | tee /tmp/chunkscan_p10_compile.log

compile_exit=${PIPESTATUS[0]}
echo "compile_exit=$compile_exit"
```

应看到 R1/R2/R3 的 S1/S3/P6 全部 `COMPILE PASS`，最后为：

```text
P10 COMPILE-ONLY PASS: ...
compile_exit=0
```

## 6. 快速真机预检

快速模式仍覆盖完整 36 点，但只运行一轮、5 次预热、至少 0.5 秒和 5 个样本。

```bash
timeout 7200s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_benchmark_p10.py \
  --quick \
  2>&1 | tee /tmp/chunkscan_p10_quick.log

quick_exit=${PIPESTATUS[0]}
echo "quick_exit=$quick_exit"
```

通过标志：

```text
P10 SUBSET PASS: ...
quick_exit=0
```

快速结果只能验证框架和数量级，不能作为正式性能结论。

## 7. 正式性能矩阵

运行前尽量关闭同机其他高负载任务。该步骤可能运行数小时，中间每个点完成时
才输出一行，不要因短时间无输出而中断。

```bash
timeout 28800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_benchmark_p10.py \
  --stages s1,s3,p6 \
  --shapes r1,r2,r3 \
  --core-counts 1,2,4,8 \
  --rounds 7 \
  --warmup 100 \
  --min-seconds 5 \
  --min-runs 20 \
  2>&1 | tee /tmp/chunkscan_p10_full.log

full_exit=${PIPESTATUS[0]}
echo "full_exit=$full_exit"
```

正式通过标志：

```text
P10 PASS: ...
full_exit=0
```

若出现 `P10 FAIL_UNSTABLE`，不要挑选最快轮次，应保留证据并重新排查系统负载。

## 8. 自动验收

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_multicore_benchmark_p10/result.json"
)
result = json.loads(path.read_text())

assert result["status"] == "PASS"
assert result["stage"] == "P10"
assert result["full_protocol"] is True
assert result["performance_validated"] is True
assert result["physical_gdma_bdc_overlap_validated"] is False
assert result["trace_status"] == "NOT_CAPTURED"
assert result["invalid_points"] == []
assert len(result["summaries"]) == 36
assert len(result["correctness_before"]) == 36
assert len(result["correctness_after"]) == 36

for key, summary in result["summaries"].items():
    assert summary["round_count"] == 7
    assert summary["stability"] != "INVALID_CV_GT_10_PERCENT"
    assert summary["median_of_round_p50_us"] > 0
    assert summary["multicore_speedup_vs_same_stage_c1"] > 0
    assert summary["parallel_efficiency"] > 0
    assert summary["pipeline_gain_vs_s1_same_core"] > 0
    assert summary["combined_gain_vs_s1_c1"] > 0
    assert summary["output_elements_per_second"] > 0

for key, before in result["correctness_before"].items():
    after = result["correctness_after"][key]
    assert before["close"] and after["close"]
    assert before["guard_rows_unchanged"]
    assert after["guard_rows_unchanged"]
    assert before["output_sha256"] == after["output_sha256"]

print("P10 RESULT PASS")
print("runtime retries =", result["runtime_module_load_retry_count"])
print("run_dir =", result["run_dir"])
PY
```

## 9. 查看核心结果

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_multicore_benchmark_p10/result.json"
).read_text())

print("shape stage cores p50_us pipeline_gain multicore_speedup efficiency combined_gain")
for key in sorted(result["summaries"]):
    item = result["summaries"][key]
    print(
        item["shape"], item["stage"], item["core_num"],
        f'{item["median_of_round_p50_us"]:.3f}',
        f'{item["pipeline_gain_vs_s1_same_core"]:.4f}',
        f'{item["multicore_speedup_vs_same_stage_c1"]:.4f}',
        f'{item["parallel_efficiency"]:.4f}',
        f'{item["combined_gain_vs_s1_c1"]:.4f}',
    )

print("best =", result["best_configuration_by_shape"])
PY
```

完整产物位于 `run_dir`：

- `result.json`：验收状态和边界；
- `summary.json`、`summary.csv`：36 点汇总；
- `samples.jsonl`：全部原始样本；
- `report.md`：自动生成的 Markdown 性能报告；
- 各 shape/stage 目录：生成源码、二进制哈希、每轮 JSON 和正确性 JSON。

## 10. 结论边界

P10 只能给出真实 BM1690 上的同步调用延迟、流水线收益和多核 scaling。
若没有硬件 trace，不能把性能提升表述为已直接证明 GDMA/BDC 物理重叠。
BM1690e 四核结果必须在独立仓库重新验证，不能直接继承本结果。
