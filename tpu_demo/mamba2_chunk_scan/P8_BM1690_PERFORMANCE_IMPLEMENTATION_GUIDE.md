# P8 BM1690 单核性能测试实施指南

P8 只增加正式 benchmark 入口，不修改已经通过的 S0/S1/S2/S3/P6
正确性文件。当前指标是一轮中每个同步 kernel wrapper 调用的延迟：包含
host launch、设备执行和 stream synchronize，不包含编译、模块加载、设备
内存分配、H2D 或 D2H。

## 1. 新增文件

- `main_template_device_bench.cpp`：在一次设备初始化和 H2D 后执行预热与
  至少指定时长的重复测量，将所有原始样本写入 JSON；
- `test_chunk_scan_device_benchmark_p8.py`：编译 S1/S2/S3/P6、执行前后
  正确性门、随机化每轮顺序、汇总统计和相对 S1 的流水线收益；
- 本指南。

由于新增源码接近 1000 行，推荐通过只包含这些新文件的 Git 提交交付，
不要在 ToDesk 终端逐行重打。

## 2. 进入现有 BM1690 工作树

```bash
cd ~/ChunkScan-bm1690-test-86add2f
git status --short
git rev-parse HEAD
```

`3rdparty/tvm` 显示小写 `m` 是已经应用 `patches/tvm.patch` 的预期状态，
不要清理、重置或再次应用该补丁。

取得 P8 交付提交后执行：

```bash
git fetch origin main
git cherry-pick <P8_COMMIT>
```

P8 提交只新增文件，不修改已经在该服务器上通过的四个真机源文件。

## 3. 恢复本次终端的环境变量

每次新开终端都需要重新执行：

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
```

确认关键路径：

```bash
test -x .venv/bin/python
test -f build/libtilelang_module.so
test -f build/tvm/libtvm.so
test -f "$PPL_PROJECT_ROOT/runtime/bm1690/lib/libbm1690.a"
test -f "$CHUNKSCAN_DEVICE_RUNTIME_ROOT/lib/libtpuv7_rt.so"
test -x "$CHUNKSCAN_RISCV_TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-gcc"
echo "P8 environment paths PASS"
```

## 4. 静态检查

```bash
./.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_benchmark_p8.py

./.venv/bin/python - <<'PY'
from pathlib import Path

path = Path("tpu_demo/mamba2_chunk_scan/main_template_device_bench.cpp")
text = path.read_text()
text.format(
    arg_declarations="// args",
    device_declarations="// devices",
    malloc_statements="// mallocs",
    memcpy_s2d_statements="// h2d",
    memcpy_d2s_statements="// d2h",
    free_statements="// frees",
    kernel_call="",
    pure_kernel_call="rst = main_kernel(0,0,0,0,0,0,0,0);",
)
print("P8 C++ template format PASS")
PY

git diff --check
```

`git diff --check` 不应输出任何内容。

记录 runtime event 能力；P8 暂不使用 async/event 计时：

```bash
nm -D "$CHUNKSCAN_DEVICE_RUNTIME_ROOT/lib/libtpuv7_rt.so" | \
  grep -E 'tpuRtEvent(Create|Record|Synchronize|ElapsedTime)'
```

缺少某个 event 符号不会阻塞 P8 的同步调用延迟测试，但必须保留输出。

## 5. 只编译，不启动 TPU kernel

```bash
timeout 1800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_benchmark_p8.py \
  all --compile-only 2>&1 | tee /tmp/chunkscan_p8_compile.log

echo "compile_exit=${PIPESTATUS[0]}"
```

通过标志：

```text
S1 COMPILE PASS
S2 COMPILE PASS
S3 COMPILE PASS
P6 COMPILE PASS
P8 COMPILE PASS: ...
compile_exit=0
```

若这里失败，不要继续 quick 或 full benchmark。

## 6. 快速真机测试

`--quick` 使用 5 次预热、每候选至少 0.5 秒、1 轮，目的是快速发现
编译、设备运行、正确性或 JSON 产物问题，不用于报告性能。

```bash
timeout 1800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_benchmark_p8.py \
  all --quick 2>&1 | tee /tmp/chunkscan_p8_quick.log

echo "quick_exit=${PIPESTATUS[0]}"
```

通过条件：

- S1/S2/S3/P6 都出现 `CORRECTNESS BEFORE PASS`；
- 四个候选都出现 `ROUND 1 PASS`；
- 四个候选都出现 `CORRECTNESS AFTER PASS`；
- 最终出现 `P8 PASS` 且 `quick_exit=0`。

## 7. 正式 P8 测量

默认配置是每候选 100 次预热、每轮至少 5 秒、7 个独立轮次。
每轮候选顺序会确定性打乱，预计纯测量时间至少约 140 秒，编译时间另计。

```bash
timeout 3600s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_benchmark_p8.py \
  all 2>&1 | tee /tmp/chunkscan_p8_full.log

echo "full_exit=${PIPESTATUS[0]}"
```

不要同时在该卡上运行其他测试。不要把 `--quick` 结果当作正式结果。

## 8. 自动验收最终 JSON

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_benchmark_p8/result.json"
)
result = json.loads(path.read_text())

assert result["status"] == "PASS", result["status"]
assert result["rounds"] == 7
assert result["minimum_seconds_per_round"] >= 5.0
assert set(result["summaries"]) == {"s1", "s2", "s3", "p6"}

for stage, summary in result["summaries"].items():
    assert summary["round_count"] == 7, (stage, summary)
    assert summary["total_measurement_seconds"] >= 35.0, (stage, summary)
    assert summary["total_sample_count"] > 0, (stage, summary)
    assert summary["stability"] != "INVALID_CV_GT_10_PERCENT", (
        stage,
        summary,
    )
    assert summary["median_of_round_p50_us"] > 0.0
    assert summary["pipeline_speedup_vs_s1"] > 0.0

for group in ("correctness_before", "correctness_after"):
    for stage, check in result[group].items():
        assert check["close"], (group, stage, check)
        assert check["nan_count"] == 0
        assert check["inf_count"] == 0

print("P8 RESULT PASS")
print("run_dir =", result["run_dir"])
for stage, summary in result["summaries"].items():
    print(
        stage,
        "p50_us =", summary["median_of_round_p50_us"],
        "p95_us =", summary["median_of_round_p95_us"],
        "cv =", summary["round_p50_cv"],
        "speedup_vs_s1 =", summary["pipeline_speedup_vs_s1"],
        "stability =", summary["stability"],
    )
PY
```

查看便于人工比较的表格：

```bash
result_dir=$(./.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("tpu_demo/mamba2_chunk_scan/artifacts/device_benchmark_p8/result.json")
print(json.loads(path.read_text())["run_dir"])
PY
)

column -s, -t "$result_dir/summary.csv"
```

## 9. P8 通过条件与结论边界

P8 完整通过需要：

1. compile-only、quick 和 full 三次命令退出码均为 0；
2. 每个候选测试前后均通过 all-terms CPU oracle；
3. 每个候选具有 7 轮且每轮至少 5 秒的原始样本；
4. 每个候选跨轮 p50 的 CV 不超过 10%；5%–10% 只标记警告；
5. `samples.jsonl`、`summary.json`、`summary.csv`、最终 `result.json`、
   环境清单、编译清单和逐轮 JSON 均存在。

P8 只能报告固定小形状上的单核同步调用延迟，以及 S2/S3/P6 相对 S1 的
初步变化。它不能报告多核加速、async throughput 或物理 GDMA/BDC 重叠。

## 10. 后续融合阶段

为加快推进，后续由原 P9–P13 合并为两个管理阶段：

- **P9：参数化形状 + 多核基础设施 + 1/2/4/8 核正确性**；
- **P10：单核/多核性能矩阵 + 调优 + 硬件 trace 能力核查 + 最终收口**。

内部仍保留能力探针、正确性和性能三道停止门，但不再把它们拆成多个对外阶段。
