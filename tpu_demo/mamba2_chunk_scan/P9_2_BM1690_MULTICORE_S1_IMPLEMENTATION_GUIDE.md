# P9.2：BM1690 多核 ChunkScan S1 实现与验收

P9.1 已证明 BM1690 的 1/2/4/8 work-item runtime 语义。本阶段把该接口
接入真实 ChunkScan S1，并完成参数化形状、compact ABI、任务覆盖、数值正确性
和重复确定性检查。P9.2 不进行性能测试。

## 1. 本阶段实现内容

- 保留原有 `chunk_scan_reduction_serial.py`，新增独立 P9.2 路线；
- 任务编号固定为
  `task = (batch * Ck + chunk) * H + head`；
- 每个 work-item 执行满足
  `task % workitem_num == workitem_index` 的任务；
- 每个任务负责完整 `[L=64,P=64]` 输出，不进行跨核归约；
- 使用 task-major compact 2-D ABI，host 端显式 pack/unpack；
- `cb` 和 `C` 在第一版中按 head 复制，使每个任务的读取连续；
- 输出 payload 前后各增加一行 guard，检测越界写；
- 保持 S1 的 K16 串行归约与零个 pipeline marker。

验收形状如下：

| 名称 | `(B,Ck,H)` | 任务数 | 目的 |
| --- | --- | ---: | --- |
| R0 | `(1,2,1)` | 2 | 旧固定形状回归、核数大于任务数 |
| R1 | `(1,16,8)` | 128 | `S=1024` 的正式多核形状 |
| edge3 | `(1,1,3)` | 3 | 任务数小于 4/8 核 |
| edge10 | `(1,2,5)` | 10 | 任务数不能被 4/8 整除 |

每个形状均测试 1/2/4/8 核，以及 residual-only、state-only、scan-only、
all-terms、negative-D 和 causal-poison 六个用例。all-terms 在每种核数下重复
20 次并要求逐位确定。

## 2. 获取 P9.2 提交

先在 AutoDL 推送 P9.2 提交，然后在 BM1690 电脑执行：

```bash
cd ~/ChunkScan-bm1690-test-86add2f
git status --short
git fetch origin main
git cherry-pick <P9_2_COMMIT>
```

正常情况下，操作前后 `git status --short` 只显示：

```text
 m 3rdparty/tvm
```

这是已经应用 TVM 补丁的正常状态，不要清理，也不要再次应用补丁。如果
cherry-pick 出现冲突，停止操作并保存完整终端输出。

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

## 4. Python 与仓库静态检查

```bash
./.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_multicore_s1_p9.py \
  tpu_demo/mamba2_chunk_scan/chunk_scan_multicore_abi_p9.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s1_p9.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s1_p9_cmodel.py \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_s1_p9.py

git diff --check
```

本提交没有修改 TileLang C++，P9.1 已经成功重建后，本阶段不需要再次执行
`cmake --build`。

## 5. Lowering、LMEM 与 ABI 单元测试

```bash
./.venv/bin/python \
  testing/python/target/test_tilelang_target_chunk_scan_multicore_s1_p9.py \
  2>&1 | tee /tmp/chunkscan_p9_2_s1_unit.log

unit_exit=${PIPESTATUS[0]}
echo "unit_exit=$unit_exit"
```

通过标志：

```text
P9.2 S1 lowering and compact ABI PASS
unit_exit=0
```

该测试不启动 TPU，检查四个形状的 task loop、work-item 调用、零 pipeline
marker、LMEM 小于 256 KiB，以及 task-major pack/unpack 往返。

## 6. 全部形状只编译

```bash
timeout 3600s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s1_p9.py \
  2>&1 | tee /tmp/chunkscan_p9_2_s1_compile.log

compile_exit=${PIPESTATUS[0]}
echo "compile_exit=$compile_exit"
```

通过标志：

```text
R0 S1 COMPILE PASS tasks=2
R1 S1 COMPILE PASS tasks=128
EDGE3 S1 COMPILE PASS tasks=3
EDGE10 S1 COMPILE PASS tasks=10
P9.2 S1 COMPILE-ONLY PASS: ...
compile_exit=0
```

编译清单会检查 RISC-V 设备库、x86-64 host 库、真实 tpuv7 runtime、
work-item 查询、task loop、参数结构数量、零 pipeline marker 和 LMEM。

## 7. 小矩阵真机预检

```bash
timeout 1800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s1_p9.py \
  --run --repeat 2 --shapes r0,edge3 \
  --core-counts 1,2,4,8 \
  2>&1 | tee /tmp/chunkscan_p9_2_s1_quick.log

quick_exit=${PIPESTATUS[0]}
echo "quick_exit=$quick_exit"
```

最后应显示：

```text
P9.2 S1 SUBSET PASS (not the full acceptance matrix): ...
quick_exit=0
```

该结果只能证明预检通过，不能标记 P9.2 完成。

## 8. 正式 1/2/4/8 核正确性矩阵

```bash
timeout 7200s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_multicore_s1_p9.py \
  --run --repeat 20 \
  --shapes r0,r1,edge3,edge10 \
  --core-counts 1,2,4,8 \
  2>&1 | tee /tmp/chunkscan_p9_2_s1_full.log

full_exit=${PIPESTATUS[0]}
echo "full_exit=$full_exit"
```

每个形状都应打印四条 `N-CORE PASS`，最后必须是：

```text
P9.2 S1 PASS: ...
full_exit=0
```

## 9. 自动验收 JSON

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_multicore_s1_p9/result.json"
)
result = json.loads(path.read_text())

assert result["status"] == "PASS"
assert result["stage"] == "P9.2"
assert result["candidate"] == "S1"
assert result["chunk_scan_multicore_s1_validated"] is True
assert result["chunk_scan_multicore_s3_p6_validated"] is False
assert set(result["shape_results"]) == {"r0", "r1", "edge3", "edge10"}

case_names = {
    "residual_only",
    "state_only",
    "scan_only",
    "all_terms",
    "residual_only_negative_D",
    "causal_upper_triangle_poison",
}

for shape_name, shape_result in result["shape_results"].items():
    assert shape_result["status"] == "PASS"
    assert set(shape_result["core_results"]) == {"1", "2", "4", "8"}
    for core_text, core_result in shape_result["core_results"].items():
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
        poison = core_result["cases"]["causal_upper_triangle_poison"]
        assert poison["bitwise_equal_to_clean_scan"]
        repeats = core_result["all_terms_repetitions"]
        assert len(repeats) == 20
        assert all(item["bitwise_deterministic"] for item in repeats)
        assert all(item["guard_rows_unchanged"] for item in repeats)
        assert all(item["payload_fully_covered"] for item in repeats)

assert result["shape_results"]["edge3"]["core_results"]["8"][
    "idle_workitem_count"
] == 5
assert result["shape_results"]["edge10"]["core_results"]["8"][
    "assignment_counts"
] == [2, 2, 1, 1, 1, 1, 1, 1]

print("P9.2 MULTICORE S1 RESULT PASS")
print("run_dir =", result["run_dir"])
PY
```

## 10. P9.2 通过后的边界

P9.2 通过后可以声明：S1 已在 BM1690 上完成参数化 task-major ABI，并在
1/2/4/8 核下通过四类形状和六种语义用例。此时仍不能声明：

- S3/P6 已完成多核迁移；
- 多核或流水线性能已有结论；
- `cb/C` 的复制 ABI 已经是最终性能 ABI；
- 已直接证明 GDMA/BDC 的物理重叠。

下一内部停止门是 P9.3：把相同任务映射迁移到 S3 和 P6，并完成同一正确性
矩阵。只有 P9.3 通过后，才能进入 P10 多核性能测试。
