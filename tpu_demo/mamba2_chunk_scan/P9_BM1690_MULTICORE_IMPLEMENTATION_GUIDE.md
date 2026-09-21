# P9 BM1690 多核实现指南

P9 是融合阶段，包含形状参数化、多核运行时接口和 1/2/4/8 核
ChunkScan 正确性。本指南按内部停止门推进，但这些停止门仍属于同一个 P9：

1. P9.1：证明 BM1690 runtime 的 work-item 编号与核数语义；
2. P9.2：参数化物理 ABI 和任务索引，先完成多核 S1；
3. P9.3：迁移 S3/P6，并完成全部形状与核数的正确性矩阵。

P9.1 已在真实 BM1690 上通过 1/2/4/8 核各 20 次验收。P9.2 的实际
ChunkScan S1 实现与命令见
`P9_2_BM1690_MULTICORE_S1_IMPLEMENTATION_GUIDE.md`。

## 1. P9.1 修改内容

- `T.ppl_workitem_index()`：lower 为设备端 `tpu_workitem_index()`；
- `T.ppl_workitem_num()`：lower 为设备端 `tpu_workitem_num()`；
- `kernel_template_device_multicore.cpp`：从 `CHUNKSCAN_CORE_NUM` 读取
  1/2/4/8，将参数结构复制为对应份数，并以相同 `block_num` 启动；
- `multicore_workitem_probe_p9.py`：每个 work-item 将自身编号和总 work-item
  数写入独立常量槽位；
- `test_multicore_workitem_probe_p9.py`：编译、ELF/runtime 审计以及
  1/2/4/8 核各 20 次确定性测试；
- `test_tilelang_target_codegen_ppl_workitem.py`：不启动 TPU 的 lowering
  单元测试。

探针输出为 `[8,2]` FP32：第 `i` 个活跃行必须严格等于
`[i, core_num]`，未激活行必须保持 `-777` sentinel。

## 2. 获取提交

在 AutoDL 推送 P9.1 提交后，在 BM1690 电脑执行：

```bash
cd ~/ChunkScan-bm1690-test-86add2f
git fetch origin main
git cherry-pick <P9_WORKITEM_COMMIT>
```

`3rdparty/tvm` 的小写 `m` 是已应用 TVM 补丁的正常状态，不要清理或重新
应用补丁。

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
```

## 4. 静态检查

```bash
./.venv/bin/python -m py_compile \
  tilelang/language/customize.py \
  tilelang/language/__init__.py \
  tpu_demo/mamba2_chunk_scan/multicore_workitem_probe_p9.py \
  tpu_demo/mamba2_chunk_scan/test_multicore_workitem_probe_p9.py \
  testing/python/target/test_tilelang_target_codegen_ppl_workitem.py

git diff --check
```

## 5. 增量重编译 TileLang

本次修改包含 `src/target/codegen_ppl.cc`，所以必须重建；只复制 Python
文件而不重建会让新 DSL 调用无法发射。

```bash
cmake --build build -j 4 2>&1 | tee /tmp/chunkscan_p9_rebuild.log
rebuild_exit=${PIPESTATUS[0]}
echo "rebuild_exit=$rebuild_exit"
```

通过条件：

```text
Built target tilelang_module
rebuild_exit=0
```

不要删除 `build/` 或重新配置 CMake；使用现有成功的 build 目录增量构建。

## 6. Lowering 单元测试

```bash
./.venv/bin/python \
  testing/python/target/test_tilelang_target_codegen_ppl_workitem.py \
  2>&1 | tee /tmp/chunkscan_p9_workitem_unit.log

unit_exit=${PIPESTATUS[0]}
echo "unit_exit=$unit_exit"
```

通过标志：

```text
PPL work-item codegen PASS
unit_exit=0
```

该步骤不启动 TPU，只检查生成源码。若生成源码仍出现
`ppl.workitem_index` 或缺少 `tpu_workitem_index()`，不得继续。

## 7. 只编译真机探针

```bash
timeout 1800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_multicore_workitem_probe_p9.py \
  2>&1 | tee /tmp/chunkscan_p9_workitem_compile.log

compile_exit=${PIPESTATUS[0]}
echo "compile_exit=$compile_exit"
```

通过标志：

```text
P9 WORKITEM COMPILE PASS
P9 WORKITEM COMPILE-ONLY PASS: ...
compile_exit=0
```

编译清单还必须证明：设备库为 RISC-V、host 库为 x86-64、host 动态库
链接真实 `/opt/tpuv7/.../libtpuv7_rt.so` 且不含 emulator。

## 8. 1/2/4/8 核真机测试

```bash
timeout 1800s ./.venv/bin/python \
  tpu_demo/mamba2_chunk_scan/test_multicore_workitem_probe_p9.py \
  --run --repeat 20 --core-counts 1,2,4,8 \
  2>&1 | tee /tmp/chunkscan_p9_workitem_run.log

run_exit=${PIPESTATUS[0]}
echo "run_exit=$run_exit"
```

通过标志：

```text
P9 WORKITEM 1-CORE PASS repeat=20
P9 WORKITEM 2-CORE PASS repeat=20
P9 WORKITEM 4-CORE PASS repeat=20
P9 WORKITEM 8-CORE PASS repeat=20
P9 WORKITEM PASS: ...
run_exit=0
```

## 9. 自动验收 JSON

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/"
    "device_multicore_probe_p9/result.json"
)
result = json.loads(path.read_text())

assert result["status"] == "PASS"
assert result["physical_multicore_dispatch_validated"] is True
assert result["chunk_scan_multicore_validated"] is False
assert set(result["core_results"]) == {"1", "2", "4", "8"}

for text, core_result in result["core_results"].items():
    core_num = int(text)
    assert core_result["status"] == "PASS"
    assert core_result["repeat"] == 20
    assert core_result["expected_indices"] == list(range(core_num))
    assert core_result["expected_workitem_num"] == core_num
    assert len(core_result["repetitions"]) == 20
    assert all(item["exact"] for item in core_result["repetitions"])
    assert all(item["deterministic"] for item in core_result["repetitions"])
    assert all(
        item["inactive_sentinel_unchanged"]
        for item in core_result["repetitions"]
    )

print("P9.1 WORKITEM RESULT PASS")
print("run_dir =", result["run_dir"])
PY
```

## 10. P9.1 通过后的边界

P9.1 只证明：BM1690 host 的 `block_num`、参数结构数量以及设备端
work-item index/num 在 1/2/4/8 核下语义一致。它不证明 ChunkScan 已经多核，
也没有性能结论。

该停止门已通过。P9.2 按 `(batch, chunk, head)` 分配互不重叠的输出任务，
并首先验证多核 S1 正确性；不要把 P9.1 探针结果当成 ChunkScan 多核结果。
