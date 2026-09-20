# P7 实施指南：单入口复跑与 cmodel 证据封存

状态：已于 2026-09-20 在当前工作树完成独立验收。运行编号
`20260920T065556Z-4383`；结果见 `artifacts/p7/result.json` 和
`P7_CLOSURE_REPORT.md`。P7 不修改 ChunkScan 数学、P6 编译器、W8A16
旧路线或真实设备后端。

## 目标与边界

从仓库根目录运行一次 `./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --p7`，
按顺序复跑工具链身份、A0 的 14 个 CPU 测试、P1 的五项原语、P2
串行 S0、P3 归约分块 S1、P4 的 S2/S3、P5 六项配置矩阵、P6 的 7 个
规划器单元测试和显式 sProgram 候选。每项运行在独立子进程中，不能
并行；已有阶段测试仍负责自己的数值和结构门。

P7 新入口负责检查每个阶段的退出码、`result.json` 的 `PASS` 与本次
重新写入、必须存在的 TIR/原始目标代码/cmodel 代码/manifest/LMEM
度量，并汇总身份、源码契约哈希、P5 接受/拒绝表、P6 关键结构门和
文本证据 SHA-256。不要用 cmodel 运行时间排名或作硬件性能结论。

本阶段只让用户手动新增 `close_p7_cmodel.py`，并在 `run_cmodel.sh`
添加 `--p7` 分支；对应完整代码和修改前后对照由 Codex 在对话中提供。
Markdown 文档由 Codex 维护。不要再执行 W8A16 快照脚本，也不要覆盖
旧快照。

## 执行前检查

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan
git status --short
git diff --check
bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh
/root/autodl-tmp/tilelang-tpu/.venv/bin/python -c \
  'import pytest, torch; print(pytest.__version__, torch.__version__)'
```

如果 `pytest` 或 CPU PyTorch 缺失，先修复**当前 `PYTHON_BIN` 所指的环境**。
不要切换到另一 Python 解释器来掩盖依赖问题。保留现有所有阶段产物；
P7 的“本次重新写入”检查能识别旧结果，不需要清空 `artifacts`。

## 应用代码后静态检查

```bash
bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh
/root/autodl-tmp/tilelang-tpu/.venv/bin/python -c \
  'import ast, pathlib; ast.parse(pathlib.Path("tpu_demo/mamba2_chunk_scan/close_p7_cmodel.py").read_text())'
git diff --check
```

## 运行与验收

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --p7
```

终端逐项显示阶段名和日志路径。详细输出保存在
`tpu_demo/mamba2_chunk_scan/artifacts/p7/runs/<run-id>/`；单入口最终
状态保存在 `artifacts/p7/result.json`。进程退出码为 0、总状态为
`PASS`、14 个步骤均为 `PASS`、A0 报告 14 个测试通过、P6 规划器报告
7 个测试通过、P5 为 3 接受/3 拒绝、P6 显式契约与负向门均通过，才
能进入人工证据复核。任何一步失败应停止，保留失败日志，`result.json`
须为 `FAIL`；不能拿上一轮的阶段 `PASS` 顶替本轮。

人工复核还要确认：

1. P6 `kernel_raw.c` 保留流水线标记，而 `kernel_cmodel.c` 仅因移除
   标记而不同；P7 索引包含两者及其 SHA-256。
2. `source_contract.md`、CPU oracle、TIR、源码、manifest、LMEM
   度量、数值 `result.json` 都能从 P7 索引定位。P5 拒绝项包含具体
   原因，不能称为成功候选。
3. P7 `scope` 明确声明：已验证的是 BM1690 目标代码结构与 CPU-hosted
   cmodel 数值；尚未验证真实 TPU 执行、性能、物理流水线重叠、四核扩展
   或完整 Mamba2 模型集成。
4. `toolchain_identity.repository_dirty` 与 `tvm_dirty` 如实记录。当前
   工作树未提交时，P7 只保证**这份工作树上的单入口复跑**；准备 GitHub
   跨服务器复现仍需提交未跟踪源码和 TVM 改动，且不能只上传模拟器产物。

可用以下只读命令检查总结：

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

p = Path('tpu_demo/mamba2_chunk_scan/artifacts/p7/result.json')
r = json.loads(p.read_text(encoding='utf-8'))
print('status:', r['status'])
print('steps:', [(s['name'], s['status']) for s in r['steps']])
print('P5:', r.get('p5_configuration_outcomes'))
print('evidence files:', len(r.get('evidence_index', {})))
print('scope:', r['scope'])
PY
```

以上验收已在当前工作树完成，README 和总计划现标记 P7 为完成。
最终范围仍须明确：这是独立 `_chunk_scan_fwd`，不是
`mamba_chunk_scan_combined`；真机 BM1690E/SG2260E 迁移应作为新阶段
另行规划。
