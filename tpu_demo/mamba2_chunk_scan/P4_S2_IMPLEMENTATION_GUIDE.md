# P4.1 手工实现指南：两阶段 Load/Compute 流水线 S2

状态：已于 2026-09-13 在项目工作树复现全部 cmodel 与 raw-source 门槛，
P4.1 已关闭。P4.2 S3 继续使用本阶段产物作为直接控制组。

本阶段以 P3 的四次 `K=16` 串行归约 S1 为唯一控制组，只改变因果扫描
归约循环的排布：使用 `T.Pipelined(..., num_stages=2)`，让下一迭代的
`dA`、`dt` 和 `x` 全局到局部加载使用第二套生产者缓冲区。所有实验仍
只在 CPU 承载的 BM1690 cmodel 中执行；cmodel 时间不是性能证据。

## 变更边界

以下已验收内容必须保持不变：

- `chunk_scan_serial.py` 和 `test_chunk_scan_cmodel.py`；
- `chunk_scan_reduction_serial.py` 和
  `test_chunk_scan_reduction_cmodel.py`；
- `artifacts/serial/` 与 `artifacts/reduction_serial/` 中的既有证据；
- W8A16 路线及其快照。

P4.1 只新增两个 Python 文件，并给 `run_cmodel.sh` 增加一个入口：

- 新增 `chunk_scan_pipeline_s2.py`；
- 新增 `test_chunk_scan_pipeline_s2_cmodel.py`；
- 增加 `--pipeline-s2`，不改变任何现有入口的含义。

## 第一步：确认 P3 控制证据

从仓库根目录执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

grep -n '"status": "PASS"' \
  tpu_demo/mamba2_chunk_scan/artifacts/reduction_serial/result.json
```

若文件不存在或不是 `PASS`，先重新执行：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --reduction-serial
```

## 第二步：新增完整 S2 算子文件

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s2.py`，将以下完整候选
文件的 407 行原样粘贴进去：

- [完整 chunk_scan_pipeline_s2.py](/root/autodl-tmp/tmp/chunkscan-p4-s2-delivery/tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s2.py)

正确文件的 SHA-256 为：

```text
d17e0668e312daf4a44546aa02a205829a61f815b188aaa7c87abcd3b5de3c01
```

S1 的关键循环是：

```python
for k_blk in T.serial(REDUCE_TILES):
    # LoadY -> DecayScale -> LoadX -> GEMM -> FP32 add
```

S2 的关键循环改为：

```python
for k_blk in T.Pipelined(REDUCE_TILES, num_stages=2):
    # LoadY.dA/dt -> DecayScale -> LoadX -> GEMM -> FP32 add
```

候选中还包含两项必要的适配：

1. 四个严格下三角 `cb` 的 `K16` 切片先在流水线外物化为只读局部块，
   流水线内按 `k_blk` 选择对应块；当前 TPU pipeline rewriter 不能对条件
   分支中读取且被重复写入的同一 `cb` 缓冲区正确做版本化。
2. 被版本化的两个行缓冲区名称保留 `_shared`：
   `dA_reduce_row_fp16_shared` 和 `dt_reduce_row_fp16_shared`。当前 PPL
   `LetStmt` 代码生成路径依靠名称识别局部 tensor descriptor；删除该后缀
   会错误生成 `void *` 别名并导致 C 编译失败。

这里没有修改编译器，也没有覆盖 S1。`cb_reduce_shared` 是流水线内 BDC
计算临时量，不跨加载阶段，因此没有无依据地复制为双缓冲。

## 第三步：新增完整 P4.1 cmodel 测试

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s2_cmodel.py`，将以下
完整候选文件的 536 行原样粘贴进去：

- [完整 test_chunk_scan_pipeline_s2_cmodel.py](/root/autodl-tmp/tmp/chunkscan-p4-s2-delivery/tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s2_cmodel.py)

正确文件的 SHA-256 为：

```text
76ea6b7e0eadefee424ea81f1b6ab30b9a82bcf45c11807dcc5e6aaeb5df5fda
```

测试会顺序启动两个独立 worker：S1 串行归约控制和 S2 两阶段流水线。
二者不能加载到同一个 Python 进程。主进程随后执行三类门槛：

- S1、S2 分别与同一 CPU oracle 比较，并直接比较 S2 与 S1；
- 检查全部三项数学、负 `D`、双 chunk 非零和因果上三角污染；
- 检查原始目标代码的流水线标记、稳态循环、双缓冲声明、指令族计数、
  非法 C 轴零步长和 LMEM 上界，并确认 cmodel 源只删除了流水线标记。

## 第四步：修改 cmodel 入口

编辑 `tpu_demo/mamba2_chunk_scan/run_cmodel.sh`。

### 4.1 增加 P4.1 分支

修改前：

```bash
if [[ "${1:-}" == "--reduction-serial" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_reduction_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

修改后：

```bash
if [[ "${1:-}" == "--reduction-serial" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_reduction_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-s2" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s2_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

### 4.2 更新 usage

修改前：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial]" >&2
```

修改后：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2]" >&2
```

不要修改默认无参数分支；它必须继续运行 P2 S0。现有四个单线程环境变量
也必须原样保留。

## 第五步：静态检查

所有命令从仓库根目录顺序执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh

/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s2.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s2_cmodel.py

sha256sum \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s2.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s2_cmodel.py

git diff --check
```

两个 hash 必须与第二、三步完全相同。命令中的 `_` 前不要添加反斜杠。

## 第六步：顺序执行三次验证

0.5 核、2 GB 内存环境中不要并行运行：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --identity-only

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s2

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --reduction-serial
```

第一条确认工具链身份；第二条生成 P4.1 证据；第三条回归受保护的 P3
S1。终端中的 benchmark 和 `Single kernel execution time` 全部来自 cmodel，
不采信、不比较，也不写入性能结论。

## 第七步：检查两个结果文件

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
for relative in ("pipeline_s2/result.json", "reduction_serial/result.json"):
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    print(relative, data["status"])
PY
```

再检查 S2 结构证据：

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "tpu_demo/mamba2_chunk_scan/artifacts/pipeline_s2/result.json"
)
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data["pipeline_structure"], indent=2, sort_keys=True))
print(json.dumps(data["source_metrics"], indent=2, sort_keys=True))
PY
```

## P4.1 验收门槛

只有以下条件同时成立，S2 才能关闭并开始 S3：

- `pipeline_s2/result.json` 与 `reduction_serial/result.json` 均为 `PASS`；
- state、scan、residual、负 `D` 和 all-terms 均满足
  `s2_close_to_oracle=true`、`s2_close_to_s1=true`，且没有 NaN/Inf；
- S1 与 S2 的干净/污染输出各自 bitwise equal；
- `reduction_iterations=4`、`num_stages=2`、稳态循环 extent 为 2；
- 原始目标源中 `tpu_parallel_start/end` 各 1 个，cmodel 源中各 0 个；
- `dA`、`dt`、`x` 三个生产者各存在 `_0`、`_1` 两个静态版本；
- C 轴零步长计数为 0，LMEM 上界不超过 262,144 字节；
- 工具链路径全部属于当前 ChunkScan 仓库；
- 不使用 cmodel 时间作性能或物理重叠结论。

项目验收结果为：all-terms 的 S2-to-oracle 最大绝对误差
`6.103515625e-05`，S2-to-S1 最大绝对差为 `0`；scan-only 的
S2-to-S1 最大绝对差为 `0`；LMEM 静态上界为
`69632 / 262144` 字节。受保护的 P3 S1 回归仍为 `PASS`。
