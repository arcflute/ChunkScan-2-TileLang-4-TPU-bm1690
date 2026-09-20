# P4.2 手工实现指南：论文顺序的 ChunkScan S3

状态：已于 2026-09-14 在项目工作树复现全部 cmodel 与 raw-source 门槛，
P4.2 已关闭。其 S3 产物已经作为 P5 的直接控制与前置证据。

本阶段以已经关闭的 P4.1 S2 为唯一控制组，继续使用 `L=64`、
`K_TILE=16`、四次归约和 `num_stages=2`。S2 只把 `dA`、`dt`、`x`
作为流水线内的全局生产者，因果 `cb` 切片预先在流水线外准备；S3 则把
有效的 dense `cb` 全局加载也移入流水线，并按照 PipeThreader 论文
Fig. 6 选用的 sProg-B 顺序验证 `LoadY` 在 `LoadX` 之前。

全部执行仍限于 CPU 承载的 BM1690 cmodel。cmodel 时间不用于评价性能，
raw 目标代码的流水线结构也不等价于真实硬件上的物理重叠。

## 变更边界

以下内容必须保持不变：

- W8A16 路线及其快照；
- P2 S0、P3 S1、P4.1 S2 的 Python 文件；
- `artifacts/serial/`、`artifacts/reduction_serial/` 和
  `artifacts/pipeline_s2/` 的既有证据；
- TileLang、TVM 和 PPL 代码生成器源码。

P4.2 只进行三项源码变更：

1. 新增 `chunk_scan_pipeline_s3.py`；
2. 新增 `test_chunk_scan_pipeline_s3_cmodel.py`；
3. 给 `run_cmodel.sh` 增加独立入口 `--pipeline-s3`。

## 第一步：确认 P4.1 控制证据

从仓库根目录执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("tpu_demo/mamba2_chunk_scan/artifacts/pipeline_s2/result.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(path, data["status"])
PY
```

输出必须是 `PASS`。若不是，不要安装 S3，先重新处理 P4.1。

## 第二步：新增完整 S3 算子文件

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s3.py`，将以下完整候选
文件的 354 行原样粘贴进去：

- [完整 chunk_scan_pipeline_s3.py](/root/autodl-tmp/tmp/chunkscan-p4-s3-delivery/tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s3.py)

正确文件的 SHA-256 为：

```text
d96d7d2f3679f93dcc3a5d70ba2344adb0f64fad180b24acffe6e116dfeb276f
```

S2 与 S3 的关键差异如下。

修改前，S2 在流水线外准备四个因果 `cb` 切片，流水线内只选择局部块：

```python
T.ppl_fill(cb_causal_0, T.float16(0.0))
T.ppl_fill(cb_causal_1, T.float16(0.0))
T.ppl_fill(cb_causal_2, T.float16(0.0))
T.ppl_fill(cb_causal_3, T.float16(0.0))

# 多段 unroll 将下三角 cb 搬入 cb_causal_0 ... cb_causal_3

for k_blk in T.Pipelined(REDUCE_TILES, num_stages=2):
    if k_blk == 0:
        T.ppl_copy(cb_causal_0, cb_reduce_shared)
    elif k_blk == 1:
        T.ppl_copy(cb_causal_1, cb_reduce_shared)
    elif k_blk == 2:
        T.ppl_copy(cb_causal_2, cb_reduce_shared)
    else:
        T.ppl_copy(cb_causal_3, cb_reduce_shared)
```

修改后，S3 在流水线外只准备严格上三角校正矩阵，并在流水线内真实加载
dense `cb`；`dense - upper` 精确恢复包含对角线的因果块：

```python
T.ppl_fill(cb_upper_shared, T.float16(0.0))
for row in T.unroll(0, CHUNK_SIZE - 1):
    T.call_extern(
        "handle",
        "ppl.copy",
        region(
            cb[chunk * CHUNK_SIZE + row, row + 1],
            "r",
            1,
            CHUNK_SIZE - row - 1,
        ),
        region(
            cb_upper_shared[row, row + 1],
            "w",
            1,
            CHUNK_SIZE - row - 1,
        ),
    )

for k_blk in T.Pipelined(REDUCE_TILES, num_stages=2):
    T.ppl_copy(
        cb[chunk * CHUNK_SIZE, k_blk * REDUCE_TILE],
        cb_loaded_shared,
    )
    T.ppl_copy(
        cb_upper_shared[0, k_blk * REDUCE_TILE],
        cb_upper_reduce,
    )
    T.ppl_copy(cb_loaded_shared, cb_dense_compute_shared)
    T.ppl_subtract(
        cb_reduce_shared,
        cb_dense_compute_shared,
        cb_upper_reduce,
    )

    # LoadY.dA/dt -> decay/scale -> LoadX -> GEMM -> FP32 add
```

`cb_dense_compute_shared` 是必要的局部物化缓冲区。当前代码生成路径无法
直接给被流水线版本化的 `cb_loaded_shared` 解析稳定的 elementwise
descriptor；先做一次局部 copy 后，shape 和地址都能稳定 lowering。

不要把上三角矩阵改成从单个全 1 行向 64 个 NPU lane 做局部复制。该写法
虽然能 lowering 和编译，但 BM1690 cmodel 会在跨 lane 的 BDC copy 执行时
退出。也不要拆成四个上三角局部 tile；当前地址分配器可能把它们与后续
计算缓冲区复用，从而破坏尚未消费的 tile。候选中的单个 `[64,64]`
`cb_upper_shared` 用明确的完整生命周期规避了这两个问题。

## 第三步：新增完整 P4.2 cmodel 测试

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s3_cmodel.py`，将以下
完整候选文件的 641 行原样粘贴进去：

- [完整 test_chunk_scan_pipeline_s3_cmodel.py](/root/autodl-tmp/tmp/chunkscan-p4-s3-delivery/tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s3_cmodel.py)

正确文件的 SHA-256 为：

```text
e8a5c09f22a2b0eba357d32d3031c85ffaa24f9286bd6be249102a5659f13228
```

测试会在两个独立 worker 中分别编译和执行 S2 与 S3，然后进行以下验收：

- S2、S3 分别与同一 CPU oracle 比较，并直接比较 S3 与 S2；
- 覆盖 state-only、scan-only、正负 `D` residual-only、all-terms；
- 用值 `8.0` 污染全部 `cb` 上三角，要求 S3 干净/污染输出逐位相等；
- 检查四个流水线生产者均有 `_0`、`_1` 静态版本；
- 将 raw kernel 切分成 prologue、steady state、epilogue 并检查生产者与
  scan GEMM 的出现次数；
- 检查 steady state 的未来装载顺序确实为
  `LoadY.cb -> LoadY.dA -> LoadY.dt -> LoadX`；
- 检查 S2 raw 源中没有 `cb_loaded_shared`，而 S3 中存在；
- 检查 raw/cmodel 源的唯一差异是 cmodel 适配器删除了 parallel markers；
- 检查指令族计数、非法 C 轴零步长和 LMEM 上界。

## 为什么这里不填写显式 `order` 和 `stage`

TileLang 前端能够接收 `order` 和 `stage`，但当前 TPU 标准 lowering 路径
会先运行 `PipelinePlanning`，再运行 `InjectSoftwarePipeline`。现有
`pipeline_planning.cc` 只读取 `num_stages`，随后重新生成流水线规划；因此
把前端显式 `order/stage` 当作最终 TPU 排布证据是不可靠的。

P4.2 不提前修改编译器。S3 用可复现的 TileLang 源语句顺序表达 sProg-B，
并以 raw kernel 中的实际顺序作为硬性门槛。让编译器稳定保留或生成指定
排布属于 P6 自动化阶段。

## 第四步：修改 cmodel 入口

编辑 `tpu_demo/mamba2_chunk_scan/run_cmodel.sh`。

### 4.1 增加 P4.2 分支

修改前：

```bash
if [[ "${1:-}" == "--pipeline-s2" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s2_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

修改后：

```bash
if [[ "${1:-}" == "--pipeline-s2" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s2_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-s3" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s3_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

### 4.2 更新 usage

修改前：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2]" >&2
```

修改后：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2|"\
"--pipeline-s3]" >&2
```

不要修改默认无参数分支，也不要删除四个单线程环境变量。

## 第五步：静态检查

所有命令从仓库根目录顺序执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh

/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s3.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s3_cmodel.py

sha256sum \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_s3.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_s3_cmodel.py

git diff --check
```

两个 hash 必须与第二、三步完全相同。命令中的 `_` 前不要添加反斜杠。

## 第六步：顺序执行三次验证

当前只有 0.5 核 CPU 和 2 GB 内存，三条命令必须逐条运行，不能并行：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --identity-only

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s3

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s2
```

第二条内部会在独立进程中运行 S2 控制和 S3 实验；第三条再次回归受保护
的 P4.1 S2。终端出现的 benchmark 和 `Single kernel execution time` 都是
cmodel 时间，忽略即可。

## 第七步：检查两个结果文件

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
for relative in ("pipeline_s3/result.json", "pipeline_s2/result.json"):
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    print(relative, data["status"])
PY
```

再输出 S3 的核心结构证据：

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("tpu_demo/mamba2_chunk_scan/artifacts/pipeline_s3/result.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data["pipeline_structure"], indent=2, sort_keys=True))
print(json.dumps(data["source_metrics"], indent=2, sort_keys=True))
PY
```

## P4.2 验收门槛

只有以下条件同时成立，P4.2 才能关闭并进入 P5：

- `pipeline_s3/result.json` 与 `pipeline_s2/result.json` 均为 `PASS`；
- 所有常规用例均有 `s2_close_to_oracle=true`、
  `s3_close_to_oracle=true`、`s3_close_to_s2=true`，且无 NaN/Inf；
- S3 的干净/污染输出 bitwise equal，污染结果仍匹配 oracle 和 S2；
- `reduction_iterations=4`、`num_stages=2`、steady loop extent 为 2；
- raw parallel start/end 各 1，cmodel parallel start/end 各 0；
- `cb_loaded_shared`、`dA_reduce_row_fp16_shared`、
  `dt_reduce_row_fp16_shared`、`x_reduce_shared` 均有 `_0`、`_1`；
- prologue 中四类生产者各出现 2 次，steady 中各出现 1 次，epilogue 中
  各出现 0 次；scan GEMM 在三段中的次数依次为 0、1、2；
- steady future load 顺序为 `LoadY.cb`、`LoadY.dA`、`LoadY.dt`、`LoadX`；
- `static_local_memory_end_bytes <= 262144`，C 轴零步长计数为 0；
- 工具链路径全部属于当前 ChunkScan 仓库；
- 结果中继续明确写出 `performance_claim = NONE`。

项目验收结果是：all-terms 的 S3-to-oracle 最大绝对误差为
`6.103515625e-05`，scan-only 为 `1.52587890625e-05`，所有 S3-to-S2
最大绝对差为 `0`；上三角污染前后逐位相等；raw source 的 LMEM 静态
上界为 `69632 / 262144` 字节。P4.2 的项目结果为 `PASS`。
