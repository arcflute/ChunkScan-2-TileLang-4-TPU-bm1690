# P5 手工实现指南：依赖与 LMEM 感知的配置覆盖矩阵

状态：已于 2026-09-14 在项目工作树复现完整配置矩阵、cmodel 数值和
raw-source 门槛，P5 已关闭。P6 是下一阶段，但当前尚未启动。

P5 不搜索“最快配置”，也不使用 cmodel 时间排名。它建立一个固定、可审计
的配置矩阵：合法配置必须通过 raw 结构和 cmodel 数值门槛；非法配置必须
留下明确的依赖、循环深度、LMEM 或 lowering 拒绝原因。

## P5 固定矩阵

| 配置 | 预期结果 | 原因或新增证据 |
| --- | --- | --- |
| `s2_k16_stage2` | `ACCEPTED` | 已关闭的 P4.1 控制，`cb` 在流水线外物化 |
| `sprog_b_k16_stage2` | `ACCEPTED` | P4.2 手工参考，四个有效生产者均为双缓冲 |
| `sprog_b_k16_stage3` | `ACCEPTED` | 新增三阶段候选，四个有效生产者均为三缓冲 |
| `sprog_a_k16_stage2` | `REJECTED` | 请求 LoadX-first，但 raw 目标代码仍被重排为 LoadY-first |
| `sprog_b_k32_stage2` | `REJECTED` | 只有 2 次归约，无法形成非空两阶段 steady state |
| `sprog_b_k16_stage4` | `REJECTED` | 4 次归约等于 4 个 stage，无法形成非空 steady state |

预期拒绝是 P5 的成功结果组成部分，不是测试错误。最终矩阵必须恰好包含
3 个 `ACCEPTED` 和 3 个 `REJECTED`。

## 变更边界

以下内容不得修改：

- W8A16 路线及其快照；
- P2 S0、P3 S1、P4.1 S2、P4.2 S3 的源码和测试；
- 已生成的 `serial`、`reduction_serial`、`pipeline_s2`、`pipeline_s3`
  证据；
- TileLang、TVM 与 PPL 编译器源码。

P5 只新增两个 Python 文件，并向 `run_cmodel.sh` 增加一个独立入口：

- 新增 `chunk_scan_pipeline_p5.py`；
- 新增 `test_chunk_scan_pipeline_matrix_p5_cmodel.py`；
- 新增 `--pipeline-matrix-p5`。

## 第一步：等待并确认 P4.2 已关闭

你稍后完成 P4.2 测试后，从仓库根目录执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
for relative in ("pipeline_s2/result.json", "pipeline_s3/result.json"):
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    print(relative, data["status"])
PY
```

两个输出都必须是 `PASS`。在此之前不要执行第二步。

## 第二步：新增完整 P5 算子文件

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p5.py`，将以下完整候选
文件的 386 行原样粘贴进去：

- [完整 chunk_scan_pipeline_p5.py](/root/autodl-tmp/tmp/chunkscan-p5-delivery/tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p5.py)

正确文件的 SHA-256 为：

```text
2649b196041b02a409ba5e1713b62176bbb7199f2b2dbe494e4ac2e4fdb8706e
```

P4.2 S3 的修改前形式是固定的 sProg-B 两阶段工厂：

```python
def make_chunk_scan_pipeline_s3_kernel():
    ...
    for k_blk in T.Pipelined(REDUCE_TILES, num_stages=2):
        # LoadY.cb -> LoadY.dA/dt -> LoadX -> GEMM
```

P5 的修改后形式是独立的参数化工厂，不修改 S3：

```python
SPROG_A = "sprog_a_loadx_first"
SPROG_B = "sprog_b_loady_first"
SUPPORTED_SCHEDULES = (SPROG_A, SPROG_B)
SUPPORTED_NUM_STAGES = (2, 3)


def make_chunk_scan_pipeline_p5_kernel(schedule: str, num_stages: int):
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(f"unsupported P5 schedule: {schedule}")
    if num_stages not in SUPPORTED_NUM_STAGES:
        raise ValueError(f"unsupported P5 num_stages: {num_stages}")

    ...
    for k_blk in T.Pipelined(
        REDUCE_TILES,
        num_stages=num_stages,
    ):
        if schedule == SPROG_A:
            T.ppl_copy(
                x[chunk * CHUNK_SIZE + k_blk * REDUCE_TILE, 0],
                x_reduce_shared,
            )

        # LoadY.cb/dA/dt and DecayScale

        if schedule == SPROG_B:
            T.ppl_copy(
                x[chunk * CHUNK_SIZE + k_blk * REDUCE_TILE, 0],
                x_reduce_shared,
            )

        # GEMM and FP32 accumulation
```

这里的 sProg-A 故意保留为 lowering 探针。源码虽然把 LoadX 放在 LoadY
之前，但当前 TPU planner 最终仍生成 LoadY-first；P5 必须依据 raw 结果
拒绝它，不能依据 Python 源码意图将其标为接受。

`K32/stage2` 与 `K16/stage4` 不进入 kernel factory。它们先经过循环深度
合法性检查，并在 lowering/cmodel 之前确定性拒绝，避免对已知不可能产生
steady state 的配置浪费 0.5 核环境资源。

## 第三步：新增完整 P5 矩阵测试

修改前：该文件不存在。

修改后：手工新建
`tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_matrix_p5_cmodel.py`，
将以下完整候选文件的 548 行原样粘贴进去：

- [完整 test_chunk_scan_pipeline_matrix_p5_cmodel.py](/root/autodl-tmp/tmp/chunkscan-p5-delivery/tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_matrix_p5_cmodel.py)

正确文件的 SHA-256 为：

```text
e8b893ff5b984da9c7604c7b8b2dc2d6f66ac46c343f11e9d65adce9840de2a8
```

测试分为四部分：

1. 强制检查 P4.1 S2 与 P4.2 S3 的 `result.json` 均为 `PASS`；
2. lower sProg-A，并根据 raw steady future-load 顺序记录确定性拒绝；
3. 在 lowering 前拒绝两个 steady depth 为 0 的配置；
4. 分别在独立 worker 中运行 S3 控制与三阶段 sProg-B，检查 raw 结构、
   cmodel 源变换、全部数学项、负 `D` 和因果污染。

三阶段配置的结构门槛是：

- `reduction_iterations=4`、`num_stages=3`、steady extent 为 1；
- 编译器会展开唯一一次 steady iteration，所以 raw 中不应残留
  `for (int k_blk ...)`，但必须保留一对 parallel markers；
- `cb_loaded_shared`、`dA_reduce_row_fp16_shared`、
  `dt_reduce_row_fp16_shared`、`x_reduce_shared` 分别存在 `_0/_1/_2`；
- 四类生产者在 prologue/steady/epilogue 中的次数分别是 `3/1/0`；
- scan GEMM 在三段中的次数是 `0/1/3`；
- steady future-load 顺序仍为
  `LoadY.cb -> LoadY.dA -> LoadY.dt -> LoadX`。

## 第四步：修改 cmodel 入口

编辑 `tpu_demo/mamba2_chunk_scan/run_cmodel.sh`。

### 4.1 增加 P5 分支

修改前：

```bash
if [[ "${1:-}" == "--pipeline-s3" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s3_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

修改后：

```bash
if [[ "${1:-}" == "--pipeline-s3" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_s3_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-matrix-p5" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_matrix_p5_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

### 4.2 更新 usage

修改前：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2|"\
"--pipeline-s3]" >&2
```

修改后：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial|--pipeline-s2|"\
"--pipeline-s3|--pipeline-matrix-p5]" >&2
```

默认无参数入口与已有入口必须保持不变。

## 第五步：静态检查

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh

/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p5.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_matrix_p5_cmodel.py

sha256sum \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p5.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_matrix_p5_cmodel.py

git diff --check
```

两个 hash 必须与第二、三步相同。命令中的 `_` 前不要添加反斜杠。

## 第六步：顺序执行三次验证

在 0.5 核、2 GB 内存环境中逐条执行，禁止并行：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --identity-only

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-matrix-p5

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s3
```

第二条产生 P5 配置矩阵，第三条回归直接控制 P4.2 S3。忽略终端中所有
cmodel benchmark 和 `Single kernel execution time`。

## 第七步：检查两个结果文件

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
for relative in ("pipeline_matrix_p5/result.json", "pipeline_s3/result.json"):
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    print(relative, data["status"])

matrix = json.loads(
    (root / "pipeline_matrix_p5/result.json").read_text(encoding="utf-8")
)
print("accepted:", matrix["accepted_configurations"])
print("rejected:", matrix["rejected_configurations"])
for name, result in sorted(matrix["configuration_matrix"].items()):
    print(name, result["status"], result.get("reason_code", ""))
PY
```

两个 `result.json` 都必须是 `PASS`。矩阵列表必须是：

```text
accepted:
  s2_k16_stage2
  sprog_b_k16_stage2
  sprog_b_k16_stage3

rejected:
  sprog_a_k16_stage2          schedule_order_not_preserved
  sprog_b_k16_stage4          insufficient_loop_depth
  sprog_b_k32_stage2          insufficient_loop_depth
```

## P5 验收门槛

只有以下条件全部成立，P5 才能关闭并进入 P6：

- P4.1、P4.2 前置结果和 P5 总结果均为 `PASS`；
- `accepted_count=3`、`rejected_count=3`，配置名称与固定矩阵一致；
- 每个拒绝项包含具体 `reason_code`，且未运行不必要的 cmodel；
- sProg-A 请求顺序是 LoadX-first，实际 raw 顺序是 LoadY-first；
- 三阶段候选具有四组 `_0/_1/_2` 生产者缓冲；
- 三阶段 prologue/steady/epilogue 生产者次数为 `3/1/0`，scan GEMM 为
  `0/1/3`；
- 三阶段 raw markers 为 `1/1`，cmodel markers 为 `0/0`；
- 三阶段与 S3、CPU oracle 的所有数值门槛通过，因果污染前后逐位相等；
- 三阶段 LMEM 不超过 262,144 字节，C 轴零步长计数为 0；
- `selection_policy` 不含基于时间的排名，`performance_claim` 明确为
  `NONE`。

项目验收结果：P5 总状态为 `PASS`，接受/拒绝数量为 `3/3`；三阶段
候选对 S3 的所有最大绝对差均为 `0`，all-terms 对 oracle 最大绝对误差为
`6.103515625e-05`；raw source 的静态 LMEM 上界为
`69632 / 262144` 字节。三项预期拒绝均记录了确定性原因。
