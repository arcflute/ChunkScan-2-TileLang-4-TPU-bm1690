# P3 手工实现指南：串行归约分块 S1

状态：已于 2026-09-13 在项目工作树中复现全部门槛，P3 已关闭。

P3 的目标是在不修改 P2 串行 S0 的前提下，把因果 ChunkScan GEMM 的
源位置归约轴拆成四个 `K=16` 的串行迭代，为 P4 的手工流水线建立正确性
控制组。P3 只使用 CPU 承载的 BM1690 cmodel，不进行真实 TPU 测试，也
不使用 cmodel 时间评价性能。

## 变更边界

保留以下 P2 文件不变：

- `chunk_scan_serial.py`；
- `test_chunk_scan_cmodel.py`；
- `artifacts/serial/` 中已经通过的 S0 证据。

本阶段只新增两个 Python 文件并小范围修改一个 Shell 入口：

- 新增 `chunk_scan_reduction_serial.py`；
- 新增 `test_chunk_scan_reduction_cmodel.py`；
- 修改 `run_cmodel.sh`，固定单线程环境并增加
  `--reduction-serial` 入口。

## 第一步：确认 P2 控制证据

从仓库根目录执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan
grep -n '"status": "PASS"' \
  tpu_demo/mamba2_chunk_scan/artifacts/serial/result.json
```

如果文件不存在或没有 `PASS`，先串行执行一次 P2：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh
```

## 第二步：新增完整 S1 算子文件

手工新建：

```text
tpu_demo/mamba2_chunk_scan/chunk_scan_reduction_serial.py
```

将交付候选文件的全部 1243 行原样粘贴进去。粘贴后其 SHA-256 必须是：

```text
b5a117831865e526e58affad28bc967f29e2627df885aa5ecb8e2d12ab491081
```

这个文件不替换 S0。它保留相同七输入加一输出 ABI、固定 shape 和三项
数学，只将因果扫描项改成：

```text
for chunk in 2 chunks:
    historical_state_term
    scan_accumulator = 0
    for k_blk in 4 K16 reduction tiles:
        LoadY(cb, dA, dt)
        DecayScale
        LoadX
        K16 GEMM
        FP32 accumulate
    add historical term
    add D*x residual
    store FP16 output
```

每次 K16 GEMM 前必须显式清零 `gemm_temp`。该行是 cmodel 正确性要求，
不能省略：

```python
T.ppl_fill(gemm_temp, T.float32(0.0))
T.ppl_gemm(cb_reduce_shared, x_reduce_shared, gemm_temp)
```

## 第三步：新增完整 P3 cmodel 测试

手工新建：

```text
tpu_demo/mamba2_chunk_scan/test_chunk_scan_reduction_cmodel.py
```

将交付候选文件的全部 484 行原样粘贴进去。粘贴后其 SHA-256 必须是：

```text
fe526d01f6d5bce0408638d2f2139131605b37d20e800344ef9cd861f2168cbd
```

该测试先读取 `artifacts/serial/result.json`，确认 P2 控制证据已经关闭，
然后顺序启动两个独立 worker 进程。S0 与 S1 分别编译和加载各自的 cmodel
适配器，保存输出后由主进程直接比较 S1 与 S0，并将二者分别与同一 CPU
oracle 比较。不要把两个 cmodel 适配器加载到同一 Python 进程；当前
运行时不能可靠隔离它们。

## 第四步：修改 cmodel 入口

编辑：

```text
tpu_demo/mamba2_chunk_scan/run_cmodel.sh
```

### 4.1 固定无卡环境的 CPU 线程数

修改前：

```bash
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
export PYTHONDONTWRITEBYTECODE=1
```

修改后：

```bash
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

这里必须覆盖外部环境里的非法值 `OMP_NUM_THREADS=0` 和
`MKL_NUM_THREADS=0`。

### 4.2 增加 P3 运行分支

修改前：

```bash
if [[ "${1:-}" == "--residual-cast-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_residual_cast_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

修改后：

```bash
if [[ "${1:-}" == "--residual-cast-probe" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/microprobes/test_residual_cast_cmodel.py"
fi

if [[ "${1:-}" == "--reduction-serial" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_reduction_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

### 4.3 更新 usage

修改前：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe]" >&2
```

修改后：

```bash
    printf '%s\n' \
        "Usage: $0 [--identity-only|--broadcast-probe|"\
"--exp-probe|--causal-mask-probe|--gemm-probe|"\
"--residual-cast-probe|--reduction-serial]" >&2
```

不带参数的默认分支必须继续运行 P2 S0，不能改成 S1。

## 第五步：静态检查

以下命令全部从仓库根目录顺序执行。命令中的下划线 `_` 前不要添加
反斜杠：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh

/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_reduction_serial.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_reduction_cmodel.py

sha256sum \
  tpu_demo/mamba2_chunk_scan/chunk_scan_reduction_serial.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_reduction_cmodel.py

git diff --check
```

## 第六步：顺序执行 cmodel gate

0.5 核和 2GB 内存环境中不要并行启动测试：

```bash
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --identity-only

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --reduction-serial

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh
```

第二条验证 P3 S1；第三条重新验证受保护的 P2 S0。输出中的
`Single kernel execution time` 和 benchmark 数字来自 cmodel，全部忽略。

## 第七步：检查证据

```bash
grep -nE \
  '"status":|"max_abs_diff":|"bitwise_equal_clean_vs_poisoned_s1":|"static_local_memory_end_bytes":|"parallel_start_count":|"parallel_end_count":' \
  tpu_demo/mamba2_chunk_scan/artifacts/reduction_serial/result.json

grep -n '"status": "PASS"' \
  tpu_demo/mamba2_chunk_scan/artifacts/serial/result.json
```

P3 的关闭条件是：

1. S0 和 S1 的 `result.json` 都为 `PASS`；
2. S1 的 state-only、scan-only、正负 residual-only 和 all-terms 均通过；
3. clean/poisoned S1 输出 bitwise equal；
4. raw source 与本阶段 cmodel source 相同；
5. raw source 只有一个四次 `k_blk` 串行归约循环；
6. `parallel_start_count = parallel_end_count = 0`；
7. C 轴零 stride 数为 0；
8. 静态 LMEM 上界不超过 262144 字节；
9. 不引用任何 cmodel 时间作为性能证据。

预验证候选的 all-terms 最大绝对误差为 `6.103515625e-05`，scan-only
最大绝对误差为 `1.52587890625e-05`，静态 LMEM 上界为 69632 字节。
本地手工应用后的结果不要求逐位复刻这些浮点误差，但必须满足
`atol = rtol = 1e-2` 和上述全部结构门槛。
