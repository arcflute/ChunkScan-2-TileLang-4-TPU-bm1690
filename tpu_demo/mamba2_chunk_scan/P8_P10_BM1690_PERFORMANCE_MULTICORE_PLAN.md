# P8–P10：BM1690 性能测试与多核扩展计划

本计划接续已经完成的 P0–P7 cmodel 收口和 BM1690 单核真机正确性
验证。后续工作分成两条有先后依赖的路线：先建立可信的单核性能测量，
再扩展为 BM1690 八核执行，最后测量流水线优化与多核扩展的组合收益。

## 1. 当前基线与边界

当前真机代码已经证明 S0、S1、S2、S3、P6 在固定形状
`B=G=H=1, S=128, Ck=2, L=64, P=64, N=128` 上单核正确。
它还不能支持下列结论：

- `main_template_device.cpp` 每个用例只启动一次 kernel，控制台打印的
  单次时间容易受到首次运行、主机调度和计时粒度影响，不是正式性能数据；
- 现有 host wrapper 中 `core_num=1`、`group_num=1`、`block_num=1`；
- 现有算子使用 `T.Kernel(1, 1, is_cpu=True)`，生成设备代码中没有
  `tpu_workitem_index()`，因此现在确实是单核；
- 当前形状只有两个 chunk，工作量过小，不适合作为八核扩展或最终性能结论；
- 原始设备源码中的 `tpu_parallel_start/end` 证明排布已经发射，
  但不单独证明 GDMA 与 BDC 在硬件上发生了物理重叠。

必须保护现有正确性入口。正式性能模板、参数化算子和多核算子均采用新文件，
不覆盖已经验收的单核真机文件。

## 2. 融合后的总体阶段

| 阶段 | 目标 | 通过条件 |
| --- | --- | --- |
| P8 | 建立可信的单核计时框架 | 正确性前后检查通过，计时不含编译、分配和 H2D/D2H，原始样本可复核 |
| P9 | 参数化形状，打通多核运行时并完成多核 ChunkScan 正确性 | 非平凡形状及 1/2/4/8 核全部通过 CPU oracle、覆盖和确定性检查 |
| P10 | 测量流水线与多核组合性能，完成调优、追踪和收口 | 性能矩阵稳定，最优合法配置复测通过，结论边界完整归档 |

严格按 P8 → P9 → P10 推进。下文仍把 P9 和 P10 内部的能力探针、
正确性、性能与调优检查写开，作为停止门而不是额外的管理阶段。不能在多核正确性
通过之前运行多核性能矩阵，也不能用 smoke 形状的单次时间宣称加速。

## 3. P8：可信的单核性能测量

### P8.1 新增独立性能入口

新增下列文件，不修改现有正确性入口：

- `main_template_device_bench.cpp`：只负责已编译 kernel 的重复计时；
- `test_chunk_scan_device_benchmark_p8.py`：编译候选、执行正确性门、启动
  benchmark 并汇总 JSON；

汇总逻辑直接包含在 P8 Python 入口中，不再增加只负责转抄结果的脚本。

计时前只执行一次设备初始化、模块加载、设备内存分配和输入 H2D；
计时后才执行输出 D2H 和资源释放。计时区间内不得打印每次迭代日志，
也不得重新编译、重新加载模块、重新分配内存或重新复制输入。

### P8.2 本阶段的时间口径

P8 正式报告只使用一个口径：

**同步调用延迟**：用 `std::chrono::steady_clock` 包住一次现有 kernel
wrapper。该 wrapper 内部已经执行 stream synchronize，结果包含 host launch、
设备执行和同步开销，但不包含编译、模块加载、内存分配或 H2D/D2H。

P8 会记录 `libtpuv7_rt.so` 是否导出 event API，但 event/async 批量时间不作为
本阶段的停止门。若 P10 需要分离 host launch 与设备时间，再新增专用 async
wrapper；不得用当前同步 wrapper 冒充纯设备时间。

R0 小形状的结果可能被 host launch 与同步开销主导。launch-floor 探针并入
P10 调优分析；正式主指标始终使用未扣除的实际时间，不把“测量值减空 kernel”
作为对外性能数据。

### P8.3 采样协议

- 每个候选先执行不少于 100 次 warm-up；若频率或温度仍明显漂移，继续预热；
- 每个候选每轮连续测量至少 5 秒，与 PipeThreader 论文的最低执行时长一致；
- 至少进行 7 个独立轮次，轮换或随机化 S1/S2/S3/P6 的测试顺序；
- 保存每次原始样本，并输出 `min`、`p50`、`p90`、`p95`、`max`、
  `mean`、标准差和变异系数 `CV`；
- 主结论使用各轮 `p50` 的中位数；`min` 只作辅助，不作为主要结论；
- `CV > 5%` 时先排查系统负载和温度再重测；连续重测仍 `> 10%` 的结果标为无效；
- 每轮 benchmark 之前和全部轮次之后各做一次 all-terms 正确性检查。

### P8.4 单核候选对照

主对照必须保持相同的 K16 reduction 分块：

- S1：无流水线标记的串行 K16 基线；
- S2：两阶段手工流水线中间版本；
- S3：论文顺序的手工流水线；
- P6：编译器显式 sProg-B 排布。

S0 是完整 K64 串行实现，分块策略不同，只作为补充工程基线，不能替代
S1 与 S3/P6 的同构对照。P6 与 S3 若生成同一关键排布，性能应接近；
若差异显著，必须先比较生成源码和二进制哈希，再分析性能。

单核流水线收益定义为：

```text
pipeline_speedup(stage) = median_latency(S1) / median_latency(stage)
```

### P8.5 P8 验收产物

每次运行写入独立时间戳目录，至少包含：

- `environment.json`：主仓库、TVM、PPL、runtime、驱动、设备 ID 和文件哈希；
- `compile_manifest.json`：生成源码哈希、ELF 架构、流水线标记和 LMEM 上界；
- `correctness_before.json` 与 `correctness_after.json`；
- `samples.jsonl`：不得只保留汇总值；
- `summary.json` 与 `summary.csv`；
- 完整 stdout/stderr 和退出码。

P8 只在固定 smoke 形状上验证计时框架和候选相对关系。由于该形状工作量小，
P8 结果标为 `preliminary`，不能作为项目最终性能数字。

## 4. P9：参数化形状与有意义的单核负载

### P9.1 参数化目标

将当前源码中的固定常量改成 kernel factory 参数：

`B, S, L, Ck, G, H, P, N, reduce_tile, num_stages, core_num`。

第一版仍冻结以下条件，以限制修改面积：

- FP16 输入/输出、FP32 中间累加；
- `L=64, P=64, N=128, G=1`；
- `S=Ck*L`，无 tail chunk；
- compact contiguous ABI；
- K16 reduction 和现有 P6 排布。

先推广 `B`、`Ck` 和 `H`，不要同时引入 `L=256`、非连续 stride、tail、
多 dtype 或完整 Mamba2 combined kernel。

### P9.2 首批形状矩阵

| ID | 形状 | 用途 |
| --- | --- | --- |
| R0 | `B=1,S=128,L=64,H=1,P=64,N=128` | 现有回归与 launch-floor 对照 |
| R1 | `B=1,S=1024,L=64,H=8,P=64,N=128` | 多任务和多核映射的最小压力形状 |
| R2 | `B=1,S=4096,L=64,H=8,P=64,N=128` | 单 batch 延迟形状 |
| R3 | `B=8,S=2048,L=64,H=8,P=64,N=128` | 吞吐与带宽压力形状 |

R1–R3 是本项目为 TPU 设计的测试形状，不冒充论文的完整配置。
论文评测覆盖 batch 1/64、sequence 1k–16k；上游 TileLang 另有
`B=8,S=2048/4096,L=256` 配置。只有后续真正支持相同完整维度和 ABI 后，
才可使用“论文形状”或“上游形状”的名称。

### P9.3 正确性门

每个形状至少验证 state-only、scan-only、residual-only、all-terms、
negative-D 和 causal-poison 六个用例。必须同时检查：

- `atol=rtol=1e-2`；
- NaN/Inf 为零；
- causal-poison 与 clean scan 逐位一致；
- 每个 batch、chunk、head 的参考输出均有非零信号；
- 输出 sentinel 全部被覆盖；
- 生成代码没有越界物理视图，LMEM 低于硬件上限。

P9-A 通过后，用 P8 的协议测量 R1–R3 单核 S1/S2/S3/P6，形成可信的
单核性能基线，然后直接进入同一管理阶段的 P9-B。

## 5. P9-B：BM1690 多核运行时与编译器打通

### P9-B.1 不能采用的做法

不能仅把 `T.Kernel(1, 1, is_cpu=True)` 改成 8。当前 TPU 路径会把
CPU kernel frame 的 block 维度降低为设备函数内的普通循环，而不会自动
生成 `tpu_workitem_index()`。这样可能仍由一个物理核串行执行八份工作。

也不能只把 host 的 `block_num` 改成 8 而让每个核执行当前完整算子；
这会让八个核写同一输出并重复全部计算。

### P9-B.2 work-item 接口

为 TPU DSL 和 PPL codegen 增加最小接口：

- `T.ppl_workitem_index()` → 设备端 `tpu_workitem_index()`；
- `T.ppl_workitem_num()` → 设备端 `tpu_workitem_num()`；
- 必要时增加 `T.ppl_physical_core_id()` 仅用于诊断，不参与索引正确性。

host wrapper 由编译元数据或明确参数接收 `core_num`，设置：

```text
group_num = 1
block_num = core_num
argument_struct_count = core_num
```

每个 argument struct 可以相同，由设备端 work-item index 决定任务；
`sizeof(apis)` 必须与 `core_num` 一致。第一版只允许
`core_num in {1,2,4,8}`，超过 BM1690 已验证核数时在 host 侧拒绝。

### P9-B.3 独立多核探针

在接入 ChunkScan 前先实现一个极小探针：每个 work-item 将自己的
`workitem_index`、`workitem_num` 和可选 physical core ID 写到独立槽位。
分别用 1、2、4、8 核启动，验收条件为：

- 返回索引恰好覆盖 `[0, core_num)`，无重复、无缺失；
- 每个 work-item 看到相同的 `workitem_num=core_num`；
- 输出槽位之外的 sentinel 未改变；
- 连续运行 20 次均一致；
- 生成源码确实包含 work-item 查询，host 源码确实使用对应 block 数。

P9-B 没有通过时不得修改 ChunkScan 的任务划分。

## 6. P9-C：多核 ChunkScan 正确性实现

### P9-C.1 任务划分

第一版按彼此独立的 `(batch, chunk, head)` 划分，每个任务负责完整的
`[L,P]` 输出 tile：

```text
task_count = B * Ck * H
for task = workitem_index; task < task_count; task += workitem_num:
    b = task // (Ck * H)
    rem = task % (Ck * H)
    c = rem // H
    h = rem % H
    g = h // (H / G)
    compute output[b, c, h, :, :]
```

该划分不跨核归约，不需要核间 barrier：`cb` 和 `C` 可能被多个 head
只读共享，`prev_states` 和 `D` 按 head 读取，输出区域完全不相交。
这比在一个 `[L,P]` tile 内拆 GEMM reduction 更适合作为首个多核版本。

### P9-C.2 实现顺序

1. 先实现多核 S1，保持无流水线标记，排除流水线因素；
2. S1 在 1/2/4/8 核全部正确后迁移 S3；
3. 最后迁移 P6，并验证编译器排布仍与单核版本结构等价；
4. S2 作为诊断候选，S0 只保留回归，不阻塞主路线。

### P9-C.3 多核正确性矩阵

至少覆盖 R1、R2 和一个非整除任务数的专用形状。对每个候选执行：

- `core_num=1,2,4,8` 对 CPU oracle；
- 多核输出对同源码单核输出；
- 六个既有语义用例；
- 输出 sentinel 覆盖检查；
- 连续 20 次确定性检查；
- 任务数小于核数以及任务数不能被核数整除的边界检查；
- AddressSanitizer 不适用于设备端时，使用 guard 区和 canary 检测越界写。

只有全部 core count 正确后才能声明“BM1690 八核实现完成”。

## 7. P10-A：多核性能与流水线组合实验

### P10-A.1 核心实验矩阵

主矩阵使用 R1–R3：

```text
stage ∈ {S1, S3, P6}
core_num ∈ {1, 2, 4, 8}
shape ∈ {R1, R2, R3}
```

S2 作为补充；R0 只验证 launch floor，不进入主加速比图。每个点严格复用
P8 的 warm-up、至少 5 秒、7 轮、原始样本和正确性前后门。

### P10-A.2 报告指标

```text
pipeline_gain(c) = latency(S1, c) / latency(P6, c)
multicore_speedup(c) = latency(P6, 1) / latency(P6, c)
parallel_efficiency(c) = multicore_speedup(c) / c
combined_gain(c) = latency(S1, 1) / latency(P6, c)
output_throughput = B * S * H * P / latency
```

报告绝对延迟和置信范围，不能只报 speedup。若八核没有线性提升，分别检查：

- task 数不足或负载不均；
- host launch/sync 占比；
- 多核争用全局内存带宽；
- 多个 head 重复读取共享的 `cb/C`；
- P6 流水线改变 LMEM 占用或阻塞并发；
- 温度、频率或同机其他任务干扰。

不得为了得到更好的数字只选择最快一次运行，也不得把不同形状、不同
correctness contract 或不同数据类型的时间直接相除。

## 8. P10-B：联合调优、硬件追踪与收口

### P10-B.1 合法调优顺序

在 P10-A 基线完成后，每次只改一个轴：

1. reduction tile：在 LMEM 和依赖检查允许时比较 K8/K16/K32；
2. pipeline stage：比较 2/3，保留 P5/P6 的依赖和 LMEM 拒绝门；
3. 每核任务粒度：一个 work-item 连续处理的 `(b,c,h)` 数量；
4. `L/P/N` tile：最后再扩展，避免与多核问题同时调试。

每个候选必须先过静态结构、LMEM、单核正确性和八核正确性，再进入性能测量。
调优目标是同一形状的稳定 `p50`，不能用 cmodel 时间筛选。

### P10-B.2 物理重叠证据

先盘点目标服务器是否存在可用的 BM1690 trace/profiler。若能获得每个 kernel
的 GDMA/BDC 时间线，则归档 S1 与 P6 的同形状 trace，并检查命令重叠区间。
若无法获得硬件 trace，则最终结论只能写成：

- P6 保留指定流水线排布；
- P6 在真实 BM1690 上取得某个经过稳定测量的延迟变化；
- 尚未直接证明物理 GDMA/BDC 重叠。

性能提升本身不能替代硬件时间线证据。

### P10-B.3 最终交付

- 单核与 1/2/4/8 核全部 correctness JSON；
- 原始 benchmark 样本、环境清单、生成源码和二进制哈希；
- S1/S3/P6 的绝对延迟、流水线收益、多核加速比和效率图表；
- 最优配置及所有被拒绝配置的原因；
- 是否具有硬件 trace 的明确说明；
- BM1690 八核结论与 BM1690e 四核后续路线严格分开。

BM1690e 后续可以复用参数化算子、work-item 映射和 benchmark 框架，
但必须把允许核数限制为 4，并在独立仓库重新完成 P9–P10，不能直接继承
BM1690 八核的正确性或性能结论。

## 9. 推荐的下一步

下一步只执行 P8，不同时修改多核代码。先建立独立性能模板、runtime event
能力探针、重复测量脚本和 JSON 产物；确认 S1/S2/S3/P6 在当前单核 smoke
形状上能够稳定复测后，再进入 P9 的形状参数化。

## 10. 计划依据

- [PipeThreader OSDI 2025 论文](https://www.usenix.org/system/files/osdi25-cheng.pdf)：
  Figure 6 给出 ChunkScan reduction 轴分块与 sProgram，Section 6.1 规定先
  warm-up、随后每个 workload 至少重复执行 5 秒，Section 6.4 说明 tiling 与
  pipeline scheduling 需要联合评估。
- 本仓库 `configs.py` 固定记录了上游 TileLang v0.1.5 的 ChunkScan 默认形状、
  测试形状和调优轴；P9 的 R1–R3 明确是 TPU 项目形状，不冒充上游原配置。
- PPL v1.4.195 的 `test_add_pipeline` 多核样例使用
  `block_num=8` 与设备端 `tpu_workitem_index()`，因此 P9-B 先复现该最小机制，
  再将其接入 ChunkScan。
