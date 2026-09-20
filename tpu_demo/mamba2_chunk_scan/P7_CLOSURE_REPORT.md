# P7 cmodel 单入口验收记录

验收时间：2026-09-20。运行编号：`20260920T065556Z-4383`。

## 结论

P7 **通过当前工作树上的 cmodel-only 验收**。唯一顶层入口
`./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --p7` 顺序复跑 A0-P6，
`artifacts/p7/result.json` 为 `PASS`；14 个步骤均为 `PASS`、退出码均为
0，14 份逐项日志均存在且无未预期的失败/异常。A0 CPU oracle 为
`14 passed`；P6 规划器单元测试为 `7 passed`。

P1 五项原语、P2 S0、P3 S1、P4 S2/S3、P5 矩阵和 P6 显式候选的 11 份
`result.json` 均在此次运行期间重新写入，并非仅沿用旧 `PASS`。
每一阶段必需的 TIR、原始目标代码、cmodel 执行代码、manifest 和适用的
LMEM 度量均存在且非空。P7 索引中的 114 份文本证据 SHA-256 与当前
文件一致；31 份源码 SHA-256 也与当前文件一致。

P5 结果是 3 个接受、3 个带原因的拒绝；不能将拒绝候选计为实现成功。
P6 显式 `order/stage` 已被规划器消费，三个负向编译门均按预期拒绝；
其原始代码保留流水线标记，cmodel 代码只移除了这些标记。P6 静态
LMEM 上界为 `69632 / 262144` 字节。P6 六类数值情况均与手工 S3
逐位相同；`all_terms` 对 CPU oracle 的最大绝对误差是
`6.103515625e-05`，因果上三角污染不改变输出。

## 证据入口

- 汇总状态、范围声明、工具链身份、P5/P6 摘要与哈希索引：
  `artifacts/p7/result.json`。
- 本轮 14 份日志：
  `artifacts/p7/runs/20260920T065556Z-4383/`。
- P6 数值、结构、负向门及原始/cmodel 代码：
  `artifacts/pipeline_p6_explicit/`。
- P5 六项结果与拒绝原因：
  `artifacts/pipeline_matrix_p5/result.json`。

## 严格范围声明

本项目完成的是**独立 `_chunk_scan_fwd` 算子**的 BM1690 目标
TileLang-TPU 写法、规则驱动的流水线排布，以及 CPU-hosted cmodel
数值和生成代码结构验证。它没有运行于真实 TPU；没有验证真实硬件
同步或 GDMA/BDC 物理重叠；没有可引用的真机延迟、吞吐、加速比或
四核扩展结果；也没有实现完整 `mamba_chunk_scan_combined` 或 Mamba2
模型集成。cmodel 日志中的运行时间不是 TPU 性能数据。

工具链身份记录中 `repository_dirty=true`、`tvm_dirty=true`。因此
本次验收仅保证**当前工作树**能由单入口复跑，不表示这些未提交改动
已经能从 GitHub 在另一服务器完整重建。BM1690E/SG2260E 真机迁移
和跨服务器复现须在新的独立阶段处理。
