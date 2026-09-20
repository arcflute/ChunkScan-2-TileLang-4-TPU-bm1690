# BM1690 真机交付与验证记录

本记录对应独立的 BM1690 仓库
<https://github.com/arcflute/ChunkScan-2-TileLang-4-TPU-bm1690>。
P7 的 14/14 `PASS` 是 CPU cmodel 收口；下述结果是新增的、彼此独立的
BM1690 PCIe 真机正确性验证。两者都不是性能测量。

## 已验证范围

2026-09-20 在 Ubuntu 24.04 x86-64 主机、真实 BM1690 PCIe 设备上，
使用 PPL v1.4.195 的 `libbm1690.a`、真实 `tpuv7-runtime` 1.9.3、
RISC-V 交叉编译器以及 `sg-host-drv`，完成如下固定形状的单核验证：

`B=G=H=1, S=128, Ck=2, L=64, P=64, N=128`，输入/输出 FP16，
主要中间累加为 FP32。设备内核 `libkernel.so` 是 RISC-V ELF，主机
`main.so` 是 x86-64 ELF；后者链接到真实的 `libtpuv7_rt.so`，
没有链接 cmodel 模拟器库。

| 实现 | 真机用例 | 原始设备源码的流水线标记 | 结果 |
| --- | ---: | ---: | --- |
| S0 单核串行 | 2 | 0 | `PASS` |
| S1 K16 串行归约 | 6 | 0 | `PASS` |
| S2 双阶段流水线 | 6 | 1 对 | `PASS` |
| S3 论文顺序流水线 | 6 | 1 对 | `PASS` |
| P6 编译器显式 sProg-B 排布 | 6 | 1 对 | `PASS` |

S1、S2、S3、P6 的六个用例均覆盖 residual-only、state-only、
scan-only、all-terms、负 `D` residual，以及因果上三角污染。
所有用例对 CPU oracle 的 `atol=rtol=1e-2` 比较均通过，
无 NaN/Inf；污染用例与干净 scan 输出逐位相同。
四版 all-terms 的最大绝对误差均为 `6.103515625e-05`。
S0 的 residual-only 完全一致，all-terms 最大绝对误差同为
`6.103515625e-05`。控制台最终为 `hardware_batch_exit=0`。

验收使用的远端源码提交是 `5659001ea22f9993b69e090bf5345d43dea00e9a`，
相对于仓库基准 `86add2f2ddfaa58783bd0fb7dd7c2ddbe1395a25`
交付的邮件补丁 SHA-256 为
`c17fe60137b2d3d8046cd79b1287577a5428a93489cfda9f2f4c9729771f916a`。
四份真机 `result.json` 的状态、六个用例、源码 SHA-256 和流水线标记
已在远端文件级复核；生成的二进制、日志和 JSON 不随 Git 源码提交。

## 可复现的源码和环境边界

- `3rdparty/tvm` 固定为
  `a8a54d2b1f43c23a47f2fc08779654918eae6464`。
  干净克隆需要先初始化子模块，再检查并应用
  [`patches/tvm.patch`](../../patches/tvm.patch)；补丁 SHA-256 为
  `a0b2afc1b4adc4f25a9241c5bc2795babdf4beec44ed7569823454b2c68b4d72`。
  已打补丁的子模块不可再次正向应用。
- [`tilelang/jit/adapter/libgen.py`](../../tilelang/jit/adapter/libgen.py)
  的 PCIe 分支使用 `CHUNKSCAN_DEVICE_RUNTIME_ROOT` 与
  `CHUNKSCAN_RISCV_TOOLCHAIN_ROOT`，不再引用 emulator 头文件、库或 rpath。
  cmodel 编译分支保持独立。
- [`main_template_device.cpp`](main_template_device.cpp) 是真机专用模板，
  由 `CHUNKSCAN_DEVICE_ID` 选择设备；原有 cmodel 模板未改动。
  首次验证每个用例仅启动一次 kernel，不执行自动预热或测量循环。
- PPL SDK、交叉编译器、真实 runtime、驱动、`.venv/`、`build/` 和
  `artifacts/` 均是运行环境或生成物，不应提交到公开仓库。
  [`.env.bm1690.example`](.env.bm1690.example) 仅为路径配置模板。

在已按仓库说明构建 TileLang/TVM、准备 CPU PyTorch 环境并设置
`PPL_PROJECT_ROOT`、`CHUNKSCAN_DEVICE_RUNTIME_ROOT`、
`CHUNKSCAN_RISCV_TOOLCHAIN_ROOT`、`CHUNKSCAN_DEVICE_ID`、
`PYTHONPATH`、`TVM_LIBRARY_PATH`、真实 runtime 的 `LD_LIBRARY_PATH`
后，先不带 `--run` 编译，再带 `--run` 验证：

```bash
python tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_s0.py
python tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_s0.py --run

bash -c 'set -e; for stage in s1 s2 s3 p6; do
  python tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_pipeline.py "$stage"
done'
bash -c 'set -e; for stage in s1 s2 s3 p6; do
  python tpu_demo/mamba2_chunk_scan/test_chunk_scan_device_pipeline.py "$stage" --run
done'
```

真机结果写在 `artifacts/device_pipeline/<stage>/result.json`；
`compile_manifest.json` 记录真实 runtime 路径、设备源码 SHA-256、
ELF 架构与流水线标记数。运行时仍应逐阶段失败即停，
不要把 cmodel 的 `run_cmodel.sh` 当成真机入口。

## 尚未验证和不得推断的结论

这次只证明上述单核、固定形状在真实 BM1690 上正确运行，
以及 PCIe 编译的原始设备源码保留流水线标记。
它不证明物理 GDMA/BDC 重叠、吞吐量或加速比，也不覆盖其他形状、
八核扩展、BM1690e/SG2260e、完整 Mamba2 网络或端到端推理。
控制台的单次耗时包含当前测试封装因素，不能作为性能结论。
下一阶段 BM1690e 应在独立路线验证，不覆盖本记录。
