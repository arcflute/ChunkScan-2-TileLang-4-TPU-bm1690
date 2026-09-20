# BM1690 真机仓库交付清单（准备中）

目标仓库：<https://github.com/arcflute/ChunkScan-2-TileLang-4-TPU-bm1690>。
2026-09-20 核查时，该 GitHub 仓库为空。此文件是交付准备清单，**不是**
BM1690 真机已可运行的声明；不要把 P7 的 cmodel `PASS` 标为真机 `PASS`。

## 已具备与尚缺的条件

- 当前项目 `exp/pipethreader-chunkscan-bm1690` 工作树的 P7 cmodel
  结果已通过；算子是独立 `_chunk_scan_fwd`，固定 smoke 形状、单核。
- ChunkScan 的 Python/脚本/文档文件已加入暂存区，但尚未提交；普通
  `git push` 不会上传仅暂存、未提交的文件。
- `3rdparty/tvm` 指向提交 `a8a54d2b1f43c23a47f2fc08779654918eae6464`，
  但子模块内三个源文件还有本地修改。只提交主仓库不会带走它们。
  已更新并暂存 `patches/tvm.patch`，覆盖这三个文件；当前补丁 SHA-256
  为 `a0b2afc1b4adc4f25a9241c5bc2795babdf4beec44ed7569823454b2c68b4d72`。
  子模块源码的 `git diff --check` 和补丁反向应用检查均已通过。
- `tilelang/jit/adapter/libgen.py` 把芯片写死为 `bm1690`；现有 `pcie`
  分支仍引用 emulator 路径与 `cdm_daemon_emulator`，因此尚不是安全的
  真机编译/链接路径。`run_cmodel.sh` 仍应保持 cmodel 专用。
- 本机 PPL 包位于 `/root/autodl-tmp/ppl_v1.4.195-geb2acdd0-20250220`；
  它不属于 Git 仓库。目标服务器是否有兼容的驱动、PPL、真实设备
  runtime 和交叉编译器仍需核实，不能从“GitHub 仓库为空”推断。

## 推荐的仓库内容

保留构建所需的 TileLang 源码、仓库元数据与子模块声明；提交
`tpu_demo/mamba2_chunk_scan/` 下的算子、测试、P7 汇总程序、运行脚本
和 Markdown 文档，以及 `src/`、`tilelang/language/` 和编译器测试中的
ChunkScan/TPU 改动。为三个 TVM 子模块源文件的改动选择**一种**可复现
交付方式：

1. 推荐：更新已有 `patches/tvm.patch`，使它精确表示相对于已钉住
   TVM 提交的全部三个本地文件改动；干净克隆在
   `git submodule update --init --recursive` 后执行 `git apply --check`
   再应用。补丁 SHA-256 与基准 TVM 提交写入部署清单。
2. 或者：将改动提交到可被私域服务器访问的 TVM fork，主仓库更新
   子模块 URL 和 gitlink。没有可访问的子模块提交时，不得使用此方案。

不要上传本机 `build/`、`.venv/`、PPL release/SDK、设备库、`*.so`、
`*.o`、运行时缓存、`.pt` 输出、P7 逐项日志或含令牌/私域地址的真实
环境文件。P7 的项目报告可以提交，但生成的 `artifacts/` 应在目标
环境重新生成；原机证据如需长期保存，单独封存并注明来源。

## 建议配置接口

已新增真机专用的 `.env.bm1690.example`（仅占位值）；仍需新增
`run_device.sh`，至少配置 PPL 根目录、真实设备 runtime 根目录、
RISC-V 交叉编译器目录、Python 解释器、设备编号与芯片名。真实
`.env.bm1690` 应被忽略，不进入公开 GitHub。硬件入口不得继承
`run_cmodel.sh` 的 emulator `LD_LIBRARY_PATH`，也不得删除 P6 原始
源码里的 `tpu_parallel_start/end` 后再宣称完成流水线真机验证。

私域服务器上的单入口应依次执行：环境/设备预检、最小设备算子、
P1 原语、S0、S1、S3/P6。首次真机验证先用单核；8 核分配是后续
单独阶段。每项输出独立日志，失败即停，归档提交号、芯片/驱动/
SDK 版本、编译命令、生成源码、退出码与数值差异，便于离线带回诊断。

## 上传前的停止门

1. 真机编译路径不再引用 emulator 头文件、库或 rpath；真实 runtime
   与 SDK 的 ABI 已从目标服务器确认。
2. 所有拟交付源码均已跟踪；TVM 修改可在干净克隆中重建；构建产物
   和秘密文件未进入暂存区。
3. 保留 P7 cmodel 回归通过，但不把它当作真机结果。
4. 从新仓库的干净克隆可以初始化子模块、应用补丁、完成静态构建。
   没有硬件时不能宣称运行已通过。

在完成上述停止门前，建议仅在目标 GitHub 仓库标记“bring-up
candidate”，不要发布“BM1690 hardware validated”标签。

## 提交与私域交付顺序

1. 已整理 TVM 差异并更新 `patches/tvm.patch`；补丁包含
   `target_kind.cc`、`block_access_region_detector.cc`、
   `storage_rewrite.cc` 三个文件，且在当前已打补丁子模块上通过
   `git apply --reverse --check`。不要在当前子模块上正向再应用一次。
2. 已忽略生成的 `artifacts/`、真实 `.env.bm1690` 和 build/runtime
   二进制；源码、补丁和文档已暂存，仍需复核并提交。
3. TVM 补丁更新后已重新运行 P7，14 项均为 `PASS`。它仍是 cmodel
   证据，不是 BM1690 真机证据。
4. 为新 GitHub 仓库新增独立 remote，确认 URL 后推送；不要改用
   当前指向 `xwhzz/tilelang-tpu` 的 `origin` 做盲目推送。
5. 私域服务器干净克隆新仓库，初始化子模块，检查并应用补丁。
   然后盘点设备驱动、PPL、真实 runtime、交叉编译器，再做真机
   编译和分阶段测试。缺失任何 SDK/设备信息时，在这里停止而不是
   继续猜测编译或链接参数。
