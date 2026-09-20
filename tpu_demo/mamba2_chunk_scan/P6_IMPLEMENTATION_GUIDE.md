# P6 手工实现指南：显式 sProgram 契约与编译器流水线生成

状态：已于 2026-09-14 完成独立验收并关闭。全部 7 个编译器单元测试通过；
P6 ChunkScan cmodel、负向编译器门以及 P4.1、P4.2、P5 保护性回归均为
`PASS`。P7 尚未启动。

## 1. 本阶段到底实现什么

P4.2 已经用 TileLang 源语句顺序手工表达论文 Figure 6 的 sProg-B，现有
`PipelinePlanning` 再根据读写关系推导流水线。P5 进一步证明：即使 Python
源码请求 sProg-A 的 LoadX-first，当前规划器仍会重新生成 LoadY-first。

P6 修复这里缺少的编译器契约：

1. `T.Pipelined` 已经能够携带 `order` 与 `stage`，但非 SM90 的通用
   `PipelinePlanning` 尚未消费它们；
2. P6 让规划器在 `order` 和 `stage` 同时存在时保留这份显式 sProgram；
3. 编译器检查契约是否成对出现、长度是否正确、order 是否为排列、stage
   是否合法以及循环是否具有非空 steady state；
4. 现有 `InjectSoftwarePipeline` 继续检查数据依赖，并自动生成 buffer
   versioning、prologue、steady state 和 epilogue；
5. 现有 `AddressAssign` 继续承担 BM1690 LMEM 溢出拒绝；
6. 新 P6 ChunkScan 只给已关闭的 S3 附加显式契约，不修改 S3 的 ABI、数学、
   tile 大小或算子语句；
7. P6 只证明规则驱动的编译器生成，不实现论文中依赖真实硬件 profiler 的
   性能搜索。

论文中的 sProgram 是按执行单元与顺序组织的二维任务表示，依赖通过等待/
barrier 约束。本文在 TPU 路径上的对应关系是：显式 `order/stage` 表示已选
sProgram，TIR 数据依赖检查承担 Wait 的合法性，生成的
`tpu_parallel_start/end` 表示一个可审计的流水区域。`sync` 与 `group` 在
本候选中均为空，因为当前 BM1690 PPL 路径没有可供本实验验证的异步事件
接口；不得把空值描述成真实硬件同步已经验证。

## 2. 变更边界

本阶段只允许修改或新增：

- 修改 `src/transform/pipeline_planning.cc`；
- 修改 `testing/python/transform/test_tilelang_transform_pipeline_planning.py`；
- 新增 `tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p6.py`；
- 新增 `tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_p6_cmodel.py`；
- 修改 `tpu_demo/mamba2_chunk_scan/run_cmodel.sh`。

以下内容不得修改：

- W8A16 路线及其快照；
- `chunk_scan_serial.py`、`chunk_scan_reduction_serial.py`；
- `chunk_scan_pipeline_s2.py`、`chunk_scan_pipeline_s3.py`；
- `chunk_scan_pipeline_p5.py` 及 P2-P5 测试；
- `src/transform/inject_pipeline.cc`、`src/transform/address_assign.cc`；
- TileLang/TVM/PPL 的既有 ABI 和 codegen。

## 3. 第零步：保存状态并确认前置证据

从仓库根目录执行：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

git status --short
git diff --check

python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
for name in (
    "pipeline_s2/result.json",
    "pipeline_s3/result.json",
    "pipeline_matrix_p5/result.json",
):
    path = root / name
    result = json.loads(path.read_text(encoding="utf-8"))
    print(name, result["status"])
    assert result["status"] == "PASS"
PY

cp src/transform/pipeline_planning.cc \
  /root/autodl-tmp/pipeline_planning.cc.before-p6
cp build/libtilelang_module.so \
  /root/autodl-tmp/libtilelang_module.so.before-p6
```

三个状态必须均为 `PASS`。两个备份位于仓库外，不会污染路线；在 P6 完成前
不要删除。

## 4. 第一步：让 PipelinePlanning 消费显式契约

编辑 `src/transform/pipeline_planning.cc`。

### 4.1 增加标准库头文件

修改前：

```cpp
#include <tvm/tir/transform.h>

#include "../target/utils.h"
```

修改后：

```cpp
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../target/utils.h"
```

### 4.2 替换规划器中“分析依赖并生成 order/stage”的整个区段

定位从下面注释开始：

```cpp
    // analysis use-def chain
```

一直选择到 `copy_stage_at_end` 调整块末尾，即下面代码之后：

```cpp
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      ...
    }
```

将这一整个区段替换为：

```cpp
    auto explicit_order_anno =
        loop->annotations.Get("tl_pipeline_order");
    auto explicit_stage_anno =
        loop->annotations.Get("tl_pipeline_stage");
    bool has_explicit_order = explicit_order_anno.defined();
    bool has_explicit_stage = explicit_stage_anno.defined();
    CHECK_EQ(has_explicit_order, has_explicit_stage)
        << "ValueError: explicit pipeline schedule requires both order and "
           "stage";
    bool has_explicit_schedule =
        has_explicit_order && has_explicit_stage;

    if (has_explicit_schedule) {
      auto explicit_orders =
          Downcast<Array<Integer>>(explicit_order_anno);
      auto explicit_stages =
          Downcast<Array<Integer>>(explicit_stage_anno);
      CHECK_EQ(explicit_orders.size(), pipeline_stage_infos.size())
          << "ValueError: explicit pipeline order size must match the "
             "lowered pipeline body size";
      CHECK_EQ(explicit_stages.size(), pipeline_stage_infos.size())
          << "ValueError: explicit pipeline stage size must match the "
             "lowered pipeline body size";

      if (const auto *extent = loop->extent.as<IntImmNode>()) {
        CHECK_GT(extent->value, num_stages)
            << "ValueError: explicit pipeline schedule requires reduction "
               "iterations > num_stages to form a nonempty steady state";
      }

      std::unordered_set<int> seen_orders;
      bool has_producer_stage = false;
      bool has_consumer_stage = false;
      for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
        int order = static_cast<int>(explicit_orders[i]->value);
        int stage = static_cast<int>(explicit_stages[i]->value);
        CHECK_GE(order, 0)
            << "ValueError: explicit pipeline order must be non-negative";
        CHECK_LT(order, static_cast<int>(pipeline_stage_infos.size()))
            << "ValueError: explicit pipeline order must be a permutation "
               "of [0, body_size)";
        CHECK(seen_orders.insert(order).second)
            << "ValueError: explicit pipeline order contains a duplicate: "
            << order;
        CHECK_GE(stage, 0)
            << "ValueError: explicit pipeline stage must be non-negative";
        CHECK_LE(stage, num_stages)
            << "ValueError: explicit pipeline stage exceeds num_stages";
        has_producer_stage = has_producer_stage || stage == 0;
        has_consumer_stage = has_consumer_stage || stage == num_stages;
        pipeline_stage_infos[i].order = order;
        pipeline_stage_infos[i].stage = stage;
      }
      CHECK(has_producer_stage)
          << "ValueError: explicit pipeline schedule requires stage 0";
      CHECK(has_consumer_stage)
          << "ValueError: explicit pipeline schedule requires the terminal "
             "num_stages stage";
    } else {
      // Analyze use-def chains for the existing inferred-schedule path.
      for (auto &pinfo : pipeline_stage_infos) {
        for (int i = pinfo.original_order + 1;
             i < static_cast<int>(pipeline_body_seq->size()); i++) {
          if (!pinfo.copy_stage)
            continue;
          for (const BufferRegion &read : pipeline_stage_infos[i].reads) {
            if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                             [&](const BufferRegion &r) {
                               return r->buffer == read->buffer &&
                                      MayConflict(r->region, read->region);
                             }) != pinfo.writes.end()) {
              pinfo.last_use_stage = std::max(pinfo.last_use_stage, i);
            }
          }
          for (const BufferRegion &write : pipeline_stage_infos[i].writes) {
            if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                             [&](const BufferRegion &r) {
                               return r->buffer == write->buffer &&
                                      MayConflict(r->region, write->region);
                             }) != pinfo.writes.end()) {
              LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                            "overlapping buffer regions detected. "
                         << "Stage " << pinfo.original_order << " and stage "
                         << i << " are both writing to buffer '"
                         << write->buffer->name
                         << "' with overlapping regions. This is not "
                            "supported in pipeline planning.";
            }
          }
        }
      }

      // Preserve the original inferred scheduling behavior byte-for-byte.
      int order_idx = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.copy_stage && pinfo.last_use_stage != -1)
          continue;
        pinfo.order = order_idx++;
        pinfo.stage = num_stages;
        for (auto &pinfo_1 : pipeline_stage_infos) {
          if (pinfo_1.copy_stage &&
              pinfo_1.last_use_stage == pinfo.original_order) {
            pinfo_1.order = order_idx++;
            pinfo_1.stage = 0;
          }
        }
      }
      ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
          << "The number of stages should be equal to the number of pipeline "
             "stages. "
          << "Got " << order_idx << " stages and "
          << pipeline_stage_infos.size() << " pipeline stages.";

      // If all copies are at the end, move them to the beginning and shrink
      // the non-copy stage offset by one.  This is the pre-P6 behavior.
      int copy_stage_at_end = [&]() {
        int copy_stage_cnt = 0;
        int copy_order_min = pipeline_stage_infos.size();
        int non_copy_order_max = 0;
        for (auto &pinfo : pipeline_stage_infos) {
          if (pinfo.copy_stage) {
            copy_stage_cnt++;
            copy_order_min = std::min(copy_order_min, pinfo.order);
          } else {
            non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
          }
        }
        if (copy_order_min > non_copy_order_max)
          return copy_stage_cnt;
        return -1;
      }();
      if (copy_stage_at_end > 0 && num_stages >= 2) {
        for (auto &pinfo : pipeline_stage_infos) {
          pinfo.order =
              (pinfo.order + copy_stage_at_end) %
              pipeline_stage_infos.size();
          if (!pinfo.copy_stage)
            pinfo.stage--;
        }
      }
    }
```

注意：`else` 内的主体来自原文件，目的是保证所有没有显式契约的既有算子走
原路径。不要顺手修改其中的推导算法。

### 4.3 消费前端属性并留下可审计标记

在同一函数稍后的注释复制循环中，修改前是：

```cpp
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }
```

修改后：

```cpp
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage") {
        annotations.Set(key, value);
      }
    }
    if (has_explicit_schedule) {
      annotations.Set("tl_pipeline_explicit_schedule", Integer(1));
    }
```

这里必须删除已消费的两个 `tl_` 属性，再写入标准
`software_pipeline_order/stage`。`tl_pipeline_explicit_schedule=1` 只用于
TIR/manifest 审计，不参与排布。

## 5. 第二步：增加规划器单元测试

编辑 `testing/python/transform/test_tilelang_transform_pipeline_planning.py`。

### 5.1 增加 pytest 导入

修改前：

```python
from tilelang import tvm as tvm
import tilelang as tl
```

修改后：

```python
import pytest

from tilelang import tvm as tvm
import tilelang as tl
```

### 5.2 在文件末尾的 `if __name__ == "__main__":` 之前追加

```python
def _plan_for_tpu(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("tpu"))(mod)
    return tl.transform.PipelinePlanning()(mod)


def _single_planned_loop(mod):
    loops = []

    def visitor(node):
        if (
            isinstance(node, tvm.tir.For)
            and "software_pipeline_order" in node.annotations
        ):
            loops.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    assert len(loops) == 1
    return loops[0]


def test_explicit_pipeline_schedule_is_preserved():
    @T.prim_func
    def before(
        A: T.Tensor((4, 1), "float32"),
        C: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(
                4,
                num_stages=2,
                order=[0, 1],
                stage=[0, 2],
            ):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    loop = _single_planned_loop(_plan_for_tpu(before))
    assert [int(value) for value in loop.annotations[
        "software_pipeline_order"
    ]] == [0, 1]
    assert [int(value) for value in loop.annotations[
        "software_pipeline_stage"
    ]] == [0, 2]
    assert int(loop.annotations["tl_pipeline_explicit_schedule"]) == 1
    assert "tl_pipeline_order" not in loop.annotations
    assert "tl_pipeline_stage" not in loop.annotations
    assert "num_stages" not in loop.annotations


def test_explicit_pipeline_schedule_requires_order_and_stage():
    @T.prim_func
    def before(
        A: T.Tensor((4, 1), "float32"),
        C: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(4, num_stages=2, order=[0, 1]):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    with pytest.raises(
        (tvm.TVMError, ValueError), match="requires both order and stage"
    ):
        _plan_for_tpu(before)


def test_explicit_pipeline_schedule_rejects_duplicate_order():
    @T.prim_func
    def before(
        A: T.Tensor((4, 1), "float32"),
        C: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(
                4,
                num_stages=2,
                order=[0, 0],
                stage=[0, 2],
            ):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    with pytest.raises(
        (tvm.TVMError, ValueError), match="contains a duplicate"
    ):
        _plan_for_tpu(before)


def test_explicit_pipeline_schedule_rejects_invalid_stage():
    @T.prim_func
    def before(
        A: T.Tensor((4, 1), "float32"),
        C: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(
                4,
                num_stages=2,
                order=[0, 1],
                stage=[0, 3],
            ):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    with pytest.raises(
        (tvm.TVMError, ValueError), match="exceeds num_stages"
    ):
        _plan_for_tpu(before)


def test_explicit_pipeline_schedule_rejects_insufficient_loop_depth():
    @T.prim_func
    def before(
        A: T.Tensor((2, 1), "float32"),
        C: T.Tensor((2, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(
                2,
                num_stages=2,
                order=[0, 1],
                stage=[0, 2],
            ):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    with pytest.raises(
        (tvm.TVMError, ValueError), match="nonempty steady state"
    ):
        _plan_for_tpu(before)


def test_explicit_pipeline_schedule_rejects_dependency_violation():
    @T.prim_func
    def before(
        A: T.Tensor((4, 1), "float32"),
        C: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((1, 1), "float32")
            for i in T.Pipelined(
                4,
                num_stages=2,
                order=[1, 0],
                stage=[2, 0],
            ):
                A_shared[0, 0] = A[i, 0]
                C[i, 0] = A_shared[0, 0]

    mod = _plan_for_tpu(before)
    with pytest.raises(
        (tvm.TVMError, ValueError), match="in a later stage"
    ):
        tl.transform.InjectSoftwarePipeline()(mod)
```

这些测试必须同时证明：新显式路径可用，错误契约会失败，且依赖检查仍由
既有注入器执行。原有 `test_simple_pipeline` 仍用于保护未显式指定调度的自动
路径。

## 6. 第三步：新增 P6 ChunkScan 契约包装器

新增 `tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p6.py`。修改前：文件
不存在。修改后完整内容如下：

```python
"""Compiler-directed P6 schedule for BM1690 TileLang-TPU ChunkScan.

The mathematical kernel is the closed P4.2 S3 implementation.  P6 does not
copy or mutate that source; it attaches the designated Figure-6 sProg-B
statement order and stage assignment as an explicit compiler contract.  The
TPU pipeline passes remain responsible for dependency validation, buffer
versioning, and prologue/steady/epilogue generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import tilelang
from tilelang import tvm

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s3 import (
    NUM_STAGES,
    REDUCE_TILES,
    make_chunk_scan_pipeline_s3_kernel,
)


tir = tvm.tir

SCHEDULE_NAME = "figure6_sprog_b_explicit"
LOWERED_STATEMENT_COUNT = 22

# These arrays are the closed P4.2 S3 schedule after FrontendLegalize.  Each
# position describes one lowered statement in the reduction-loop body.
PIPELINE_ORDER: Tuple[int, ...] = (
    2, 0, 1, 3, 5, 4, 6, 7, 8, 9, 10,
    11, 12, 14, 13, 15, 16, 17, 20, 18, 19, 21,
)
PIPELINE_STAGE: Tuple[int, ...] = (
    0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2,
    2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2,
)


@dataclass(frozen=True)
class PipelineScheduleContract:
    name: str
    num_stages: int
    reduction_iterations: int
    lowered_statement_count: int
    order: Tuple[int, ...]
    stage: Tuple[int, ...]
    sync: Tuple[Tuple[int, ...], ...] = ()
    group: Tuple[Tuple[int, ...], ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "num_stages": self.num_stages,
            "reduction_iterations": self.reduction_iterations,
            "lowered_statement_count": self.lowered_statement_count,
            "order": list(self.order),
            "stage": list(self.stage),
            "sync": [list(item) for item in self.sync],
            "group": [list(item) for item in self.group],
            "paper_schedule": "Figure 6 sProg-B: LoadY before LoadX",
            "performance_claim": "NONE: cmodel timing is not TPU performance",
        }


SPROG_B_CONTRACT = PipelineScheduleContract(
    name=SCHEDULE_NAME,
    num_stages=NUM_STAGES,
    reduction_iterations=REDUCE_TILES,
    lowered_statement_count=LOWERED_STATEMENT_COUNT,
    order=PIPELINE_ORDER,
    stage=PIPELINE_STAGE,
)


def attach_explicit_pipeline_schedule(program, contract):
    """Attach one explicit schedule without changing the kernel body."""

    if len(contract.order) != contract.lowered_statement_count:
        raise ValueError("P6 order length does not match its frozen contract")
    if len(contract.stage) != contract.lowered_statement_count:
        raise ValueError("P6 stage length does not match its frozen contract")

    matched_loops = 0

    def postorder(node):
        nonlocal matched_loops
        if not isinstance(node, tir.For) or "num_stages" not in node.annotations:
            return node
        matched_loops += 1
        annotations = dict(node.annotations)
        annotations["tl_pipeline_order"] = tvm.runtime.convert(
            list(contract.order)
        )
        annotations["tl_pipeline_stage"] = tvm.runtime.convert(
            list(contract.stage)
        )
        return tir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            node.body,
            node.thread_binding,
            annotations,
            node.span,
        )

    body = tir.stmt_functor.ir_transform(
        program.body,
        None,
        postorder,
        ["tir.For"],
    )
    if matched_loops != 1:
        raise ValueError(
            f"P6 expected one pipelined loop, found {matched_loops}"
        )
    return program.with_body(body).with_attr(
        "p6_pipeline_schedule", contract.name
    )


def make_chunk_scan_pipeline_p6_kernel():
    """Return the closed S3 math with the explicit P6 compiler contract."""

    return attach_explicit_pipeline_schedule(
        make_chunk_scan_pipeline_s3_kernel(),
        SPROG_B_CONTRACT,
    )


__all__ = [
    "LOWERED_STATEMENT_COUNT",
    "PIPELINE_ORDER",
    "PIPELINE_STAGE",
    "SCHEDULE_NAME",
    "SPROG_B_CONTRACT",
    "attach_explicit_pipeline_schedule",
    "make_chunk_scan_pipeline_p6_kernel",
]
```

这里复用的是 P4.2 S3 的 PrimFunc，而不是在 P6 再维护一份三百余行的数学
副本。附加动作发生在 FrontendLegalize 之前，数组长度则针对其稳定输出的
22 个语句；编译器会在 FrontendLegalize 之后验证长度。

## 7. 第四步：新增完整 P6 编译器生成与 cmodel 测试

新增 `tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_p6_cmodel.py`。
修改前：文件不存在。修改后完整内容如下：

```python
"""Validate P6 explicit-sProgram lowering for BM1690 ChunkScan."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict

import tilelang
import tilelang.language as T
import torch
from tilelang import tvm

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import (
    LOWERED_STATEMENT_COUNT,
    PIPELINE_ORDER,
    PIPELINE_STAGE,
    SPROG_B_CONTRACT,
    make_chunk_scan_pipeline_p6_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_s3 import (
    make_chunk_scan_pipeline_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    comparison_stats,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_cmodel import (
    _make_inputs,
    _source_metrics,
    _variant_inputs,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_pipeline_matrix_p5_cmodel import (
    _pipeline_region_evidence,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_pipeline_s3_cmodel import (
    EXPECTED_DOUBLE_BUFFER_DECLARATIONS,
    EXPECTED_REDUCTION_LOOP,
    EXPECTED_SOURCE_SITE_COUNTS,
    _compile_cmodel,
    _worker_outputs,
)
from tpu_demo.mamba2_chunk_scan.toolchain_identity import (
    REPOSITORY_ROOT,
    assert_chunkscan_toolchain,
)
from tilelang.engine.phase import LowerAndLegalize


tir = tvm.tir
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_p6_explicit"
P6_RUNTIME_DIR = ARTIFACT_DIR / "runtime_p6"
S3_RUNTIME_DIR = ARTIFACT_DIR / "runtime_s3_control"
P5_RESULT_PATH = (
    ROOT / "artifacts" / "pipeline_matrix_p5" / "result.json"
)
S3_RESULT_PATH = ROOT / "artifacts" / "pipeline_s3" / "result.json"
ATOL = 1e-2
RTOL = 1e-2


@T.prim_func
def _insufficient_depth_probe(
    A: T.Tensor((2, 1), "float32"),
    C: T.Tensor((2, 1), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        A_shared = T.alloc_shared((1, 1), "float32")
        for i in T.Pipelined(
            2,
            num_stages=2,
            order=[0, 1],
            stage=[0, 2],
        ):
            A_shared[0, 0] = A[i, 0]
            C[i, 0] = A_shared[0, 0]


@T.prim_func
def _dependency_violation_probe(
    A: T.Tensor((4, 1), "float32"),
    C: T.Tensor((4, 1), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        A_shared = T.alloc_shared((1, 1), "float32")
        for i in T.Pipelined(
            4,
            num_stages=2,
            order=[1, 0],
            stage=[2, 0],
        ):
            A_shared[0, 0] = A[i, 0]
            C[i, 0] = A_shared[0, 0]


@T.prim_func
def _lmem_overflow_probe(
    C: T.Tensor((1088, 4096), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True):
        too_large = T.alloc_shared((1088, 4096), "float32")
        T.ppl_fill(too_large, T.float32(0.0))
        T.ppl_copy(too_large, C[0, 0])


def _load_pass_result(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} prerequisite: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise AssertionError(f"{label} prerequisite is not PASS: {result}")
    return result


def _bound_tpu_module(program):
    func = program.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    return tir.transform.BindTarget(tvm.target.Target("tpu"))(mod)


def _expect_compiler_error(
    gate: str,
    action: Callable[[], object],
    expected_text: str,
) -> Dict[str, object]:
    try:
        action()
    except (tvm.TVMError, ValueError) as error:
        message = str(error)
        if expected_text not in message:
            raise AssertionError(
                f"{gate} failed for the wrong reason: {message}"
            ) from error
        reason_line = next(
            line.strip()
            for line in message.splitlines()
            if expected_text in line
        )
        return {
            "status": "REJECTED_AS_EXPECTED",
            "reason_contains": expected_text,
            "reason_line": reason_line,
        }
    raise AssertionError(f"{gate} was unexpectedly accepted")


def _negative_compiler_gates() -> Dict[str, object]:
    depth_mod = _bound_tpu_module(_insufficient_depth_probe)
    depth = _expect_compiler_error(
        "insufficient_loop_depth",
        lambda: tilelang.transform.PipelinePlanning()(depth_mod),
        "nonempty steady state",
    )

    dependency_mod = tilelang.transform.PipelinePlanning()(
        _bound_tpu_module(_dependency_violation_probe)
    )
    dependency = _expect_compiler_error(
        "dependency_violation",
        lambda: tilelang.transform.InjectSoftwarePipeline()(dependency_mod),
        "in a later stage",
    )

    lmem = _expect_compiler_error(
        "lmem_overflow",
        lambda: tilelang.lower(_lmem_overflow_probe, target="tpu"),
        "BM1690 local memory allocation failed",
    )
    return {
        "insufficient_loop_depth": depth,
        "dependency_violation": dependency,
        "lmem_overflow": lmem,
    }


def _archive_planned_schedule(program) -> Dict[str, object]:
    target = tvm.target.Target("tpu", host="llvm")
    symbol = str(program.attrs["global_symbol"])
    mod = tvm.IRModule({symbol: program})
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)

    loops = []

    def visitor(node):
        if (
            isinstance(node, tir.For)
            and "software_pipeline_order" in node.annotations
        ):
            loops.append(node)

    func = next(iter(mod.functions.values()))
    tir.stmt_functor.post_order_visit(func.body, visitor)
    if len(loops) != 1:
        raise AssertionError(f"expected one planned loop, found {len(loops)}")

    loop = loops[0]
    order = [int(value) for value in loop.annotations[
        "software_pipeline_order"
    ]]
    stage = [int(value) for value in loop.annotations[
        "software_pipeline_stage"
    ]]
    marker = int(loop.annotations.get("tl_pipeline_explicit_schedule", 0))
    if order != list(PIPELINE_ORDER):
        raise AssertionError(f"P6 planner changed explicit order: {order}")
    if stage != list(PIPELINE_STAGE):
        raise AssertionError(f"P6 planner changed explicit stage: {stage}")
    if marker != 1:
        raise AssertionError("P6 planner did not record the explicit path")
    if "tl_pipeline_order" in loop.annotations:
        raise AssertionError("P6 planner did not consume tl_pipeline_order")
    if "tl_pipeline_stage" in loop.annotations:
        raise AssertionError("P6 planner did not consume tl_pipeline_stage")
    if len(order) != LOWERED_STATEMENT_COUNT:
        raise AssertionError("unexpected P6 lowered statement count")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "planned_device.tir").write_text(
        mod.script(), encoding="utf-8"
    )
    evidence = {
        "explicit_schedule_marker": marker,
        "lowered_statement_count": len(order),
        "software_pipeline_order": order,
        "software_pipeline_stage": stage,
        "frontend_order_consumed": True,
        "frontend_stage_consumed": True,
    }
    (ARTIFACT_DIR / "planner_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _archive_lowering(program) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = tilelang.lower(program, target="tpu")
    raw_source = str(artifact.kernel_source)
    metrics = _source_metrics(raw_source)
    (ARTIFACT_DIR / "lowered_host.tir").write_text(
        artifact.host_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "lowered_device.tir").write_text(
        artifact.device_mod.script(), encoding="utf-8"
    )
    (ARTIFACT_DIR / "kernel_raw.c").write_text(
        raw_source, encoding="utf-8"
    )
    (ARTIFACT_DIR / "raw_source_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"raw_source": raw_source, "source_metrics": metrics}


def _archive_p6_cmodel_sources() -> None:
    for source_name, archive_name in (
        ("kernel.c", "kernel_cmodel.c"),
        ("kernel.cpp", "kernel.cpp"),
        ("kernel.h", "kernel.h"),
        ("main.cpp", "main.cpp"),
    ):
        source = P6_RUNTIME_DIR / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ARTIFACT_DIR / archive_name)


def _worker_main(variant: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "s3":
        program = make_chunk_scan_pipeline_s3_kernel()
        runtime_dir = S3_RUNTIME_DIR
    elif variant == "p6":
        program = make_chunk_scan_pipeline_p6_kernel()
        runtime_dir = P6_RUNTIME_DIR
        _archive_lowering(program)
    else:
        raise ValueError(f"unknown P6 worker variant: {variant}")

    kernel = _compile_cmodel(program, runtime_dir)
    if variant == "p6":
        _archive_p6_cmodel_sources()
    torch.save(
        _worker_outputs(kernel),
        ARTIFACT_DIR / f"worker_{variant}_outputs.pt",
    )


def _run_worker(variant: str) -> Dict[str, torch.Tensor]:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", variant],
        check=True,
    )
    output_path = ARTIFACT_DIR / f"worker_{variant}_outputs.pt"
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    return torch.load(output_path, map_location="cpu", weights_only=True)


def _inputs_for_case(name: str):
    base = _make_inputs(seed=20260910)
    if name != "residual_only_negative_D":
        return _variant_inputs(base, name)
    negative = list(_variant_inputs(base, "residual_only"))
    negative[-1].fill_(-0.28125)
    return tuple(negative)


def _numerical_results(
    s3_outputs: Dict[str, torch.Tensor],
    p6_outputs: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for name in (
        "state_only",
        "scan_only",
        "residual_only",
        "residual_only_negative_D",
        "all_terms",
    ):
        expected = chunk_scan_reference(*_inputs_for_case(name))
        s3 = s3_outputs[name]
        p6 = p6_outputs[name]
        oracle_stats = comparison_stats(p6, expected)
        control_stats = comparison_stats(p6, s3)
        result = {
            "p6_close_to_oracle": bool(
                torch.allclose(
                    p6.float(), expected.float(), atol=ATOL, rtol=RTOL
                )
            ),
            "p6_bitwise_equal_to_s3": bool(torch.equal(p6, s3)),
            "p6_vs_oracle": oracle_stats,
            "p6_vs_s3": control_stats,
            "expected_nonzero_count": int(torch.count_nonzero(expected).item()),
        }
        if (
            not result["p6_close_to_oracle"]
            or not result["p6_bitwise_equal_to_s3"]
            or result["expected_nonzero_count"] == 0
            or oracle_stats["nan_count"]
            or oracle_stats["inf_count"]
            or control_stats["max_abs_diff"] != 0.0
        ):
            raise AssertionError(f"P6 numerical mismatch: {name}: {result}")
        results[name] = result

    expected = chunk_scan_reference(
        *_variant_inputs(_make_inputs(seed=20260910), "scan_only")
    )
    p6_clean = p6_outputs["causal_clean"]
    p6_poisoned = p6_outputs["causal_poisoned"]
    s3_poisoned = s3_outputs["causal_poisoned"]
    causal = {
        "p6_clean_equals_poisoned": bool(torch.equal(p6_clean, p6_poisoned)),
        "p6_poisoned_equals_s3": bool(torch.equal(p6_poisoned, s3_poisoned)),
        "p6_poisoned_close_to_oracle": bool(
            torch.allclose(
                p6_poisoned.float(),
                expected.float(),
                atol=ATOL,
                rtol=RTOL,
            )
        ),
        "p6_poisoned_vs_oracle": comparison_stats(p6_poisoned, expected),
        "p6_poisoned_vs_s3": comparison_stats(p6_poisoned, s3_poisoned),
    }
    if not all(
        causal[key]
        for key in (
            "p6_clean_equals_poisoned",
            "p6_poisoned_equals_s3",
            "p6_poisoned_close_to_oracle",
        )
    ):
        raise AssertionError(f"P6 causal gate failed: {causal}")
    results["causal_upper_triangle_poison"] = causal
    return results


def _raw_matches_manual_reference(
    raw_source: str,
    metrics: Dict[str, object],
    s3_result: Dict[str, object],
) -> Dict[str, object]:
    actual_counts = {
        name: metrics[name] for name in EXPECTED_SOURCE_SITE_COUNTS
    }
    if actual_counts != EXPECTED_SOURCE_SITE_COUNTS:
        raise AssertionError(f"unexpected P6 source metrics: {actual_counts}")
    if raw_source.count(EXPECTED_REDUCTION_LOOP) != 1:
        raise AssertionError("P6 must contain one two-iteration steady loop")
    for declaration in EXPECTED_DOUBLE_BUFFER_DECLARATIONS:
        if raw_source.count(declaration) != 1:
            raise AssertionError(
                f"P6 missing double-buffer declaration: {declaration}"
            )

    p6_region = _pipeline_region_evidence(raw_source, 2)
    manual_region = s3_result["pipeline_structure"]["raw_region_evidence"]
    for key in (
        "producer_sites",
        "scan_gemm_sites",
        "steady_future_load_order",
    ):
        if p6_region[key] != manual_region[key]:
            raise AssertionError(
                f"P6 differs from manual S3 for {key}: {p6_region[key]}"
            )

    manual_metrics = s3_result["source_metrics"]
    compared_metric_names = list(EXPECTED_SOURCE_SITE_COUNTS) + [
        "static_local_memory_end_bytes",
        "bm1690_local_memory_limit_bytes",
    ]
    for name in compared_metric_names:
        if metrics[name] != manual_metrics[name]:
            raise AssertionError(
                f"P6 differs from manual S3 metric {name}: "
                f"{metrics[name]} != {manual_metrics[name]}"
            )
    if metrics["static_local_memory_end_bytes"] > metrics[
        "bm1690_local_memory_limit_bytes"
    ]:
        raise AssertionError(f"P6 exceeds LMEM: {metrics}")
    return {
        "matched_region_fields": [
            "producer_sites",
            "scan_gemm_sites",
            "steady_future_load_order",
        ],
        "matched_source_metrics": compared_metric_names,
        "raw_region_evidence": p6_region,
        "manual_reference": str(S3_RESULT_PATH),
    }


def _strip_parallel_markers(raw_source: str) -> str:
    return raw_source.replace(
        "      tpu_parallel_start(); \n", ""
    ).replace(
        "      tpu_parallel_end(); \n", ""
    ).replace(
        "tpu_parallel_start(); \n", ""
    ).replace(
        "tpu_parallel_end(); \n", ""
    )


def test_pipeline_p6_explicit_schedule_cmodel() -> None:
    toolchain = assert_chunkscan_toolchain()
    print(json.dumps(toolchain, indent=2, sort_keys=True), flush=True)
    p5_result = _load_pass_result(P5_RESULT_PATH, "P5 matrix")
    s3_result = _load_pass_result(S3_RESULT_PATH, "P4.2 S3")

    p6_program = make_chunk_scan_pipeline_p6_kernel()
    planner_evidence = _archive_planned_schedule(p6_program)
    negative_gates = _negative_compiler_gates()

    s3_outputs = _run_worker("s3")
    p6_outputs = _run_worker("p6")

    raw_source = (ARTIFACT_DIR / "kernel_raw.c").read_text(encoding="utf-8")
    metrics = _source_metrics(raw_source)
    manual_comparison = _raw_matches_manual_reference(
        raw_source, metrics, s3_result
    )
    numerical = _numerical_results(s3_outputs, p6_outputs)

    cmodel_source = (ARTIFACT_DIR / "kernel_cmodel.c").read_text(
        encoding="utf-8"
    )
    cmodel_equals_raw = cmodel_source == _strip_parallel_markers(raw_source)
    if not cmodel_equals_raw:
        raise AssertionError(
            "P6 cmodel source differs by more than parallel-marker removal"
        )

    result = {
        "status": "PASS",
        "stage": "P6",
        "variant": "compiler_explicit_sprog_b",
        "objective": "compiler-generated designated ChunkScan pipeline",
        "schedule_contract": SPROG_B_CONTRACT.as_dict(),
        "planner_evidence": planner_evidence,
        "negative_compiler_gates": negative_gates,
        "manual_s3_structural_comparison": manual_comparison,
        "numerical_results": numerical,
        "source_metrics": metrics,
        "cmodel_equals_raw_after_marker_strip": cmodel_equals_raw,
        "prerequisites": {
            "pipeline_s3": {
                "path": str(S3_RESULT_PATH),
                "status": s3_result["status"],
            },
            "pipeline_matrix_p5": {
                "path": str(P5_RESULT_PATH),
                "status": p5_result["status"],
            },
        },
        "toolchain": toolchain,
        "selection_policy": (
            "fixed paper-designated sProg-B contract; no timing search"
        ),
        "performance_claim": "NONE: cmodel timing is not TPU performance",
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (ARTIFACT_DIR / "manifest.json").write_text(
        serialized, encoding="utf-8"
    )
    (ARTIFACT_DIR / "result.json").write_text(
        serialized, encoding="utf-8"
    )
    print(serialized, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
    elif len(sys.argv) == 1:
        test_pipeline_p6_explicit_schedule_cmodel()
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [--worker s3|p6]")
```

该测试同时保存：显式契约、PipelinePlanning 后的 TIR、raw PPL 源码、cmodel
实际执行源码、LMEM/指令计数、三项编译器拒绝证据、S3 结构比较和全部数值
结果。

## 8. 第五步：修改统一入口

编辑 `tpu_demo/mamba2_chunk_scan/run_cmodel.sh`。

### 8.1 在 P5 分支之后增加两个 P6 分支

修改前：

```bash
if [[ "${1:-}" == "--pipeline-matrix-p5" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_matrix_p5_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

修改后：

```bash
if [[ "${1:-}" == "--pipeline-matrix-p5" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_matrix_p5_cmodel.py"
fi

if [[ "${1:-}" == "--pipeline-planner-p6-unit" ]]; then
    exec "${PYTHON_BIN}" -m pytest -q \
        "${CHUNKSCAN_TOOLCHAIN_ROOT}/testing/python/transform/test_tilelang_transform_pipeline_planning.py"
fi

if [[ "${1:-}" == "--pipeline-p6" ]]; then
    exec "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/test_chunk_scan_pipeline_p6_cmodel.py"
fi

if [[ "$#" -ne 0 ]]; then
```

### 8.2 修改 Usage

修改前：

```bash
"--pipeline-s3|--pipeline-matrix-p5]" >&2
```

修改后：

```bash
"--pipeline-s3|--pipeline-matrix-p5|"\
"--pipeline-planner-p6-unit|--pipeline-p6]" >&2
```

## 9. 第六步：只做静态检查

先不要构建：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

bash -n tpu_demo/mamba2_chunk_scan/run_cmodel.sh

PYTHONDONTWRITEBYTECODE=1 \
  /root/autodl-tmp/tilelang-tpu/.venv/bin/python -m py_compile \
  tpu_demo/mamba2_chunk_scan/chunk_scan_pipeline_p6.py \
  tpu_demo/mamba2_chunk_scan/test_chunk_scan_pipeline_p6_cmodel.py \
  testing/python/transform/test_tilelang_transform_pipeline_planning.py

git diff --check
```

如果任何命令失败，停在本步，不要继续构建。

## 10. 第七步：低资源模式重建 TileLang

本阶段修改了 C++ pass，必须重建 `libtilelang_module.so`。0.5 核、2 GB
内存下只允许单任务构建：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

CMAKE_BUILD_PARALLEL_LEVEL=1 \
  cmake --build build --target tilelang_module -j1
```

不要使用 `-j2`、`-j4` 或裸 `make -j`。成功后确认动态库时间戳已更新：

```bash
stat build/libtilelang_module.so
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --identity-only
```

身份输出中的 `tilelang_library` 必须仍指向当前仓库的
`build/libtilelang_module.so`。

## 11. 第八步：运行测试与保护性回归

第一次运行编译器单元测试前，先确认 `run_cmodel.sh` 所选择的同一个
Python 环境中安装了 `pytest`：

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python -c \
  'import pytest; print(pytest.__version__)'
```

如果报告 `No module named pytest`，只安装当前单元测试所需的最小依赖；
版本下限与仓库 `requirements-test.txt` 保持一致：

```bash
/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m pip install \
  --no-cache-dir 'pytest>=6.2.4'

/root/autodl-tmp/tilelang-tpu/.venv/bin/python -m pytest --version
```

这里必须使用上述虚拟环境的 `python -m pip`，不能使用未指定解释器的
`pip`，否则可能把 `pytest` 安装到另一个 Python 环境。当前测试命令没有
使用并行测试参数，因此无须为了这一项额外安装 `pytest-xdist`。在 2 GB
内存的无卡环境中，也无须重新安装整个 `requirements-test.txt`。

无卡环境如果仍安装了 CUDA 工具链，`determine_target("auto")` 会选择
CUDA；因为无法探测真实 GPU 架构，TVM 会回退到 `sm_50`。原有
`test_simple_pipeline` 却固定期望只有 `sm_80` 及以上才会生成的
`software_pipeline_async_stages`，会造成与 P6 无关的环境相关失败。为使
这项旧 CUDA 自动规划回归可重复，应在
`testing/python/transform/test_tilelang_transform_pipeline_planning.py`
中删除未再使用的导入：

```python
from tilelang.utils.target import determine_target
```

并把：

```python
auto_target = tvm.target.Target(determine_target("auto"))
```

改为显式的编译目标：

```python
auto_target = tvm.target.Target("cuda -arch=sm_80")
```

该测试只执行 IR 变换和结构比较，不会执行 CUDA kernel，因此固定编译目标
不需要 GPU，也不改变本项目只在 TPU cmodel 上执行 ChunkScan 的约束。

严格按顺序运行，每次等待前一条完全退出：

```bash
cd /root/autodl-tmp/tilelang-tpu-pipethreader-chunkscan

./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-planner-p6-unit
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s2
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-s3
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-matrix-p5
./tpu_demo/mamba2_chunk_scan/run_cmodel.sh --pipeline-p6
```

为什么要重新运行 P4.1-P5：P6 改变的是共享编译器动态库，旧 JSON 只能证明
旧动态库；关闭 P6 前必须证明新显式路径没有破坏旧自动路径。

不要记录或比较终端中的 benchmark、`Single kernel execution time`，这些都
不是 TPU 性能。

## 12. 第九步：检查结果

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("tpu_demo/mamba2_chunk_scan/artifacts")
paths = {
    "P4.1": root / "pipeline_s2/result.json",
    "P4.2": root / "pipeline_s3/result.json",
    "P5": root / "pipeline_matrix_p5/result.json",
    "P6": root / "pipeline_p6_explicit/result.json",
}
for name, path in paths.items():
    result = json.loads(path.read_text(encoding="utf-8"))
    print(name, result["status"], path)
    assert result["status"] == "PASS"

p6 = json.loads(paths["P6"].read_text(encoding="utf-8"))
print("statement_count =", p6["planner_evidence"]["lowered_statement_count"])
print("explicit_marker =", p6["planner_evidence"]["explicit_schedule_marker"])
print("negative_gates =", {
    key: value["status"]
    for key, value in p6["negative_compiler_gates"].items()
})
print("load_order =", p6["manual_s3_structural_comparison"]
      ["raw_region_evidence"]["steady_future_load_order"])
print("lmem =", p6["source_metrics"]["static_local_memory_end_bytes"],
      "/", p6["source_metrics"]["bm1690_local_memory_limit_bytes"])
print("performance_claim =", p6["performance_claim"])
PY
```

再检查关键产物：

```bash
find tpu_demo/mamba2_chunk_scan/artifacts/pipeline_p6_explicit \
  -maxdepth 2 -type f -printf '%p\n' | sort

git diff --check
```

## 13. P6 验收门槛

只有以下条件全部成立，P6 才能关闭并进入 P7：

- 规划器单元测试全部通过，原有自动规划测试没有回归；
- P4.1、P4.2、P5 在新动态库下重新运行后仍为 `PASS`；
- P6 `result.json` 为 `PASS`；
- planned TIR 中恰好一个显式调度标记，22 项 order/stage 与冻结契约一致，
  且两个前端 `tl_` 属性已被消费；
- 循环深度不足、依赖倒置和 LMEM 溢出都由编译器确定性拒绝；
- raw source 保持一个平衡流水区域、双缓冲、`2/1/0` 生产者排布、
  `0/1/2` scan-GEMM 排布以及
  `LoadY.cb -> LoadY.dA -> LoadY.dt -> LoadX`；
- P6 与手工 S3 的指令族计数、stage partition、buffer versioning 和 LMEM
  指标一致；
- state、scan、正负 D residual、all-terms 均通过 oracle，且 P6 与 S3 输出
  逐位相等；
- clean/poisoned 因果用例逐位相等；
- cmodel 源与 raw 源仅相差两个并行标记；
- `performance_claim` 仍为 `NONE`，没有真实 TPU 性能或物理重叠结论。

本次独立验收已保留 `pipeline_p6_explicit/result.json` 及全部生成产物；上述
门槛全部成立，因此 P6 已关闭。P7 仍需作为单独阶段启动和验收。
