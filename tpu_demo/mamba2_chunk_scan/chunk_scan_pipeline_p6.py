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