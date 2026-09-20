# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing

auto_target = tvm.target.Target("cuda -arch=sm_80")


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_simple_pipeline():

    @T.prim_func
    def before(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    @T.prim_func
    def after(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.serial(
                    32,
                    annotations={
                        "software_pipeline_async_stages": [0],
                        "software_pipeline_order": [0, 1, 2],
                        "software_pipeline_stage": [3, 3, 3]
                    }):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before, after)

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

if __name__ == "__main__":
    tilelang.testing.main()
