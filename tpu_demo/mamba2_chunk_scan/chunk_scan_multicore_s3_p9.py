"""Task-major multi-core BM1690 TileLang-TPU ChunkScan S3.

P9.3 keeps the P9.2 compact ABI and work-item ownership rule, but replaces
the serial K16 reduction loop with the two-stage paper-ordered S3 pipeline.
Every task owns a complete [L, P] output tile, so no cross-core reduction or
synchronisation is required.
"""

import tilelang.language as T
from tilelang.language.copy import region

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ACCUM_DTYPE,
    CHUNK_SIZE,
    DSTATE,
    DTYPE,
    HEADDIM,
    OUTPUT_GUARD_ROWS,
    REDUCE_TILE,
    REDUCE_TILES,
    ChunkScanMulticoreConfig,
    physical_shapes,
)


NUM_STAGES = 2


def make_chunk_scan_multicore_s3_kernel(
    config: ChunkScanMulticoreConfig,
):
    """Return task-major S3 with runtime 1/2/4/8 work-item mapping."""

    shapes = physical_shapes(config)
    task_count = config.task_count

    @T.prim_func
    def main_kernel_inner(
        cb: T.Tensor(shapes["cb"], DTYPE),
        x: T.Tensor(shapes["x"], DTYPE),
        dt: T.Tensor(shapes["dt"], DTYPE),
        dA_cumsum: T.Tensor(shapes["dA_cumsum"], DTYPE),
        C: T.Tensor(shapes["C"], DTYPE),
        prev_states: T.Tensor(shapes["prev_states"], DTYPE),
        D: T.Tensor(shapes["D"], DTYPE),
        out: T.Tensor(shapes["out"], DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            C_shared = T.alloc_shared((CHUNK_SIZE, DSTATE), DTYPE)
            states_shared = T.alloc_shared((HEADDIM, DSTATE), DTYPE)
            x_shared = T.alloc_shared((CHUNK_SIZE, HEADDIM), DTYPE)
            x_fp32 = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)

            cb_loaded_shared = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_upper_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), DTYPE
            )
            cb_upper_reduce = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_dense_compute_shared = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_reduce_shared = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            x_reduce_shared = T.alloc_shared(
                (REDUCE_TILE, HEADDIM), DTYPE
            )
            matrix_reduce_scratch_fp32 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )
            matrix_scratch_fp32 = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), ACCUM_DTYPE
            )

            dA_col_fp16 = T.alloc_shared((CHUNK_SIZE, 1), DTYPE)
            dA_col_fp32 = T.alloc_shared((CHUNK_SIZE, 1), ACCUM_DTYPE)
            dA_exp_col_fp32 = T.alloc_shared(
                (CHUNK_SIZE, 1), ACCUM_DTYPE
            )
            dA_row_fp16 = T.alloc_shared((1, CHUNK_SIZE), DTYPE)
            dA_row_fp32 = T.alloc_shared((1, CHUNK_SIZE), ACCUM_DTYPE)
            dA_reduce_row_fp16_shared = T.alloc_shared(
                (1, REDUCE_TILE), DTYPE
            )
            dA_reduce_row_fp32 = T.alloc_shared(
                (1, REDUCE_TILE), ACCUM_DTYPE
            )
            dt_reduce_row_fp16_shared = T.alloc_shared(
                (1, REDUCE_TILE), DTYPE
            )
            dt_reduce_row_fp32 = T.alloc_shared(
                (1, REDUCE_TILE), ACCUM_DTYPE
            )

            historical = T.alloc_shared(
                (CHUNK_SIZE, HEADDIM), ACCUM_DTYPE
            )
            scan_or_residual = T.alloc_shared(
                (CHUNK_SIZE, HEADDIM), ACCUM_DTYPE
            )
            gemm_temp = T.alloc_shared(
                (CHUNK_SIZE, HEADDIM), ACCUM_DTYPE
            )
            decay_scores = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )

            exp_work0 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )
            exp_work1 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )
            exp_vector_work0 = T.alloc_shared(
                (CHUNK_SIZE, 1), ACCUM_DTYPE
            )
            exp_vector_work1 = T.alloc_shared(
                (CHUNK_SIZE, 1), ACCUM_DTYPE
            )
            exp_coeff = T.alloc_shared((64, 32), ACCUM_DTYPE)
            exp_table = T.alloc_shared((64, 192), ACCUM_DTYPE)

            D_fp16 = T.alloc_shared((1, 1), DTYPE)
            D_fp32 = T.alloc_shared((1, 1), ACCUM_DTYPE)
            out_fp16 = T.alloc_shared((CHUNK_SIZE, HEADDIM), DTYPE)

            workitem_index = T.ppl_workitem_index()
            workitem_num = T.ppl_workitem_num()

            for task in T.serial(task_count):
                if task % workitem_num == workitem_index:
                    T.ppl_copy(D[task, 0], D_fp16)
                    T.ppl_copy(D_fp16, D_fp32)

                    # Historical-state contribution.
                    T.ppl_copy(C[task * CHUNK_SIZE, 0], C_shared)
                    T.ppl_copy(
                        prev_states[task * HEADDIM, 0], states_shared
                    )
                    T.ppl_fill(historical, T.float32(0.0))
                    T.ppl_gemm(
                        C_shared,
                        states_shared,
                        historical,
                        transpose_B=True,
                    )
                    T.ppl_copy(dA_cumsum[task, 0], dA_row_fp16)
                    for index in T.serial(CHUNK_SIZE):
                        T.call_extern(
                            "handle",
                            "ppl.copy",
                            region(dA_cumsum[task, index], "r", 1, 1),
                            region(dA_col_fp16[index, 0], "w", 1, 1),
                        )
                    T.ppl_copy(dA_col_fp16, dA_col_fp32)
                    T.ppl_copy(dA_col_fp16, dA_exp_col_fp32)
                    T.ppl_exp2(
                        dA_exp_col_fp32,
                        exp_vector_work0,
                        exp_vector_work1,
                        exp_coeff,
                        exp_table,
                    )
                    T.ppl_mul(historical, historical, dA_exp_col_fp32)

                    # Strict-upper correction is immutable across K16 tiles.
                    T.ppl_fill(scan_or_residual, T.float32(0.0))
                    T.ppl_fill(cb_upper_shared, T.float16(0.0))
                    for row in T.unroll(0, CHUNK_SIZE - 1):
                        T.call_extern(
                            "handle",
                            "ppl.copy",
                            region(
                                cb[
                                    task * CHUNK_SIZE + row,
                                    row + 1,
                                ],
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

                    # Figure-6 sProg-B order: LoadY before LoadX.
                    for k_blk in T.Pipelined(
                        REDUCE_TILES, num_stages=NUM_STAGES
                    ):
                        T.ppl_copy(
                            cb[
                                task * CHUNK_SIZE,
                                k_blk * REDUCE_TILE,
                            ],
                            cb_loaded_shared,
                        )
                        T.ppl_copy(
                            cb_upper_shared[0, k_blk * REDUCE_TILE],
                            cb_upper_reduce,
                        )
                        T.ppl_copy(
                            cb_loaded_shared, cb_dense_compute_shared
                        )
                        T.ppl_subtract(
                            cb_reduce_shared,
                            cb_dense_compute_shared,
                            cb_upper_reduce,
                        )

                        T.ppl_copy(
                            dA_cumsum[task, k_blk * REDUCE_TILE],
                            dA_reduce_row_fp16_shared,
                        )
                        T.ppl_copy(
                            dA_reduce_row_fp16_shared,
                            dA_reduce_row_fp32,
                        )
                        T.ppl_fill(decay_scores, T.float32(0.0))
                        T.ppl_add(
                            decay_scores, decay_scores, dA_col_fp32
                        )
                        T.ppl_npu_bcast(
                            matrix_reduce_scratch_fp32,
                            dA_reduce_row_fp32,
                        )
                        T.ppl_subtract(
                            decay_scores,
                            decay_scores,
                            matrix_reduce_scratch_fp32,
                        )
                        T.ppl_exp2(
                            decay_scores,
                            exp_work0,
                            exp_work1,
                            exp_coeff,
                            exp_table,
                        )

                        T.ppl_copy(
                            cb_reduce_shared, matrix_reduce_scratch_fp32
                        )
                        T.ppl_mul(
                            decay_scores,
                            decay_scores,
                            matrix_reduce_scratch_fp32,
                        )
                        T.ppl_copy(
                            dt[task, k_blk * REDUCE_TILE],
                            dt_reduce_row_fp16_shared,
                        )
                        T.ppl_copy(
                            dt_reduce_row_fp16_shared,
                            dt_reduce_row_fp32,
                        )
                        T.ppl_npu_bcast(
                            matrix_reduce_scratch_fp32,
                            dt_reduce_row_fp32,
                        )
                        T.ppl_mul(
                            decay_scores,
                            decay_scores,
                            matrix_reduce_scratch_fp32,
                        )

                        T.ppl_copy(decay_scores, cb_reduce_shared)
                        T.ppl_copy(
                            x[
                                task * CHUNK_SIZE
                                + k_blk * REDUCE_TILE,
                                0,
                            ],
                            x_reduce_shared,
                        )
                        T.ppl_fill(gemm_temp, T.float32(0.0))
                        T.ppl_gemm(
                            cb_reduce_shared, x_reduce_shared, gemm_temp
                        )
                        T.ppl_add(
                            scan_or_residual,
                            scan_or_residual,
                            gemm_temp,
                        )

                    T.ppl_add(historical, historical, scan_or_residual)

                    # D*x residual remains outside the pipelined reduction.
                    T.ppl_copy(
                        x[task * CHUNK_SIZE, 0], x_shared
                    )
                    T.ppl_copy(x_shared, x_fp32)
                    T.ppl_fill(dA_row_fp32, T.float32(0.0))
                    T.ppl_add(dA_row_fp32, dA_row_fp32, D_fp32)
                    T.ppl_npu_bcast(
                        matrix_scratch_fp32, dA_row_fp32
                    )
                    T.ppl_mul(
                        scan_or_residual,
                        x_fp32,
                        matrix_scratch_fp32,
                    )
                    T.ppl_add(
                        historical, historical, scan_or_residual
                    )
                    T.ppl_copy(historical, out_fp16)
                    T.ppl_copy(
                        out_fp16,
                        out[
                            OUTPUT_GUARD_ROWS
                            + task * CHUNK_SIZE,
                            0,
                        ],
                    )

    return main_kernel_inner


__all__ = [
    "NUM_STAGES",
    "make_chunk_scan_multicore_s3_kernel",
]
