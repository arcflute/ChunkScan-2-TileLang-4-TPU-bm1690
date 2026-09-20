"""Two-stage pipelined BM1690 TileLang-TPU ChunkScan S2.

The public Python-side contract remains the seven A0 logical inputs plus one
output.  The fixed B=G=H=1 smoke tensors are flattened to 2-D views immediately
before launch because the reused W8A16 PPL code generator only materializes
2-D and 4-D global tensor descriptors.  Flattening is a stride-preserving ABI
lowering; it does not remove any term from the standalone forward.

S2 preserves the complete S0 ABI and math while scheduling the four ordered
K16 causal-scan reduction tiles with ``T.Pipelined(..., num_stages=2)``.
The global-to-local loads of dA, dt, and X are double buffered.  The causal
cb tiles are materialized before the pipeline because the current TPU
pipeline planner cannot version a conditionally selected shared buffer.
"""

import tilelang.language as T
from tilelang.language.copy import region


BATCH = 1
SEQLEN = 128
NCHUNKS = 2
CHUNK_SIZE = 64
NGROUPS = 1
NHEADS = 1
HEADDIM = 64
DSTATE = 128
DTYPE = "float16"
ACCUM_DTYPE = "float32"
CORE_NUM = 1
NUM_STAGES = 2
REDUCE_TILE = 16
REDUCE_TILES = CHUNK_SIZE // REDUCE_TILE


CB_PHYSICAL_SHAPE = (NCHUNKS * CHUNK_SIZE, CHUNK_SIZE)
X_PHYSICAL_SHAPE = (SEQLEN, HEADDIM)
DT_PHYSICAL_SHAPE = (NCHUNKS, CHUNK_SIZE)
DA_PHYSICAL_SHAPE = (NCHUNKS, CHUNK_SIZE)
C_PHYSICAL_SHAPE = (SEQLEN, DSTATE)
STATES_PHYSICAL_SHAPE = (NCHUNKS * HEADDIM, DSTATE)
D_PHYSICAL_SHAPE = (1, 1)
OUT_PHYSICAL_SHAPE = (SEQLEN, HEADDIM)


PHYSICAL_SHAPES = {
    "cb": CB_PHYSICAL_SHAPE,
    "x": X_PHYSICAL_SHAPE,
    "dt": DT_PHYSICAL_SHAPE,
    "dA_cumsum": DA_PHYSICAL_SHAPE,
    "C": C_PHYSICAL_SHAPE,
    "prev_states": STATES_PHYSICAL_SHAPE,
    "D": D_PHYSICAL_SHAPE,
    "out": OUT_PHYSICAL_SHAPE,
}


def make_chunk_scan_pipeline_s2_kernel():
    """Return the one-core, two-stage reduction-pipelined S2 kernel."""

    @T.prim_func
    def main_kernel_inner(
        cb: T.Tensor(CB_PHYSICAL_SHAPE, DTYPE),
        x: T.Tensor(X_PHYSICAL_SHAPE, DTYPE),
        dt: T.Tensor(DT_PHYSICAL_SHAPE, DTYPE),
        dA_cumsum: T.Tensor(DA_PHYSICAL_SHAPE, DTYPE),
        C: T.Tensor(C_PHYSICAL_SHAPE, DTYPE),
        prev_states: T.Tensor(STATES_PHYSICAL_SHAPE, DTYPE),
        D: T.Tensor(D_PHYSICAL_SHAPE, DTYPE),
        out: T.Tensor(OUT_PHYSICAL_SHAPE, DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            C_shared = T.alloc_shared((CHUNK_SIZE, DSTATE), DTYPE)
            states_shared = T.alloc_shared((HEADDIM, DSTATE), DTYPE)

            # Full x buffers are retained for the D*x residual after the
            # reduction-tiled scan loop.
            x_shared = T.alloc_shared((CHUNK_SIZE, HEADDIM), DTYPE)
            x_fp32 = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)

            # Paper-aligned scan reduction tiles: Y is the transformed
            # cb/dA/dt tile and X is the corresponding sequence tile.
            cb_causal_0 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_causal_1 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_causal_2 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), DTYPE
            )
            cb_causal_3 = T.alloc_shared(
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
                (CHUNK_SIZE, 1),
                ACCUM_DTYPE,
            )
            dA_row_fp16 = T.alloc_shared((1, CHUNK_SIZE), DTYPE)
            dA_row_fp32 = T.alloc_shared((1, CHUNK_SIZE), ACCUM_DTYPE)
            dA_reduce_row_fp16_shared = T.alloc_shared((1, REDUCE_TILE), DTYPE)
            dA_reduce_row_fp32 = T.alloc_shared(
                (1, REDUCE_TILE), ACCUM_DTYPE
            )
            dt_reduce_row_fp16_shared = T.alloc_shared((1, REDUCE_TILE), DTYPE)
            dt_reduce_row_fp32 = T.alloc_shared(
                (1, REDUCE_TILE), ACCUM_DTYPE
            )

            historical = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)
            scan_or_residual = T.alloc_shared(
                (CHUNK_SIZE, HEADDIM), ACCUM_DTYPE
            )
            gemm_temp = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)
            decay_scores = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )

            exp_work0 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )
            exp_work1 = T.alloc_shared(
                (CHUNK_SIZE, REDUCE_TILE), ACCUM_DTYPE
            )
            exp_vector_work0 = T.alloc_shared((CHUNK_SIZE, 1), ACCUM_DTYPE)
            exp_vector_work1 = T.alloc_shared((CHUNK_SIZE, 1), ACCUM_DTYPE)
            exp_coeff = T.alloc_shared((64, 32), ACCUM_DTYPE)
            exp_table = T.alloc_shared((64, 192), ACCUM_DTYPE)

            D_fp16 = T.alloc_shared((1, 1), DTYPE)
            D_fp32 = T.alloc_shared((1, 1), ACCUM_DTYPE)
            out_fp16 = T.alloc_shared((CHUNK_SIZE, HEADDIM), DTYPE)

            T.ppl_copy(D[0, 0], D_fp16)
            T.ppl_copy(D_fp16, D_fp32)

            for chunk in T.serial(NCHUNKS):
                # 1) Historical state: exp(dA_q) * (C_q @ prev_state^T).
                T.ppl_copy(C[chunk * CHUNK_SIZE, 0], C_shared)
                T.ppl_copy(
                    prev_states[chunk * HEADDIM, 0],
                    states_shared,
                )
                T.ppl_fill(historical, T.float32(0.0))
                T.ppl_gemm(
                    C_shared,
                    states_shared,
                    historical,
                    transpose_B=True,
                )
                # The compact logical [B,H,Ck,L] input is exposed as the
                # physical row view [Ck,L].  A bounded serial GDMA scatter
                # converts that row to the local [L,1] column layout.
                T.ppl_copy(dA_cumsum[chunk, 0], dA_row_fp16)
                for index in T.serial(CHUNK_SIZE):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(dA_cumsum[chunk, index], "r", 1, 1),
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

                # 2) Causal chunk scan with four two-stage-pipelined K16
                # reductions.
                # The local CB tile starts at zero and receives only q >= s
                # entries, so poisoned upper-triangle values remain invisible.
                T.ppl_fill(scan_or_residual, T.float32(0.0))

                # Current TPU pipeline rewriting cannot version a buffer that
                # is read from inside the four causal selection branches.
                # Prepare the immutable lower-triangular K16 CB slices once
                # per chunk, before entering the Load/Compute pipeline.
                T.ppl_fill(cb_causal_0, T.float16(0.0))
                T.ppl_fill(cb_causal_1, T.float16(0.0))
                T.ppl_fill(cb_causal_2, T.float16(0.0))
                T.ppl_fill(cb_causal_3, T.float16(0.0))

                for row in T.unroll(0, 16):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 0],
                            "r",
                            1,
                            row + 1,
                        ),
                        region(cb_causal_0[row, 0], "w", 1, row + 1),
                    )
                for row in T.unroll(16, CHUNK_SIZE):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 0],
                            "r",
                            1,
                            REDUCE_TILE,
                        ),
                        region(
                            cb_causal_0[row, 0], "w", 1, REDUCE_TILE
                        ),
                    )

                for row in T.unroll(16, 32):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 16],
                            "r",
                            1,
                            row - 15,
                        ),
                        region(cb_causal_1[row, 0], "w", 1, row - 15),
                    )
                for row in T.unroll(32, CHUNK_SIZE):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 16],
                            "r",
                            1,
                            REDUCE_TILE,
                        ),
                        region(
                            cb_causal_1[row, 0], "w", 1, REDUCE_TILE
                        ),
                    )

                for row in T.unroll(32, 48):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 32],
                            "r",
                            1,
                            row - 31,
                        ),
                        region(cb_causal_2[row, 0], "w", 1, row - 31),
                    )
                for row in T.unroll(48, CHUNK_SIZE):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 32],
                            "r",
                            1,
                            REDUCE_TILE,
                        ),
                        region(
                            cb_causal_2[row, 0], "w", 1, REDUCE_TILE
                        ),
                    )

                for row in T.unroll(48, CHUNK_SIZE):
                    T.call_extern(
                        "handle",
                        "ppl.copy",
                        region(
                            cb[chunk * CHUNK_SIZE + row, 48],
                            "r",
                            1,
                            row - 47,
                        ),
                        region(cb_causal_3[row, 0], "w", 1, row - 47),
                    )

                for k_blk in T.Pipelined(REDUCE_TILES, num_stages=2):
                    # The causal CB slices are prepared before the pipeline.
                    # Select the immutable slice for this reduction iteration.
                    if k_blk == 0:
                        T.ppl_copy(cb_causal_0, cb_reduce_shared)
                    elif k_blk == 1:
                        T.ppl_copy(cb_causal_1, cb_reduce_shared)
                    elif k_blk == 2:
                        T.ppl_copy(cb_causal_2, cb_reduce_shared)
                    else:
                        T.ppl_copy(cb_causal_3, cb_reduce_shared)

                    # LoadY tail: compact dA[s] and dt[s] rows for this K tile.
                    T.ppl_copy(
                        dA_cumsum[chunk, k_blk * REDUCE_TILE],
                        dA_reduce_row_fp16_shared,
                    )
                    T.ppl_copy(dA_reduce_row_fp16_shared, dA_reduce_row_fp32)

                    # decay[q,s] = exp(dA[q] - dA[s]) for this tile.
                    T.ppl_fill(decay_scores, T.float32(0.0))
                    T.ppl_add(decay_scores, decay_scores, dA_col_fp32)
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

                    T.ppl_copy(cb_reduce_shared, matrix_reduce_scratch_fp32)
                    T.ppl_mul(
                        decay_scores,
                        decay_scores,
                        matrix_reduce_scratch_fp32,
                    )

                    T.ppl_copy(
                        dt[chunk, k_blk * REDUCE_TILE],
                        dt_reduce_row_fp16_shared,
                    )
                    T.ppl_copy(dt_reduce_row_fp16_shared, dt_reduce_row_fp32)
                    T.ppl_npu_bcast(
                        matrix_reduce_scratch_fp32,
                        dt_reduce_row_fp32,
                    )
                    T.ppl_mul(
                        decay_scores,
                        decay_scores,
                        matrix_reduce_scratch_fp32,
                    )

                    # LoadX and accumulate one K16 GEMM contribution.
                    T.ppl_copy(decay_scores, cb_reduce_shared)
                    T.ppl_copy(
                        x[chunk * CHUNK_SIZE + k_blk * REDUCE_TILE, 0],
                        x_reduce_shared,
                    )
                    T.ppl_fill(gemm_temp, T.float32(0.0))
                    T.ppl_gemm(cb_reduce_shared, x_reduce_shared, gemm_temp)
                    T.ppl_add(scan_or_residual, scan_or_residual, gemm_temp)

                T.ppl_add(historical, historical, scan_or_residual)

                # 3) D*x residual.  Keep it outside the scan reduction loop so
                # S2 differs from S1 only in causal-tile materialization and
                # the two-stage causal-scan reduction schedule.
                T.ppl_copy(x[chunk * CHUNK_SIZE, 0], x_shared)
                T.ppl_copy(x_shared, x_fp32)
                T.ppl_fill(dA_row_fp32, T.float32(0.0))
                T.ppl_add(dA_row_fp32, dA_row_fp32, D_fp32)
                T.ppl_npu_bcast(matrix_scratch_fp32, dA_row_fp32)
                T.ppl_mul(
                    scan_or_residual,
                    x_fp32,
                    matrix_scratch_fp32,
                )
                T.ppl_add(historical, historical, scan_or_residual)
                T.ppl_copy(historical, out_fp16)
                T.ppl_copy(out_fp16, out[chunk * CHUNK_SIZE, 0])

    return main_kernel_inner


__all__ = [
    "make_chunk_scan_pipeline_s2_kernel",
    "PHYSICAL_SHAPES",
    "BATCH",
    "SEQLEN",
    "NCHUNKS",
    "CHUNK_SIZE",
    "NGROUPS",
    "NHEADS",
    "HEADDIM",
    "DSTATE",
    "DTYPE",
    "ACCUM_DTYPE",
    "CORE_NUM",
    "NUM_STAGES",
    "REDUCE_TILE",
    "REDUCE_TILES",
]
