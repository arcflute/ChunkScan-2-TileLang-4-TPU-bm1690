"""Serial BM1690 TileLang-TPU implementation of standalone ChunkScan.

The public Python-side contract remains the seven A0 logical inputs plus one
output.  The fixed B=G=H=1 smoke tensors are flattened to 2-D views immediately
before launch because the reused W8A16 PPL code generator only materializes
2-D and 4-D global tensor descriptors.  Flattening is a stride-preserving ABI
lowering; it does not remove any term from the standalone forward.
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
NUM_STAGES = 1


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


def make_chunk_scan_serial_kernel():
    """Return the fixed-shape, one-core, single-buffer ChunkScan S0 kernel."""

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
            x_shared = T.alloc_shared((CHUNK_SIZE, HEADDIM), DTYPE)
            x_fp32 = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)

            cb_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), DTYPE)
            matrix_scratch_fp32 = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE),
                ACCUM_DTYPE,
            )

            dA_col_fp16 = T.alloc_shared((CHUNK_SIZE, 1), DTYPE)
            dA_col_fp32 = T.alloc_shared((CHUNK_SIZE, 1), ACCUM_DTYPE)
            dA_exp_col_fp32 = T.alloc_shared(
                (CHUNK_SIZE, 1),
                ACCUM_DTYPE,
            )
            dA_row_fp16 = T.alloc_shared((1, CHUNK_SIZE), DTYPE)
            dA_row_fp32 = T.alloc_shared((1, CHUNK_SIZE), ACCUM_DTYPE)
            dt_row_fp16 = T.alloc_shared((1, CHUNK_SIZE), DTYPE)
            dt_row_fp32 = T.alloc_shared((1, CHUNK_SIZE), ACCUM_DTYPE)

            historical = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)
            scan_or_residual = T.alloc_shared((CHUNK_SIZE, HEADDIM), ACCUM_DTYPE)
            decay_scores = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), ACCUM_DTYPE)

            exp_work0 = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), ACCUM_DTYPE)
            exp_work1 = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), ACCUM_DTYPE)
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

                # 2) Causal chunk scan.  Start with an all-zero CB tile and
                # materialize only the lower-triangular prefixes.
                T.ppl_fill(cb_shared, T.float16(0.0))
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 0, 0], "r", 1, 1),
                    region(cb_shared[0, 0], "w", 1, 1),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 1, 0], "r", 1, 2),
                    region(cb_shared[1, 0], "w", 1, 2),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 2, 0], "r", 1, 3),
                    region(cb_shared[2, 0], "w", 1, 3),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 3, 0], "r", 1, 4),
                    region(cb_shared[3, 0], "w", 1, 4),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 4, 0], "r", 1, 5),
                    region(cb_shared[4, 0], "w", 1, 5),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 5, 0], "r", 1, 6),
                    region(cb_shared[5, 0], "w", 1, 6),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 6, 0], "r", 1, 7),
                    region(cb_shared[6, 0], "w", 1, 7),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 7, 0], "r", 1, 8),
                    region(cb_shared[7, 0], "w", 1, 8),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 8, 0], "r", 1, 9),
                    region(cb_shared[8, 0], "w", 1, 9),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 9, 0], "r", 1, 10),
                    region(cb_shared[9, 0], "w", 1, 10),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 10, 0], "r", 1, 11),
                    region(cb_shared[10, 0], "w", 1, 11),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 11, 0], "r", 1, 12),
                    region(cb_shared[11, 0], "w", 1, 12),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 12, 0], "r", 1, 13),
                    region(cb_shared[12, 0], "w", 1, 13),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 13, 0], "r", 1, 14),
                    region(cb_shared[13, 0], "w", 1, 14),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 14, 0], "r", 1, 15),
                    region(cb_shared[14, 0], "w", 1, 15),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 15, 0], "r", 1, 16),
                    region(cb_shared[15, 0], "w", 1, 16),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 16, 0], "r", 1, 17),
                    region(cb_shared[16, 0], "w", 1, 17),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 17, 0], "r", 1, 18),
                    region(cb_shared[17, 0], "w", 1, 18),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 18, 0], "r", 1, 19),
                    region(cb_shared[18, 0], "w", 1, 19),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 19, 0], "r", 1, 20),
                    region(cb_shared[19, 0], "w", 1, 20),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 20, 0], "r", 1, 21),
                    region(cb_shared[20, 0], "w", 1, 21),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 21, 0], "r", 1, 22),
                    region(cb_shared[21, 0], "w", 1, 22),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 22, 0], "r", 1, 23),
                    region(cb_shared[22, 0], "w", 1, 23),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 23, 0], "r", 1, 24),
                    region(cb_shared[23, 0], "w", 1, 24),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 24, 0], "r", 1, 25),
                    region(cb_shared[24, 0], "w", 1, 25),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 25, 0], "r", 1, 26),
                    region(cb_shared[25, 0], "w", 1, 26),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 26, 0], "r", 1, 27),
                    region(cb_shared[26, 0], "w", 1, 27),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 27, 0], "r", 1, 28),
                    region(cb_shared[27, 0], "w", 1, 28),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 28, 0], "r", 1, 29),
                    region(cb_shared[28, 0], "w", 1, 29),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 29, 0], "r", 1, 30),
                    region(cb_shared[29, 0], "w", 1, 30),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 30, 0], "r", 1, 31),
                    region(cb_shared[30, 0], "w", 1, 31),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 31, 0], "r", 1, 32),
                    region(cb_shared[31, 0], "w", 1, 32),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 32, 0], "r", 1, 33),
                    region(cb_shared[32, 0], "w", 1, 33),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 33, 0], "r", 1, 34),
                    region(cb_shared[33, 0], "w", 1, 34),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 34, 0], "r", 1, 35),
                    region(cb_shared[34, 0], "w", 1, 35),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 35, 0], "r", 1, 36),
                    region(cb_shared[35, 0], "w", 1, 36),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 36, 0], "r", 1, 37),
                    region(cb_shared[36, 0], "w", 1, 37),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 37, 0], "r", 1, 38),
                    region(cb_shared[37, 0], "w", 1, 38),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 38, 0], "r", 1, 39),
                    region(cb_shared[38, 0], "w", 1, 39),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 39, 0], "r", 1, 40),
                    region(cb_shared[39, 0], "w", 1, 40),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 40, 0], "r", 1, 41),
                    region(cb_shared[40, 0], "w", 1, 41),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 41, 0], "r", 1, 42),
                    region(cb_shared[41, 0], "w", 1, 42),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 42, 0], "r", 1, 43),
                    region(cb_shared[42, 0], "w", 1, 43),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 43, 0], "r", 1, 44),
                    region(cb_shared[43, 0], "w", 1, 44),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 44, 0], "r", 1, 45),
                    region(cb_shared[44, 0], "w", 1, 45),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 45, 0], "r", 1, 46),
                    region(cb_shared[45, 0], "w", 1, 46),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 46, 0], "r", 1, 47),
                    region(cb_shared[46, 0], "w", 1, 47),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 47, 0], "r", 1, 48),
                    region(cb_shared[47, 0], "w", 1, 48),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 48, 0], "r", 1, 49),
                    region(cb_shared[48, 0], "w", 1, 49),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 49, 0], "r", 1, 50),
                    region(cb_shared[49, 0], "w", 1, 50),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 50, 0], "r", 1, 51),
                    region(cb_shared[50, 0], "w", 1, 51),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 51, 0], "r", 1, 52),
                    region(cb_shared[51, 0], "w", 1, 52),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 52, 0], "r", 1, 53),
                    region(cb_shared[52, 0], "w", 1, 53),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 53, 0], "r", 1, 54),
                    region(cb_shared[53, 0], "w", 1, 54),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 54, 0], "r", 1, 55),
                    region(cb_shared[54, 0], "w", 1, 55),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 55, 0], "r", 1, 56),
                    region(cb_shared[55, 0], "w", 1, 56),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 56, 0], "r", 1, 57),
                    region(cb_shared[56, 0], "w", 1, 57),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 57, 0], "r", 1, 58),
                    region(cb_shared[57, 0], "w", 1, 58),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 58, 0], "r", 1, 59),
                    region(cb_shared[58, 0], "w", 1, 59),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 59, 0], "r", 1, 60),
                    region(cb_shared[59, 0], "w", 1, 60),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 60, 0], "r", 1, 61),
                    region(cb_shared[60, 0], "w", 1, 61),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 61, 0], "r", 1, 62),
                    region(cb_shared[61, 0], "w", 1, 62),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 62, 0], "r", 1, 63),
                    region(cb_shared[62, 0], "w", 1, 63),
                )
                T.call_extern(
                    "handle",
                    "ppl.copy",
                    region(cb[chunk * CHUNK_SIZE + 63, 0], "r", 1, 64),
                    region(cb_shared[63, 0], "w", 1, 64),
                )

                # decay_scores[q,s] = exp(dA[q] - dA[s]).
                T.ppl_copy(dA_row_fp16, dA_row_fp32)
                T.ppl_fill(decay_scores, T.float32(0.0))
                T.ppl_add(decay_scores, decay_scores, dA_col_fp32)
                T.ppl_npu_bcast(matrix_scratch_fp32, dA_row_fp32)
                T.ppl_subtract(
                    decay_scores,
                    decay_scores,
                    matrix_scratch_fp32,
                )
                T.ppl_exp2(
                    decay_scores,
                    exp_work0,
                    exp_work1,
                    exp_coeff,
                    exp_table,
                )

                T.ppl_copy(cb_shared, matrix_scratch_fp32)
                T.ppl_mul(
                    decay_scores,
                    decay_scores,
                    matrix_scratch_fp32,
                )

                # Right scale by the contiguous per-chunk dt[s] row.
                T.ppl_copy(dt[chunk, 0], dt_row_fp16)
                T.ppl_copy(dt_row_fp16, dt_row_fp32)
                T.ppl_npu_bcast(matrix_scratch_fp32, dt_row_fp32)
                T.ppl_mul(
                    decay_scores,
                    decay_scores,
                    matrix_scratch_fp32,
                )

                # Cast the score tile back to FP16 for the BDC matrix product.
                T.ppl_copy(decay_scores, cb_shared)
                T.ppl_copy(x[chunk * CHUNK_SIZE, 0], x_shared)
                T.ppl_fill(scan_or_residual, T.float32(0.0))
                T.ppl_gemm(cb_shared, x_shared, scan_or_residual)
                T.ppl_add(historical, historical, scan_or_residual)

                # 3) D*x residual.  First expand the dynamic scalar along W,
                # then materialize it across NPU lanes.
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
    "make_chunk_scan_serial_kernel",
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
]
