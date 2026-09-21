"""Minimal BM1690 work-item dispatch probe for P9.

Each active work-item writes its index and the launch work-item count to a
dedicated output row.  All addresses are compile-time constants so the probe
does not weaken the existing PPL global-region bounds checks.
"""

import tilelang.language as T


MAX_CORES = 8
PROBE_COLUMNS = 2
DTYPE = "float32"
OUTPUT_SHAPE = (MAX_CORES, PROBE_COLUMNS)


def make_multicore_workitem_probe_p9_kernel():
    """Return the physical 1/2/4/8-core runtime-dispatch probe."""

    @T.macro
    def write_row(
        output: T.Tensor(OUTPUT_SHAPE, DTYPE),
        index_value: T.Tensor((1, 1), DTYPE),
        count_value: T.Tensor((1, 1), DTYPE),
        row: T.int32,
        row_value: T.float32,
    ):
        T.ppl_fill(index_value, row_value)
        T.ppl_copy(index_value, output[row, 0])

        workitem_num = T.ppl_workitem_num()
        if workitem_num == 1:
            T.ppl_fill(count_value, T.float32(1.0))
        elif workitem_num == 2:
            T.ppl_fill(count_value, T.float32(2.0))
        elif workitem_num == 4:
            T.ppl_fill(count_value, T.float32(4.0))
        elif workitem_num == 8:
            T.ppl_fill(count_value, T.float32(8.0))
        else:
            T.ppl_fill(count_value, T.float32(-1.0))
        T.ppl_copy(count_value, output[row, 1])

    @T.prim_func
    def main_kernel_inner(
        probe_out: T.Tensor(OUTPUT_SHAPE, DTYPE),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            index_value = T.alloc_shared((1, 1), DTYPE)
            count_value = T.alloc_shared((1, 1), DTYPE)
            workitem_index = T.ppl_workitem_index()

            if workitem_index == 0:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    0,
                    T.float32(0.0),
                )
            elif workitem_index == 1:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    1,
                    T.float32(1.0),
                )
            elif workitem_index == 2:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    2,
                    T.float32(2.0),
                )
            elif workitem_index == 3:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    3,
                    T.float32(3.0),
                )
            elif workitem_index == 4:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    4,
                    T.float32(4.0),
                )
            elif workitem_index == 5:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    5,
                    T.float32(5.0),
                )
            elif workitem_index == 6:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    6,
                    T.float32(6.0),
                )
            elif workitem_index == 7:
                write_row(
                    probe_out,
                    index_value,
                    count_value,
                    7,
                    T.float32(7.0),
                )

    return main_kernel_inner


__all__ = [
    "DTYPE",
    "MAX_CORES",
    "OUTPUT_SHAPE",
    "PROBE_COLUMNS",
    "make_multicore_workitem_probe_p9_kernel",
]
