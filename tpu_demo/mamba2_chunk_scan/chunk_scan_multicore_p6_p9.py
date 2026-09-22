"""Explicit P6 schedule for the P9.3 multi-core ChunkScan kernel."""

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    ChunkScanMulticoreConfig,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s3_p9 import (
    make_chunk_scan_multicore_s3_kernel,
)
from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import (
    SPROG_B_CONTRACT,
    attach_explicit_pipeline_schedule,
)


def make_chunk_scan_multicore_p6_kernel(
    config: ChunkScanMulticoreConfig,
):
    """Attach the frozen Figure-6 sProg-B contract to multi-core S3."""

    return attach_explicit_pipeline_schedule(
        make_chunk_scan_multicore_s3_kernel(config),
        SPROG_B_CONTRACT,
    )


__all__ = ["make_chunk_scan_multicore_p6_kernel"]
