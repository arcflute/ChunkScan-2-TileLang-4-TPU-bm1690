"""Pinned source configurations and project-local A0 test configurations."""

TILELANG_V015_COMMIT = "a32009bf1e314b514c07389123648ba19009f3a5"
TILELANG_EXAMPLE = "examples/linear_attention/example_mamba_chunk_scan.py"
TILELANG_TEST = "testing/python/kernel/test_tilelang_kernel_flash_linear_attention.py"


# These entries are transcribed from the fixed TileLang v0.1.5 sources.  The
# collection name intentionally says "PAPER_OR_TILELANG": no entry is labeled
# as a paper configuration unless the paper itself supplies it.  The pinned
# source files below supply TileLang configurations, not PipeThreader CC IDs.
PAPER_OR_TILELANG_CONFIGS = {
    "tilelang_v0_1_5_autotune_space": {
        "source": f"{TILELANG_EXAMPLE}:66-82",
        "commit": TILELANG_V015_COMMIT,
        "kind": "TileLang source configuration; not labeled as a paper CC configuration",
        "block_M": (64, 128, 256),
        "block_N": (32, 64),
        "block_K": (64, 128, 256),
        "block_Dstate": (128,),
        "num_stages": (1, 2, 3, 4, 5),
        "threads_formula": "2 * block_M",
    },
    "tilelang_v0_1_5_cli_defaults": {
        "source": f"{TILELANG_EXAMPLE}:220-231",
        "commit": TILELANG_V015_COMMIT,
        "kind": "TileLang benchmark/CLI default shape; not labeled as a paper shape",
        "batch": 8,
        "seqlen": 4096,
        "chunk_size": 256,
        "ngroups": 1,
        "nheads": 80,
        "headdim": 64,
        "dstate": 128,
        "dtype": "float16",
    },
    "tilelang_v0_1_5_non_tuned_launch": {
        "source": f"{TILELANG_EXAMPLE}:234-247",
        "commit": TILELANG_V015_COMMIT,
        "kind": "TileLang non-tuned benchmark launch; not labeled as a paper CC configuration",
        "block_M": 64,
        "block_N": 64,
        "block_K": 64,
        "block_Dstate": 128,
        "num_stages": 2,
        "threads": 128,
        "correctness_atol": 1e-2,
        "correctness_rtol": 1e-2,
        "benchmark_warmup": 500,
    },
    "tilelang_v0_1_5_test_chunk_scan": {
        "source": f"{TILELANG_TEST}:318-332",
        "commit": TILELANG_V015_COMMIT,
        "kind": "TileLang unit-test shape and launch; not labeled as a paper shape",
        "batch": 8,
        "seqlen": 2048,
        "chunk_size": 256,
        "ngroups": 1,
        "nheads": 8,
        "headdim": 64,
        "dstate": 128,
        "dtype": "float16",
        "block_M": 64,
        "block_N": 64,
        "block_K": 64,
        "block_Dstate": 128,
        "num_stages": 2,
        "threads": 128,
        "correctness_source": f"{TILELANG_TEST}:184",
        "correctness_atol": 1e-2,
        "correctness_rtol": 1e-2,
        "max_mismatched_ratio": 0.05,
    },
}


# Small project-authored shapes for CPU unit tests.  They are not paper or
# upstream TileLang benchmark configurations.
CPU_TEST_CONFIGS = {
    "two_chunk_unit": {
        "source": "this project: A0 CPU oracle test design",
        "kind": "project test configuration",
        "batch": 1,
        "seqlen": 6,
        "chunk_size": 3,
        "ngroups": 2,
        "nheads": 4,
        "headdim": 2,
        "dstate": 3,
        "dtype": "float16",
    },
    "scalar_two_chunk": {
        "source": "this project: A0 chunk-state indexing test design",
        "kind": "project test configuration",
        "batch": 1,
        "seqlen": 2,
        "chunk_size": 1,
        "ngroups": 1,
        "nheads": 1,
        "headdim": 1,
        "dstate": 1,
        "dtype": "float16",
    },
}


# This is a project recommendation, not a PipeThreader paper configuration and
# not a TileLang v0.1.5 benchmark configuration.  A0 uses it only for CPU-side
# interface, shape, and dtype validation.
TPU_SMOKE_CONFIG = {
    "source": "this project: recommended BM1690 standalone ChunkScan smoke shape",
    "kind": "project-recommended configuration; not yet tested on TPU or cmodel",
    "batch": 1,
    "seqlen": 128,
    "chunk_size": 64,
    "nchunks": 2,
    "ngroups": 1,
    "nheads": 1,
    "headdim": 64,
    "dstate": 128,
    "dtype": "float16",
    "D_shape": (1,),
    "block_M": 64,
    "block_N": 64,
    "block_K": 64,
    "block_Dstate": 128,
    "num_stages": 1,
    "core_num": 1,
}
