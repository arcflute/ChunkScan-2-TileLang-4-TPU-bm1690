"""Structural gate for the P10 S1/S3/P6 performance matrix."""

import contextlib
import io

import tilelang

from tpu_demo.mamba2_chunk_scan.chunk_scan_pipeline_p6 import SCHEDULE_NAME
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_benchmark_p10 import (
    SHAPES,
    STAGES,
    add_comparative_metrics,
    summarize_point,
)
from tpu_demo.mamba2_chunk_scan.test_chunk_scan_multicore_s1_p9 import (
    static_local_memory_end,
)


def test_p10_shapes_and_lowering():
    expected_tasks = {"r1": 128, "r2": 512, "r3": 2048}
    for shape_name, config in SHAPES.items():
        assert config.task_count == expected_tasks[shape_name]
        for stage, builder in STAGES.items():
            program = builder(config)
            schedule = program.attrs.get("p6_pipeline_schedule")
            if stage == "p6":
                assert str(schedule) == SCHEDULE_NAME
            else:
                assert schedule is None

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                artifact = tilelang.lower(program, target="tpu")
            source = str(artifact.kernel_source)
            assert (
                f"for (int task = 0; task < {config.task_count}; ++task)"
                in source
            )
            assert source.count("tpu_workitem_index()") == 1
            assert source.count("tpu_workitem_num()") == 1
            expected_markers = 0 if stage == "s1" else 1
            assert source.count("tpu_parallel_start(") == expected_markers
            assert source.count("tpu_parallel_end(") == expected_markers
            local_end = static_local_memory_end(source)
            assert local_end is not None
            assert local_end <= 256 * 1024


def test_p10_summary_metrics():
    summaries = {}
    latencies = {
        ("s1", 1): 400.0,
        ("s1", 2): 220.0,
        ("s3", 1): 320.0,
        ("s3", 2): 180.0,
        ("p6", 1): 300.0,
        ("p6", 2): 160.0,
    }
    for (stage, core), latency in latencies.items():
        rounds = [
            {
                "p50_us": latency,
                "p95_us": latency * 1.1,
                "sample_count": 20,
                "measurement_wall_seconds": 5.0,
            }
            for _ in range(7)
        ]
        key = f"r1:{stage}:c{core}"
        summaries[key] = summarize_point(
            "r1", stage, core, rounds, SHAPES["r1"]
        )
    best = add_comparative_metrics(
        summaries,
        ("r1",),
        ("s1", "s3", "p6"),
        (1, 2),
    )
    assert best["r1"]["stage"] == "p6"
    assert best["r1"]["core_num"] == 2
    assert summaries["r1:p6:c2"][
        "multicore_speedup_vs_same_stage_c1"
    ] == 1.875
    assert summaries["r1:p6:c2"][
        "pipeline_gain_vs_s1_same_core"
    ] == 1.375
    assert summaries["r1:p6:c2"]["combined_gain_vs_s1_c1"] == 2.5
    assert summaries["r1:p6:c2"]["stability"] == "STABLE"


if __name__ == "__main__":
    test_p10_shapes_and_lowering()
    test_p10_summary_metrics()
    print("P10 R1/R2/R3 S1/S3/P6 lowering PASS")
