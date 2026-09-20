"""Single-entry, cmodel-only closure for standalone Mamba2 ChunkScan."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARTIFACTS = ROOT / "artifacts"
P7_DIR = ARTIFACTS / "p7"
RESULT_PATH = P7_DIR / "result.json"
RUNNER = ROOT / "run_cmodel.sh"

STAGE_DIRS = (
    "p1_broadcast_2d",
    "p1_exp",
    "p1_causal_mask",
    "p1_gemm",
    "p1_residual_cast",
    "serial",
    "reduction_serial",
    "pipeline_s2",
    "pipeline_s3",
    "pipeline_matrix_p5",
    "pipeline_p6_explicit",
)


def runner(option: str) -> list[str]:
    return [str(RUNNER), option]


STEPS = (
    ("P0_identity", runner("--identity-only"), (), None),
    (
        "A0_cpu_oracle",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tpu_demo/mamba2_chunk_scan/test_reference.py",
        ],
        (),
        14,
    ),
    ("P1_broadcast", runner("--broadcast-probe"), ("p1_broadcast_2d",), None),
    ("P1_exp", runner("--exp-probe"), ("p1_exp",), None),
    ("P1_causal_mask", runner("--causal-mask-probe"), ("p1_causal_mask",), None),
    ("P1_gemm", runner("--gemm-probe"), ("p1_gemm",), None),
    (
        "P1_residual_cast",
        runner("--residual-cast-probe"),
        ("p1_residual_cast",),
        None,
    ),
    ("P2_S0", [str(RUNNER)], ("serial",), None),
    ("P3_S1", runner("--reduction-serial"), ("reduction_serial",), None),
    ("P4_S2", runner("--pipeline-s2"), ("pipeline_s2",), None),
    ("P4_S3", runner("--pipeline-s3"), ("pipeline_s3",), None),
    (
        "P5_matrix",
        runner("--pipeline-matrix-p5"),
        ("pipeline_matrix_p5",),
        None,
    ),
    (
        "P6_planner_unit",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "testing/python/transform/"
            "test_tilelang_transform_pipeline_planning.py",
        ],
        (),
        7,
    ),
    (
        "P6_explicit",
        runner("--pipeline-p6"),
        ("pipeline_p6_explicit",),
        None,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_result(report: dict) -> None:
    P7_DIR.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_name("result.json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, RESULT_PATH)


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def previous_mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.is_file() else -1


def required_files(stage: str) -> tuple[str, ...]:
    if stage == "pipeline_matrix_p5":
        return (
            "result.json",
            "configuration_matrix.json",
            "sprog_a_k16_stage2/kernel_raw.c",
            "sprog_a_k16_stage2/lowered_device.tir",
            "sprog_a_k16_stage2/raw_source_metrics.json",
            "sprog_b_k16_stage3/kernel_raw.c",
            "sprog_b_k16_stage3/kernel_cmodel.c",
            "sprog_b_k16_stage3/lowered_device.tir",
            "sprog_b_k16_stage3/raw_source_metrics.json",
        )

    files = (
        "result.json",
        "manifest.json",
        "lowered_host.tir",
        "lowered_device.tir",
        "kernel_raw.c",
        "kernel_cmodel.c",
    )
    if not stage.startswith("p1_"):
        files += ("raw_source_metrics.json",)
    if stage == "pipeline_p6_explicit":
        files += ("planned_device.tir",)
    return files


def check_stage(stage: str, old_mtime: int) -> None:
    directory = ARTIFACTS / stage
    result_path = directory / "result.json"

    if not result_path.is_file():
        raise RuntimeError(f"{stage}: missing result.json")
    if result_path.stat().st_mtime_ns <= old_mtime:
        raise RuntimeError(f"{stage}: result.json was not rewritten in this run")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"{stage}: result.json is not PASS")

    for name in required_files(stage):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{stage}: missing or empty evidence: {path}")


def parse_identity(log_path: Path) -> dict:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()

    for offset, character in enumerate(content):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "repository_root" in candidate:
            return candidate

    raise RuntimeError("P0 identity log has no toolchain JSON")


def check_pytest_count(log_path: Path, expected: int) -> None:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\b(\d+) passed\b", content)
    if not matches or int(matches[-1]) != expected:
        raise RuntimeError(
            f"{log_path.name}: expected {expected} passed tests"
        )


def source_hashes() -> dict[str, str]:
    paths = set(ROOT.glob("*.py"))
    paths.update((ROOT / "microprobes").glob("*.py"))
    paths.update(
        {
            ROOT / "source_contract.md",
            ROOT / "stask_mapping.md",
            ROOT / "run_cmodel.sh",
            REPO / "src/transform/pipeline_planning.cc",
            REPO / "src/target/codegen_ppl.cc",
            REPO / "tilelang/jit/adapter/libgen.py",
            REPO / "3rdparty/tvm/src/target/target_kind.cc",
            REPO / "3rdparty/tvm/src/tir/analysis/"
            "block_access_region_detector.cc",
            REPO / "3rdparty/tvm/src/tir/transforms/storage_rewrite.cc",
        }
    )

    result = {}
    for path in sorted(paths):
        if not path.is_file():
            raise RuntimeError(f"missing frozen source: {path}")
        result[relative(path)] = sha256(path)
    return result


def validate_configuration_matrix() -> dict:
    path = ARTIFACTS / "pipeline_matrix_p5" / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = data["configuration_matrix"]

    accepted = {
        "s2_k16_stage2",
        "sprog_b_k16_stage2",
        "sprog_b_k16_stage3",
    }
    rejected = {
        "sprog_a_k16_stage2",
        "sprog_b_k16_stage4",
        "sprog_b_k32_stage2",
    }
    if set(data["accepted_configurations"]) != accepted:
        raise RuntimeError("P5 accepted configuration set changed")
    if set(data["rejected_configurations"]) != rejected:
        raise RuntimeError("P5 rejected configuration set changed")
    if data["accepted_count"] != 3 or data["rejected_count"] != 3:
        raise RuntimeError("P5 is not a 3-accepted/3-rejected matrix")
    if set(matrix) != accepted | rejected:
        raise RuntimeError("P5 matrix does not contain exactly six entries")

    summary = {}
    for name in sorted(matrix):
        item = matrix[name]
        expected = "ACCEPTED" if name in accepted else "REJECTED"
        if item.get("status") != expected:
            raise RuntimeError(f"P5 {name}: unexpected status")
        if expected == "REJECTED" and not item.get("reason_code"):
            raise RuntimeError(f"P5 {name}: rejection has no reason")
        summary[name] = {
            "status": expected,
            "reason_code": item.get("reason_code"),
        }
    return summary


def validate_p6() -> dict:
    directory = ARTIFACTS / "pipeline_p6_explicit"
    data = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    planner = data["planner_evidence"]
    gates = data["negative_compiler_gates"]
    metrics = data["source_metrics"]

    if planner.get("explicit_schedule_marker") != 1:
        raise RuntimeError("P6 explicit schedule marker is missing")
    if not planner.get("frontend_order_consumed"):
        raise RuntimeError("P6 frontend order was not consumed")
    if not planner.get("frontend_stage_consumed"):
        raise RuntimeError("P6 frontend stage was not consumed")
    if set(gates) != {
        "dependency_violation",
        "insufficient_loop_depth",
        "lmem_overflow",
    }:
        raise RuntimeError("P6 negative gate set changed")
    if any(
        gate.get("status") != "REJECTED_AS_EXPECTED"
        for gate in gates.values()
    ):
        raise RuntimeError("P6 negative compiler gate failed")

    used = int(metrics["static_local_memory_end_bytes"])
    limit = int(metrics["bm1690_local_memory_limit_bytes"])
    if not 0 < used <= limit:
        raise RuntimeError("P6 LMEM bound is invalid")

    raw = (directory / "kernel_raw.c").read_text(encoding="utf-8")
    cmodel = (directory / "kernel_cmodel.c").read_text(encoding="utf-8")
    if "tpu_parallel_start();" not in raw:
        raise RuntimeError("P6 raw target source has no pipeline marker")
    if "tpu_parallel_start();" in cmodel:
        raise RuntimeError("P6 cmodel source still has a pipeline marker")

    stripped = raw
    for marker in (
        "      tpu_parallel_start(); \n",
        "      tpu_parallel_end(); \n",
        "tpu_parallel_start(); \n",
        "tpu_parallel_end(); \n",
    ):
        stripped = stripped.replace(marker, "")
    if stripped != cmodel:
        raise RuntimeError("P6 raw/cmodel source difference is not marker-only")
    if not data.get("cmodel_equals_raw_after_marker_strip"):
        raise RuntimeError("P6 result did not record marker-only adaptation")

    comparison = data["manual_s3_structural_comparison"]
    if not comparison.get("matched_region_fields"):
        raise RuntimeError("P6 manual S3 structural comparison is missing")
    return {
        "explicit_schedule_marker": 1,
        "frontend_order_consumed": True,
        "frontend_stage_consumed": True,
        "negative_compiler_gates": sorted(gates),
        "static_local_memory_end_bytes": used,
        "bm1690_local_memory_limit_bytes": limit,
        "raw_cmodel_difference": "parallel markers only",
    }


def evidence_index() -> dict[str, dict]:
    index = {}
    extensions = {".json", ".tir", ".c", ".h", ".cpp"}

    for stage in STAGE_DIRS:
        directory = ARTIFACTS / stage
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in extensions:
                continue
            parent_parts = path.relative_to(directory).parts[:-1]
            if any(part.startswith("runtime") for part in parent_parts):
                continue
            index[relative(path)] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }

    return index


def show_log_tail(log_path: Path) -> None:
    if not log_path.is_file():
        return
    print(f"Last lines of {log_path}:", flush=True)
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in deque(stream, maxlen=30):
            print(line.rstrip(), flush=True)


def main() -> int:
    P7_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id += f"-{os.getpid()}"
    run_dir = P7_DIR / "runs" / run_id

    report = {
        "stage": "P7",
        "status": "RUNNING",
        "run_id": run_id,
        "steps": [],
        "scope": {
            "operator": "standalone _chunk_scan_fwd",
            "validated": (
                "BM1690 target-code structure and CPU-hosted cmodel "
                "functional correctness"
            ),
            "not_evaluated": [
                "physical TPU execution",
                "latency, throughput, or speedup",
                "physical pipeline overlap or synchronization",
                "four-core scaling",
                "mamba_chunk_scan_combined or full-model integration",
            ],
        },
    }
    write_result(report)

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        report["source_sha256"] = source_hashes()
        write_result(report)

        for name, command, stage_names, expected_tests in STEPS:
            number = len(report["steps"]) + 1
            log_path = run_dir / f"{number:02d}_{name}.log"
            old_mtimes = {
                stage: previous_mtime(ARTIFACTS / stage / "result.json")
                for stage in stage_names
            }

            step = {
                "name": name,
                "status": "RUNNING",
                "log": relative(log_path),
                "results": [
                    relative(ARTIFACTS / stage / "result.json")
                    for stage in stage_names
                ],
            }
            report["steps"].append(step)
            write_result(report)
            print(f"[{number:02d}/{len(STEPS)}] {name}: {log_path}", flush=True)

            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=REPO,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            step["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{name} exited with code {completed.returncode}"
                )

            if expected_tests is not None:
                check_pytest_count(log_path, expected_tests)
            for stage in stage_names:
                check_stage(stage, old_mtimes[stage])

            if name == "P0_identity":
                identity = parse_identity(log_path)
                if Path(identity["repository_root"]).resolve() != REPO:
                    raise RuntimeError("P0 identity points to another repository")
                report["toolchain_identity"] = identity

            step["status"] = "PASS"
            write_result(report)

        report["p5_configuration_outcomes"] = (
            validate_configuration_matrix()
        )
        report["p6_closure"] = validate_p6()
        report["evidence_index"] = evidence_index()
        report["status"] = "PASS"
        write_result(report)
        print(f"P7 PASS: {RESULT_PATH}", flush=True)
        return 0

    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["steps"] and report["steps"][-1]["status"] == "RUNNING":
            report["steps"][-1]["status"] = "FAIL"
            log_path = REPO / report["steps"][-1]["log"]
            show_log_tail(log_path)
        write_result(report)
        print(f"P7 FAIL: {error}", file=sys.stderr, flush=True)
        print(f"See {RESULT_PATH}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())