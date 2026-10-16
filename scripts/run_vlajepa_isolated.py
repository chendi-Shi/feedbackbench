"""Four independent VLA-JEPA episodes, one fresh CLI process per episode."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vlajepa_compact as compact

DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_isolated_closed_loop"
SOURCES = ("scripts/run_vlajepa_isolated.py", "docs/VLAJEPA_ISOLATED_PROTOCOL.md")
WORKER_SCHEMA = "feedbackbench.vlajepa_isolated_worker.v1"
_RUN_ONE_STARTED = False


def separate_output(output):
    output = compact.separate_output(output)
    previous = compact.DEFAULT_OUTPUT.resolve()
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("The previous compact experiment must remain separate")
    return output


def configured_runner():
    runner = compact.configured_runner()
    prior_cases, prior_spec = runner.cases_for_study, runner.study_spec
    runner.SOURCES = (*runner.SOURCES, *SOURCES)

    def cases():
        result = prior_cases()
        for case in result:
            case["model"] = "vlajepa_isolated"
            case["split"] = "isolated_closed_loop_feasibility"
            case["id"] = runner.episode_id(case)
        return result

    def spec():
        result = prior_spec()
        result["name"] = "VLA-JEPA isolated-process four-task feasibility"
        result["process_isolation"] = {
            "episodes_per_process": 1, "fresh_cli_process_required": True,
            "policy_loads_per_episode": 1, "environment_loads_per_episode": 1,
            "numeric_physics_and_memory_budgets": "unchanged from compact protocol",
            "previous_results_imported": False,
            "worker_evidence": "workers/worker_NNNN.json; PID plus process creation time and sealed-case binding"}
        result["limitations"].extend([
            "New four-case matrix; earlier compact three successes and one unsealed attempt remain separate",
            "Process startup and model loading occur once per episode; rollout wall time still excludes model loading",
            "Frozen LIBERO reset may swallow repeated native-warning exceptions; a hung reset requires external timeout and retained failure evidence"])
        return result

    runner.cases_for_study, runner.study_spec = cases, spec
    return runner


def process_identity():
    import psutil
    process = psutil.Process(os.getpid())
    return {"pid": process.pid, "process_create_time": process.create_time()}


def verify_workers(runner, output, study):
    """Validate worker provenance after execution, never inside the base runner."""
    output = Path(output)
    workers, identities, completed_cases = [], set(), set()
    expected_script = runner.file_sha(Path(__file__))
    results = {item["id"]: item for item in study.get("results", [])}
    for path in sorted((output / "workers").glob("worker_*.json")):
        row = runner.read_json(path)
        runner.require(row.get("schema") == WORKER_SCHEMA and row.get("protocol_fingerprint") == study["protocol_fingerprint"]
                       and row.get("script_sha256") == expected_script, "Worker provenance differs")
        pid, created = row.get("pid"), row.get("process_create_time")
        runner.require(type(pid) is int and pid > 0 and type(created) in (int, float)
                       and math.isfinite(created) and created > 0, "Worker process identity is invalid")
        identity = (pid, created)
        runner.require(identity not in identities, "Two workers reused the same process")
        identities.add(identity)
        runner.require(row.get("status") in {"started", "complete", "blocked_preflight", "failed"}, "Worker status differs")
        before = row["statuses_before"]
        runner.require(set(before) == set(study["statuses"]) and all(value in {"complete", "missing"} for value in before.values()),
                       "Worker initial episode inventory differs")
        case_id = row["case_id"]
        runner.require(before.get(case_id) == "missing" and row["completed_before"] == sum(value == "complete" for value in before.values()),
                       "Worker did not start a missing episode")
        for binding in row.get("run_records", []):
            runner.require(runner.file_sha(runner.project_file(binding["path"])) == binding["sha256"], "Worker run audit changed")
        if row["status"] == "complete":
            after = row["statuses_after"]
            expected = {**before, case_id: "complete"}
            runner.require(after == expected and row["completed_after"] == row["completed_before"] + 1
                           and row["newly_completed"] == [case_id] and case_id not in completed_cases,
                           "Worker must complete exactly one distinct episode")
            runner.require(case_id in results and len(row["run_records"]) == 1
                           and results[case_id]["model_info"]["run_record"] == row["run_records"][0]["path"],
                           "Sealed episode does not reference its worker run")
            completed_cases.add(case_id)
        workers.append({"path": runner.base.relative(path), "sha256": runner.file_sha(path), **row})
    expected_completed = {key for key, value in study["statuses"].items() if value == "complete"}
    runner.require(completed_cases == expected_completed, "Completed episode lacks a finalized independent worker")
    return workers


def report(runner, output):
    study = runner.verify_study(output, partial=True)
    workers = verify_workers(runner, output, study)
    result = runner.write_report(output)
    path = Path(output) / "report.json"
    saved = runner.read_json(path)
    saved["worker_audits"] = workers
    compact.reliable_write_json(runner.write_json, path, saved)
    result["worker_audits"] = workers
    return result


def _create_worker(runner, output, record):
    directory = Path(output) / "workers"
    directory.mkdir(exist_ok=True)
    index = 1
    while True:
        path = directory / f"worker_{index:04d}.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return path
        except FileExistsError:
            index += 1


def run_one(runner, output):
    """Attempt the next missing case once; invoke a new CLI process for each case."""
    global _RUN_ONE_STARTED
    output = separate_output(output)
    runner.require(not _RUN_ONE_STARTED, "run-one requires a fresh process for every invocation")
    before = runner.verify_study(output, partial=True)
    runner.require(all(value in {"complete", "missing"} for value in before["statuses"].values()),
                   "Unsealed/error episode retained; automatic retry forbidden")
    runner.require(not (output / "active.lock").exists(), "Existing active/stale lock retained")
    previous = verify_workers(runner, output, before)
    if before["complete"]:
        return report(runner, output)
    identity = process_identity()
    runner.require(all((row["pid"], row["process_create_time"]) != (identity["pid"], identity["process_create_time"])
                       for row in previous), "Worker process identity has already been used")
    case_id = next(key for key, value in before["statuses"].items() if value == "missing")
    record = {"schema": WORKER_SCHEMA, "status": "started", "started_at": runner.utc_now(), **identity,
              "protocol_fingerprint": before["protocol_fingerprint"], "script_sha256": runner.file_sha(Path(__file__)),
              "case_id": case_id, "completed_before": before["completed"], "statuses_before": before["statuses"],
              "newly_completed": [], "run_records": []}
    path = _create_worker(runner, output, record)
    _RUN_ONE_STARTED = True
    run_before = set((output / "runs").glob("run_*.json"))
    try:
        original = runner.seal_episode.__globals__["write_json"]
        writer = lambda destination, value: compact.reliable_write_json(original, destination, value)
        with patch.dict(runner.seal_episode.__globals__, {"write_json": writer}):
            outcome = compact.run(runner, output, limit=1)
        after = runner.verify_study(output, partial=True)
        record.update(completed_after=after["completed"], statuses_after=after["statuses"])
        record["run_records"] = [{"path": runner.base.relative(item), "sha256": runner.file_sha(item)}
                                 for item in sorted(set((output / "runs").glob("run_*.json")) - run_before)]
        runner.require(len(record["run_records"]) == 1, "Expected exactly one base-run audit")
        if outcome.get("status") == "blocked_preflight":
            runner.require(after["statuses"] == before["statuses"], "Blocked worker changed episode evidence")
            record["status"] = "blocked_preflight"
        else:
            runner.require(after["statuses"] == {**before["statuses"], case_id: "complete"}
                           and after["completed"] == before["completed"] + 1, "Worker must seal exactly its one missing case")
            record.update(status="complete", newly_completed=[case_id])
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=runner.portable_error(error))
        raise
    finally:
        record["finished_at"] = runner.utc_now()
        compact.reliable_write_json(runner.write_json, path, record)
    if record["status"] == "blocked_preflight":
        return {**outcome, "worker": runner.base.relative(path)}
    return report(runner, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run-one", "report", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    output = separate_output(args.output)
    runner = configured_runner()
    if args.command == "freeze":
        result = runner.run(output, freeze_only=True)
    elif args.command == "run-one":
        result = run_one(runner, output)
    elif args.command == "report":
        result = report(runner, output)
    else:
        result = runner.verify_study(output, partial=args.partial)
        result["worker_audits"] = verify_workers(runner, output, result)
        result.pop("results")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
