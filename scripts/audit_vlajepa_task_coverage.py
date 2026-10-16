"""Index task evidence across separate VLA-JEPA budgets without pooling outcomes.

This read-only audit validates original protocols, source hashes, receipts and
numeric traces. It never copies/reseals episodes or invokes a model/environment.
Run after the independently budgeted LIBERO-10 diagnostic has terminated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": checksum,
            "size_bytes": path.stat().st_size}


def process_is_alive(pid, created):
    import psutil
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - created) < 0.001
    except psutil.NoSuchProcess:
        return False


def run_evidence(output):
    """Keep all run/stop records, interpreting stale labels without editing them."""
    records = []
    for path in sorted((output / "runs").glob("run_*.json")):
        row = read(path)
        item = {**binding(path), "raw_status": row["status"]}
        if path.name.endswith(".memory_stop.json"):
            require(row["status"] == "stopped_memory_guard", "Unexpected memory sidecar status")
            require(not process_is_alive(row["pid"], row["process_create_time"]),
                    "Bound experiment process is still running; wait before auditing")
            item.update(effective_status="stopped_memory_guard", process_exited=True,
                        rss_bytes=row["rss_bytes"], available_bytes=row["available_bytes"],
                        limits=row["limits"], pid=row["pid"], process_create_time=row["process_create_time"])
        else:
            stop_path = path.with_name(path.stem + ".memory_stop.json")
            item["effective_status"] = "stopped_memory_guard" if stop_path.exists() else row["status"]
            item["memory"] = {key: value for key, value in row.get("memory", {}).items() if key != "samples"}
            item["load_quantize_seconds"] = row.get("load_quantize_seconds")
            item["weights_loaded"] = row.get("weights_loaded", False)
        records.append(item)
    lock = output / "active.lock"
    if lock.exists():
        lock_row = read(lock)
        matched = [row for row in records if row.get("pid") == lock_row["pid"]
                   and row.get("effective_status") == "stopped_memory_guard" and row.get("process_exited")]
        require(bool(matched), "Unresolved active/stale lock; do not bind a potentially active run")
    return {"records": records, "retained_lock": binding(lock) if lock.exists() else None,
            "stale_status_explanation": "A memory-guard os._exit bypasses finally; raw started/running_episodes labels and locks are retained, not interpreted as an active process."}


def protocol_evidence(runner, output, study):
    protocol = runner.verify_protocol(output)
    sources = []
    for name, expected in protocol["sources"].items():
        source = binding(ROOT / name)
        require(source["sha256"] == expected, "Frozen source changed: " + name)
        sources.append(source)
    return protocol, {
        "protocol": binding(output / "protocol.json"), "fingerprint": protocol["fingerprint"],
        "inputs_snapshot": binding(output / "inputs.json"), "source_hashes_verified": sources,
        "resource_budget": protocol["spec"]["memory_guard"],
        "own_denominator": {key: study[key] for key in ("planned", "completed", "complete", "successes", "task_failures")},
        "case_statuses": study["statuses"], "run_evidence": run_evidence(output),
    }


def episode_index(runner, output, protocol, case):
    directory = output / "episodes" / case["id"]
    common = {"suite": case["suite"], "task_id": case["task_id"], "init_state_id": case["init_state_id"],
              "seed": case["seed"], "case_id": case["id"], "protocol": binding(output / "protocol.json"),
              "protocol_fingerprint": protocol["fingerprint"], "resource_budget": protocol["spec"]["memory_guard"],
              "model_variant": protocol["spec"]["variant"], "episode_directory": directory.relative_to(ROOT).as_posix()}
    if not (directory / "receipt.json").is_file():
        files = {path.name: binding(path) for path in sorted(directory.iterdir()) if path.is_file()} if directory.exists() else {}
        actions = directory / "executed_steps.jsonl"
        action_rows = [json.loads(line) for line in actions.read_text(encoding="utf-8").splitlines()] if actions.exists() else []
        return {**common, "evidence_status": "unsealed" if directory.exists() else "missing",
                "completed": False, "official_success": None, "recorded_policy_actions": len(action_rows),
                "retained_files": files, "classification": "Incomplete infrastructure/attempt evidence; not a completed task failure or success."}
    result = runner.verify_episode(directory, case, protocol)
    receipt = read(directory / "receipt.json")
    files = {}
    for name, expected in receipt["files"].items():
        item = binding(directory / name)
        require(item["sha256"] == expected, "Receipt file changed: " + name)
        files[name] = item
    calls = read(directory / "calls.json")
    nfe = 0
    for reference in calls:
        call = read(directory / reference["record"])
        require(call["status"] == "complete" and call["forward_returned"] is True
                and call["action_head_nfe"] == 4 and len(call["quantized_module_calls"]) == 300
                and all(value > 0 for value in call["quantized_module_calls"].values()),
                "Real model execution evidence differs")
        nfe += call["action_head_nfe"]
    arena = [json.loads(line) for line in (directory / "arena_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    initialization = read(directory / "arena_initialization.json")
    require(len(arena) == result["steps"] + 1 and initialization["native_warnings"] == []
            and all(all(number == 0 for number in row["warnings"]) for row in arena),
            "Incomplete arena trace or warning")
    return {**common, "evidence_status": "original_sealed_episode_verified", "completed": True,
            "official_success": result["success"], "task": result["task"], "steps": result["steps"],
            "horizon": result["horizon"], "terminated": result["terminated"], "truncated": result["truncated"],
            "inference_calls": len(calls), "actual_action_head_nfe": nfe,
            "all_300_quantized_modules_executed_per_call": True,
            "policy_inference_seconds": result["policy_inference_seconds"], "episode_wall_seconds": result["wall_seconds"],
            "all_observed_arena_warnings_zero": True, "arena_bytes": 268435456,
            "maximum_used_arena_bytes": max(row["maxuse_arena_bytes"] for row in arena),
            "receipt": binding(directory / "receipt.json"), "receipt_files_verified": files,
            "numeric_trace_verification": "Original strict verifier checked executed action/state trajectories, latest observation hashes, actual model chunks, noise/RNG hashes, fixed-seven replans and terminal/timing evidence.",
            "model_run_record": result["model_info"]["run_record"],
            "classification": "One completed original episode in its own resource protocol; indexed here without copying or resealing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vlajepa-task-coverage-20261016.json")
    args = parser.parse_args()
    destination = args.output.resolve()
    require(destination.is_relative_to((ROOT / "artifacts").resolve()), "Output must stay under artifacts")
    require(not destination.exists(), "Prior audit is immutable; choose a new output name")
    from embodied_research.runtime import configure_platform
    configure_platform()
    import run_vlajepa_isolated as isolated
    import run_vlajepa_long_horizon as long_horizon

    old_output, new_output = isolated.DEFAULT_OUTPUT, long_horizon.DEFAULT_OUTPUT
    old_runner, new_runner = isolated.configured_runner(), long_horizon.configured_runner()
    old_study, new_study = old_runner.verify_study(old_output, partial=True), new_runner.verify_study(new_output, partial=True)
    workers = isolated.verify_workers(old_runner, old_output, old_study)
    require(old_study["planned"] == 4 and old_study["completed"] == old_study["successes"] == 3
            and not old_study["complete"], "Expected preserved three-complete/one-unsealed isolated matrix")
    require(new_study["planned"] == 1, "Long-task protocol denominator must remain one")
    for worker in workers:
        require(not process_is_alive(worker["pid"], worker["process_create_time"]), "Isolated worker still running")
    old_protocol, old_proof = protocol_evidence(old_runner, old_output, old_study)
    new_protocol, new_proof = protocol_evidence(new_runner, new_output, new_study)
    new_runs = new_proof["run_evidence"]["records"]
    require(bool(new_runs) and all(row["effective_status"] in {
        "complete", "stopped_after_requested_limit", "blocked_preflight", "error", "stopped_memory_guard"
    } for row in new_runs), "Long-task execution has not reached an auditable terminal state")
    require(old_proof["resource_budget"] == {"minimum_available_gib": 3.5, "maximum_rss_gib": 5.5,
            "minimum_runtime_available_gib": 0.5, "sampling_seconds": 1}, "Isolated budget changed")
    require(new_proof["resource_budget"] == {"minimum_available_gib": 5.75, "maximum_rss_gib": 6.0,
            "minimum_runtime_available_gib": 0.5, "sampling_seconds": 1}, "Single-task budget changed")
    selected_suites = ("libero_spatial", "libero_object", "libero_goal")
    cases = []
    for suite in selected_suites:
        case = next(case for case in old_protocol["spec"]["cases"] if case["suite"] == suite)
        row = episode_index(old_runner, old_output, old_protocol, case)
        require(row["completed"] and row["official_success"] is True, "Expected isolated successful episode missing")
        cases.append(row)
    require(len(new_protocol["spec"]["cases"]) == 1, "Expected one long-task case")
    long_case = new_protocol["spec"]["cases"][0]
    require((long_case["suite"], long_case["task_id"], long_case["init_state_id"], long_case["seed"])
            == ("libero_10", 1, 41, 1000), "Long-task identity changed")
    cases.append(episode_index(new_runner, new_output, new_protocol, long_case))
    retained_history, historical_bindings = [], {}

    def verify_retained_bindings(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                name, expected = value["path"], value["sha256"]
                candidate = (ROOT / name).resolve()
                require(candidate.is_relative_to(ROOT) and candidate.is_file(), "Historical binding is unsafe or missing")
                actual = binding(candidate)
                require(actual["sha256"] == expected, "Historical evidence changed: " + name)
                historical_bindings[name] = actual
            for child in value.values():
                verify_retained_bindings(child)
        elif isinstance(value, list):
            for child in value:
                verify_retained_bindings(child)

    for name in ("artifacts/vlajepa-compact-audit-20261016.json", "artifacts/vlajepa-isolated-audit-20261016.json",
                 "artifacts/vlajepa-process-isolation-consistency-20261016.json"):
        retained_history.append(binding(ROOT / name))
        verify_retained_bindings(read(ROOT / name))
    old_interrupted = old_output / "episodes/vlajepa_isolated__libero_10__1__41__clean__native4_chunk7"
    require(not (old_interrupted / "receipt.json").exists(), "Old interrupted task unexpectedly resealed")
    retained_old_attempt = {path.name: binding(path) for path in sorted(old_interrupted.iterdir()) if path.is_file()}
    require("torch" not in sys.modules, "Read-only audit unexpectedly imported Torch")
    result = {
        "schema": "feedbackbench.cross_protocol_task_evidence_index.v1",
        "status": "verified_cross_protocol_index", "audited_at": datetime.now(timezone.utc).isoformat(),
        "audit_source": binding(__file__), "torch_imported": False,
        "scope": "Index existing sealed task evidence under two distinct resource protocols, or explicitly retain incomplete long-task evidence. This is not a new cohort or a pooled benchmark.",
        "cross_budget_protocols": True, "single_protocol_four_episode_completed": False,
        "combined_success_rate": None, "combined_success_rate_permitted": False,
        "indexed_tasks": 4, "tasks_with_completed_original_evidence": sum(row["completed"] for row in cases),
        "protocols": {"isolated_5_5_gib": old_proof, "selected_long_task_6_gib": new_proof},
        "isolated_worker_proof": [{key: row[key] for key in ("path", "sha256", "pid", "process_create_time",
                                                               "case_id", "status", "run_records")} for row in workers],
        "raw_worker_statuses_are_historical": True,
        "task_evidence": cases, "previous_audit_bindings": retained_history,
        "historical_read_only_bindings_verified": list(historical_bindings.values()),
        "retained_original_isolated_long_attempt": retained_old_attempt,
        "original_episodes_copied_or_resealed": False,
        "limitations": ["The four-case 5.5 GiB protocol remains incomplete even if the separate 6 GiB selected task completes.",
                        "Do not report four of four, 100 percent, or any combined success rate across these different budgets.",
                        "The selected long task follows a known memory interruption; it is a targeted resource diagnostic, not an independent held-out trial.",
                        "Earlier blocked, interrupted and unsealed attempts remain preserved and linked.",
                        "Exact results are known-task feasibility evidence, not SOTA, full LIBERO, quantization quality, unseen generalization or physical robot validation.",
                        "Simulation pauses for inference; rollout wall time excludes model loading, and one-second memory sampling does not establish instantaneous peaks or stable throughput."]}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "tasks_with_completed_original_evidence": result["tasks_with_completed_original_evidence"],
                      "single_protocol_four_episode_completed": False, "combined_success_rate": None,
                      "long_protocol_complete": new_study["complete"], "torch_imported": False}))


if __name__ == "__main__":
    main()
