"""Public reproduction v1: one separately registered development episode using local tiny checkpoints.

The production evaluator never accepts a foreign training protocol. This wrapper
explicitly verifies two protocols and excludes its single integration episode
from every formal research denominator. It launches no subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from action_campaign_common import ROOT, DEADLINE, deadline_reached, digest, read, resolve, seal, verify_protocol, write_json
import evaluate_action_adaptation as worker
from run_action_adaptation_probe import matching


CASE_ID = "libero_spatial_t0_i0_clean_lora_s17"
METHOD = {"method_id": "lora_s17", "family": "lora", "training_run_id": "lora_s17"}
LANGUAGE = "pick up the black bowl between the plate and the ramekin and place it on the plate"
CASE = {"case_id": CASE_ID, "method_id": "lora_s17", "suite": "libero_spatial", "task_id": 0,
        "task_uid": "libero_spatial:0", "language": LANGUAGE, "initial_state_id": 0, "condition": "clean"}
REQUIRED_SOURCES = {
    "scripts/reproduce_action_development.py",
    "scripts/evaluate_action_development_public.py", "scripts/evaluate_action_adaptation.py",
    "scripts/action_campaign_common.py", "scripts/budget_campaign_common.py", "scripts/budget_campaign_common_v5.py",
    "scripts/run_action_campaign.py", "scripts/run_action_adaptation_probe.py",
    "scripts/budget_startup_memory.py", "scripts/libero_arena_budget.py",
    "src/embodied_research/action_adaptation.py", "src/embodied_research/action_head_baseline.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py", "src/embodied_research/simulator.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    "tests/test_evaluate_action_adaptation.py", "tests/test_reproduce_action_development.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_file(record):
    path = resolve(record["path"])
    require(digest(path) == record["sha256"], f"Bound development input changed: {record['path']}")
    return path


def verify_snapshot(index_path, source_files):
    """Check historical bytes; current live files are not historical evidence."""
    index_path = Path(index_path).resolve()
    index = read(index_path)
    require(set(index) == set(source_files) and bool(index), "Source snapshot coverage differs")
    require(len(set(index.values())) == len(index), "Source snapshot maps duplicate files")
    expected_files = {"index.json"}
    for original, stored in index.items():
        require(isinstance(stored, str) and Path(stored).name == stored, "Unsafe source snapshot filename")
        path = resolve(stored, index_path.parent)
        require(digest(path) == source_files[original], f"Historical source snapshot changed: {original}")
        expected_files.add(stored)
    actual = {path.name for path in index_path.parent.iterdir()}
    require(actual == expected_files, "Source snapshot contains missing or unbound files")
    return {"index_sha256": digest(index_path), "verified_files": len(index)}


def verify_plan(protocol_path, expected_sha):
    protocol_path = Path(protocol_path).resolve()
    campaign = protocol_path.parent
    require(campaign.is_relative_to((ROOT / "outputs/action_adaptation").resolve()), "Development campaign escapes owned area")
    protocol = verify_protocol(protocol_path, expected_sha)
    require(protocol.get("development_only") is True and protocol.get("excluded_from_formal_study") is True
            and protocol.get("study_result") is False
            and protocol.get("purpose_id") == "action_development_public_v1",
            "This entrypoint accepts only the separately registered development experiment")
    require(REQUIRED_SOURCES.issubset(protocol["source_files"]), "Development evaluator dependencies are not all bound")
    evaluation = protocol["evaluation"]
    require(evaluation["methods"] == [METHOD] and evaluation["cases"] == [CASE]
            and evaluation.get("official_horizon") == 280, "Development evaluation is restricted to one exact case")
    case, method = worker.select_case(protocol, CASE_ID)
    require(protocol["resources"]["torch_threads"] == 2 and protocol["resources"]["interop_threads"] == 1
            and protocol["resources"]["evaluation_max_seconds"] == 7200, "Development resource limits differ")
    current_index = bound_file(protocol["source_snapshot_index"])
    require(current_index == campaign / "source_snapshot/index.json", "Wrong development source snapshot")
    current_snapshot = verify_snapshot(current_index, protocol["source_files"])
    history = protocol["historical_training"]
    training_path = bound_file(history["protocol"])
    training = read(training_path)
    require(training.get("schema") == "feedbackbench.action_adaptation_train.v1" and training.get("frozen") is True
            and training.get("development_only") is True and training.get("study_result") is False
            and training.get("deadline_utc") == DEADLINE,
            "Historical training is not the frozen development experiment")
    require(set(training["runs"]) == {"lora_s17", "output_head_s17"}
            and all(run["steps"] == 2 and run["initialization_seed"] == 17 for run in training["runs"].values()),
            "Expected the two-step development checkpoints")
    require(protocol["runs"] == training["runs"] and protocol["model"] == training["model"]
            and protocol["data"] == training["data"], "Development evaluation changed historical training inputs")
    history_index = bound_file(history["source_snapshot_index"])
    require(history_index == training_path.parent / "source_snapshot/index.json", "Wrong historical source snapshot")
    historical_snapshot = verify_snapshot(history_index, training["source_files"])
    manifest_path = bound_file(history["trained_manifest"])
    require(manifest_path == training_path.parent / "trained_manifest.json", "Wrong historical trained manifest")
    trained = worker.verify_trained_manifest(manifest_path, history["trained_manifest"]["sha256"], training,
                                            history["protocol"]["sha256"])
    audit = {"evaluation_protocol_sha256": expected_sha, "training_protocol_sha256": history["protocol"]["sha256"],
             "trained_manifest_sha256": history["trained_manifest"]["sha256"],
             "historical_source_snapshot": historical_snapshot, "evaluation_source_snapshot": current_snapshot,
             "historical_source_verification": "original snapshot bytes against original protocol; never substituted by live files",
             "excluded_from_formal_study": True, "development_only": True, "study_result": False}
    return protocol, case, method, trained, audit


def require_new_attempt(campaign, output, protocol_sha):
    """Allow only the live supervisor's precreated log directory, never a retry."""
    require(not output.exists(), "Development evaluation was already attempted; preserve and diagnose it")
    evaluation = campaign / "evaluation"
    require(evaluation.exists(), "The live exclusive supervisor must create the launch directory before this worker starts")
    case_directory = evaluation / CASE_ID
    launch = output.with_name(output.name + ".launch")
    require(set(evaluation.iterdir()) == {case_directory} and case_directory.is_dir()
            and set(case_directory.iterdir()) == {launch} and launch.is_dir(),
            "Development evaluation was already attempted or contains another case")
    files = {path.name for path in launch.iterdir()}
    require({"stdout.log", "stderr.log"}.issubset(files) and files.issubset({"stdout.log", "stderr.log", "resources.jsonl"})
            and all(path.is_file() for path in launch.iterdir()), "Development launch is not a fresh supervisor log")
    state = read(campaign / "status.json")
    lock = read(ROOT / "outputs/action_adaptation/worker.lock")
    require(state.get("state") in ("launching", "running") and state.get("protocol_sha256") == protocol_sha
            and state.get("current") == output.relative_to(campaign).as_posix(), "Supervisor launch binding differs")
    require(lock.get("owner") == state.get("owner") and resolve(lock.get("campaign", "")) == campaign
            and matching(state["owner"]) is not None, "Development launch owner is not the live exclusive supervisor")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--case-id", choices=[CASE_ID], default=CASE_ID)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not deadline_reached(), "Original development deadline has expired; a separately registered new protocol is required")
    protocol, case, method, trained, audit = verify_plan(args.protocol, args.protocol_sha256)
    campaign = args.protocol.resolve().parent
    output = args.output.resolve()
    require(output == campaign / "evaluation" / CASE_ID / "attempt_001", "Development output is fixed and never retried")
    require_new_attempt(campaign, output, args.protocol_sha256)
    args.trained_manifest_sha256 = audit["trained_manifest_sha256"]
    require(not deadline_reached(), "Original development deadline expired during preflight; no attempt may start")
    session = worker.EvaluationSession(output, 7200)
    result = None
    try:
        write_json(output / "development_binding.json", audit)
        session.begin()
        result = worker.execute_episode(args, protocol, case, method, trained, session)
        require(result["status"] == "complete" and result["horizon"] == 280 and result["case"] == CASE
                and result["protocol_sha256"] == args.protocol_sha256
                and result["trained_manifest_sha256"] == args.trained_manifest_sha256
                and result["NFE"] == 10 * result["decision_count"], "Development execution contract changed")
        result.update(audit)
        result["purpose"] = "Real-model/renderer/checkpoint/NFE/sealing integration only; no efficacy claim"
        metadata = read(output / "metadata.json")
        metadata.update(audit)
        write_json(output / "metadata.json", metadata)
        write_json(output / "report.json", result)
        session.finish("complete", official_success=result["official_success"], steps=result["steps"],
                       excluded_from_formal_study=True)
        receipt_sha = seal(output, {"evaluation_protocol_sha256": args.protocol_sha256,
                                   "training_protocol_sha256": audit["training_protocol_sha256"],
                                   "trained_manifest_sha256": audit["trained_manifest_sha256"],
                                   "case_id": CASE_ID, "excluded_from_formal_study": True})
        print(json.dumps({"status": "complete", "receipt_sha256": receipt_sha, "excluded_from_formal_study": True}))
        return 0
    except BaseException as error:
        if not (output / "receipt.json").exists():
            try:
                session.finish("needs_attention", error=repr(error), traceback=traceback.format_exc(),
                               excluded_from_formal_study=True, may_retry=False)
            except BaseException as persistence_error:
                print(json.dumps({"error": repr(error), "persistence_error": repr(persistence_error)}), flush=True)
        print(json.dumps({"status": "needs_attention", "error": repr(error), "may_retry": False}), flush=True)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform
    configure_platform()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    raise SystemExit(main())
