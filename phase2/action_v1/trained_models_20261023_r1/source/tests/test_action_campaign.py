"""Protect whole-demo splits, fixed denominators and sealed recovery evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import action_campaign_common as common
import prepare_action_campaign as prepare
import report_action_campaign as report
import run_action_campaign as supervisor


def test_schedule_is_balanced_reproducible_and_has_no_validation_leakage():
    demos = [{"id": f"{task}_{index}", "task": task, "split": "train" if index < 3 else "validation", "frame_count": 100}
             for task in prepare.TASKS for index in range(5)]
    a = prepare.make_schedule(demos, 17, steps=200)
    assert a == prepare.make_schedule(demos, 17, steps=200)
    assert a != prepare.make_schedule(demos, 23, steps=200)
    assert set(a["task_sample_counts"].values()) == {50}
    train = {d["id"] for d in demos if d["split"] == "train"}
    val = {d["id"] for d in demos if d["split"] == "validation"}
    assert all(i["demo_id"] in train and 0 <= i["observation_index"] < 99 for i in a["train"])
    assert all(i["demo_id"] in val for i in a["validation"])
    assert len(a["validation"]) == 24


def test_all_methods_have_identical_prospective_case_matrix():
    evaluation = prepare.make_evaluation()
    assert len(evaluation["cases"]) == 280
    assert len({case["case_id"] for case in evaluation["cases"]}) == 280
    groups = []
    for method in evaluation["methods"]:
        groups.append({(c["suite"], c["initial_state_id"], c["condition"]) for c in evaluation["cases"]
                       if c["method_id"] == method["method_id"]})
    assert all(group == groups[0] for group in groups)
    assert len(groups[0]) == 40


def test_contrasts_wait_for_all_pairs_and_do_not_drop_failures():
    rows = {}
    for case in prepare.make_evaluation()["cases"]:
        rows[(case["suite"], case["initial_state_id"], case["condition"], case["method_id"])] = case["method_id"] == "frozen"
    result = report.paired_summary(rows, "lora", "frozen", "clean")
    assert result["estimate"] == -1
    assert result["descriptive_task_bootstrap_95"] == [-1, -1]
    assert result["task_groups"] == 4
    rows.pop(("libero_spatial", 5, "clean", "lora_s17"))
    assert report.paired_summary(rows, "lora", "frozen", "clean")["estimate"] is None


def make_receipt(path, *, step=100):
    path.mkdir(parents=True)
    binding = {"protocol_sha256": "protocol", "run_id": "lora_s17"}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    files = {}
    for name in ("adapter.npz", "optimizer.npz", "rng.npz", "state.json", "history.json"):
        (path / name).write_bytes(name.encode())
        files[name] = common.digest(path / name)
    receipt = {"schema": "feedbackbench.action_adaptation_checkpoint.v1", "completed_steps": step,
               "binding": binding, "binding_sha256": hashlib.sha256(canonical(binding).encode()).hexdigest(), "files": files}
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    (path / "receipt.json").write_text(json.dumps(receipt))


def test_checkpoint_tampering_and_cross_run_are_rejected(tmp_path):
    make_receipt(tmp_path / "step_000100")
    assert supervisor.checkpoint_receipt(tmp_path / "step_000100", "protocol", "lora_s17")["completed_steps"] == 100
    with pytest.raises(ValueError, match="another"):
        supervisor.checkpoint_receipt(tmp_path / "step_000100", "protocol", "lora_s23")
    (tmp_path / "step_000100" / "optimizer.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        supervisor.checkpoint_receipt(tmp_path / "step_000100", "protocol", "lora_s17")


def test_incomplete_newer_checkpoint_is_never_resumed(tmp_path):
    complete = tmp_path / "attempt_001/checkpoints/step_000100"
    partial = tmp_path / "attempt_001/checkpoints/step_000200.partial_uuid"
    make_receipt(complete)
    make_receipt(partial, step=200)
    assert supervisor.latest_checkpoint(tmp_path, "protocol", "lora_s17") == complete


def test_sealed_failed_episode_remains_failed_and_immutable(tmp_path):
    binding = {"case_id": "one", "protocol_sha256": "protocol"}
    common.write_json(tmp_path / "report.json", {"status": "complete", "official_success": False})
    common.seal(tmp_path, binding)
    common.verify_seal(tmp_path, binding)
    assert common.read(tmp_path / "report.json")["official_success"] is False
    common.write_json(tmp_path / "report.json", {"status": "complete", "official_success": True})
    with pytest.raises(ValueError, match="changed"):
        common.verify_seal(tmp_path, binding)


def test_paths_cannot_escape_evidence_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        common.resolve("../another_project", tmp_path)


def test_completed_failure_without_receipt_must_not_be_retried(tmp_path):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    common.write_json(attempt / "report.json", {"status": "complete", "official_success": False})
    with pytest.raises(RuntimeError, match="never rerun"):
        supervisor.refuse_unsealed_complete(tmp_path)


def test_report_cannot_count_a_nonterminal_or_differently_bound_case():
    case = {"suite": "libero_spatial"}
    protocol = {"evaluation": {"horizons": {"libero_spatial": 280}}}
    result = {"status": "complete", "case": case, "official_success": False,
              "steps": 280, "horizon": 280, "terminated": False, "truncated": True,
              "protocol_sha256": "protocol", "trained_manifest_sha256": "trained"}
    report.verify_episode_report(result, case, protocol, "protocol", "trained")
    for changes in ({"steps": 1, "truncated": False}, {"protocol_sha256": "other"}, {"horizon": 10}):
        with pytest.raises(ValueError, match="denominator"):
            report.verify_episode_report({**result, **changes}, case, protocol, "protocol", "trained")
