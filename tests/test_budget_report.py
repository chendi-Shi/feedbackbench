"""Sealed synthetic receipts test aggregation without any simulation or model."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import budget_campaign_common as common
    SPEC = importlib.util.spec_from_file_location("budget_report_test", SCRIPTS / "budget_report.py")
    report = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(report)
finally:
    sys.path.pop(0)


@pytest.fixture
def campaign(tmp_path):
    methods = [{"method_id": "fixed_2", "family": "fixed", "feature_mode": None, "seed": None, "budget": 2},
               {"method_id": "full_seed_0", "family": "learned", "feature_mode": "full", "seed": 0, "budget": None}]
    collection = [{"case_id": "collect:train:0", "task_uid": "train", "split": "train", "duplicate_group": "train_group",
                   "initial_state_id": 0}]
    cases = []
    for task in ("a", "b"):
        for initial in (41, 43):
            for method in methods:
                cases.append({"case_id": f"eval:{task}:{initial}:{method['method_id']}", "task_uid": task,
                              "initial_state_id": initial, "split": "test", "duplicate_group": task,
                              "method_id": method["method_id"]})
    protocol = {"collection": {"cases": collection}, "evaluation": {"cases": cases, "methods": methods},
                "training": {"seeds": [0], "feature_modes": ["full"]}}
    protocol["campaign_fingerprint"] = common.fingerprint(protocol)
    common.write_json(tmp_path / "protocol.json", protocol)
    return tmp_path, protocol


def _seal(campaign, protocol, phase, case, *, success=True, decisions=2, nfe=4, latency=6, case_override=None):
    directory = campaign / phase / common.case_directory(case)
    directory.mkdir(parents=True)
    method = next(row for row in protocol["evaluation"]["methods"] if row["method_id"] == case["method_id"]) if phase == "evaluation" else None
    metadata = {"case_id": case["case_id"], "task_key": case["task_uid"], "case": case_override or case,
                "phase": phase, "split": case["split"], "method": method,
                "campaign_fingerprint": protocol["campaign_fingerprint"]}
    if phase == "evaluation":
        if not (campaign / "trained_manifest.json").exists():
            _training_complete(campaign, protocol)
        manifest = common.read_json(campaign / "trained_manifest.json")
        selector = None
        if method["family"] != "fixed":
            variant = method["feature_mode"] if method["family"] == "learned" else "full"
            relative = f"trained/{variant}_seed{method['seed']}/selector.pt"
            selector = {"path": relative, "sha256": manifest["files"][relative]["sha256"]}
        metadata["trained_binding"] = {"manifest_sha256": common.digest(campaign / "trained_manifest.json"),
                                       "selector": selector, "calibration_budget_probabilities": None}
    result = {"complete": True, "success": success, "terminated": success, "truncated": not success,
              "steps": decisions * 10, "decisions": decisions, "actual_nfe": nfe,
              "decision_wall_seconds": latency, "episode_wall_seconds": latency + 2,
              "controller_overhead_seconds": .2, "budget_frequencies": {"2": decisions, "4": 0, "10": 0}}
    common.write_json(directory / "metadata.json", metadata)
    common.write_json(directory / "result.json", result)
    (directory / "trajectory.npz").write_bytes(b"synthetic receipt inventory; not loaded")
    (directory / "samples.npz").write_bytes(b"synthetic receipt inventory; not loaded")
    common.seal_episode(directory)
    return directory


def _training_complete(campaign, protocol):
    checkpoint = campaign / "trained/full_seed0/selector.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint; never deserialized")
    summary = {"status": "complete", "binding": {"campaign_fingerprint": protocol["campaign_fingerprint"]},
               "models": [{"variant": "full", "seed": 0, "checkpoint": "full_seed0/selector.pt",
                           "checkpoint_sha256": common.digest(checkpoint)}]}
    common.write_json(campaign / "trained/summary.json", summary)
    common.write_json(campaign / "trained_manifest.json", {
        "schema": "feedbackbench.budget_trained_manifest.v1", "campaign_fingerprint": protocol["campaign_fingerprint"],
        "controller_count": 1, "test_opened_after_freeze": True,
        "files": {path.relative_to(campaign).as_posix(): {"sha256": common.digest(path), "bytes": path.stat().st_size}
                  for path in (campaign / "trained").rglob("*") if path.is_file()}})


def _complete(campaign, protocol):
    for case in protocol["collection"]["cases"]:
        _seal(campaign, protocol, "collection", case)
    for case in protocol["evaluation"]["cases"]:
        _seal(campaign, protocol, "evaluation", case,
              success=case["task_uid"] == "a", decisions=2 if case["task_uid"] == "a" else 10,
              nfe=4 if case["task_uid"] == "a" else 100, latency=6 if case["task_uid"] == "a" else 90)


def test_empty_campaign_is_partial_without_confidence_intervals(campaign):
    path, protocol = campaign
    result = report.generate_report(path)
    assert result["status"] == "partial"
    assert result["progress"]["evaluation"]["missing"] == 8
    assert all(row["full_study_ci95"] is None for row in result["methods"] + result["paired_differences"])
    assert all(row["point_estimates"]["success"] is None for row in result["methods"])
    assert not (path / "report.json").exists()


def test_completed_failures_remain_in_denominator_and_task_weighted_means(campaign):
    path, protocol = campaign
    _complete(path, protocol)
    result = report.generate_report(path)
    assert result["status"] == "complete"
    for method in result["methods"]:
        assert method["completed_successes"] == 2
        assert method["completed_task_failures"] == 2
        assert method["point_estimates"]["success"] == .5
        # Equal tasks, not pooled 208 NFE / 24 decisions.
        assert method["point_estimates"]["nfe_per_decision"] == 6
        assert method["point_estimates"]["latency_seconds_per_decision"] == 6
        assert method["full_study_ci95"]["success"] == [0, 1]
    paired = result["paired_differences"][0]
    assert paired["paired_episodes"] == 4
    assert paired["both_failed_pairs_included"] == 2
    assert all(value == 0 for value in paired["point_estimates"].values())
    assert all(value == [0, 0] for value in paired["full_study_ci95"].values())


def test_partial_method_never_gets_full_study_ci_even_when_another_method_is_complete(campaign):
    path, protocol = campaign
    for case in protocol["evaluation"]["cases"]:
        if case["method_id"] == "fixed_2" or case["initial_state_id"] == 41:
            _seal(path, protocol, "evaluation", case)
    result = report.generate_report(path)
    assert result["status"] == "partial"
    assert [row["complete"] for row in result["methods"]] == [True, False]
    assert all(row["full_study_ci95"] is None for row in result["methods"])
    assert result["paired_differences"][0]["paired_episodes"] == 2
    assert result["paired_differences"][0]["full_study_ci95"] is None


def test_tampered_receipt_is_excluded_and_prevents_complete_claim(campaign):
    path, protocol = campaign
    _complete(path, protocol)
    case = protocol["evaluation"]["cases"][0]
    directory = path / "evaluation" / common.case_directory(case)
    (directory / "trajectory.npz").write_bytes(b"changed")
    result = report.generate_report(path)
    assert result["status"] == "partial"
    assert result["progress"]["evaluation"]["verified_complete"] == 7
    assert len(result["progress"]["evaluation"]["invalid"]) == 1
    assert all(row["full_study_ci95"] is None for row in result["methods"])


def test_sealed_wrong_initial_state_metadata_is_rejected(campaign):
    path, protocol = campaign
    case = protocol["evaluation"]["cases"][0]
    _seal(path, protocol, "evaluation", case, case_override={**case, "initial_state_id": 1})
    result = report.generate_report(path)
    assert result["progress"]["evaluation"]["verified_complete"] == 0
    assert "metadata differs" in result["progress"]["evaluation"]["invalid"][0]["error"]


def test_resource_attempt_never_becomes_task_failure(campaign):
    path, protocol = campaign
    case = protocol["evaluation"]["cases"][0]
    attempt = path / "attempts/evaluation" / common.case_directory(case) / "0001"
    common.write_json(attempt / "failure.json", {"error_type": "ResourceWait", "error": "process/system memory guard", "steps": 11})
    result = report.generate_report(path)
    assert result["attempts"]["failed_unsealed_attempts"] == 1
    assert result["attempts"]["resource_or_deadline_attempts"] == 1
    assert result["progress"]["evaluation"]["completed_task_failures"] == 0
    assert result["attempts"]["included_in_success_denominator"] is False


def test_supervisor_killed_worker_is_reported_when_worker_could_not_write_failure(campaign):
    path, protocol = campaign
    case = protocol["evaluation"]["cases"][0]
    attempt = path / "attempts/evaluation" / common.case_directory(case) / "001"
    attempt.mkdir(parents=True)
    common.write_json(path / "runs" / f"evaluation-{common.case_directory(case)}-001.json",
                      {"case_id": case["case_id"], "phase": "evaluation", "stop_reason": "resource_guard", "exit_code": 1})
    result = report.generate_report(path)
    assert result["attempts"]["failed_unsealed_attempts"] == 1
    assert result["attempts"]["resource_or_deadline_attempts"] == 1
    assert result["progress"]["evaluation"]["completed_task_failures"] == 0


def test_all_episodes_without_frozen_training_manifest_stay_partial(campaign):
    path, protocol = campaign
    _complete(path, protocol)
    (path / "trained_manifest.json").unlink()
    result = report.generate_report(path)
    assert result["status"] == "partial"
    assert result["progress"]["evaluation"]["complete"] is False
    assert result["progress"]["evaluation"]["verified_complete"] == 0
    assert result["training"]["complete"] is False
    assert all(row["full_study_ci95"] is None for row in result["methods"])


@pytest.mark.parametrize("mutation", ["missing", "manifest_sha", "selector_path", "selector_sha", "random_probs"])
def test_sealed_evaluation_rejects_wrong_trained_binding(campaign, mutation):
    path, protocol = campaign
    case = next(case for case in protocol["evaluation"]["cases"] if case["method_id"] == "full_seed_0")
    directory = _seal(path, protocol, "evaluation", case)
    metadata = common.read_json(directory / "metadata.json")
    if mutation == "missing":
        metadata.pop("trained_binding")
    elif mutation == "manifest_sha":
        metadata["trained_binding"]["manifest_sha256"] = "0" * 64
    elif mutation == "selector_path":
        metadata["trained_binding"]["selector"]["path"] = "trained/full_seed1/selector.pt"
    elif mutation == "selector_sha":
        metadata["trained_binding"]["selector"]["sha256"] = "0" * 64
    else:
        metadata["trained_binding"]["calibration_budget_probabilities"] = [.5, .5, 0]
    common.write_json(directory / "metadata.json", metadata)
    (directory / "receipt.json").unlink()
    common.seal_episode(directory)
    result = report.generate_report(path)
    assert result["progress"]["evaluation"]["verified_complete"] == 0
    assert len(result["progress"]["evaluation"]["invalid"]) == 1


def test_fixed_baseline_requires_manifest_binding_but_no_selector(campaign):
    path, protocol = campaign
    case = protocol["evaluation"]["cases"][0]
    directory = _seal(path, protocol, "evaluation", case)
    metadata = common.read_json(directory / "metadata.json")
    metadata["trained_binding"]["selector"] = {"path": "trained/full_seed0/selector.pt", "sha256": "0" * 64}
    common.write_json(directory / "metadata.json", metadata)
    (directory / "receipt.json").unlink()
    common.seal_episode(directory)
    assert report.generate_report(path)["progress"]["evaluation"]["verified_complete"] == 0


def test_random_baseline_uses_matching_full_seed_and_valid_probability_vector(campaign):
    path, protocol = campaign
    _training_complete(path, protocol)
    method = {"family": "random_calibrated", "feature_mode": None, "seed": 0}
    relative = "trained/full_seed0/selector.pt"
    binding = {"manifest_sha256": common.digest(path / "trained_manifest.json"),
               "selector": {"path": relative, "sha256": common.digest(path / relative)},
               "calibration_budget_probabilities": [.2, .3, .5]}
    report._validate_trained_binding(path, method, binding)
    binding["calibration_budget_probabilities"] = [.5, .5, .5]
    with pytest.raises(ValueError, match="sum to one"):
        report._validate_trained_binding(path, method, binding)


def test_duplicate_groups_are_not_counted_as_independent_tasks():
    rows = [{"case": {"task_uid": task, "duplicate_group": group}, "metrics": {name: value for name in report.METRICS}}
            for task, group, value in [("a", "same", 1), ("b", "same", 1), ("c", "other", 0)]]
    result = report._statistics(rows, True)
    assert result["observed_tasks"] == 3
    assert result["observed_task_groups"] == 2
    assert result["point_estimates"]["success"] == .5


def test_single_group_cannot_produce_informative_task_bootstrap():
    rows = [{"case": {"task_uid": "a", "duplicate_group": "a"}, "metrics": dict.fromkeys(report.METRICS, 1)}]
    assert report._statistics(rows, True)["full_study_ci95"] is None


@pytest.mark.parametrize("field,value", [("success", 1), ("decisions", 0), ("actual_nfe", -1),
                                        ("decision_wall_seconds", float("nan")), ("episode_wall_seconds", float("inf"))])
def test_invalid_result_numbers_are_rejected(field, value):
    result = {"complete": True, "success": True, "terminated": True, "truncated": False,
              "steps": 1, "decisions": 1, "actual_nfe": 2, "decision_wall_seconds": 1,
              "episode_wall_seconds": 1, "controller_overhead_seconds": 0,
              "budget_frequencies": {"2": 1, "4": 0, "10": 0}}
    result[field] = value
    with pytest.raises(ValueError):
        report._validate_result(result)


def test_fingerprint_tampering_is_rejected(campaign):
    path, protocol = campaign
    protocol["training"]["seeds"] = [3]
    common.write_json(path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="fingerprint changed"):
        report.generate_report(path)


def test_cli_writes_requested_report_without_model_imports(campaign):
    path, protocol = campaign
    output = path / "review/report.json"
    code = ("import sys;sys.path.insert(0,sys.argv[1]);import budget_report;"
            "budget_report.main(['--campaign',sys.argv[2],'--output',sys.argv[3]]);"
            "assert not ({'torch','numpy','libero','mujoco','robosuite'} & sys.modules.keys())")
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS), str(path), str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "partial"
