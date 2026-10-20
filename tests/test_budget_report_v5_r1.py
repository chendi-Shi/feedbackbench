"""Reporting-only probability roundoff regression; no trained model loading."""
from copy import deepcopy
import importlib.util
import json
import sys
from types import ModuleType

import pytest

from test_budget_report import SCRIPTS, campaign, common, _seal, _training_complete


def load_report(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
try:
    original = load_report("budget_report_preserved_r1_test", "budget_report.py")
    revised = load_report("budget_report_r1_test", "budget_report_v5_r1.py")
finally:
    sys.path.pop(0)


@pytest.fixture
def random_binding(campaign):
    path, protocol = campaign
    _training_complete(path, protocol)
    relative = "trained/full_seed0/selector.pt"
    method = {"method_id": "random_seed_0", "family": "random_calibrated",
              "feature_mode": None, "seed": 0, "budget": None}
    binding = {"manifest_sha256": common.digest(path / "trained_manifest.json"),
               "selector": {"path": relative, "sha256": common.digest(path / relative)},
               "calibration_budget_probabilities": [0.0, 1.0000000000000004, 0.0]}
    return path, method, binding


def test_real_roundoff_case_accepted_without_mutation_and_original_still_rejects(random_binding):
    path, method, binding = random_binding
    before = deepcopy(binding)
    revised._validate_trained_binding(path, method, binding)
    assert binding == before
    assert binding["calibration_budget_probabilities"][1] > 1
    with pytest.raises(ValueError, match="probabilities must sum to one"):
        original._validate_trained_binding(path, method, binding)


@pytest.mark.parametrize("probabilities", [
    [0.0, 1.01, 0.0],
    [0.0, 1.0000000011, 0.0],
    [-1e-12, 1.0, 0.0],
    [float("nan"), 1.0, 0.0],
    [0.0, float("inf"), 0.0],
    [0.2, 0.2, 0.2],
    [0.5000000006, 0.5000000006, 0.0],
    [False, True, 0.0],
    [0.0, 1.0],
    (0.0, 1.0, 0.0),
    None,
])
def test_invalid_probabilities_still_rejected(random_binding, probabilities):
    path, method, binding = random_binding
    binding["calibration_budget_probabilities"] = probabilities
    with pytest.raises(ValueError):
        revised._validate_trained_binding(path, method, binding)


def test_valid_interior_distribution_is_accepted_unchanged(random_binding):
    path, method, binding = random_binding
    binding["calibration_budget_probabilities"] = [0.2, 0.3, 0.5]
    before = deepcopy(binding)
    original._validate_trained_binding(path, method, binding)
    revised._validate_trained_binding(path, method, binding)
    assert binding == before


@pytest.mark.parametrize("mutation", ["manifest_hash", "selector_path", "selector_hash", "checkpoint_bytes"])
def test_roundoff_does_not_bypass_frozen_binding_checks(random_binding, mutation):
    path, method, binding = random_binding
    if mutation == "manifest_hash":
        binding["manifest_sha256"] = "0" * 64
    elif mutation == "selector_path":
        binding["selector"]["path"] = "trained/full_seed1/selector.pt"
    elif mutation == "selector_hash":
        binding["selector"]["sha256"] = "0" * 64
    else:
        (path / binding["selector"]["path"]).write_bytes(b"tampered synthetic checkpoint")
    with pytest.raises(ValueError, match="binding mismatch|selector path/hash"):
        revised._validate_trained_binding(path, method, binding)


def test_completed_random_failure_is_restored_to_denominator_without_rewriting_evidence(campaign):
    path, protocol = campaign
    method = {"method_id": "random_seed_0", "family": "random_calibrated",
              "feature_mode": None, "seed": 0, "budget": None}
    case = {**protocol["evaluation"]["cases"][0], "case_id": "eval:a:41:random_seed_0",
            "method_id": method["method_id"]}
    protocol["evaluation"]["methods"].append(method)
    protocol["evaluation"]["cases"].append(case)
    protocol["campaign_fingerprint"] = common.fingerprint(protocol)
    common.write_json(path / "protocol.json", protocol)
    directory = _seal(path, protocol, "evaluation", case, success=False)
    metadata = common.read_json(directory / "metadata.json")
    metadata["trained_binding"]["calibration_budget_probabilities"] = [0.0, 1.0000000000000004, 0.0]
    common.write_json(directory / "metadata.json", metadata)
    (directory / "receipt.json").unlink()
    common.seal_episode(directory)
    before = {item.name: item.read_bytes() for item in directory.iterdir()}
    old = original.generate_report(path)
    new = revised.generate_report(path)
    assert old["progress"]["evaluation"]["verified_complete"] == 0
    assert len(old["progress"]["evaluation"]["invalid"]) == 1
    assert new["progress"]["evaluation"]["verified_complete"] == 1
    assert new["progress"]["evaluation"]["completed_task_failures"] == 1
    assert new["progress"]["evaluation"]["invalid"] == []
    assert new["progress"]["evaluation"]["planned"] == old["progress"]["evaluation"]["planned"]
    assert new["statistical_contract"] == old["statistical_contract"]
    assert new["status"] == "partial" and not new["claims"]["full_planned_study_complete"]
    assert {item.name: item.read_bytes() for item in directory.iterdir()} == before


def test_core_cli_delegates_all_arguments_to_checked_entrypoint(campaign, monkeypatch):
    path, protocol = campaign
    original_bytes = b"previous report evidence must remain unchanged\n"
    (path / "report.json").write_bytes(original_bytes)
    calls = []
    checked_entrypoint = ModuleType("run_budget_report_r1")
    checked_entrypoint.main = lambda argv: calls.append(argv) or 17
    monkeypatch.setitem(sys.modules, "run_budget_report_r1", checked_entrypoint)
    arguments = ["--campaign", str(path), "--output", str(path / "report.json")]
    assert revised.main(arguments) == 17
    assert calls == [arguments]
    assert (path / "report.json").read_bytes() == original_bytes
    assert not (path / "report_r1.json").exists()
