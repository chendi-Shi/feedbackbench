"""Tiny sealed fixtures verify the reporting amendment entry point without models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_budget_report import SCRIPTS, common, _seal


sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("reporting_revision_entrypoint_tested", SCRIPTS / "run_budget_report_r1.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


@pytest.fixture
def revision(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    campaign = repository / "outputs/learned_compute/v5"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "frozen_source.py").write_text("# synthetic frozen experiment source\n", encoding="utf-8")
    for name in ("run_budget_report_r1.py", "budget_report_v5_r1.py"):
        (scripts / name).write_bytes((SCRIPTS / name).read_bytes())
    monkeypatch.setattr(entry, "__file__", str(scripts / "run_budget_report_r1.py"))
    monkeypatch.setattr(entry.budget_report_v5_r1, "__file__", str(scripts / "budget_report_v5_r1.py"))
    method = {"method_id": "fixed_2", "family": "fixed", "feature_mode": None, "seed": None, "budget": 2}
    collect = {"case_id": "collect:train:0", "task_uid": "train", "split": "train",
               "duplicate_group": "train", "initial_state_id": 0}
    evaluate = {"case_id": "eval:test:41:fixed_2", "task_uid": "test", "split": "test",
                "duplicate_group": "test", "initial_state_id": 41, "method_id": "fixed_2"}
    protocol = {"collection": {"cases": [collect]}, "evaluation": {"cases": [evaluate], "methods": [method]},
                "training": {"seeds": [0], "feature_modes": ["full"]},
                "sources": {"scripts/frozen_source.py": common.digest(scripts / "frozen_source.py")}}
    protocol["campaign_fingerprint"] = common.fingerprint(protocol)
    common.write_json(campaign / "protocol.json", protocol)
    directories = {"collection": _seal(campaign, protocol, "collection", collect),
                   "evaluation": _seal(campaign, protocol, "evaluation", evaluate)}
    evidence = {"trained_manifest_sha256": common.digest(campaign / "trained_manifest.json")}
    for phase, case in (("collection", collect), ("evaluation", evaluate)):
        evidence[phase] = [{"case_id": case["case_id"], "directory": directories[phase].name,
                            "receipt_sha256": common.digest(directories[phase] / "receipt.json")}]
    amendment = campaign / "reporting_r1"
    common.write_json(amendment / "pre_revision_evidence.json", evidence)
    common.write_json(amendment / "previous_report.json", {"status": "preserved synthetic report"})
    revision = {"schema": "feedbackbench.reporting_revision.v1", "revision_id": "v5-report-r1",
                "base_protocol_sha256": common.digest(campaign / "protocol.json"),
                "base_campaign_fingerprint": protocol["campaign_fingerprint"],
                "sources": {f"scripts/{name}": common.digest(scripts / name)
                            for name in ("run_budget_report_r1.py", "budget_report_v5_r1.py")},
                "preserved_files": {path.name: {"sha256": common.digest(path), "bytes": path.stat().st_size}
                                    for path in amendment.iterdir()}}
    revision["revision_fingerprint"] = common.canonical_hash(revision)
    common.write_json(amendment / "protocol.json", revision)
    return {"campaign": campaign, "repository": repository, "revision": revision,
            "directories": directories, "amendment": amendment}


def test_valid_revision_checks_and_report_provenance(revision):
    campaign = revision["campaign"]
    assert entry.verify_reporting_revision(campaign) == revision["revision"]
    report = entry.generate_report(campaign)
    assert report["reporting_revision"]["fingerprint"] == revision["revision"]["revision_fingerprint"]
    assert report["reporting_revision"]["changed_after_test_started"] is True
    assert report["reporting_revision"]["prior_results_rewritten"] is False
    assert report["reporting_revision"]["completed_episodes_rerun"] is False
    assert report["progress"]["evaluation"]["verified_complete"] == 1
    assert not (campaign / "report_r1.json").exists()  # Pure report generation is read-only.


def test_revision_fingerprint_tampering_is_rejected(revision):
    path = revision["amendment"] / "protocol.json"
    value = common.read_json(path)
    value["base_campaign_fingerprint"] = "tampered"
    common.write_json(path, value)
    with pytest.raises(ValueError, match="Reporting protocol fingerprint mismatch"):
        entry.verify_reporting_revision(revision["campaign"])


def test_original_experiment_protocol_tampering_is_rejected(revision):
    path = revision["campaign"] / "protocol.json"
    value = common.read_json(path)
    value["training"]["seeds"].append(1)
    common.write_json(path, value)
    with pytest.raises(ValueError, match="Original experiment protocol changed"):
        entry.verify_reporting_revision(revision["campaign"])


@pytest.mark.parametrize("name,pattern", [
    ("frozen_source.py", "Original experiment source changed"),
    ("run_budget_report_r1.py", "Reporting source changed"),
    ("budget_report_v5_r1.py", "Reporting source changed"),
])
def test_bound_source_tampering_is_rejected(revision, name, pattern):
    (revision["repository"] / "scripts" / name).write_bytes(b"tampered source")
    with pytest.raises(ValueError, match=pattern):
        entry.verify_reporting_revision(revision["campaign"])


@pytest.mark.parametrize("implementation", [False, True])
def test_loaded_module_must_match_frozen_reporting_source(revision, monkeypatch, implementation):
    wrong = revision["repository"] / "different.py"
    wrong.write_bytes(b"different loaded source")
    monkeypatch.setattr(entry.budget_report_v5_r1 if implementation else entry, "__file__", str(wrong))
    with pytest.raises(ValueError, match="Loaded reporting .* differs"):
        entry.verify_reporting_revision(revision["campaign"])


@pytest.mark.parametrize("name", ["previous_report.json", "pre_revision_evidence.json"])
def test_preserved_evidence_tampering_is_rejected(revision, name):
    (revision["amendment"] / name).write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Preserved revision evidence changed"):
        entry.verify_reporting_revision(revision["campaign"])


def test_trained_manifest_tampering_is_rejected(revision):
    (revision["campaign"] / "trained_manifest.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen trained manifest changed"):
        entry.verify_reporting_revision(revision["campaign"])


@pytest.mark.parametrize("phase", ["collection", "evaluation"])
def test_pre_revision_receipt_tampering_is_rejected(revision, phase):
    (revision["directories"][phase] / "receipt.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="Pre-revision episode changed"):
        entry.verify_reporting_revision(revision["campaign"])


@pytest.mark.parametrize("target", ["report.json", "protocol.json", "reporting_r1/protocol.json", "../report_r1.json"])
def test_dangerous_output_is_rejected_before_report_generation(revision, monkeypatch, target):
    campaign = revision["campaign"]
    before = {str(path): path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
    monkeypatch.setattr(entry, "generate_report", lambda *args: pytest.fail("output guard must run first"))
    with pytest.raises(ValueError, match="Only the campaign root report_r1.json"):
        entry.main(["--campaign", str(campaign), "--output", str(campaign / target)])
    assert {str(path): path.read_bytes() for path in campaign.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("module", ["run_budget_report_r1", "budget_report_v5_r1"])
def test_formal_cli_and_core_delegation_preserve_original_report_without_model_imports(revision, module):
    campaign = revision["campaign"]
    preserved = b"original report evidence\n"
    (campaign / "report.json").write_bytes(preserved)
    code = ("import importlib,sys;sys.path.insert(0,sys.argv[1]);"
            "entry=importlib.import_module(sys.argv[3]);entry.main(['--campaign',sys.argv[2]]);"
            "loaded={'torch','numpy','libero','mujoco','robosuite'} & sys.modules.keys();"
            "raise SystemExit(2 if loaded else 0)")
    process = subprocess.run([sys.executable, "-c", code, str(SCRIPTS), str(campaign), module],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert (campaign / "report.json").read_bytes() == preserved
    assert common.read_json(campaign / "report_r1.json")["reporting_revision"]["id"] == "v5-report-r1"


@pytest.mark.parametrize("operation", ["require", "wrapper_cli", "core_cli", "protocol"])
def test_optimized_python_cannot_disable_guards(revision, operation):
    campaign = revision["campaign"]
    value = common.read_json(revision["amendment"] / "protocol.json")
    value["revision_fingerprint"] = "0" * 64
    common.write_json(revision["amendment"] / "protocol.json", value)
    code = """
import sys
sys.path.insert(0,sys.argv[1])
import run_budget_report_r1 as entry
import budget_report_v5_r1 as core
operation=sys.argv[3]
try:
    if operation=='require': entry.require(False,'explicit guard')
    elif operation=='protocol': entry.verify_reporting_revision(sys.argv[2])
    else:
        module=entry if operation=='wrapper_cli' else core
        module.main(['--campaign',sys.argv[2],'--output',sys.argv[2]+'/report.json'])
except ValueError as error:
    print(str(error))
    raise SystemExit(0)
raise SystemExit('Guard was bypassed under -O')
"""
    process = subprocess.run([sys.executable, "-O", "-c", code, str(SCRIPTS), str(campaign), operation],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    expected = ("explicit guard" if operation == "require" else "Reporting protocol fingerprint mismatch"
                if operation == "protocol" else "Only the campaign root report_r1.json")
    assert expected in process.stdout
    assert not (campaign / "report_r1.json").exists()
