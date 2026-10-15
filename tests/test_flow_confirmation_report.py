"""Synthetic prospective-report arithmetic and export integrity; no model execution.

Raw protocol/receipt/input validation belongs to test_flow_confirmation.py and
the runner's verify_study. This suite feeds an explicit synthetic verified study.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from embodied_research.experiment import write_json


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flow_confirmation_report_under_test",
                                            ROOT / "scripts/summarize_flow_confirmation.py")
reporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reporter
spec.loader.exec_module(reporter)


@pytest.fixture
def study(monkeypatch):
    cases, records, statuses = [], {}, {}
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for suite in suites:
        for initial in (41, 43):
            for method in ("full10", "noise2"):
                case_id = f"smolvla__{suite}__1__{initial}__clean__{method}"
                case = {"id": case_id, "model": "smolvla", "suite": suite, "task_id": 1,
                        "init_state_id": initial, "method": method, "seed": 1000,
                        "condition": "clean", "split": "confirmation"}
                success = not (suite == suites[0] and initial == 43 and method == "noise2")
                steps = 21 if success else 31
                nfe = 10 if method == "full10" else 2
                calls = []
                for index, _ in enumerate(range(0, steps, 10)):
                    call = {"call_index": index, "mode": method, "hit": None, "nfe": nfe,
                            "noise_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
                    call.update({name: 10. if method == "full10" else 6. for name in reporter.support.TIMINGS})
                    calls.append(call)
                record = {**case, "success": success, "steps": steps, "horizon": 31,
                          "inference_calls": len(calls), "wall_seconds": 100. if method == "full10" else 80. if success else 999.,
                          "policy_inference_seconds": sum(c["inference_seconds"] for c in calls)}
                records[case_id] = {"case": case, "record": record, "calls": calls,
                                    "actions": np.full((steps, 7), 0. if method == "full10" else .2, dtype=np.float32),
                                    "result_sha256": "1" * 64, "receipt_sha256": "2" * 64,
                                    "artifact_sha256": {"result.json": "1" * 64}}
                cases.append(case)
                statuses[case_id] = "complete"
    protocol = {"spec": {"cases": cases, "name": "Synthetic flow confirmation", "question": "Synthetic question"},
                "fingerprint": "3" * 64, "sources": {"fixture": "4" * 64}, "runtime_versions": {},
                "artifacts": {"models": {}}, "origin": {"scope": "synthetic earlier exploratory evidence"}}
    monkeypatch.setattr(reporter, "_implementation_hashes", lambda: {"synthetic_reporter": "5" * 64})
    return {"protocol": protocol, "records": records, "statuses": statuses, "complete": True,
            "evidence_sha256": {"protocol.json": "6" * 64}, "pairing_audit": {"paired_initial_blocks": 8}}


def _load_fixture(monkeypatch, study):
    def load(output, *, partial=False, root=None):
        if not study["complete"] and not partial:
            raise reporter.IntegrityError("incomplete study; --partial required")
        return copy.deepcopy(study)
    monkeypatch.setattr(reporter, "load_study", load)


@pytest.fixture
def exported(tmp_path, monkeypatch, study):
    _load_fixture(monkeypatch, study)
    write_json(tmp_path / "protocol.json", study["protocol"])
    reporter.build_report(tmp_path)
    return tmp_path, study


def _reseal(output, name):
    manifest = reporter.read_json(output / "exports.json")
    manifest["files"][name] = reporter.sha(output / name)
    write_json(output / "exports.json", manifest)


def test_primary_direction_task_clusters_and_denominators(study):
    result = reporter.summarize(study)
    comp = result["comparison"]
    assert result["counts"] == {"planned": 16, "complete": 16, "missing": 0, "unsealed": 0, "error": 0}
    assert comp["planned_pairs"] == comp["completed_pairs"] == 8
    assert comp["observed_task_clusters"] == 4
    assert comp["macro_success_difference"] == -.125
    assert comp["success_task_cluster_95ci"] == [-.375, 0.]
    assert comp["difference_direction"] == "noise2 minus full10"
    assert result["uncertainty"]["seed"] == 20261015
    assert [g["successes"] for g in result["groups"]] == [8, 7]
    assert [g["total_nfe"] for g in result["groups"]] == [240, 50]
    assert [g["replans"] for g in result["groups"]] == [24, 25]
    assert len(result["episodes"]) == 16 and len(result["calls"]) == 49


def test_long_failed_wall_excluded_only_from_conditional_wall_summary(study):
    result = reporter.summarize(study)
    comp = result["comparison"]
    assert comp["both_success_pairs"] == 7
    assert comp["mean_wall_difference_both_success_only"] == -20.
    failed = [p for p in comp["per_pair"] if not p["candidate_success"]]
    assert len(failed) == 1 and failed[0]["candidate_wall_seconds"] == 999.
    assert failed[0]["wall_difference_both_success_only"] is None
    assert any(e.get("wall_seconds") == 999. for e in result["episodes"])
    assert "按结果筛选" in reporter._markdown(result)


def test_short_success_preserves_outcome_and_marks_first_ten_missing(study):
    row = next(r for r in study["records"].values() if r["case"]["method"] == "noise2")
    row["actions"] = np.full((5, 7), 9., dtype=np.float32)
    row["record"].update(steps=5, inference_calls=1)
    row["calls"] = row["calls"][:1]
    result = reporter.summarize(study)
    comp = result["comparison"]
    pair = next(p for p in comp["per_pair"] if p["candidate_id"] == row["case"]["id"])
    assert pair["candidate_success"] is True
    assert pair["actual_available_steps"] == 5 and pair["first_action_steps_compared"] == 0
    assert pair["first_actions_rmse_all_dims"] is None
    assert comp["first_10_complete_pairs"] == 7 and comp["first_10_missing_pairs"] == 1
    assert comp["mean_first_actions_rmse_all_dims"] == pytest.approx(.2)
    assert result["groups"][1]["successes"] == 7


@pytest.mark.parametrize("status", ["missing", "unsealed", "error"])
def test_partial_keeps_full_plan_and_requires_explicit_flag(study, monkeypatch, tmp_path, status):
    case_id = next(iter(study["records"]))
    del study["records"][case_id]
    study["statuses"][case_id] = status
    study["complete"] = False
    _load_fixture(monkeypatch, study)
    with pytest.raises(reporter.IntegrityError, match="partial"):
        reporter.load_study(tmp_path)
    result = reporter.summarize(reporter.load_study(tmp_path, partial=True))
    assert result["counts"]["planned"] == 16 and result["counts"][status] == 1
    assert result["comparison"]["completed_pairs"] == 7
    assert result["comparison"]["planned_pairs"] == 8
    assert result["status"] == "PARTIAL" and len(result["episodes"]) == 16


def test_empty_partial_does_not_invent_success_or_time(study):
    study.update(records={}, complete=False)
    study["statuses"] = {key: "missing" for key in study["statuses"]}
    result = reporter.summarize(study)
    assert result["counts"]["missing"] == 16
    assert result["comparison"]["macro_success_difference"] is None
    assert result["comparison"]["success_task_cluster_95ci"] is None
    assert result["comparison"]["mean_wall_difference_both_success_only"] is None
    assert all(g["observed_macro_task_success"] is None for g in result["groups"])


def test_bootstrap_zero_width_not_equivalence_and_one_task_has_no_ci(study):
    assert reporter._cluster_ci([0, 0, 0, 0]) == [0., 0.]
    assert reporter._cluster_ci([0]) is None
    assert "不证明等价" in reporter._markdown(reporter.summarize(study))


@pytest.mark.parametrize("kind", ["old_initial", "old_split", "wrong_method", "missing_case"])
def test_foreign_or_reduced_plan_rejected(study, kind):
    case = study["protocol"]["spec"]["cases"][0]
    if kind == "old_initial":
        case["init_state_id"] = 11
    elif kind == "old_split":
        case["split"] = "heldout"
    elif kind == "wrong_method":
        case["method"] = "cache2"
    else:
        study["protocol"]["spec"]["cases"].pop()
    with pytest.raises(reporter.IntegrityError):
        reporter.summarize(study)


def test_build_exports_and_readonly_validation(exported, monkeypatch):
    output, _ = exported
    original = {p.relative_to(output).as_posix(): (p.stat().st_mtime_ns, reporter.sha(p))
                for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(reporter, "_plots", lambda *_: pytest.fail("read-only validation must not plot"))
    result = reporter.validate_report(output)
    assert result["status"] == "complete"
    seal = reporter.read_json(output / "exports.json")
    assert set(seal["files"]) == set(reporter.EXPORT_FILES)
    assert len(seal["files"]) == 10
    assert original == {p.relative_to(output).as_posix(): (p.stat().st_mtime_ns, reporter.sha(p))
                        for p in output.rglob("*") if p.is_file()}


def test_changed_export_bytes_rejected(exported):
    output, _ = exported
    (output / "figures/flow_success_budget.png").write_bytes(b"altered")
    with pytest.raises(reporter.IntegrityError, match="hash mismatch"):
        reporter.validate_report(output)


def test_csv_tampering_rejected_even_if_export_manifest_is_resealed(exported):
    output, _ = exported
    (output / "episodes.csv").write_text("invented success\n", encoding="utf-8")
    _reseal(output, "episodes.csv")
    with pytest.raises(reporter.IntegrityError, match="CSV differs"):
        reporter.validate_report(output)


def test_statistic_tampering_rejected_even_if_export_manifest_is_resealed(exported):
    output, _ = exported
    report = reporter.read_json(output / "report.json")
    report["comparison"]["macro_success_difference"] = .9
    write_json(output / "report.json", report)
    _reseal(output, "report.json")
    with pytest.raises(reporter.IntegrityError, match="differs from verified"):
        reporter.validate_report(output)


def test_changed_report_implementation_requires_regeneration(exported, monkeypatch):
    output, _ = exported
    monkeypatch.setattr(reporter, "_implementation_hashes", lambda: {"synthetic_reporter": "7" * 64})
    with pytest.raises(reporter.IntegrityError, match="differs from verified"):
        reporter.validate_report(output)


def test_timestamp_only_change_is_allowed_after_reseal(exported):
    output, _ = exported
    report = reporter.read_json(output / "report.json")
    report["generated_at"] = "2026-10-15T00:00:00+00:00"
    write_json(output / "report.json", report)
    _reseal(output, "report.json")
    assert reporter.validate_report(output)["status"] == "complete"


def test_external_local_markdown_reference_rejected_even_after_reseal(exported):
    output, _ = exported
    path = output / "report.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n![escape](../outside.png)\n", encoding="utf-8")
    _reseal(output, "report.md")
    with pytest.raises(reporter.IntegrityError, match="Missing or out-of-stage"):
        reporter.validate_report(output)


def test_export_whitelist_rejects_extra_entries(exported):
    output, _ = exported
    seal = reporter.read_json(output / "exports.json")
    seal["files"]["../outside"] = "a" * 64
    write_json(output / "exports.json", seal)
    with pytest.raises(reporter.IntegrityError, match="inventory"):
        reporter.validate_report(output)


def test_real_runner_report_interface_with_synthetic_sealed_evidence(tmp_path):
    fixture_spec = importlib.util.spec_from_file_location("flow_report_runner_fixture",
                                                        Path(__file__).with_name("test_flow_confirmation.py"))
    fixtures = importlib.util.module_from_spec(fixture_spec)
    sys.modules[fixture_spec.name] = fixtures
    fixture_spec.loader.exec_module(fixtures)
    root, output, protocol = fixtures.study_fixture(tmp_path)
    for case in fixtures.runner.cases_for_study():
        fixtures.complete_case(output, case, protocol)
    verified = reporter.load_study(output, root=root)
    assert verified["complete"] and verified["pairing_audit"]["paired_initial_blocks"] == 8
    reporter.build_report(output, root=root)
    result = reporter.validate_report(output, root=root)
    assert result["comparison"]["completed_pairs"] == 8
    assert result["comparison"]["macro_success_difference"] == 0.
    assert result["comparison"]["first_10_complete_pairs"] == 8
    assert [g["total_nfe"] for g in result["groups"]] == [80, 16]
