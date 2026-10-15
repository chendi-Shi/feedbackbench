import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_research import reporting


SPEC = importlib.util.spec_from_file_location(
    "campaign_driver", Path(__file__).resolve().parents[1] / "scripts/run_campaign.py",
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    for name in campaign.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {name}", encoding="utf-8")
    events = []
    counts = dict(zip([stem for _, stem in campaign.STAGES], [40, 32, 128, 32, 12, 4]))
    evidence = {"protocol.json": "development-raw-hash"}
    selection = {"selected": "adaptive_medium", "development_protocol_fingerprint": "dev-fingerprint",
                 "development_inputs_sha256": campaign.digest_json(evidence), "scores": []}

    def generate(project_root, selected=None):
        events.append(("generate", selected))
        plans = project_root / "plans"
        plans.mkdir(exist_ok=True)
        stages = campaign.STAGES[:2] if selected is None else campaign.STAGES
        for index, (_, stem) in enumerate(stages):
            plan = {"name": stem}
            if index >= 2:
                plan["selected"] = selected
            (plans / f"{stem}.json").write_text(json.dumps(plan), encoding="utf-8")

    def run(plan, output, *, root):
        events.append(("run", output.name))
        assert plan.parent.name == "frozen_plans"
        output.mkdir(parents=True, exist_ok=True)
        (output / "protocol.json").write_text("{}", encoding="utf-8")
        return {"completed": counts[output.name], "total": counts[output.name]}

    def report(output, partial=False):
        events.append(("report", output.name, partial))
        destination = output / "report.html"
        destination.write_text("test report", encoding="utf-8")
        return destination

    def select(output):
        events.append(("select", output.name))
        return dict(selection)

    def flow(base, candidate, output):
        events.append(("flow", base.name, candidate.name))
        path = output / "flow_comparison.html"
        path.write_text("flow paired report", encoding="utf-8")
        return path

    def delay(base, candidate, output):
        events.append(("delay", base.name, candidate.name))
        path = output / "delay_comparison.html"
        path.write_text("delay paired report", encoding="utf-8")
        return path

    monkeypatch.setattr(campaign, "_generate_plans", generate)
    monkeypatch.setattr(campaign, "run_plan", run)
    monkeypatch.setattr(campaign, "build_report", report)
    monkeypatch.setattr(campaign, "select_development", select)
    monkeypatch.setattr(campaign, "load_experiment", lambda _: SimpleNamespace(
        protocol={"fingerprint": "dev-fingerprint"}, input_sha256=evidence))
    monkeypatch.setattr(reporting, "build_flow_report", flow)
    monkeypatch.setattr(reporting, "build_delay_report", delay)
    return SimpleNamespace(root=root, output=tmp_path / "campaign", events=events,
                           evidence=evidence, selection=selection, generate=generate)


def test_development_stop_freezes_all_test_plans_without_running_any_test(harness):
    result = campaign.run_campaign(harness.output, through="development", project_root=harness.root)
    assert result.is_file()
    assert [entry[1] for entry in harness.events if entry[0] == "run"] == ["10_coverage", "20_development"]
    assert len(list((harness.output / "frozen_plans").glob("*.json"))) == 6
    assert (harness.output / "development_selection.json").is_file()
    assert harness.events.index(("select", "20_development")) > harness.events.index(("report", "20_development", False))
    status = json.loads((harness.output / "campaign_status.json").read_text())
    assert status["status"] == "complete_through_requested_stage"
    assert "integration_smoke" not in status["stages"]
    assert "不能额外计作独立试验" in result.read_text(encoding="utf-8")


def test_full_campaign_is_serial_and_runs_cross_stage_flow_check_last(harness):
    campaign.run_campaign(harness.output, project_root=harness.root)
    assert [entry[1] for entry in harness.events if entry[0] == "run"] == [stem for _, stem in campaign.STAGES]
    assert harness.events[-1] == ("flow", "50_smolvla", "60_flow_steps")
    status = json.loads((harness.output / "campaign_status.json").read_text())
    assert sum(item["completed"] for item in status["stages"].values()) == 248
    assert (harness.output / "flow_comparison.html").is_file()
    assert harness.events.index(("delay", "30_methods", "40_delay")) < harness.events.index(("run", "50_smolvla"))
    assert 'href="delay_comparison.html"' in (harness.output / "report.html").read_text(encoding="utf-8")


def test_resume_keeps_selection_and_frozen_plans_byte_identical(harness):
    campaign.run_campaign(harness.output, through="development", project_root=harness.root)
    selection_bytes = (harness.output / "development_selection.json").read_bytes()
    plans = {path.name: path.read_bytes() for path in (harness.output / "frozen_plans").glob("*.json")}
    campaign.run_campaign(harness.output, through="methods", project_root=harness.root)
    assert (harness.output / "development_selection.json").read_bytes() == selection_bytes
    assert {path.name: path.read_bytes() for path in (harness.output / "frozen_plans").glob("*.json")} == plans


def test_source_change_cannot_silently_regenerate_after_results(harness):
    campaign.run_campaign(harness.output, through="coverage", project_root=harness.root)
    (harness.root / "scripts/make_plans.py").write_text("changed", encoding="utf-8")
    previous_events = list(harness.events)
    with pytest.raises(ValueError, match="changed"):
        campaign.run_campaign(harness.output, through="methods", project_root=harness.root)
    assert harness.events == previous_events


def test_changed_development_evidence_cannot_reselect_into_existing_test_plans(harness):
    campaign.run_campaign(harness.output, through="development", project_root=harness.root)
    harness.evidence["protocol.json"] = "different"
    harness.events.clear()
    with pytest.raises(ValueError, match="development evidence changed"):
        campaign.run_campaign(harness.output, through="methods", project_root=harness.root)
    assert not any(item[0] == "select" for item in harness.events)
    assert not any(item == ("run", "30_methods") for item in harness.events)


def test_failed_stage_stops_before_selection_and_heldout_execution(harness, monkeypatch):
    def fail(plan, output, **kwargs):
        harness.events.append(("run", output.name))
        output.mkdir(parents=True, exist_ok=True)
        (output / "protocol.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("simulator failure")
    monkeypatch.setattr(campaign, "run_plan", fail)
    with pytest.raises(RuntimeError, match="simulator failure"):
        campaign.run_campaign(harness.output, project_root=harness.root)
    assert not any(item[0] == "select" for item in harness.events)
    assert ("report", "10_coverage", True) in harness.events
    status = json.loads((harness.output / "campaign_status.json").read_text())
    assert status["status"] == "error"


def test_cannot_adopt_preexisting_results_as_if_preregistered(harness):
    existing = harness.output / "10_coverage"
    existing.mkdir(parents=True)
    (existing / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="retrospectively"):
        campaign.run_campaign(harness.output, project_root=harness.root)
    assert not harness.events


def test_invalid_stage_rejected_without_starting_campaign(harness):
    with pytest.raises(ValueError, match="Unknown"):
        campaign.run_campaign(harness.output, through="best_results", project_root=harness.root)
    assert not harness.output.exists()
