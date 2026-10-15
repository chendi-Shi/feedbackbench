import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from embodied_research.experiment import digest_json, expand_plan, source_fingerprint, write_json
from embodied_research.scheduling import SchedulerConfig
from embodied_research.selection import select_development
from embodied_research.statistics import ExperimentIntegrityError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_summary", ROOT / "scripts/summarize_campaign.py")
summary_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_tool)


def method(name, *, ensemble=True, threshold=0.08):
    if name == "upstream":
        return {"name": name, "scheduler": None}
    if name.startswith("fixed"):
        return {"name": name, "num_steps": 5 if name == "fixed15_steps5" else 10,
                "scheduler": {"mode": "fixed", "fixed_interval": int(name.split("_")[0][5:]),
                              "temporal_ensemble": ensemble}}
    config = {"mode": "adaptive", "vision_threshold": threshold, "gripper_threshold": 0.01,
              "temporal_ensemble": ensemble}
    disabled = {"no_vision": "use_vision", "no_gripper": "use_gripper", "no_ensemble": "temporal_ensemble"}.get(name)
    if disabled:
        config[disabled] = False
    return {"name": name, "scheduler": config}


@pytest.fixture
def complete_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    source_names = [f"src/embodied_research/{name}" for name in summary_tool.RUNTIME_SOURCES]
    source_names += ["src/embodied_research/statistics.py", "src/embodied_research/selection.py"]
    freeze = {"stage_order": [name for name, _ in summary_tool.STAGES], "integration_smoke_included": False,
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}}
    write_json(campaign / "campaign_freeze.json", freeze)
    definitions = {
        "coverage": ("coverage-fixture", "minerva", [method("upstream")], [0], 0, "clean"),
        "development": ("development-calibration", "minerva", [method("fixed8"), method("adaptive_low", threshold=.015),
                         method("adaptive_medium", threshold=.04), method("adaptive_high")], [0], 1, "clean"),
        "methods": ("methods-fixture", "minerva", [method(name) for name in
                     ("fixed1", "fixed4", "fixed8", "fixed15", "adaptive", "no_vision", "no_gripper", "no_ensemble")],
                    [1, 2, 3, 4], 11, "clean"),
        "delay": ("delay-fixture", "minerva", [method("fixed8"), method("adaptive")], [1, 2, 3, 4], 11, "delay4"),
        "smolvla": ("smol-fixture", "smolvla", [method(name, ensemble=False) for name in ("fixed8", "fixed15", "adaptive")],
                    [1, 2, 3, 4], 11, "clean"),
        "flow_steps": ("flow-fixture", "smolvla", [method("fixed15_steps5", ensemble=False)], [1, 2, 3, 4], 11, "clean"),
    }
    status = {"status": "complete_through_requested_stage", "through": "flow_steps", "stages": {}}
    for stage_name, stem in summary_tool.STAGES:
        name, model, methods, tasks, initial, condition = definitions[stage_name]
        plan = {"name": name, "models": [{"name": model, "methods": methods}],
                "tasks": [{"suite": "fixture_suite", "task_id": task} for task in tasks], "initial_states": [initial],
                "conditions": [condition], "selection_rule": "fixed development-only fixture rule", "split": stage_name}
        write_json(campaign / "frozen_plans" / f"{stem}.json", plan)
        cases = expand_plan(plan)
        provenance = {"plan": plan, "sources": source_fingerprint(ROOT), "runtime_versions": {"fixture": "1"},
                      "artifacts": {"weights": model}}
        protocol = {**provenance, "cases": cases, "fingerprint": digest_json(provenance)}
        write_json(campaign / stem / "protocol.json", protocol)
        for case in cases:
            scheduler = None if case["scheduler"] is None else asdict(SchedulerConfig(**case["scheduler"], horizon=20))
            calls = {"adaptive_high": 1, "adaptive_medium": 2, "adaptive_low": 3}.get(case["method"], 4)
            result = {**case, "status": "complete", "protocol_fingerprint": protocol["fingerprint"],
                      "task": f"task {case['task_id']}", "success": True, "steps": 20, "horizon": 20,
                      "inference_calls": calls, "policy_inference_seconds": float(calls), "preprocess_seconds": .1,
                      "scheduler_seconds": .1, "simulator_seconds": 1.0, "wall_seconds": 6.0,
                      "model_info": {"model_sha256": ("a" if model == "minerva" else "b") * 64,
                                     "revision": "fixture", "num_steps": case["num_steps"] if model == "smolvla" else None},
                      "scheduler_config": scheduler, "environment": {"fixture": True},
                      "initial_image_sha256": "c" * 64, "initial_robot_state": [0.0] * 8}
            write_json(campaign / stem / "episodes" / case["id"] / "result.json", result)
        status["stages"][stage_name] = {"status": "complete", "completed": len(cases), "total": len(cases)}
    write_json(campaign / "campaign_status.json", status)
    selection = select_development(campaign / "20_development")
    write_json(campaign / "development_selection.json", selection)
    return campaign


def test_final_summary_is_complete_stage_separated_and_raw_read_only(complete_campaign, tmp_path):
    before = {path: path.read_bytes() for path in complete_campaign.rglob("*.json")}
    document = tmp_path / "docs" / "RESULTS.md"
    result = summary_tool.summarize_campaign(complete_campaign, markdown_path=document)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["counts"] == {"total_completed_episodes": 61, "coverage_episodes": 1,
                                 "development_episodes": 4, "held_out_episodes": 56,
                                 "integration_smoke_episodes_included": 0}
    assert "pooled_success" not in payload
    assert payload["selected_thresholds"] == {"vision_threshold": .08, "gripper_threshold": .01}
    assert payload["development_selection"]["selected"] == "adaptive_high"
    assert len(payload["stages"]["methods"]["groups"]) == 8
    contrasts = payload["comparisons"]
    assert contrasts["main_adaptive_vs_fixed8"]["task_cluster_95ci"] == [0, 0]
    assert {row["method"] for row in contrasts["all_main_methods_vs_fixed8"]} == {
        "fixed1", "fixed4", "fixed15", "adaptive", "no_vision", "no_gripper", "no_ensemble",
    }
    assert len(contrasts["direct_ablations"]) == 3
    assert len(contrasts["delay"]["comparisons"]) == 2
    assert len(contrasts["smolvla_vs_fixed8"]) == 2
    assert contrasts["flow_steps"]["paired_episodes"] == 4
    assert len(payload["stage_protocol_fingerprints"]) == 6
    assert payload["input_sha256"]["campaign_freeze.json"] == hashlib.sha256(before[complete_campaign / "campaign_freeze.json"]).hexdigest()
    text = document.read_text(encoding="utf-8")
    assert "61 个计划回合" in text
    assert "不证明等效、非劣效或真实不确定性为零" in text
    assert "../campaign/30_methods/protocol.json" in text
    assert "248" not in text  # Sizes are read from evidence, never hard-coded.
    for path, raw in before.items():
        assert path.read_bytes() == raw


def test_markdown_keeps_all_declared_contrasts_and_their_observed_direction(complete_campaign):
    # A non-adaptive method has a distinct negative result. It must remain visible
    # in the final Markdown instead of surviving only in the machine-readable JSON.
    for path in (complete_campaign / "30_methods/episodes").glob("*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["method"] == "fixed15":
            record["success"] = False
            record["inference_calls"] = 2
            write_json(path, record)
    summary_tool.summarize_campaign(complete_campaign)
    text = (complete_campaign / "RESULTS.md").read_text(encoding="utf-8")
    section = text.split("## MINERVA 保留初态：质量与推理预算", 1)[1].split("## 组件消融", 1)[0]
    section = section.split("下表完整列出所有预声明候选对 fixed8 的配对比较", 1)[1]
    rows = [line.split("|")[1].strip() for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| 方法 |")]
    assert len(rows) == 7
    assert set(rows) == {"fixed1", "fixed4", "fixed15", "adaptive", "no_vision", "no_gripper", "no_ensemble"}
    assert "| fixed15 | 4 / 4 | -100.0 | [-100.0, -100.0] | -2.00 |" in section
    assert "| adaptive | 4 / 4 | +0.0 | [+0.0, +0.0] | +0.00 |" in section


@pytest.mark.parametrize("mutation,expected", [
    ("unfinished", "all six"), ("missing", "requires all planned"), ("counts", "counts differ"),
    ("selection", "development selection"), ("source", "source changed"), ("plan", "frozen plan"),
    ("fingerprint", "provenance"), ("duplicate", "duplicate"),
])
def test_summary_refuses_incomplete_or_changed_evidence_without_creating_outputs(complete_campaign, mutation, expected):
    path = complete_campaign / "campaign_status.json"
    if mutation in {"unfinished", "counts"}:
        payload = json.loads(path.read_text())
        if mutation == "unfinished":
            payload["stages"]["flow_steps"]["status"] = "running"
        else:
            payload["stages"]["methods"]["completed"] += 1
        write_json(path, payload)
    elif mutation == "missing":
        next((complete_campaign / "60_flow_steps/episodes").glob("*/result.json")).unlink()
    elif mutation == "selection":
        path = complete_campaign / "development_selection.json"
        payload = json.loads(path.read_text())
        payload["selected"] = "adaptive_low"
        write_json(path, payload)
    elif mutation == "source":
        path = complete_campaign / "campaign_freeze.json"
        payload = json.loads(path.read_text())
        payload["source_sha256"]["src/embodied_research/statistics.py"] = "0" * 64
        write_json(path, payload)
    elif mutation == "plan":
        path = complete_campaign / "frozen_plans/30_methods.json"
        payload = json.loads(path.read_text())
        payload["name"] = "changed-after-results"
        write_json(path, payload)
    elif mutation == "fingerprint":
        path = complete_campaign / "30_methods/protocol.json"
        payload = json.loads(path.read_text())
        payload["fingerprint"] = "incorrect"
        write_json(path, payload)
    elif mutation == "duplicate":
        path = next((complete_campaign / "30_methods/episodes").glob("*/result.json"))
        duplicate = copy.deepcopy(json.loads(path.read_text()))
        write_json(path.parent.parent / "zzz_duplicate" / "result.json", duplicate)
    with pytest.raises(ExperimentIntegrityError, match=expected):
        summary_tool.summarize_campaign(complete_campaign)
    assert not (complete_campaign / "campaign_summary.json").exists()
    assert not (complete_campaign / "RESULTS.md").exists()
