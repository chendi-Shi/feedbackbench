import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest

from embodied_research import reporting
from embodied_research.experiment import write_json
from embodied_research.scheduling import ChunkScheduler, SchedulerConfig
from embodied_research.statistics import ExperimentIntegrityError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trigger_analyzer", ROOT / "scripts/analyze_triggers.py")
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def record(*, steps=6, fixed=False):
    config = SchedulerConfig(mode="fixed" if fixed else "adaptive", fixed_interval=4, min_interval=1,
                             max_interval=4, vision_threshold=.1, gripper_threshold=.1, horizon=10)
    scheduler = ChunkScheduler(config)
    for step in range(steps):
        value = 0.0 if step == 0 else .2
        observation = {"observation.images.image": np.full((3, 2, 2), value, dtype=np.float32),
                       "observation.state": np.full(8, value, dtype=np.float32)}
        if scheduler.should_replan(observation):
            scheduler.submit_chunk(np.zeros((4, 7), dtype=np.float32))
        scheduler.next_action()
    decisions = scheduler.diagnostics()
    return {"id": "fixed" if fixed else "adaptive", "model": "fixture", "suite": "spatial", "task_id": 1,
            "task": "fixture task", "init_state_id": 0, "seed": 1, "condition": "clean",
            "method": "fixed4" if fixed else "adaptive", "status": "complete", "success": True,
            "steps": steps, "horizon": 10, "inference_calls": decisions["inference_requests"],
            "policy_inference_seconds": .3, "preprocess_seconds": .01, "scheduler_seconds": .01,
            "simulator_seconds": .2, "wall_seconds": .7, "model_info": {"fixture": True},
            "scheduler_config": asdict(config), "decisions": decisions}


def corpus(tmp_path, rows, *, missing=False):
    cases = copy.deepcopy(rows)
    if missing:
        cases.append({**copy.deepcopy(rows[0]), "id": "missing", "seed": 2})
    write_json(tmp_path / "protocol.json", {"cases": cases, "plan": {"name": "trigger-fixture"}})
    for row in rows:
        write_json(tmp_path / "episodes" / row["id"] / "result.json", row)


def test_overlapping_reasons_are_not_double_counted_as_calls_and_final_span_is_censored():
    row = record()
    result = analyzer.validate_episode_log(row)
    assert result["inference_calls"] == 3
    assert result["plan_steps"] == [0, 1, 5]
    assert result["exclusive_reason_combinations"] == {
        "initial": 1, "gripper_change + vision_change": 1, "chunk_exhausted": 1,
    }
    assert sum(result["marginal_reason_counts"].values()) == 4
    assert sum(result["exclusive_reason_combinations"].values()) == 3
    assert result["completed_replan_intervals"] == [1, 4]
    assert result["executed_chunk_spans"] == [1, 4, 1]
    assert sum(result["executed_chunk_spans"]) == row["steps"]
    assert result["final_censored_span"] == 1
    assert result["completed_interval_summary"]["mean"] == 2.5
    assert result["executed_span_summary"]["mean"] == 2


def test_single_request_has_no_complete_interval_even_when_final_span_reaches_chunk_length():
    result = analyzer.validate_episode_log(record(steps=4, fixed=True))
    assert result["completed_replan_intervals"] == []
    assert result["completed_interval_summary"]["mean"] is None
    assert result["final_censored_span"] == 4
    assert result["executed_chunk_spans"] == [4]


@pytest.mark.parametrize("mutation,expected", [
    ("calls", "inference count"), ("events", "one scheduler event"), ("reasons", "request/reasons"),
    ("marginals", "trigger_counts"), ("plans", "plan_steps"), ("complete_spans", "intervals/executed"),
    ("executed_spans", "intervals/executed"), ("age", "steps_since_plan"),
    ("pending", "pending"), ("actions", "return an action"), ("predictions", "prediction counts"),
    ("delta", "logged features/interval"), ("duplicate_reason", "duplicate trigger"),
])
def test_log_contradictions_are_rejected(mutation, expected):
    row = record()
    log = row["decisions"]
    if mutation == "calls":
        row["inference_calls"] += 1
    elif mutation == "events":
        log["events"].pop()
    elif mutation == "reasons":
        log["events"][1]["requested"] = False
    elif mutation == "marginals":
        log["trigger_counts"]["vision_change"] += 1
    elif mutation == "plans":
        log["plan_steps"][1] = 2
    elif mutation == "complete_spans":
        log["completed_replan_intervals"].append(1)
    elif mutation == "executed_spans":
        log["executed_chunk_spans"][-1] += 1
    elif mutation == "age":
        log["events"][2]["steps_since_plan"] = 2
    elif mutation == "pending":
        log["pending_inference"] = True
    elif mutation == "actions":
        log["events"][2]["action_returned"] = False
    elif mutation == "predictions":
        log["events"][1]["active_predictions"] = 1
    elif mutation == "delta":
        log["events"][1]["image_delta"] = 0
    elif mutation == "duplicate_reason":
        log["events"][1]["reasons"].append("vision_change")
    with pytest.raises(ExperimentIntegrityError, match=expected):
        analyzer.validate_episode_log(row)


def test_stage_exports_real_patterns_hashes_and_conditional_report_link(tmp_path, monkeypatch):
    corpus(tmp_path, [record(), record(steps=4, fixed=True)])
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    path = analyzer.analyze_triggers(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["partial"] is False
    assert len(result["per_task"]) == len(result["episodes"]) == 2
    assert result["analysis_type"] == "post_hoc_scheduler_diagnostic"
    for group in result["groups"]:
        assert sum(group["exclusive_reason_combinations"].values()) == group["inference_calls"]
        assert group["final_censored_spans"]["count"] == group["episodes"]
    for raw_path, raw in before.items():
        assert raw_path.read_bytes() == raw
        key = raw_path.relative_to(tmp_path).as_posix()
        assert result["stages"][tmp_path.name]["input_sha256"][key] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / "figures/trigger_combinations.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (tmp_path / "figures/trigger_combinations.svg").read_text(encoding="utf-8")
    assert "gripper_change + vision_change" in svg
    assert "calls=3" in svg
    text = (tmp_path / "trigger_diagnostics.html").read_text(encoding="utf-8")
    assert "边际原因计数不能相加" in text
    assert 'href="episodes/adaptive/result.json"' in text
    monkeypatch.setattr(reporting, "_plot_figures", lambda root, summary: [])
    report = reporting.build_report(tmp_path).read_text(encoding="utf-8")
    assert 'href="trigger_diagnostics.html"' in report
    assert "不作为信号导致成功或失败的因果证据" in report


def test_missing_episodes_require_explicit_partial_but_invalid_logs_never_do(tmp_path, monkeypatch):
    corpus(tmp_path, [record()], missing=True)
    monkeypatch.setattr(analyzer, "_plot", lambda *_: None)
    with pytest.raises(ExperimentIntegrityError, match="requires all planned"):
        analyzer.analyze_triggers(tmp_path)
    assert not (tmp_path / "trigger_diagnostics.json").exists()
    analyzer.analyze_triggers(tmp_path, partial=True)
    result = json.loads((tmp_path / "trigger_diagnostics.json").read_text(encoding="utf-8"))
    assert result["partial"] is True
    assert result["missing_episodes"] == 1
    assert "部分事后诊断 · 尚未完成" in (tmp_path / "trigger_diagnostics.html").read_text(encoding="utf-8")
    row = record()
    row["decisions"]["trigger_counts"]["initial"] = 2
    corpus(tmp_path, [row], missing=True)
    with pytest.raises(ExperimentIntegrityError, match="trigger_counts"):
        analyzer.analyze_triggers(tmp_path, partial=True)


def test_diagnostic_paths_are_portable_and_html_links_survive_relocation(tmp_path, monkeypatch):
    stage = tmp_path / "Users" / "private-fixture-user" / "development"
    corpus(stage, [record()])
    (stage / "report.html").write_text("<!doctype html><title>Fixture report</title>", encoding="utf-8")
    monkeypatch.setattr(analyzer, "_plot", lambda *_: None)
    result = json.loads(analyzer.analyze_triggers(stage).read_text(encoding="utf-8"))
    assert result["episodes"][0]["source_directory"] == "."
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            assert not Path(value).is_absolute()
            assert not PureWindowsPath(value).is_absolute()
            assert "private-fixture-user" not in value
            assert str(tmp_path) not in value
            assert tmp_path.as_posix() not in value

    relocated = tmp_path / "portable-evidence"
    shutil.copytree(stage, relocated)
    text = (relocated / "trigger_diagnostics.html").read_text(encoding="utf-8")
    links = re.findall(r'href="([^"]+)"', text)
    assert "episodes/adaptive/result.json" in links
    for href in links:
        parsed = urlsplit(href)
        assert not parsed.scheme and not parsed.netloc
        target = (relocated / unquote(parsed.path)).resolve()
        assert target.is_relative_to(relocated.resolve())
        assert target.is_file()
