"""Single-task accounting and scoped runtime integration without model imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("long_horizon_tested", ROOT / "scripts/run_vlajepa_long_horizon.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def test_import_does_not_load_model_or_simulation_runtime():
    result = subprocess.run([sys.executable, "-c", "import runpy,sys;runpy.run_path(sys.argv[1]);"
                             "assert not ({'torch','mujoco','robosuite','libero'} & sys.modules.keys())",
                             str(ROOT / "scripts/run_vlajepa_long_horizon.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.compact.paged, "ROOT", tmp_path)
    monkeypatch.setattr(entry.compact.paged, "DEFAULT_OUTPUT", tmp_path / "outputs/frontier/vlajepa_paged_closed_loop")
    monkeypatch.setattr(entry.compact, "DEFAULT_OUTPUT", tmp_path / "outputs/frontier/vlajepa_compact_closed_loop")
    target = tmp_path / "outputs/frontier/vlajepa_long_horizon"
    target.mkdir(parents=True)
    return tmp_path, target


@pytest.mark.parametrize("name", ["vlajepa_closed_loop", "vlajepa_paged_closed_loop", "vlajepa_compact_closed_loop", "vlajepa_isolated_closed_loop"])
@pytest.mark.parametrize("child", ["", "child"])
def test_all_four_previous_outputs_are_rejected(workspace, name, child):
    root, _ = workspace
    with pytest.raises(ValueError, match="separate"):
        entry.separate_output(root / "outputs/frontier" / name / child)


def test_ancestor_and_outside_project_are_rejected_but_new_sibling_is_allowed(workspace):
    root, target = workspace
    assert entry.separate_output(target) == target.resolve()
    for path in (root, root / "outputs/frontier", root.parent / "outside"):
        with pytest.raises(ValueError):
            entry.separate_output(path)


@pytest.fixture
def fake(workspace, monkeypatch):
    root, target = workspace
    cases = [{"id": f"old-{suite}", "suite": suite, "task_id": 1, "init_state_id": 41, "seed": 1000,
              "model": "vlajepa_compact", "split": "compact_closed_loop_feasibility", "condition": "clean",
              "method": "native4_chunk7", "num_steps": 4, "record_video": True,
              "scheduler": {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False}}
             for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    state = {"protocol_fingerprint": "a" * 64, "planned": 4, "complete": False, "completed": 0,
             "successes": 0, "task_failures": 0, "statuses": {"vlajepa_long_horizon-libero_10": "missing"},
             "results": [], "limitations": ["Independent one-case diagnostic"]}
    seal_globals = {"write_json": write_json}
    exec("def seal_episode(path):\n    write_json(path, {'sealed': True})\n", seal_globals)
    partial_calls = []

    def prior_verify(output, *, partial=False):
        partial_calls.append(partial)
        return deepcopy(state)

    runner = SimpleNamespace(require=require, SOURCES=tuple(f"old-{index}.py" for index in range(20)),
                             cases_for_study=lambda: deepcopy(cases), episode_id=lambda row: row["model"] + "-" + row["suite"],
                             verify_study=prior_verify, write_json=write_json, seal_episode=seal_globals["seal_episode"],
                             read_json=lambda path: json.loads(Path(path).read_text()),
                             file_sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    runner.study_spec = lambda: {"name": "old", "cases": runner.cases_for_study(),
                                  "execution": {"threads": 2, "flow_steps": 4, "chunk_size": 7},
                                  "arena": {"bytes": 268435456, "nconmax": 5000},
                                  "limitations": ["four known tasks, prior limitation", "3.5 GiB gate prior limitation", "no original BF16 comparison"]}
    monkeypatch.setattr(entry.compact, "configured_runner", lambda: runner)
    configured = entry.configured_runner()
    return SimpleNamespace(runner=configured, cases=cases, state=state, target=target, partial_calls=partial_calls)


def complete(fake, *, success=True):
    case = fake.runner.cases_for_study()[0]
    row = {"id": case["id"], "suite": "libero_10", "task": "Known long task", "success": success,
           "steps": 100, "horizon": 520, "inference_calls": 15, "policy_inference_seconds": 120.0,
           "wall_seconds": 130.0, "model_info": {"action_head_nfe": 60}}
    fake.state.update(completed=1, successes=int(success), task_failures=int(not success),
                      statuses={case["id"]: "complete"}, results=[row])


def test_configuration_keeps_exact_long_case_and_changes_only_identity_and_resource_budget(fake):
    runner = fake.runner
    assert len(runner.SOURCES) == 22 and runner.SOURCES[-2:] == entry.SOURCES
    assert (runner.MIN_FREE_GIB, runner.MAX_RSS_GIB, runner.MIN_RUNTIME_FREE_GIB) == (5.75, 6.0, 0.5)
    specification = runner.study_spec()
    selected = specification["cases"]
    assert len(selected) == 1
    assert selected[0]["model"] == "vlajepa_long_horizon"
    assert selected[0]["split"] == "long_horizon_resource_feasibility"
    assert all(selected[0][key] == value for key, value in fake.cases[-1].items() if key not in {"id", "model", "split"})
    assert specification["execution"] == {"threads": 2, "flow_steps": 4, "chunk_size": 7}
    assert specification["arena"] == {"bytes": 268435456, "nconmax": 5000}
    assert specification["resource_diagnostic"]["previous_results_imported"] is False
    assert specification["memory_guard"] == {"minimum_available_gib": 5.75, "maximum_rss_gib": 6.0,
                                              "minimum_runtime_available_gib": 0.5, "sampling_seconds": 1}
    assert not any(item.startswith(("four known tasks,", "3.5 GiB gate")) for item in specification["limitations"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed"])
def test_wrong_inherited_case_fails_closed(fake, change):
    if change == "missing":
        fake.cases.pop()
    elif change == "duplicate":
        fake.cases.append(deepcopy(fake.cases[-1]))
    else:
        fake.cases[-1]["seed"] = 1001
    with pytest.raises(ValueError):
        fake.runner.cases_for_study()


@pytest.mark.parametrize("status", ["missing", "unsealed", "error"])
def test_partial_statuses_are_retained_and_never_complete(fake, status):
    fake.state["statuses"]["vlajepa_long_horizon-libero_10"] = status
    result = fake.runner.verify_study(fake.target, partial=True)
    assert result["planned"] == 1 and result["complete"] is False and result["completed"] == 0
    assert list(result["statuses"].values()) == [status]
    with pytest.raises(ValueError, match="partial must be explicit"):
        fake.runner.verify_study(fake.target)
    assert fake.partial_calls == [True, True]


@pytest.mark.parametrize("success", [True, False])
def test_one_completed_episode_is_complete_independently_of_task_success(fake, success):
    complete(fake, success=success)
    result = fake.runner.verify_study(fake.target)
    assert result["planned"] == result["completed"] == 1 and result["complete"] is True
    assert result["successes"] == int(success) and result["task_failures"] == int(not success)


@pytest.mark.parametrize("change", ["old_status", "old_result", "extra_result", "wrong_count", "inconsistent_status"])
def test_cannot_import_other_cases_or_inconsistent_completion(fake, change):
    complete(fake)
    if change == "old_status":
        fake.state["statuses"]["old-case"] = "complete"
    elif change == "old_result":
        fake.state["results"][0]["id"] = "old-case"
    elif change == "extra_result":
        fake.state["results"].append(deepcopy(fake.state["results"][0]))
    elif change == "wrong_count":
        fake.state["completed"] = 4
    else:
        fake.state["statuses"]["vlajepa_long_horizon-libero_10"] = "unsealed"
    with pytest.raises(ValueError):
        fake.runner.verify_study(fake.target, partial=True)


@pytest.mark.parametrize("finished", [False, True])
def test_json_csv_markdown_all_use_one_case_and_preserve_run_evidence(fake, finished):
    if finished:
        complete(fake)
    write_json(fake.target / "runs/run_0001.json", {"status": "complete" if finished else "blocked_preflight",
        "weights_loaded": finished, "memory": {"samples": [{"large": "raw only"}], "peak_rss_bytes": 123}})
    result = fake.runner.write_report(fake.target)
    saved = json.loads((fake.target / "report.json").read_text())
    markdown = (fake.target / "report.md").read_text()
    assert saved["planned"] == result["planned"] == 1
    assert saved["complete"] == finished and saved["completed"] == int(finished)
    assert "Planned **1** episode" in markdown and "计划 **4**" not in markdown
    assert "not a combined four-of-four result" in markdown
    assert saved["run_audits"][0]["memory"] == {"peak_rss_bytes": 123}
    assert len((fake.target / "episodes.csv").read_text().splitlines()) == 1 + int(finished)


@pytest.mark.parametrize("mode", ["complete", "blocked_preflight", "error"])
def test_scoped_progress_and_seal_patch_restores_after_all_outcomes(fake, monkeypatch, mode):
    called = []
    source_progress = {"current": "case", "ordinal": 1, "planned": 4}

    def execute(runner, output, *, limit):
        called.append(limit)
        assert runner.write_json is not write_json
        assert runner.seal_episode.__globals__["write_json"] is not write_json
        runner.write_json(output / "progress.json", source_progress)
        runner.write_json(output / "unrelated.json", {"planned": 4})
        runner.seal_episode(output / "receipt.json")
        if mode == "error":
            raise RuntimeError("controlled runtime error")
        if mode == "blocked_preflight":
            return {"status": mode}
        complete(fake)
        return runner.write_report(output)

    monkeypatch.setattr(entry.compact, "run", execute)
    if mode == "error":
        with pytest.raises(RuntimeError):
            entry.run(fake.runner, fake.target)
    else:
        result = entry.run(fake.runner, fake.target)
        assert result.get("planned", 1) == 1
    assert called == [1]
    assert json.loads((fake.target / "progress.json").read_text())["planned"] == 1
    assert json.loads((fake.target / "unrelated.json").read_text())["planned"] == 4
    assert source_progress["planned"] == 4
    assert fake.runner.write_json is write_json and fake.runner.seal_episode.__globals__["write_json"] is write_json


def test_seal_permission_retry_does_not_rerun_episode(fake, monkeypatch):
    writes, executions = [], []

    def transient(path, value):
        writes.append(path)
        if len(writes) < 3:
            raise PermissionError("transient atomic rename")
        write_json(path, value)

    fake.runner.seal_episode.__globals__["write_json"] = transient
    monkeypatch.setattr(entry.compact.time, "sleep", lambda delay: None)

    def execute(runner, output, *, limit):
        executions.append(limit)
        runner.seal_episode(output / "receipt.json")
        return {"status": "blocked_preflight"}

    monkeypatch.setattr(entry.compact, "run", execute)
    entry.run(fake.runner, fake.target)
    assert len(writes) == 3 and executions == [1]
    assert fake.runner.seal_episode.__globals__["write_json"] is transient
