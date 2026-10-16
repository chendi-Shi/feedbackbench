"""Independent-worker evidence contracts with no model or simulator imports."""
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
SPEC = importlib.util.spec_from_file_location("isolated_runner_tested", ROOT / "scripts/run_vlajepa_isolated.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_import_is_lightweight():
    result = subprocess.run([sys.executable, "-c", "import runpy,sys;runpy.run_path(sys.argv[1]);"
                             "assert not ({'torch','mujoco','robosuite','libero'} & sys.modules.keys())",
                             str(ROOT / "scripts/run_vlajepa_isolated.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.compact.paged, "ROOT", tmp_path)
    monkeypatch.setattr(entry.compact.paged, "DEFAULT_OUTPUT", tmp_path / "outputs/frontier/vlajepa_paged_closed_loop")
    monkeypatch.setattr(entry.compact, "DEFAULT_OUTPUT", tmp_path / "outputs/frontier/vlajepa_compact_closed_loop")
    monkeypatch.setattr(entry, "_RUN_ONE_STARTED", False)
    target = tmp_path / "outputs/frontier/vlajepa_isolated_closed_loop"
    target.mkdir(parents=True)
    return tmp_path, target


@pytest.mark.parametrize("name", ["vlajepa_closed_loop", "vlajepa_paged_closed_loop", "vlajepa_compact_closed_loop"])
@pytest.mark.parametrize("child", ["", "child"])
def test_all_old_studies_are_rejected(workspace, name, child):
    root, _ = workspace
    with pytest.raises(ValueError, match="separate"):
        entry.separate_output(root / "outputs/frontier" / name / child)


def test_new_sibling_is_accepted(workspace):
    _, target = workspace
    assert entry.separate_output(target) == target.resolve()


def test_configured_runner_preserves_four_cases_and_only_changes_identity_and_process_spec(monkeypatch):
    cases = [{"id": f"old-{suite}", "model": "vlajepa_compact", "split": "compact_closed_loop_feasibility",
              "suite": suite, "task_id": 1, "init_state_id": 41, "seed": 1000, "num_steps": 4,
              "scheduler": {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False}}
             for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    runner = SimpleNamespace(SOURCES=("old.py",), cases_for_study=lambda: deepcopy(cases),
                             episode_id=lambda row: row["model"] + "-" + row["suite"])
    runner.study_spec = lambda: {"cases": runner.cases_for_study(), "limitations": [], "arena": {"bytes": 268435456},
                                  "memory_guard": {"minimum_available_gib": 3.5, "maximum_rss_gib": 5.5}}
    monkeypatch.setattr(entry.compact, "configured_runner", lambda: runner)
    configured = entry.configured_runner()
    specification = configured.study_spec()
    assert configured.SOURCES == ("old.py", *entry.SOURCES)
    assert specification["process_isolation"]["episodes_per_process"] == 1
    assert specification["process_isolation"]["previous_results_imported"] is False
    assert specification["arena"] == {"bytes": 268435456}
    for before, after in zip(cases, specification["cases"]):
        assert after["model"] == "vlajepa_isolated"
        assert after["split"] == "isolated_closed_loop_feasibility"
        assert all(after[key] == value for key, value in before.items() if key not in {"id", "model", "split"})


@pytest.fixture
def fake_runner(workspace, monkeypatch):
    root, target = workspace
    state = {"protocol_fingerprint": "f" * 64, "complete": False, "planned": 4, "completed": 0,
             "statuses": {f"case-{i}": "missing" for i in range(4)}, "results": []}
    seal_globals = {"write_json": write_json}
    exec("def seal_episode(path):\n    write_json(path, {'sealed': True})\n", seal_globals)
    runner = SimpleNamespace(require=require, verify_study=lambda *args, **kwargs: deepcopy(state),
                             read_json=lambda path: json.loads(Path(path).read_text()), file_sha=sha,
                             project_file=lambda path: root / path, portable_error=str, utc_now=lambda: "fixed-test-time",
                             write_json=write_json, seal_episode=seal_globals["seal_episode"],
                             base=SimpleNamespace(relative=lambda path: Path(path).relative_to(root).as_posix()))

    def write_report(output):
        write_json(Path(output) / "report.json", state)
        return {key: value for key, value in deepcopy(state).items() if key != "results"}

    runner.write_report = write_report
    monkeypatch.setattr(entry, "process_identity", lambda: {"pid": 123, "process_create_time": 100.0})
    return SimpleNamespace(runner=runner, state=state, root=root, target=target, calls=[], mode="complete")


def install_execution(fake, monkeypatch):
    def execute(runner, output, limit):
        fake.calls.append(limit)
        assert limit == 1
        assert runner.seal_episode.__globals__["write_json"] is not write_json
        index = len(list((output / "runs").glob("run_*.json"))) + 1
        run_path = output / "runs" / f"run_{index:04d}.json"
        write_json(run_path, {"status": fake.mode})
        if fake.mode == "blocked_preflight":
            return {"status": "blocked_preflight"}
        if fake.mode == "error":
            raise RuntimeError("controlled failure")
        count = 2 if fake.mode == "two_cases" else 1
        for case_id in [key for key, value in fake.state["statuses"].items() if value == "missing"][:count]:
            fake.state["statuses"][case_id] = "complete"
            fake.state["results"].append({"id": case_id, "model_info": {"run_record": runner.base.relative(run_path)}})
            fake.state["completed"] += 1
        fake.state["complete"] = fake.state["completed"] == 4
        # The base runner verifies/reports before its outer worker is finalized.
        runner.verify_study(output, partial=True)
        runner.write_report(output)
        return {"completed": fake.state["completed"]}

    monkeypatch.setattr(entry.compact, "run", execute)


def test_one_case_per_fresh_process_is_bound_and_seal_writer_restored(fake_runner, monkeypatch):
    fake = fake_runner
    install_execution(fake, monkeypatch)
    result = entry.run_one(fake.runner, fake.target)
    assert result["completed"] == 1 and len(result["worker_audits"]) == 1
    worker = result["worker_audits"][0]
    assert worker["pid"] == 123 and worker["process_create_time"] == 100.0
    assert worker["status"] == "complete" and worker["newly_completed"] == ["case-0"]
    assert fake.runner.seal_episode.__globals__["write_json"] is write_json
    with pytest.raises(ValueError, match="fresh process"):
        entry.run_one(fake.runner, fake.target)
    assert fake.calls == [1]


def test_four_new_processes_complete_full_matrix_with_distinct_audits(fake_runner, monkeypatch):
    fake = fake_runner
    install_execution(fake, monkeypatch)
    for index in range(4):
        monkeypatch.setattr(entry, "_RUN_ONE_STARTED", False)
        monkeypatch.setattr(entry, "process_identity", lambda index=index: {"pid": 123 + index, "process_create_time": 100.0 + index})
        result = entry.run_one(fake.runner, fake.target)
        assert result["completed"] == index + 1
    assert result["complete"] is True and len(result["worker_audits"]) == 4
    assert fake.calls == [1, 1, 1, 1]


@pytest.mark.parametrize("bad", ["unsealed", "error", "lock"])
def test_partial_attempt_or_lock_is_preserved_without_model_execution(fake_runner, monkeypatch, bad):
    fake = fake_runner
    install_execution(fake, monkeypatch)
    if bad == "lock":
        (fake.target / "active.lock").write_text("retained lock")
    else:
        fake.state["statuses"]["case-0"] = bad
    with pytest.raises(ValueError):
        entry.run_one(fake.runner, fake.target)
    assert fake.calls == [] and not (fake.target / "workers").exists()


@pytest.mark.parametrize("mode", ["blocked_preflight", "error", "two_cases"])
def test_noncompletion_and_overexecution_are_never_recorded_as_one_completed_worker(fake_runner, monkeypatch, mode):
    fake = fake_runner
    fake.mode = mode
    install_execution(fake, monkeypatch)
    if mode == "blocked_preflight":
        result = entry.run_one(fake.runner, fake.target)
        assert result["status"] == mode
    else:
        with pytest.raises((ValueError, RuntimeError)):
            entry.run_one(fake.runner, fake.target)
    row = json.loads((fake.target / "workers/worker_0001.json").read_text())
    assert row["status"] == (mode if mode == "blocked_preflight" else "failed")
    assert row["newly_completed"] == []
    assert fake.runner.seal_episode.__globals__["write_json"] is write_json


@pytest.mark.parametrize("bad", ["script", "run", "missing_worker", "wrong_case", "same_process"])
def test_worker_validation_rejects_changed_or_missing_provenance(fake_runner, monkeypatch, bad):
    fake = fake_runner
    install_execution(fake, monkeypatch)
    entry.run_one(fake.runner, fake.target)
    path = fake.target / "workers/worker_0001.json"
    row = json.loads(path.read_text())
    if bad == "script":
        row["script_sha256"] = "0" * 64
    elif bad == "run":
        (fake.root / row["run_records"][0]["path"]).write_text("changed")
    elif bad == "missing_worker":
        row["status"] = "started"
    elif bad == "wrong_case":
        row["newly_completed"] = ["case-3"]
    else:
        write_json(fake.target / "workers/worker_0002.json", row)
    write_json(path, row)
    with pytest.raises(ValueError):
        entry.verify_workers(fake.runner, fake.target, deepcopy(fake.state))
