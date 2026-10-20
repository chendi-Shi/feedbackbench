"""Model-free process ownership and persistent-I/O supervision regressions."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_budget_campaign_v5 as campaign


def denied():
    error = PermissionError("synthetic Windows sharing denial")
    error.winerror = 5
    return error


def fake_tree(monkeypatch, *, cooperative=False, control=None):
    """The redirector exits, its observed interpreter later spawns a child."""
    import psutil
    state = {"tick": 0, "terminated": [], "statuses": [], "waited": False,
             "launches": 0, "cooperative": cooperative}
    rss = {101: 1, 102: 3, 103: 7}

    class FakeProcess:
        def __init__(self, pid):
            if pid not in rss:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid

        def create_time(self):
            return float(self.pid)

        def is_running(self):
            if self.pid in state["terminated"]:
                return False
            if cooperative and state["tick"] and control is not None and control.exists():
                return False
            return self.pid != 101 or state["tick"] == 0

        def children(self, recursive=False):
            assert recursive
            if not self.is_running():
                raise psutil.NoSuchProcess(self.pid)
            if self.pid == 101:
                return [FakeProcess(102)]
            return [FakeProcess(103)] if self.pid == 102 and state["tick"] else []

        def memory_info(self):
            return SimpleNamespace(rss=rss[self.pid])

        def terminate(self):
            assert self.is_running()
            state["terminated"].append(self.pid)

    class FakeLauncher:
        pid = 101

        def poll(self):
            return None if state["tick"] == 0 and 101 not in state["terminated"] else 0

        def wait(self):
            assert not FakeProcess(102).is_running()
            state["waited"] = True
            return 0

    def launch(*args, **kwargs):
        state["launches"] += 1
        return FakeLauncher()

    def sleep(seconds):
        state["tick"] += 1
        assert state["tick"] < 5, "persistent I/O failure abandoned the owned worker"

    monkeypatch.setattr(psutil, "Process", FakeProcess)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1000))
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, timeout:
                        ([p for p in processes if not p.is_running()],
                         [p for p in processes if p.is_running()]))
    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: float(state["tick"] * 5))
    monkeypatch.setattr(campaign.time, "sleep", sleep)
    monkeypatch.setattr(campaign, "deadline_reached", lambda protocol: False)
    return state


def limits(**extra):
    return {"resource_limits": {"max_process_rss_bytes": 100,
            "min_system_available_bytes": 1, "resource_stop_grace_seconds": 5, **extra}}


@pytest.mark.parametrize("control_denied,cooperative", [(False, True), (False, False), (True, False)])
def test_status_failure_keeps_monitoring_and_reaps_only_owned_tree(tmp_path, monkeypatch, control_denied, cooperative):
    attempt = tmp_path / "attempts/collection/case/001"
    attempt.mkdir(parents=True)
    control = tmp_path / "controls/collection/case/001.json"
    state = fake_tree(monkeypatch, cooperative=cooperative, control=control)
    real_write = campaign.write_json

    def status_failure(*args, **details):
        state["statuses"].append(details)
        raise denied()

    def write(path, value):
        if control_denied and Path(path) == control:
            raise denied()
        return real_write(path, value)

    monkeypatch.setattr(campaign, "status_write", status_failure)
    monkeypatch.setattr(campaign, "write_json", write)
    audit = campaign.monitor_child(tmp_path, limits(), {"pid": 900, "created": 9.0},
        ["owned-python", "worker.py"], "collection", tmp_path / "runs/collection-case-001.json",
        tmp_path / "logs/case.log", attempt, {"case_id": "case"})
    assert audit["action_required"] and audit["children_reaped"]
    assert audit["stop_reason"] == "supervisor_io_failure"
    assert state["waited"] and state["launches"] == 1
    assert len(state["statuses"]) == 1  # Do not repeatedly hit a blocked status file.
    assert {row["operation"] for row in audit["io_errors"]} == (
        {"status_write", "control_write"} if control_denied else {"status_write"})
    if cooperative:
        assert state["terminated"] == []
    else:
        assert state["terminated"] == [103, 102]
        assert audit["peak_rss_bytes"] == 10
    assert campaign.read_json(tmp_path / "runs/collection-case-001.json")["action_required"]
    assert campaign.recorded_owned_processes(tmp_path) == audit["owned_processes"]
    assert list((tmp_path / "diagnostics").glob("*.json"))


def test_resource_guard_still_monitors_orphaned_descendant(tmp_path, monkeypatch):
    state = fake_tree(monkeypatch)
    monkeypatch.setattr(campaign, "status_write", lambda *args, **details: state["statuses"].append(details))
    audit = campaign.monitor_child(tmp_path, limits(max_process_rss_bytes=8, resource_stop_grace_seconds=0),
        {"pid": 900, "created": 9.0}, ["owned-python", "worker.py"], "training",
        tmp_path / "runs/training-001.json", tmp_path / "logs/training-001.log")
    assert audit["stop_reason"] == "resource_guard" and not audit["action_required"]
    assert audit["peak_rss_bytes"] == 10
    assert state["terminated"] == [103, 102] and state["waited"]


def test_training_status_failure_without_control_file_is_reaped(tmp_path, monkeypatch):
    state = fake_tree(monkeypatch)
    monkeypatch.setattr(campaign, "status_write", lambda *args, **kwargs: (_ for _ in ()).throw(denied()))
    audit = campaign.monitor_child(tmp_path, limits(), {"pid": 900, "created": 9.0},
        ["owned-python", "train.py"], "training", tmp_path / "runs/training-001.json",
        tmp_path / "logs/training-001.log")
    assert audit["action_required"] and audit["children_reaped"]
    assert state["terminated"] == [103, 102] and not (tmp_path / "controls").exists()


def test_run_audit_denial_preserves_fallback_and_requires_attention(tmp_path, monkeypatch):
    state = fake_tree(monkeypatch)
    monkeypatch.setattr(campaign, "status_write", lambda *args, **kwargs: None)
    run_path = tmp_path / "runs/training-001.json"
    real_write = campaign.write_json

    def write(path, value):
        if Path(path) == run_path:
            raise denied()
        return real_write(path, value)

    monkeypatch.setattr(campaign, "write_json", write)
    audit = campaign.monitor_child(tmp_path, limits(max_process_rss_bytes=8, resource_stop_grace_seconds=0),
        {"pid": 900, "created": 9.0}, ["owned-python", "train.py"], "training", run_path,
        tmp_path / "logs/training-001.log")
    assert audit["action_required"] and audit["children_reaped"] and state["waited"]
    assert campaign.read_json(tmp_path / audit["fallback_audit"])["owned_processes"] == audit["owned_processes"]
    assert audit["io_errors"][-1]["operation"] == "run_audit_write"


@pytest.mark.parametrize("stale_lock", [False, True])
def test_ownership_journal_blocks_orphan_restart_when_status_is_stale(tmp_path, monkeypatch, stale_lock):
    worker = {"pid": 103, "created": 103.0}
    directory = tmp_path / "process_ownership"
    directory.mkdir()
    (directory / "case.jsonl").write_text(json.dumps({"owned_processes": [worker]}) + "\n", encoding="utf-8")
    campaign.write_json(tmp_path / "status.json", {"state": "waiting_resources"})
    if stale_lock:
        campaign.write_json(tmp_path / "supervisor.lock", {"pid": 800, "created": 8.0})
    monkeypatch.setattr(campaign, "process_identity", lambda pid=None: {"pid": 900, "created": 9.0})
    monkeypatch.setattr(campaign, "is_alive", lambda identity: identity == worker)
    with pytest.raises(RuntimeError, match="worker"):
        with campaign.campaign_lock(tmp_path):
            pytest.fail("live orphan must prevent another model launch")
    assert not (tmp_path / "supervisor.lock").exists()


@pytest.mark.parametrize("payload", ["", "{", '{"owned_processes": []}\n'])
def test_invalid_ownership_receipt_fails_closed(tmp_path, payload):
    directory = tmp_path / "process_ownership"
    directory.mkdir()
    (directory / "case.jsonl").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        campaign.recorded_owned_processes(tmp_path)


def supervise_fixture(tmp_path, monkeypatch, *, phase="collection"):
    import psutil
    case = {"case_id": "case", "task_uid": "task"}
    protocol = {"collection": {"cases": [case] if phase == "collection" else []},
                "evaluation": {"cases": []}, "resource_limits": {
                    "initial_min_system_available_bytes": 1, "minimum_free_disk_bytes": 1,
                    "max_attempts_per_case": 3, "retry_wait_seconds": 0},
                "tasks": [{"task_uid": "task", "path": "task.bddl", "initial_path": "init.dat",
                           "bddl_sha": "hash", "hash": "hash"}]}

    @contextmanager
    def lock(path):
        yield {"pid": 900, "created": 9.0}

    monkeypatch.setattr(campaign, "verify_protocol", lambda path: protocol)
    monkeypatch.setattr(campaign, "environment_python", lambda root: "owned-python")
    monkeypatch.setattr(campaign, "campaign_lock", lock)
    monkeypatch.setattr(campaign, "deadline_reached", lambda protocol: False)
    monkeypatch.setattr(campaign, "complete_cases", lambda *args: {})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * 1024**3))
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(campaign, "digest", lambda path: "hash")
    monkeypatch.setattr(campaign, "build_report", lambda path: {"status": "partial"})
    return case


@pytest.mark.parametrize("phase", ["collection", "training"])
def test_persistent_io_error_never_automatically_retries_work(tmp_path, monkeypatch, phase):
    supervise_fixture(tmp_path, monkeypatch, phase=phase)
    calls = []

    def monitor(*args):
        calls.append(args)
        return {"action_required": True, "stop_reason": "supervisor_io_failure",
                "children_reaped": True, "exit_code": 0}

    monkeypatch.setattr(campaign, "monitor_child", monitor)
    result = campaign.supervise(tmp_path)
    assert len(calls) == 1 and result["state"] == "needs_attention"
    assert campaign.read_json(tmp_path / "status.json")["reason"] == "supervisor_io_failure"


def test_preoutput_training_resource_stops_have_unique_attempts_and_three_run_cap(tmp_path, monkeypatch):
    supervise_fixture(tmp_path, monkeypatch, phase="training")
    calls = []

    def monitor(root, protocol, identity, command, phase, run_path, log_path):
        calls.append(run_path.stem)
        directory = root / "process_ownership"
        directory.mkdir(exist_ok=True)
        (directory / (run_path.stem + ".jsonl")).write_text(
            json.dumps({"owned_processes": [{"pid": 101, "created": 101.0}]}) + "\n", encoding="utf-8")
        return {"action_required": False, "stop_reason": "resource_guard", "exit_code": 3}

    monkeypatch.setattr(campaign, "monitor_child", monitor)
    result = campaign.supervise(tmp_path)
    assert result["status"] == "partial"
    assert calls == ["training-001", "training-002", "training-003"]
    assert campaign.read_json(tmp_path / "status.json")["reason"] == "training attempt limit reached"


@pytest.mark.parametrize("success", [False, True])
def test_sealed_case_recovery_preserves_outcome_without_model_launch(tmp_path, monkeypatch, success):
    case = supervise_fixture(tmp_path, monkeypatch)
    attempt = tmp_path / "attempts/collection" / campaign.case_directory(case) / "001"
    attempt.mkdir(parents=True)
    campaign.write_json(attempt / "receipt.json", {"synthetic": "strict verifier called below"})
    calls = []

    def verify(path, actual_case, protocol, strict=False):
        assert strict and actual_case == case and path == attempt
        calls.append(path)
        return {}, {"complete": True, "success": success}

    monkeypatch.setattr(campaign, "verify_episode", verify)
    monkeypatch.setattr(campaign, "monitor_child", lambda *args: pytest.fail("sealed case must never rerun"))
    assert campaign.supervise(tmp_path, max_episodes=1)["status"] == "partial"
    assert calls == [attempt]
    assert (tmp_path / "collection" / campaign.case_directory(case) / "receipt.json").exists()
    assert not attempt.exists()
