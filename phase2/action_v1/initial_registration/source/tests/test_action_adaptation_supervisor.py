"""Ensure probe cleanup cannot terminate a reused or unrelated PID."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import psutil
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("action_probe_supervisor_test", SCRIPTS / "run_action_adaptation_probe.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_reused_pid_is_never_terminated(monkeypatch):
    actions = []
    reused = SimpleNamespace(create_time=lambda: 200.0, is_running=lambda: True,
                             terminate=lambda: actions.append("terminate"))
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: reused)
    monkeypatch.setattr(supervisor.psutil, "wait_procs", lambda values, timeout: ([], []))
    assert supervisor.stop_owned([{"pid": 123, "created": 100.0}]) == []
    assert not actions


def test_only_recorded_exact_identity_is_stopped(monkeypatch):
    actions = []
    process = SimpleNamespace(pid=123, create_time=lambda: 100.0, is_running=lambda: True,
                              terminate=lambda: actions.append(123))
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(supervisor.psutil, "wait_procs", lambda values, timeout: (values, []))
    record = {"pid": 123, "created": 100.0}
    assert supervisor.stop_owned([record]) == [record]
    assert actions == [123]


def test_disappearing_descendant_is_not_misidentified(monkeypatch):
    def vanished():
        raise psutil.NoSuchProcess(456)

    parent = SimpleNamespace(pid=123, children=lambda recursive: [SimpleNamespace(pid=456, create_time=vanished)])
    monkeypatch.setattr(supervisor, "matching", lambda record: parent)
    records = [{"pid": 123, "created": 100.0}]
    supervisor.collect_owned(records[0], records)
    assert records == [{"pid": 123, "created": 100.0}]


def test_unreadable_identity_fails_closed(monkeypatch):
    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(supervisor.psutil, "Process", denied)
    with pytest.raises(psutil.AccessDenied):
        supervisor.matching({"pid": 123, "created": 100.0})
