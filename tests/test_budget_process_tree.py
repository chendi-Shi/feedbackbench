"""Small process-identity regressions; no model, Torch, or simulator imports."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_budget_campaign as campaign


@pytest.mark.parametrize("stale_lock", [False, True])
def test_live_worker_blocks_restart_even_without_supervisor_lock(tmp_path, monkeypatch, stale_lock):
    supervisor = {"pid": 901, "created": 9.0}
    worker = {"pid": 124, "created": 2.0}
    campaign.write_json(tmp_path / "status.json", {"child_processes": [worker]})
    if stale_lock:
        campaign.write_json(tmp_path / "supervisor.lock", {"pid": 900, "created": 8.0})
    monkeypatch.setattr(campaign, "process_identity", lambda pid=None: supervisor)
    monkeypatch.setattr(campaign, "is_alive", lambda identity: identity == worker)
    with pytest.raises(RuntimeError, match="child|worker"):
        with campaign.campaign_lock(tmp_path):
            pytest.fail("a previous model worker is still alive")
    # A failed acquisition may retain the old lock, but must not leave a newly
    # created lock falsely claiming that this supervisor is running.
    lock = tmp_path / "supervisor.lock"
    assert not lock.exists() or campaign.read_json(lock) != supervisor


def test_orphaned_owned_worker_discovers_descendants_and_enforces_tree_rss(tmp_path, monkeypatch):
    """A venv launcher exits; its observed child later spawns another child."""
    import psutil

    state = {"tick": 0, "terminated": [], "statuses": [], "waited": False}
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
            return self.pid != 101 or state["tick"] == 0

        def ppid(self):
            return {101: 1, 102: 101, 103: 102}[self.pid]

        def parent(self):
            return None if self.pid == 101 else FakeProcess(self.ppid())

        def children(self, recursive=False):
            assert recursive is True
            if not self.is_running():
                raise psutil.NoSuchProcess(self.pid)
            if self.pid == 101:
                return [FakeProcess(102)]
            if self.pid == 102 and state["tick"] > 0:
                return [FakeProcess(103)]
            return []

        def memory_info(self):
            return SimpleNamespace(rss=rss[self.pid])

        def terminate(self):
            assert self.is_running(), "identity/liveness must be checked before termination"
            state["terminated"].append(self.pid)

    class FakeLauncher:
        pid = 101

        def poll(self):
            return None if state["tick"] == 0 else 0

        def wait(self):
            assert 102 in state["terminated"] and 103 in state["terminated"]
            state["waited"] = True
            return 0

    def advance(seconds):
        state["tick"] += 1
        assert state["tick"] < 4, "orphan grandchild was missed or the resource stop did not occur"

    def wait_procs(processes, timeout):
        return ([p for p in processes if not p.is_running()],
                [p for p in processes if p.is_running()])

    monkeypatch.setattr(psutil, "Process", FakeProcess)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1000))
    monkeypatch.setattr(psutil, "wait_procs", wait_procs)
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: FakeLauncher())
    monkeypatch.setattr(campaign.time, "monotonic", lambda: float(state["tick"] * 5))
    monkeypatch.setattr(campaign.time, "sleep", advance)
    monkeypatch.setattr(campaign, "deadline_reached", lambda protocol: False)
    monkeypatch.setattr(campaign, "status_write", lambda *args, **details: state["statuses"].append(details))
    protocol = {"resource_limits": {"max_process_rss_bytes": 8,
                "min_system_available_bytes": 1, "resource_stop_grace_seconds": 0}}
    audit = campaign.monitor_child(tmp_path, protocol, {"pid": 900, "created": 9.0},
                                   ["owned-python", "worker.py"], "collection",
                                   tmp_path / "run.json", tmp_path / "log.txt")
    assert audit["stop_reason"] == "resource_guard"
    assert audit["peak_rss_bytes"] == 10
    assert state["terminated"] == [103, 102]
    assert state["waited"]
    assert {row["pid"] for row in audit["owned_processes"]} == {101, 102, 103}
    assert any({row["pid"] for row in status["child_processes"]} == {102, 103}
               for status in state["statuses"])
