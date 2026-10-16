"""Operator state-machine tests: all child processes and memory readings are fake."""
from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/complete_vlajepa_stage.py"
SPEC = importlib.util.spec_from_file_location("complete_vlajepa_stage_tested", PATH)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
GIB = 1024**3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def study(tmp_path):
    output = tmp_path / "outputs/stage"
    output.mkdir(parents=True)
    sources = {}
    for name in (entry.RUNNER, entry.LAUNCHER):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# never executed in these tests\n", encoding="utf-8")
        sources[name] = entry.file_sha(path)
    write_json(output / "inputs.json", {"synthetic": True})
    protocol = {"schema": "feedbackbench.vlajepa_closed_loop_protocol.v1", "sources": sources,
                "inputs_sha256": entry.file_sha(output / "inputs.json"),
                "spec": {"cases": [{"id": "case_" + str(i)} for i in range(4)],
                         "memory_guard": {"minimum_available_gib": 5.0}}}
    protocol["fingerprint"] = entry.digest_json(protocol)
    write_json(output / "protocol.json", protocol)
    return tmp_path, output, protocol


def seal_all(output, protocol):
    for case in protocol["spec"]["cases"]:
        write_json(output / "episodes" / case["id"] / "result.json", {"synthetic": True})
        write_json(output / "episodes" / case["id"] / "receipt.json", {"synthetic": True})


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class FakeChildren:
    def __init__(self, study, *, completes=True, codes=None, validation_complete=None):
        self.root, self.output, self.protocol = study
        self.completes, self.codes = completes, codes or {}
        self.validation_complete = validation_complete
        self.names, self.commands = [], []

    def __call__(self, command, *, cwd, env, stdout, stderr, check):
        assert cwd == self.root and check is False
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert command[0] == entry.sys.executable
        assert command[1:3] == [str(self.root / entry.LAUNCHER), str(self.root / entry.RUNNER)]
        assert "--partial" not in command
        name = "report" if "--report" in command else "validate" if "--validate-only" in command else "run"
        self.names.append(name)
        self.commands.append(command)
        if name == "run" and self.completes:
            seal_all(self.output, self.protocol)
        count = sum((self.output / "episodes" / case["id"] / "receipt.json").exists()
                    for case in self.protocol["spec"]["cases"])
        summary = {"complete": count == 4, "planned": 4, "completed": count,
                   "protocol_fingerprint": self.protocol["fingerprint"], "successes": count, "task_failures": 0,
                   "statuses": {case["id"]: "complete" if count == 4 else "missing"
                                for case in self.protocol["spec"]["cases"]}}
        if name == "report":
            write_json(self.output / "report.json", summary)
        if name == "validate" and self.validation_complete is not None:
            summary["complete"] = self.validation_complete
        stdout.write(json.dumps(summary).encode())
        stderr.write(b"synthetic child only\n")
        return SimpleNamespace(returncode=self.codes.get(name, 0 if count == 4 or name != "validate" else 1))


def invoke(study, children, *, memory=lambda: 6 * GIB, clock=None, **kwargs):
    root, output, _ = study
    clock = clock or FakeClock()
    return entry.complete_stage(output, root=root, memory=memory, monotonic=clock.now,
                                sleep=clock.sleep, subprocess_run=children, **kwargs)


def test_blocked_memory_launches_nothing_and_each_invocation_has_new_logs(study):
    children = FakeChildren(study)
    original = (study[1] / "protocol.json").read_bytes()
    first = invoke(study, children, memory=lambda: 5 * GIB - 1)
    second = invoke(study, children, memory=lambda: 5 * GIB - 1)
    assert first["exit_code"] == second["exit_code"] == 3
    assert first["status"] == "blocked_memory_wait_expired"
    assert first["operator_record"] != second["operator_record"]
    assert not children.names and (study[1] / "protocol.json").read_bytes() == original


def test_wait_is_finite_and_never_sleeps_past_deadline(study):
    children, clock = FakeChildren(study), FakeClock()
    result = invoke(study, children, memory=lambda: 4 * GIB, clock=clock, wait_seconds=65, poll_seconds=30)
    assert result["exit_code"] == 3 and not children.names
    assert clock.sleeps == [30, 30, 5]
    assert [sample["elapsed_seconds"] for sample in result["memory_samples"]] == [0, 30, 60, 65]


def test_ready_after_wait_runs_then_reports_then_strictly_validates(study):
    children, clock = FakeChildren(study), FakeClock()
    readings = iter([4 * GIB, 5 * GIB])
    result = invoke(study, children, memory=lambda: next(readings), clock=clock, wait_seconds=60)
    assert result["status"] == "complete_verified" and result["exit_code"] == 0
    assert result["completed"] == 4 and clock.sleeps == [30]
    assert children.names == ["run", "report", "validate"]


def test_runner_zero_exit_does_not_turn_missing_episodes_into_success(study):
    children = FakeChildren(study, completes=False)
    result = invoke(study, children)
    assert result["exit_code"] != 0 and result["status"] == "validation_failed_or_incomplete"
    assert children.names == ["run", "report", "validate"]


@pytest.mark.parametrize("error", [False, True])
def test_partial_or_error_is_never_automatically_retried(study, error):
    directory = study[1] / "episodes/case_0"
    write_json(directory / ("error.json" if error else "attempt.json"), {})
    children = FakeChildren(study)
    result = invoke(study, children)
    assert result["exit_code"] == 4 and result["status"] == "partial_or_error_requires_review"
    assert not children.names and directory.exists()


def test_existing_lock_is_preserved_and_blocks_launch(study):
    lock = study[1] / "active.lock"
    lock.write_text("retained", encoding="utf-8")
    children = FakeChildren(study)
    result = invoke(study, children)
    assert result["exit_code"] == 4 and not children.names
    assert lock.read_text() == "retained"


def test_already_sealed_stage_only_runs_consumers_without_loading_or_memory_wait(study):
    seal_all(study[1], study[2])
    children = FakeChildren(study)

    def no_memory_query():
        raise AssertionError("No model will be loaded")

    result = invoke(study, children, memory=no_memory_query)
    assert result["exit_code"] == 0 and children.names == ["report", "validate"]


@pytest.mark.parametrize("name,code", [("run", 7), ("run", 3), ("report", 5), ("validate", 2)])
def test_child_errors_propagate_even_when_other_evidence_looks_complete(study, name, code):
    children = FakeChildren(study, codes={name: code})
    result = invoke(study, children)
    assert result["exit_code"] == code and result["status"] != "complete_verified"
    assert children.names == ["run", "report", "validate"]


def test_validation_zero_without_complete_payload_is_not_completion(study):
    result = invoke(study, FakeChildren(study, validation_complete=False))
    assert result["exit_code"] == 1 and result["status"] == "error_not_complete"


def test_frozen_runner_hash_mismatch_blocks_all_subprocesses(study):
    (study[0] / entry.RUNNER).write_text("changed", encoding="utf-8")
    children = FakeChildren(study)
    result = invoke(study, children)
    assert result["exit_code"] == 1 and not children.names


def test_changed_state_while_waiting_does_not_launch_competing_runner(study):
    clock, children = FakeClock(), FakeChildren(study)
    readings = iter([4 * GIB, 6 * GIB])

    def sleeping(seconds):
        clock.sleep(seconds)
        (study[1] / "active.lock").write_text("another invocation", encoding="utf-8")

    result = entry.complete_stage(study[1], root=study[0], wait_seconds=60, memory=lambda: next(readings),
                                  monotonic=clock.now, sleep=sleeping, subprocess_run=children)
    assert result["exit_code"] == 1 and not children.names


def test_atomic_status_replace_retries_temporary_sharing_denial(tmp_path):
    pending, original = tmp_path / "status.json.tmp", tmp_path / "status.json"
    pending.write_text("new status", encoding="utf-8")
    original.write_text("old status", encoding="utf-8")
    attempts, sleeps = [], []

    def temporarily_locked(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, "synthetic sharing denial")
        entry.os.replace(source, destination)

    entry.replace_with_retry(pending, original, replace=temporarily_locked, sleep=sleeps.append)
    assert len(attempts) == 3 and sleeps == list(entry.REPLACE_DELAYS[:2])
    assert original.read_text() == "new status" and not pending.exists()


def test_atomic_status_replace_permanent_denial_fails_without_damaging_old_status(tmp_path):
    pending, original = tmp_path / "status.json.tmp", tmp_path / "status.json"
    pending.write_text("new status", encoding="utf-8")
    original.write_text("old status", encoding="utf-8")
    attempts, sleeps = [], []

    def permanently_locked(source, destination):
        attempts.append(1)
        raise PermissionError(errno.EACCES, "synthetic persistent denial")

    with pytest.raises(PermissionError, match="persistent denial"):
        entry.replace_with_retry(pending, original, replace=permanently_locked, sleep=sleeps.append)
    assert len(attempts) == len(entry.REPLACE_DELAYS) + 1 and sleeps == list(entry.REPLACE_DELAYS)
    assert original.read_text() == "old status" and pending.read_text() == "new status"


def test_status_failure_after_child_exit_saves_new_terminal_record_and_returns_nonzero(study, monkeypatch):
    children = FakeChildren(study, completes=False, codes={"run": 3})

    def reject_finished_child_status(source, destination):
        pending = entry.read_json(source)
        if pending["children"] and pending["children"][-1].get("returncode") == 3:
            raise PermissionError(errno.EACCES, "synthetic persistent status replacement denial")
        entry.os.replace(source, destination)

    monkeypatch.setattr(entry, "replace_with_retry", reject_finished_child_status)
    result = invoke(study, children)
    assert result["status"] == "status_record_write_failed" and result["exit_code"] != 0
    assert result["failed_phase"] == "running_frozen_stage" and children.names == ["run"]
    assert result["primary_status_is_stale"] and result["terminal_record_persisted"]
    terminal = study[0] / result["operator_record"]
    assert terminal.name.startswith("terminal_failure_")
    assert entry.read_json(terminal)["children"][-1]["returncode"] == 3
    old_status = entry.read_json(study[0] / result["previous_operator_record"])
    assert old_status["status"] == "running_frozen_stage" and "returncode" not in old_status["children"][-1]
