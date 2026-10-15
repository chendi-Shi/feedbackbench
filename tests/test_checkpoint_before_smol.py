"""Pure stdlib fixtures and fake processes; no real process control or psutil."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "checkpoint_before_smol", Path(__file__).resolve().parents[1] / "scripts/checkpoint_before_smol.py")
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed_project(tmp_path, monkeypatch):
    # Production requires exactly 232 cases. One per stage exercises every
    # integrity boundary without creating thousands of redundant fixture files.
    stages = tuple((name, stem, 1) for name, stem, _ in checkpoint.STAGES)
    monkeypatch.setattr(checkpoint, "STAGES", stages)
    root = tmp_path / "project"
    campaign = root / "outputs/campaign"
    campaign.mkdir(parents=True)
    source_hashes = {}
    for name in checkpoint.FROZEN_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unchanged fixture\n")
        source_hashes[name] = checkpoint._hash(path)
    write_json(campaign / "campaign_freeze.json", {"source_sha256": source_hashes})
    write_json(campaign / "campaign_status.json", {
        "status": "running", "through": "flow_steps", "started_at": "original timestamp",
        "stages": {name: {"status": "complete", "completed": 1, "total": 1} for name, _, _ in stages}})
    (campaign / "report.html").write_text("original index")
    stage_inputs, fingerprints = {}, {}
    for name, stem, _ in stages:
        directory = campaign / stem
        directory.mkdir()
        case = {"id": "example", "record_video": False}
        provenance = {"plan": {"name": name}, "runtime_versions": {}, "artifacts": {},
                      "sources": {"experiment.py": source_hashes["src/embodied_research/experiment.py"]}}
        fingerprint = checkpoint._digest(provenance)
        write_json(directory / "protocol.json", {**provenance, "fingerprint": fingerprint, "cases": [case]})
        write_json(campaign / "frozen_plans" / f"{stem}.json", provenance["plan"])
        write_json(directory / "progress.json", {"finished": True, "completed": 1, "total": 1})
        write_json(directory / "runtime_audit.json", {"preserve": "original diagnostic bytes"})
        (directory / "report.html").write_text("original report")
        episode = directory / "episodes/example"
        write_json(episode / "result.json", {**case, "status": "complete", "protocol_fingerprint": fingerprint})
        for filename in ("trajectory.npz", "initial.png", "initial_wrist.png", "final.png"):
            (episode / filename).write_bytes(b"original raw evidence")
        inputs = {"protocol.json": checkpoint._hash(directory / "protocol.json"),
                  "episodes/example/result.json": checkpoint._hash(episode / "result.json")}
        write_json(directory / "summary.json", {"partial": False, "input_sha256": inputs})
        stage_inputs[stem] = inputs
        fingerprints[stem] = fingerprint
    write_json(campaign / "delay_comparison.json", {
        "baseline_input_sha256": stage_inputs["30_methods"], "delayed_input_sha256": stage_inputs["40_delay"]})
    (campaign / "delay_comparison.html").write_text("original comparison")
    write_json(campaign / "development_selection.json", {
        "development_protocol_fingerprint": fingerprints["20_development"],
        "development_inputs_sha256": checkpoint._digest(stage_inputs["20_development"])})
    return root


class FakeProcess:
    def __init__(self, root):
        self.root = root
        self.pid = 1234
        self.created = 123.5
        self.command = ["python.exe", "scripts/run_campaign.py", "--through", "flow_steps"]
        self.alive = True
        self.suspended = False
        self.calls = []
        self.on_suspend = lambda: None
        self.terminate_error = False
        self.wait_error = False

    def is_running(self):
        return self.alive

    def create_time(self):
        return self.created

    def cwd(self):
        return str(self.root)

    def cmdline(self):
        return self.command.copy()

    def status(self):
        return "stopped" if self.suspended else "running"

    def suspend(self):
        self.calls.append("suspend")
        self.suspended = True
        self.on_suspend()

    def resume(self):
        self.calls.append("resume")
        self.suspended = False

    def terminate(self):
        self.calls.append("terminate")
        assert self.suspended
        assert (self.root / checkpoint.AUDIT).is_file()
        if self.terminate_error:
            raise OSError("termination denied")
        if not self.wait_error:
            self.alive = False

    def wait(self, timeout):
        self.calls.append("wait")
        assert timeout == 30
        if self.wait_error:
            raise TimeoutError("process still alive")
        return 0


def stop(root, process):
    return checkpoint.checkpoint_and_stop(root, 1234, 123.5, process_factory=lambda pid: process)


def test_production_denominator_is_232():
    assert [count for _, _, count in checkpoint.STAGES] == [40, 32, 128, 32]


def test_safe_stop_preserves_raw_bytes_and_portable_original_snapshots(completed_project):
    root = completed_project
    process = FakeProcess(root)
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    receipt = stop(root, process)
    assert process.calls == ["suspend", "terminate", "wait"]
    assert receipt["status"] == "stopped" and not process.alive
    assert all((root / relative).read_bytes() == raw for relative, raw in before.items())
    audit_raw = (root / checkpoint.AUDIT).read_bytes()
    audit = json.loads(audit_raw)
    assert str(root).encode() not in audit_raw
    assert hashlib.sha256(audit_raw).hexdigest() == receipt["audit_sha256"]
    assert len(audit["raw_episode_sha256"]) == 20
    for relative, checksum in audit["snapshot_source_sha256"].items():
        raw = (root / checkpoint.CHECKPOINT / relative).read_bytes()
        assert raw == before[relative]
        assert hashlib.sha256(raw).hexdigest() == checksum
    assert not list((root / checkpoint.CHECKPOINT).rglob("trajectory.npz"))
    assert "not standalone report entry points" in (root / checkpoint.CHECKPOINT / "README.md").read_text()


def test_protocol_only_smol_start_is_not_a_formal_attempt(completed_project):
    root = completed_project
    campaign = root / "outputs/campaign"
    write_json(campaign / "50_smolvla/protocol.json", {"created_before_first_progress": True})
    status = checkpoint._read(campaign / "campaign_status.json")
    status["stages"]["smolvla"] = {"status": "running"}
    write_json(campaign / "campaign_status.json", status)
    receipt = stop(root, FakeProcess(root))
    assert receipt["status"] == "stopped"
    snapshot = root / checkpoint.CHECKPOINT / "outputs/campaign/50_smolvla/protocol.json"
    assert snapshot.read_bytes() == (campaign / "50_smolvla/protocol.json").read_bytes()


@pytest.mark.parametrize("fault", ["creation", "pid", "cwd", "entrypoint", "launcher", "output", "through", "already_stopped"])
def test_wrong_identity_never_suspends_or_stops(completed_project, fault):
    root = completed_project
    process = FakeProcess(root)
    if fault == "creation":
        process.created += 1
    elif fault == "pid":
        process.pid += 1
    elif fault == "cwd":
        process.root = root.parent
    elif fault == "entrypoint":
        process.command[1] = "other.py"
    elif fault == "launcher":
        process.command.insert(1, "scripts/run.py")
    elif fault == "output":
        process.command += ["--output", "outputs/other"]
    elif fault == "through":
        process.command[-1] = "delay"
    else:
        process.suspended = True
    with pytest.raises(ValueError):
        stop(root, process)
    assert process.calls == []
    assert not (root / checkpoint.AUDIT).exists()


@pytest.mark.parametrize("fault", ["smol_progress", "smol_tmp", "smol_episodes", "flow", "stage_status",
                                  "progress", "raw_missing", "extra_episode", "source", "summary", "comparison"])
def test_post_suspend_gate_failures_resume_without_stopping(completed_project, fault):
    root = completed_project
    campaign = root / "outputs/campaign"
    process = FakeProcess(root)

    def race():
        # Mutate on suspension to demonstrate the pre-stop check observes the
        # frozen state, not a stale preliminary status read.
        if fault.startswith("smol_"):
            name = {"smol_progress": "progress.json", "smol_tmp": "progress.json.tmp", "smol_episodes": "episodes"}[fault]
            path = campaign / "50_smolvla" / name
            path.parent.mkdir()
            path.mkdir() if fault == "smol_episodes" else path.write_text("{}")
        elif fault == "flow":
            (campaign / "60_flow_steps").mkdir()
        elif fault == "stage_status":
            path = campaign / "campaign_status.json"
            data = checkpoint._read(path)
            data["stages"]["delay"]["status"] = "running"
            write_json(path, data)
        elif fault == "progress":
            write_json(campaign / "40_delay/progress.json", {"finished": False, "completed": 1, "total": 1})
        elif fault == "raw_missing":
            (campaign / "40_delay/episodes/example/trajectory.npz").unlink()
        elif fault == "extra_episode":
            (campaign / "40_delay/episodes/unexpected").mkdir()
        elif fault == "source":
            (root / "scripts/run_campaign.py").write_text("changed")
        elif fault == "summary":
            write_json(campaign / "40_delay/summary.json", {"partial": True})
        else:
            write_json(campaign / "delay_comparison.json", {})

    process.on_suspend = race
    with pytest.raises((ValueError, FileNotFoundError)):
        stop(root, process)
    assert process.calls == ["suspend", "resume"]
    assert process.alive and not process.suspended
    assert not (root / checkpoint.AUDIT).exists()


@pytest.mark.parametrize("when", ["snapshot", "audit"])
def test_write_failure_resumes_and_retains_partial_files(completed_project, monkeypatch, when):
    root = completed_project
    process = FakeProcess(root)
    write = checkpoint._write_new

    def fail(path, raw):
        if (when == "audit") == (path == root / checkpoint.AUDIT):
            path.write_bytes(b"partial forensic evidence")
            raise OSError("simulated full disk")
        write(path, raw)

    monkeypatch.setattr(checkpoint, "_write_new", fail)
    with pytest.raises(OSError, match="full disk"):
        stop(root, process)
    assert process.calls == ["suspend", "resume"]
    assert process.alive and not process.suspended
    assert (root / checkpoint.CHECKPOINT).exists()
    with pytest.raises(FileExistsError):
        stop(root, process)


def test_identity_rechecked_after_durable_audit_before_termination(completed_project, monkeypatch):
    root = completed_project
    process = FakeProcess(root)
    write = checkpoint._write_new

    def mutate_after_audit(path, raw):
        write(path, raw)
        if path == root / checkpoint.AUDIT:
            process.command.append("--unexpected")

    monkeypatch.setattr(checkpoint, "_write_new", mutate_after_audit)
    with pytest.raises(ValueError):
        stop(root, process)
    assert process.calls == ["suspend", "resume"]
    assert (root / checkpoint.AUDIT).exists()


def test_gate_rechecked_after_audit_write_before_termination(completed_project, monkeypatch):
    root = completed_project
    process = FakeProcess(root)
    write = checkpoint._write_new

    def concurrent_writer(path, raw):
        write(path, raw)
        if path == root / checkpoint.AUDIT:
            write_json(root / "outputs/campaign/50_smolvla/progress.json", {"current": "started"})

    monkeypatch.setattr(checkpoint, "_write_new", concurrent_writer)
    with pytest.raises(ValueError, match="attempt evidence"):
        stop(root, process)
    assert process.calls == ["suspend", "resume"]


def test_partial_suspend_failure_attempts_resume(completed_project):
    root = completed_project
    process = FakeProcess(root)

    def fail():
        raise OSError("suspend failed after a partial operation")

    process.on_suspend = fail
    with pytest.raises(OSError, match="partial operation"):
        stop(root, process)
    assert process.calls == ["suspend", "resume"]
    assert process.alive and not process.suspended


def test_pid_reuse_never_resumes_or_terminates_replacement(completed_project):
    root = completed_project
    process = FakeProcess(root)

    def replace():
        process.created += 1

    process.on_suspend = replace
    with pytest.raises(ValueError, match="creation time"):
        stop(root, process)
    assert process.calls == ["suspend"]


@pytest.mark.parametrize("operation", ["terminate", "wait"])
def test_control_failure_resumes_original_live_process(completed_project, operation):
    root = completed_project
    process = FakeProcess(root)
    process.terminate_error = operation == "terminate"
    process.wait_error = operation == "wait"
    with pytest.raises((OSError, TimeoutError)):
        stop(root, process)
    assert process.calls[-1] == "resume"
    assert process.alive and not process.suspended


def test_existing_audit_is_preserved_without_touching_process(completed_project):
    root = completed_project
    (root / checkpoint.AUDIT).write_bytes(b"existing evidence")
    process = FakeProcess(root)
    with pytest.raises(FileExistsError):
        stop(root, process)
    assert process.calls == []
    assert (root / checkpoint.AUDIT).read_bytes() == b"existing evidence"
