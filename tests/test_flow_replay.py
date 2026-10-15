"""Saved-action diagnostics tested with a tiny deterministic fake simulator."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/replay_flow_episode.py"
SPEC = importlib.util.spec_from_file_location("flow_replay_tested", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for name in replay.SOURCE_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT / name).read_bytes())
    study_dir = root / "outputs/frontier/flow_confirmation"
    case = {"id": "smolvla__libero_spatial__1__43__clean__noise2", "split": "confirmation"}
    original = study_dir / "episodes" / case["id"]
    original.mkdir(parents=True)
    states = np.repeat(np.arange(4, dtype=np.float32)[:, None] * np.float32(.1), 8, axis=1)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    np.savez_compressed(original / "trajectory.npz", actions=actions, states=states[:-1], final_state=states[-1])
    front = np.zeros((2, 2, 3), dtype=np.uint8)
    Image.fromarray(front).save(original / "initial.png")
    Image.fromarray(front + 9).save(original / "initial_wrist.png")
    Image.fromarray(front + 3).save(original / "final.png")
    environment = {"image_size": 2, "suite": "libero_spatial", "task_id": 1}
    result = {**case, "status": "complete", "suite": "libero_spatial", "task_id": 1,
              "seed": 1000, "init_state_id": 43, "steps": 3, "horizon": 3,
              "success": False, "terminated": False, "truncated": True, "environment": environment,
              "initial_robot_state": states[0].tolist(), "final_robot_state": states[-1].tolist(),
              "initial_image_sha256": hashlib.sha256(front.tobytes()).hexdigest(),
              "protocol_fingerprint": "fixture-protocol", "wall_seconds": 2500.,
              "policy_inference_seconds": 200., "simulator_seconds": 2200.}
    replay.runner.write_json(original / "result.json", result)
    replay.runner.write_json(original / "calls.json", [{"synthetic": "preserve alongside source trajectory"}])
    files = {path.name: replay.runner.file_sha(path) for path in original.iterdir()}
    replay.runner.write_json(original / "receipt.json", {"files": files})
    protocol = {"fingerprint": "fixture-protocol", "runtime_versions": {"numpy": "fixture"},
                "artifacts": {"pinned": "fixture"}, "initial_states": {"pinned": "fixture"}}
    replay.runner.write_json(study_dir / "protocol.json", protocol)
    verified = []

    def verify_study(directory, *, partial, root):
        assert directory == study_dir and partial is False
        verified.append(directory)
        receipt = replay.runner.read_json(original / "receipt.json")
        for name, expected in receipt["files"].items():
            if replay.runner.file_sha(original / name) != expected:
                raise ValueError("fixture source receipt mismatch")
        row = {"case": case, "record": replay.runner.read_json(original / "result.json"),
               "receipt_sha256": replay.runner.file_sha(original / "receipt.json"),
               "result_sha256": replay.runner.file_sha(original / "result.json"),
               "artifact_sha256": receipt["files"]}
        return {"complete": True, "protocol": protocol,
                "records": {case["id"]: row, **{f"other-{index}": {} for index in range(15)}}}

    monkeypatch.setattr(replay.runner, "verify_study", verify_study)
    monkeypatch.setattr(replay.runner, "versions", lambda: protocol["runtime_versions"])
    monkeypatch.setattr(replay.runner, "artifact_provenance", lambda *args: protocol["artifacts"])
    monkeypatch.setattr(replay.runner, "initial_state_manifest", lambda: protocol["initial_states"])

    class Simulator:
        resets = 0
        closed = False
        drift = 0.
        early_stop = False
        mutate_source = False

        def __init__(self, suite, task_id, image_size):
            assert (suite, task_id, image_size) == ("libero_spatial", 1, 2)
            self.index = 0

        def metadata(self):
            return environment

        def frame(self, camera="image"):
            return front + (9 if camera == "image2" else self.index)

        def reset(self, *, seed, init_state_id):
            assert (seed, init_state_id) == (1000, 43)
            type(self).resets += 1
            return {"observation.state": states[0].copy()}

        def step(self, action):
            assert np.array_equal(action, actions[self.index])
            self.index += 1
            state = states[self.index].copy()
            state[0] += np.float32(type(self).drift)
            if type(self).mutate_source and self.index == 3:
                (original / "calls.json").write_text("changed during replay")
            truncated = self.index == 3 or type(self).early_stop
            return {"observation.state": state}, 0., False, truncated, {"is_success": False}

        def close(self):
            type(self).closed = True

    output = root / "outputs/replays/flow-spatial43-noise2"
    return root, original, output, Simulator, verified


def test_exact_replay_records_all_states_and_preserves_originals(fixture):
    root, original, output, simulator, verified = fixture
    before = {path.name: path.read_bytes() for path in original.iterdir()}
    audit = replay.replay(original, output, root=root, simulator_factory=simulator)
    assert audit["status"] == "passed_exact_replay"
    assert audit["all_robot_states_exact"] and audit["max_state_absolute_difference"] == 0
    assert audit["new_policy_inference_calls"] == 0 and audit["excluded_from_benchmark_counts"] is True
    assert audit["success"] is False and audit["steps"] == 3
    assert audit["original_timings"]["simulator_seconds"] == 2200.
    assert "No cause" in audit["causal_attribution"]
    assert simulator.resets == 1 and simulator.closed and len(verified) >= 3
    assert before == {path.name: path.read_bytes() for path in original.iterdir()}
    assert replay.validate_replay(output, root=root) == audit


@pytest.mark.parametrize("fault", ["drift", "early_stop", "source_mutation"])
def test_nonidentical_replay_is_retained_as_failed_diagnostic(fixture, fault):
    root, original, output, simulator, _ = fixture
    if fault == "drift":
        simulator.drift = 1e-8  # Within the old helper's tolerance, but not exact.
    elif fault == "early_stop":
        simulator.early_stop = True
    else:
        simulator.mutate_source = True
    with pytest.raises((ValueError, AssertionError)):
        replay.replay(original, output, root=root, simulator_factory=simulator)
    audit = replay.runner.read_json(output / "flow_replay_audit.json")
    assert audit["status"] == "failed" and audit["excluded_from_benchmark_counts"] is True
    assert audit["new_policy_inference_calls"] == 0 and simulator.closed
    assert (output / "attempt.json").exists() and (output / "wrapper_source.py").exists()
    with pytest.raises(ValueError, match="incomplete or failed"):
        replay.validate_replay(output, root=root)


def test_refuses_formal_output_existing_output_and_runtime_drift(fixture, monkeypatch):
    root, original, output, simulator, _ = fixture
    with pytest.raises(ValueError):
        replay.replay(original, original.parent.parent / "diagnostic", root=root, simulator_factory=simulator)
    monkeypatch.setattr(replay.runner, "versions", lambda: {"numpy": "changed"})
    with pytest.raises(ValueError, match="runtime versions"):
        replay.replay(original, output, root=root, simulator_factory=simulator)
    assert not output.exists() and simulator.resets == 0
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="new diagnostic"):
        replay.replay(original, output, root=root, simulator_factory=simulator)


def test_final_validation_failure_does_not_publish_pass_claims(fixture, monkeypatch):
    root, original, output, simulator, _ = fixture

    def reject_final_validation(*args, **kwargs):
        raise ValueError("final export verification failed")

    monkeypatch.setattr(replay, "validate_replay", reject_final_validation)
    with pytest.raises(ValueError, match="final export verification"):
        replay.replay(original, output, root=root, simulator_factory=simulator)
    audit = replay.runner.read_json(output / "flow_replay_audit.json")
    assert audit["status"] == "failed" and simulator.closed
    assert audit["excluded_from_benchmark_counts"] is True and audit["new_policy_inference_calls"] == 0
    for key in ("all_robot_states_exact", "actions_exact", "initial_image_equal", "initial_wrist_image_equal",
                "final_image_equal", "terminal_flags_equal", "success_equal", "source_evidence_unchanged",
                "max_state_absolute_difference", "success"):
        assert key not in audit
    assert audit["files_sha256"]["replayed_trajectory.npz"] == replay.runner.file_sha(output / "replayed_trajectory.npz")


def test_active_scored_run_blocks_diagnostic_before_simulator(fixture):
    root, original, output, simulator, _ = fixture
    (original.parent.parent / "active.lock").write_text("active")
    with pytest.raises(ValueError, match="scored study process"):
        replay.replay(original, output, root=root, simulator_factory=simulator)
    assert not output.exists() and simulator.resets == 0


@pytest.mark.parametrize("name", ["wrapper_source.py", "base_helper_source.py", "base/replay.gif", "attempt.json"])
def test_export_or_source_tampering_rejected(fixture, name):
    root, original, output, simulator, _ = fixture
    replay.replay(original, output, root=root, simulator_factory=simulator)
    path = output / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        replay.validate_replay(output, root=root)


@pytest.mark.parametrize("kind", ["state", "action", "flags", "image", "scope"])
def test_rehashed_exports_cannot_hide_nonidentical_replay(fixture, kind):
    root, original, output, simulator, _ = fixture
    replay.replay(original, output, root=root, simulator_factory=simulator)
    audit_path = output / "flow_replay_audit.json"
    audit = replay.runner.read_json(audit_path)
    if kind in {"state", "action", "flags"}:
        trajectory = output / "replayed_trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        if kind == "state":
            arrays["states"][1, 0] += np.float32(1e-8)
        elif kind == "action":
            arrays["actions"][0, 0] += 1
        else:
            arrays["flags"][-1, 2] = True
        np.savez_compressed(trajectory, **arrays)
        audit["files_sha256"]["replayed_trajectory.npz"] = replay.runner.file_sha(trajectory)
    elif kind == "image":
        path = output / "final.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
        audit["files_sha256"]["final.png"] = replay.runner.file_sha(path)
    else:
        audit["excluded_from_benchmark_counts"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        replay.validate_replay(output, root=root)


def test_original_receipt_and_helper_source_bindings_rechecked(fixture):
    root, original, output, simulator, _ = fixture
    replay.replay(original, output, root=root, simulator_factory=simulator)
    (root / "scripts/replay_episode.py").write_text("changed helper")
    with pytest.raises(ValueError, match="implementation changed"):
        replay.validate_replay(output, root=root)
