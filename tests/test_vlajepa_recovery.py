"""Recovery boundary and accounting contracts; no model or simulator imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_runner_tested", ROOT / "scripts/run_vlajepa_recovery.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
FEATURES = ("observation.images.image", "observation.images.image2", "observation.state")
ARRAYS = ("model_actions", "environment_actions", "noise", "rng_state", "input_robot_state")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def validate_observation(observation):
    require(set(observation) == {*FEATURES, "task"}, "Unexpected observation fields")
    for key in FEATURES:
        value = observation[key]
        require(isinstance(value, np.ndarray) and value.dtype == np.float32 and np.isfinite(value).all(),
                "Expected finite float32 observation")
        require(value.shape == ((8,) if key == FEATURES[2] else (3, 360, 360)), "Unexpected observation shape")


@pytest.fixture
def runner():
    return SimpleNamespace(require=require, FEATURES=FEATURES, array_sha=array_sha, file_sha=sha,
                           read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
                           validate_observation=validate_observation, write_json=write_json)


@pytest.fixture
def observation():
    return {FEATURES[0]: np.zeros((3, 360, 360), dtype=np.float32),
            FEATURES[1]: np.full((3, 360, 360), 0.25, dtype=np.float32),
            FEATURES[2]: np.arange(8, dtype=np.float32) / 10,
            "task": "Put both objects into the basket"}


@pytest.fixture
def saved_row(observation):
    return {"step": 398, "action": [0.0] * 6 + [1.0],
            "next_robot_state": observation[FEATURES[2]].tolist(),
            "next_camera_sha256": {key: array_sha(observation[key]) for key in FEATURES[:2]},
            "terminated": False, "truncated": False, "success": False}


def test_import_does_not_load_model_or_simulator():
    code = ("import runpy,sys;runpy.run_path(sys.argv[1]);"
            "assert not ({'torch','mujoco','robosuite','libero'} & sys.modules.keys())")
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/run_vlajepa_recovery.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_exact_observation_accepts_saved_state_and_two_camera_hashes(runner, observation, saved_row):
    entry.exact_observation(runner, observation, saved_row)


@pytest.mark.parametrize("feature", FEATURES)
def test_exact_observation_rejects_even_one_changed_float(runner, observation, saved_row, feature):
    value = observation[feature]
    value.flat[0] = np.nextafter(value.flat[0], np.float32(1.0))
    with pytest.raises(ValueError):
        entry.exact_observation(runner, observation, saved_row)


def test_exact_observation_requires_all_eight_state_values(runner, observation, saved_row):
    observation[FEATURES[2]] = observation[FEATURES[2]][:7]
    with pytest.raises(ValueError):
        entry.exact_observation(runner, observation, saved_row)


def test_exact_step_checks_observation_and_all_terminal_flags(runner, observation, saved_row):
    entry.exact_step(runner, observation, False, False, {"is_success": False}, saved_row)


@pytest.mark.parametrize("flag", ["terminated", "truncated", "success"])
def test_exact_step_rejects_each_changed_terminal_flag(runner, observation, saved_row, flag):
    flags = {"terminated": False, "truncated": False, "success": False}
    flags[flag] = True
    with pytest.raises(ValueError):
        entry.exact_step(runner, observation, flags["terminated"], flags["truncated"],
                         {"is_success": flags["success"]}, saved_row)


def test_exact_step_rejects_changed_observation_even_when_flags_match(runner, observation, saved_row):
    observation[FEATURES[2]][0] = 0.5
    with pytest.raises(ValueError):
        entry.exact_step(runner, observation, False, False, {"is_success": False}, saved_row)


@pytest.fixture
def boundary(tmp_path, runner):
    new_dir, source_dir = tmp_path / "new", tmp_path / "source"
    new_dir.mkdir()
    source_dir.mkdir()
    model = np.linspace(-1, 1, 49, dtype=np.float32).reshape(1, 7, 7)
    environment = model.copy()
    environment[..., 6] = 1.0
    arrays = {"model_actions": model, "environment_actions": environment,
              "noise": np.arange(49, dtype=np.float32).reshape(1, 7, 7) / 100,
              "rng_state": np.arange(5056, dtype=np.uint16).astype(np.uint8),
              "input_robot_state": np.arange(8, dtype=np.float32) / 10}
    record = {"call_index": 56, "status": "complete", "forward_returned": True,
              "input_sha256": {key: str(index + 1) * 64 for index, key in enumerate(FEATURES)},
              "task": "Put both objects into the basket", "action_head_nfe": 4,
              "rng_state_sha256": array_sha(arrays["rng_state"]), "noise_sha256": array_sha(arrays["noise"]),
              "quantized_module_calls": {f"qwen.module_{index:03d}": 1 for index in range(300)},
              "forward_seconds": 4.0, "preprocess_seconds": 0.01, "postprocess_seconds": 0.02}
    for directory, index in ((source_dir, 56), (new_dir, 0)):
        path = directory / f"call_{index:04d}.npz"
        np.savez_compressed(path, **arrays)
        row = deepcopy(record)
        row.update(call_index=index, artifact_sha256=sha(path))
        if index == 0:
            row.update(forward_seconds=8.0, preprocess_seconds=0.02, postprocess_seconds=0.04)
        write_json(path.with_suffix(".json"), row)
    return SimpleNamespace(runner=runner, new_dir=new_dir, source_dir=source_dir, arrays=arrays)


def update_boundary_record(boundary, change):
    path = boundary.new_dir / "call_0000.json"
    row = boundary.runner.read_json(path)
    change(row)
    write_json(path, row)


def update_boundary_arrays(boundary, arrays):
    path = boundary.new_dir / "call_0000.npz"
    np.savez_compressed(path, **arrays)
    def update(row):
        row["artifact_sha256"] = sha(path)
        for array_name, field in (("noise", "noise_sha256"), ("rng_state", "rng_state_sha256")):
            if array_name in arrays:
                row[field] = array_sha(arrays[array_name])
    update_boundary_record(boundary, update)


def test_boundary_accepts_exact_numeric_recompute_with_different_timings(boundary):
    entry.compare_boundary(boundary.runner, boundary.new_dir, boundary.source_dir)


@pytest.mark.parametrize("name", ARRAYS)
def test_boundary_rejects_each_changed_array_even_after_rehashing_archive(boundary, name):
    arrays = {key: value.copy() for key, value in boundary.arrays.items()}
    arrays[name].flat[0] += 1
    update_boundary_arrays(boundary, arrays)
    with pytest.raises(ValueError):
        entry.compare_boundary(boundary.runner, boundary.new_dir, boundary.source_dir)


def test_boundary_rejects_different_array_dtype_even_for_same_values(boundary):
    arrays = {key: value.copy() for key, value in boundary.arrays.items()}
    arrays["model_actions"] = arrays["model_actions"].astype(np.float64)
    update_boundary_arrays(boundary, arrays)
    with pytest.raises(ValueError):
        entry.compare_boundary(boundary.runner, boundary.new_dir, boundary.source_dir)


@pytest.mark.parametrize("change", ["missing_array", "extra_array"])
def test_boundary_requires_exact_five_array_inventory(boundary, change):
    arrays = {key: value.copy() for key, value in boundary.arrays.items()}
    if change == "missing_array":
        arrays.pop("input_robot_state")
    else:
        arrays["untracked"] = np.zeros(1, dtype=np.float32)
    update_boundary_arrays(boundary, arrays)
    with pytest.raises(ValueError):
        entry.compare_boundary(boundary.runner, boundary.new_dir, boundary.source_dir)


@pytest.mark.parametrize("field", ["quantized_module_calls", "input_sha256", "task", "action_head_nfe"])
def test_boundary_rejects_changed_runtime_contract(boundary, field):
    def change(row):
        if field == "quantized_module_calls":
            row[field]["qwen.module_000"] = 2
        elif field == "input_sha256":
            row[field][FEATURES[0]] = "0" * 64
        elif field == "task":
            row[field] = "Different instruction"
        else:
            row[field] = 3
    update_boundary_record(boundary, change)
    with pytest.raises(ValueError):
        entry.compare_boundary(boundary.runner, boundary.new_dir, boundary.source_dir)


@pytest.mark.parametrize("step,expected", [(1, "saved_action_replay"), (392, "saved_action_replay"),
                                           (393, "recomputed_prefix_overlap"), (398, "recomputed_prefix_overlap"),
                                           (399, "new_continuation"), (520, "new_continuation")])
def test_recovery_phase_boundaries(step, expected):
    assert entry.phase_for_step(step) == expected


@pytest.mark.parametrize("steps,calls", [(399, 1), (400, 2), (406, 2), (407, 3), (518, 18), (519, 19), (520, 19)])
def test_recovery_counts_exclude_replay_and_boundary_call_from_new_work(steps, calls):
    counts = entry.recovery_counts(steps, calls)
    assert counts["independent_benchmark_samples"] == 0
    assert counts["replayed_steps"] == 392
    assert counts["recomputed_overlap_steps"] == 6
    assert counts["new_execution_steps"] == steps - 398
    assert counts["boundary_recomputed_forwards"] == 1
    assert counts["new_policy_forwards"] == calls - 1


@pytest.mark.parametrize("steps,calls", [(398, 1), (521, 19), (399, 0), (399, 2), (520, 18), (520, 20)])
def test_recovery_counts_reject_invalid_horizon_or_forward_denominator(steps, calls):
    with pytest.raises(ValueError):
        entry.recovery_counts(steps, calls)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("name", ["vlajepa_closed_loop", "vlajepa_paged_closed_loop", "vlajepa_compact_closed_loop",
                                  "vlajepa_isolated_closed_loop", "vlajepa_long_horizon"])
@pytest.mark.parametrize("child", ["", "child"])
def test_recovery_cannot_write_into_any_old_experiment(workspace, name, child):
    with pytest.raises(ValueError):
        entry.separate_output(workspace / "outputs/frontier" / name / child)


def test_recovery_output_accepts_new_sibling_and_rejects_ancestors(workspace):
    target = workspace / "outputs/frontier/vlajepa_recovery"
    assert entry.separate_output(target) == target.resolve()
    for target in (workspace, workspace / "outputs/frontier", workspace.parent / "outside"):
        with pytest.raises(ValueError):
            entry.separate_output(target)


@pytest.fixture
def source_evidence(tmp_path, monkeypatch, runner):
    source_root = tmp_path / "old"
    first = source_root / "runs/run_0001.json"
    second = source_root / "episodes/old-case/call_0056.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b'{"status":"interrupted"}\n')
    second.write_bytes(b"Preserved source bytes\n")
    monkeypatch.setattr(entry, "SOURCE_OUTPUT", source_root)
    runner.base = SimpleNamespace(relative=lambda path: Path(path).relative_to(tmp_path).as_posix())
    frozen = {"files": entry.source_inventory(runner)}
    return SimpleNamespace(runner=runner, root=source_root, first=first, second=second, frozen=frozen)


def test_source_inventory_binds_every_original_file_and_byte_length(source_evidence):
    fake = source_evidence
    assert fake.frozen["files"] == {
        "old/runs/run_0001.json": {"sha256": sha(fake.first), "bytes": fake.first.stat().st_size},
        "old/episodes/old-case/call_0056.npz": {"sha256": sha(fake.second), "bytes": fake.second.stat().st_size}}
    entry.verify_source_inventory(fake.runner, fake.frozen)


@pytest.mark.parametrize("change", ["modified", "added", "missing"])
def test_source_inventory_rejects_original_evidence_changes(source_evidence, change):
    fake = source_evidence
    if change == "modified":
        fake.second.write_bytes(b"Modified source bytes\n")
    elif change == "added":
        (fake.root / "new-unfrozen.txt").write_bytes(b"new")
    else:
        fake.second.unlink()
    with pytest.raises(ValueError):
        entry.verify_source_inventory(fake.runner, fake.frozen)


def tiny_observation(step):
    value = np.float32(step / 1024)
    return {FEATURES[0]: np.full((3, 2, 2), value, dtype=np.float32),
            FEATURES[1]: np.full((3, 2, 2), value / 2, dtype=np.float32),
            FEATURES[2]: np.full(8, value, dtype=np.float32), "task": "Known test task"}


def tiny_actions(start_step):
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:, 0] = np.arange(start_step + 1, start_step + 8, dtype=np.float32) / 1000
    actions[:, 6] = 1.0
    return actions


@pytest.fixture
def rollout(tmp_path, monkeypatch, runner):
    source_root = tmp_path / "old"
    source_dir = source_root / "episodes" / "old-case"
    source_dir.mkdir(parents=True)
    destination = tmp_path / "new"
    destination.mkdir()
    source = {"episode_id": "old-case", "files": {"frozen-test-file": {"sha256": "a" * 64, "bytes": 1}}}
    before = np.arange(32, dtype=np.uint8)
    context = SimpleNamespace(rng=np.zeros(32, dtype=np.uint8), restore_calls=[], boundary_checks=[],
                              inventory_checks=[], success_step=None, bad_observation_step=None)

    def inventory(unused_runner):
        context.inventory_checks.append(True)
        return deepcopy(source["files"])

    def restore(value):
        context.restore_calls.append(value.copy())
        context.rng = value.copy()

    def boundary_check(unused_runner, actual, original):
        context.boundary_checks.append((Path(actual), Path(original)))
        return {"status": "exact_recomputed_boundary", "source_call": 56, "actual_call": 0,
                "arrays_equal": sorted(ARRAYS), "module_call_counts_equal": True, "action_head_nfe": 4}

    monkeypatch.setattr(entry, "SOURCE_OUTPUT", source_root)
    monkeypatch.setattr(entry, "source_inventory", inventory)
    monkeypatch.setattr(entry, "restore_rng", restore)
    monkeypatch.setattr(entry, "current_rng", lambda: context.rng.copy())
    monkeypatch.setattr(entry, "compare_boundary", boundary_check)
    runner.digest_json = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    runner.utc_now = lambda: "test-finished"
    saved = []
    for step in range(1, 399):
        obs = tiny_observation(step)
        saved.append({"step": step, "action": tiny_actions(((step - 1) // 7) * 7)[(step - 1) % 7].tolist(),
                      "next_robot_state": obs[FEATURES[2]].tolist(),
                      "next_camera_sha256": {key: array_sha(obs[key]) for key in FEATURES[:2]},
                      "terminated": False, "truncated": False, "success": False})
    log_path = source_dir / "executed_steps.jsonl"
    log_path.write_text("".join(json.dumps(row) + "\n" for row in saved), encoding="utf-8")
    write_json(source_dir / "call_0000.json", {"input_sha256": {key: array_sha(tiny_observation(0)[key]) for key in FEATURES}})
    for index in range(57):
        np.savez_compressed(source_dir / f"call_{index:04d}.npz", environment_actions=tiny_actions(index * 7)[None],
                            rng_state=before)

    class Simulator:
        horizon = 520
        task = "Known test task"

        def __init__(self):
            self.steps = 0
            self.actions = []
            self.resets = []

        def reset(self, **kwargs):
            self.resets.append(kwargs)
            self.steps = 0
            return tiny_observation(0)

        def step(self, action):
            expected = tiny_actions((self.steps // 7) * 7)[self.steps % 7]
            require(np.array_equal(action, expected), "Fake simulator received wrong chronological action")
            self.actions.append(action.copy())
            self.steps += 1
            observation = tiny_observation(self.steps)
            if self.steps == context.bad_observation_step:
                observation[FEATURES[2]][0] += 0.1
            success = self.steps == context.success_step
            return observation, float(success), success, self.steps == self.horizon and not success, {"is_success": success}

        def frame(self, key="image"):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def metadata(self):
            return {"backend": "lightweight-test"}

    simulator = Simulator()

    class Adapter:
        def __init__(self):
            self.inference_count = 0
            self.forward_steps = []
            self.forward_rng = []
            self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

        def reset(self, seed):
            self.inference_count = 0
            self.reset_seed = seed

        def observe(self, observation):
            self.observation = observation
            self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

        def predict_chunk(self):
            self.forward_steps.append(simulator.steps)
            self.forward_rng.append(context.rng.copy())
            actions = tiny_actions(simulator.steps)
            index = self.inference_count
            write_json(destination / f"call_{index:04d}.json", {"call_index": index, "status": "complete"})
            np.savez_compressed(destination / f"call_{index:04d}.npz", environment_actions=actions[None],
                                rng_state=context.rng)
            context.rng = context.rng + np.uint8(1)
            self.inference_count += 1
            self.last_timing = {"inference_seconds": 0.001, "preprocess_seconds": 0.0001}
            return actions

        @property
        def metadata(self):
            return {"inference_count": self.inference_count, "action_head_nfe": self.inference_count * 4}

    policy = Adapter()
    case = {"id": "recovered-case", "seed": 1000, "init_state_id": 41, "condition": "clean", "record_video": False,
            "scheduler": {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False}}
    return SimpleNamespace(runner=runner, source=source, source_dir=source_dir, destination=destination,
                           simulator=simulator, policy=policy, case=case, context=context, saved=saved,
                           before=before, log_path=log_path)


def run_fake_rollout(fake):
    return entry.run_recovery_episode(fake.runner, fake.source, fake.simulator, fake.policy, fake.case, fake.destination)


@pytest.mark.parametrize("success_step", [399, 427, None])
def test_rollout_finishes_success_or_horizon_using_real_fixed_scheduler(rollout, success_step):
    rollout.context.success_step = success_step
    result = run_fake_rollout(rollout)
    steps = success_step or 520
    calls = (steps + 6) // 7 - 56
    assert result["status"] == "complete"
    assert result["steps"] == steps
    assert result["terminated"] is (success_step is not None)
    assert result["truncated"] is (success_step is None)
    assert result["success"] is (success_step is not None)
    assert result["inference_calls"] == calls
    assert result["independent_benchmark_episodes"] == 0
    assert result["recovery_counts"]["replayed_steps"] == 392
    assert result["recovery_counts"]["recomputed_overlap_steps"] == 6
    assert result["recovery_counts"]["new_execution_steps"] == steps - 398
    assert result["recovery_counts"]["new_policy_forwards"] == calls - 1
    assert rollout.policy.forward_steps == list(range(392, steps, 7))
    assert 398 not in rollout.policy.forward_steps  # The 399th action consumes the existing chunk.
    assert np.array_equal(rollout.simulator.actions[398], tiny_actions(392)[6])
    assert len(rollout.context.restore_calls) == len(rollout.context.boundary_checks) == 1
    assert np.array_equal(rollout.policy.forward_rng[0], rollout.before)
    assert len(rollout.context.inventory_checks) == 2
    checks = [json.loads(line) for line in (rollout.destination / "recovery_checks.jsonl").read_text().splitlines()]
    assert len(checks) == 399 and checks[0]["phase"] == "initial"
    assert sum(row["phase"] == "saved_action_replay" for row in checks) == 392
    assert sum(row["phase"] == "recomputed_prefix_overlap" for row in checks) == 6
    with np.load(rollout.destination / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["actions"].shape == (steps, 7)
        assert trajectory["replan_steps"].tolist() == rollout.policy.forward_steps
        assert np.count_nonzero(trajectory["model_seconds"][:392]) == 0
        assert np.count_nonzero(trajectory["model_seconds"]) == calls


@pytest.mark.parametrize("mismatch", ["saved_action", "saved_observation"])
def test_replay_mismatch_stops_before_any_model_forward(rollout, mismatch):
    if mismatch == "saved_action":
        rollout.saved[9]["action"][0] += 0.1
        rollout.log_path.write_text("".join(json.dumps(row) + "\n" for row in rollout.saved), encoding="utf-8")
    else:
        rollout.context.bad_observation_step = 10
    with pytest.raises(ValueError):
        run_fake_rollout(rollout)
    assert rollout.policy.inference_count == 0
    assert rollout.simulator.steps <= 10
    assert rollout.context.restore_calls == rollout.context.boundary_checks == []
    assert not (rollout.destination / "recovery.json").exists()


def test_wrong_restored_rng_stops_before_boundary_model_forward(rollout, monkeypatch):
    monkeypatch.setattr(entry, "restore_rng", lambda before: None)
    with pytest.raises(ValueError, match="RNG"):
        run_fake_rollout(rollout)
    assert rollout.simulator.steps == 392
    assert rollout.policy.inference_count == 0
    assert rollout.context.boundary_checks == []


def test_overlap_mismatch_stops_before_any_later_model_forward(rollout):
    rollout.context.bad_observation_step = 395
    with pytest.raises(ValueError):
        run_fake_rollout(rollout)
    assert rollout.simulator.steps == 395
    assert rollout.policy.inference_count == 1
    assert rollout.policy.forward_steps == [392]
    assert not (rollout.destination / "recovery.json").exists()


@pytest.fixture
def study(tmp_path, runner):
    output = tmp_path / "recovery"
    output.mkdir()
    case = {"id": "recovery-case", "suite": "libero_10", "task_id": 1, "init_state_id": 41, "seed": 1000}
    protocol = {"fingerprint": "f" * 64, "spec": {"cases": [case], "limitations": ["Recovery is not an independent sample"]}}
    result = {**case, "success": False, "steps": 520, "recovery_counts": entry.recovery_counts(520, 19)}
    verified = []
    runner.verify_protocol = lambda destination: deepcopy(protocol)
    runner.cases_for_study = lambda: [deepcopy(case)]

    def verify_episode(directory, supplied_case, supplied_protocol):
        verified.append((Path(directory), supplied_case, supplied_protocol))
        return deepcopy(result)

    runner.verify_episode = verify_episode
    return SimpleNamespace(runner=runner, output=output, case=case, protocol=protocol, result=result,
                           directory=output / "episodes" / case["id"], verified=verified)


@pytest.mark.parametrize("status", ["missing", "unsealed", "error"])
def test_incomplete_recovery_study_requires_explicit_partial(study, status):
    if status != "missing":
        study.directory.mkdir(parents=True)
    if status == "error":
        write_json(study.directory / "error.json", {"error": "retained failure"})
    result = entry.verify_study(study.runner, study.output, partial=True)
    assert result["complete"] is False
    assert result["planned"] == 1 and result["completed"] == 0
    assert result["independent_benchmark_episodes"] == 0
    assert result["successes"] == result["task_failures"] == 0
    assert result["statuses"] == {study.case["id"]: status}
    assert result["results"] == [] and result["recovery_counts"] is None
    assert study.verified == []
    with pytest.raises(ValueError, match="partial"):
        entry.verify_study(study.runner, study.output, partial=False)


@pytest.mark.parametrize("success", [False, True])
def test_sealed_recovery_is_complete_even_when_official_task_fails(study, success):
    write_json(study.directory / "receipt.json", {"test": "verified by fake episode verifier"})
    study.result["success"] = success
    result = entry.verify_study(study.runner, study.output)
    assert result["complete"] is True
    assert result["planned"] == result["completed"] == 1
    assert result["independent_benchmark_episodes"] == 0
    assert result["evaluation_scope"] == "recovery_of_existing_interrupted_trajectory"
    assert result["successes"] == int(success)
    assert result["task_failures"] == int(not success)
    assert result["statuses"] == {study.case["id"]: "complete"}
    assert result["recovery_counts"] == study.result["recovery_counts"]
    assert len(study.verified) == 1


@pytest.mark.parametrize("entry_type", ["directory", "file"])
def test_recovery_study_rejects_unplanned_case_inventory(study, entry_type):
    unexpected = study.output / "episodes" / "another-case"
    unexpected.parent.mkdir(parents=True)
    if entry_type == "directory":
        unexpected.mkdir()
    else:
        unexpected.write_bytes(b"unplanned")
    with pytest.raises(ValueError, match="Unplanned"):
        entry.verify_study(study.runner, study.output, partial=True)
    assert study.verified == []


@pytest.mark.parametrize("change", ["missing_case", "extra_case", "wrong_case"])
def test_recovery_study_rejects_changed_protocol_case_set(study, change):
    cases = study.protocol["spec"]["cases"]
    if change == "missing_case":
        cases.clear()
    elif change == "extra_case":
        cases.append({**study.case, "id": "extra-case"})
    else:
        cases[0] = {**study.case, "id": "wrong-case"}
    with pytest.raises(ValueError, match="one case"):
        entry.verify_study(study.runner, study.output, partial=True)
