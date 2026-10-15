"""Closed-loop contracts using synthetic evidence only; never load Torch or weights."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_vlajepa_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("vlajepa_closed_loop_runner_test", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def observation(size=360, step=0):
    return {"observation.images.image": np.full((3, size, size), np.float32(20 / 255), dtype=np.float32),
            "observation.images.image2": np.full((3, size, size), np.float32(40 / 255), dtype=np.float32),
            "observation.state": np.full(8, step, dtype=np.float32), "task": "synthetic instruction"}


def test_design_has_four_known_tasks_and_native_seven_step_execution():
    cases = runner.cases_for_study()
    assert len(cases) == len({case["id"] for case in cases}) == 4
    assert [case["suite"] for case in cases] == list(runner.SUITES)
    assert [runner.HORIZONS[case["suite"]] for case in cases] == [280, 280, 300, 520]
    for case in cases:
        assert (case["task_id"], case["init_state_id"], case["seed"], case["num_steps"]) == (1, 41, 1000, 4)
        assert case["scheduler"] == {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False}
    assert runner.study_spec()["execution"]["warmup_calls"] == 0


@pytest.mark.parametrize("bad", ["success", "reward", "object_state", "missing_wrist", "task", "dtype", "shape", "state_shape", "nan", "range"])
def test_policy_input_rejects_privileged_labels_and_invalid_camera_state_contract(bad):
    obs = observation()
    if bad in {"success", "reward", "object_state"}:
        obs[bad] = 1
    elif bad == "missing_wrist":
        del obs[runner.FEATURES[1]]
    elif bad == "task":
        obs["task"] = "   "
    elif bad == "dtype":
        obs[runner.FEATURES[0]] = obs[runner.FEATURES[0]].astype(np.float64)
    elif bad == "shape":
        obs[runner.FEATURES[0]] = np.zeros((360, 360, 3), dtype=np.float32)
    elif bad == "state_shape":
        obs[runner.FEATURES[2]] = np.zeros((1, 8), dtype=np.float32)
    elif bad == "nan":
        obs[runner.FEATURES[2]][0] = np.nan
    else:
        obs[runner.FEATURES[0]][0, 0, 0] = np.float32(1.01)
    with pytest.raises(ValueError):
        runner.validate_observation(obs)


def test_adapter_observe_uses_current_packet_and_clears_per_tick_timing():
    adapter = runner.ClosedLoopAdapter(None, None, None, None, {})
    first, second = observation(step=1), observation(step=2)
    adapter.observe(first)
    adapter.last_timing = {"inference_seconds": 9.0, "preprocess_seconds": 2.0}
    adapter.observe(second)
    assert adapter.observation is second
    assert adapter.last_timing == {"inference_seconds": 0.0, "preprocess_seconds": 0.0}


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "gripper"])
def test_environment_action_chunk_requires_native_shape_finite_fp32_binary_gripper(bad):
    value = np.zeros((1, 7, 7), dtype=np.float32)
    value[..., 6] = 1
    if bad == "shape":
        value = value[0]
    elif bad == "dtype":
        value = value.astype(np.float64)
    elif bad == "nan":
        value[0, 2, 3] = np.nan
    else:
        value[0, 2, 6] = 0
    with pytest.raises(ValueError):
        runner.validate_action_chunk(value, environment=True)


def test_action_validation_does_not_add_clipping_or_gripper_transforms():
    value = np.full((1, 7, 7), 2.0, dtype=np.float32)
    value[..., 6] = [-1, 1, -1, 1, -1, 1, -1]
    saved = value.copy()
    runner.validate_action_chunk(value, environment=True)
    assert np.array_equal(value, saved)
    raw = np.zeros((1, 7, 7), dtype=np.float32)
    runner.validate_action_chunk(raw)  # Native normalized gripper need not already be binary.


class FakeSimulator:
    """Small deterministic observation source, not a fake claim of model inference."""
    task = "synthetic instruction"

    def __init__(self, horizon, success_at=None):
        self.horizon, self.success_at, self.steps = horizon, success_at, 0
        self.actions = []

    def reset(self, *, seed, init_state_id):
        assert (seed, init_state_id) == (1000, 41)
        self.steps, self.actions = 0, []
        return observation(size=2)

    def step(self, action):
        assert self.steps < self.horizon and (self.success_at is None or self.steps < self.success_at)
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        success = self.steps == self.success_at
        return observation(size=2, step=self.steps), float(success), success, self.steps == self.horizon and not success, {"is_success": success}

    def frame(self, camera="image"):
        return np.full((2, 2, 3), 20 if camera == "image" else 40, dtype=np.uint8)

    def metadata(self):
        return {"synthetic": True}


class FakePolicy:
    def reset(self, *, seed):
        assert seed == 1000
        self.planning_states, self.inference_count = [], 0
        self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

    def observe(self, obs):
        self.obs = obs
        self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

    def predict_chunk(self):
        start = int(self.obs["observation.state"][0])
        self.planning_states.append(start)
        self.inference_count += 1
        self.last_timing = {"inference_seconds": 0.001, "preprocess_seconds": 0.0001}
        actions = np.repeat(np.arange(start, start + 7, dtype=np.float32)[:, None], 7, axis=1)
        actions[:, 6] = 1
        return actions

    @property
    def metadata(self):
        return {"synthetic": True, "inference_count": self.inference_count}


@pytest.mark.parametrize("horizon,success_at,expected_calls,tail", [(280, None, 40, 7), (300, None, 43, 6), (520, None, 75, 2), (280, 8, 2, 1)])
def test_existing_executor_replans_from_current_state_and_stops_at_tail_or_success(tmp_path, horizon, success_at, expected_calls, tail):
    case = deepcopy(runner.cases_for_study()[0])
    case["record_video"] = False
    simulator, policy = FakeSimulator(horizon, success_at), FakePolicy()
    result = runner.run_episode(runner.RecordedSimulator(simulator, tmp_path), policy, case, tmp_path)
    steps = success_at or horizon
    assert result["steps"] == steps and result["inference_calls"] == expected_calls
    assert result["success"] is (success_at is not None)
    assert policy.planning_states == list(range(0, steps, 7))
    assert steps - policy.planning_states[-1] == tail
    assert np.array_equal(np.asarray(simulator.actions)[:, 0], np.arange(steps, dtype=np.float32))
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as data:
        assert np.array_equal(data["states"][:, 0], np.arange(steps, dtype=np.float32))
        assert data["final_state"][0] == steps
        assert np.count_nonzero(data["preprocessing_seconds"]) == expected_calls
    rows = [json.loads(line) for line in (tmp_path / "executed_steps.jsonl").read_text().splitlines()]
    assert len(rows) == steps and rows[0]["step"] == 1 and rows[-1]["step"] == steps
    assert rows[0]["next_robot_state"] == [1.0] * 8
    assert rows[-1]["success"] is (success_at is not None)
    assert rows[-1]["next_camera_sha256"] == {key: runner.array_sha(observation(size=2)[key]) for key in runner.FEATURES[:2]}
    with pytest.raises(ValueError, match="already exists"):
        runner.RecordedSimulator(simulator, tmp_path)


def complete_evidence(tmp_path, monkeypatch, *, steps=8, success=True, suite="libero_spatial"):
    """Write real receipt/hash/NPZ/image plumbing with explicitly synthetic payloads."""
    root = tmp_path / "project"
    root.mkdir()
    official_root = root / "frontier/vendor/lerobot"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner.base, "SOURCE", official_root)
    modules = {}
    for name in ("lerobot", "environment", "policy"):
        path = official_root / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n", encoding="utf-8")
        modules[name] = {"path": path.relative_to(root).as_posix(), "sha256": runner.file_sha(path)}
    loading_path = root / "outputs/study/runs/loading_0001.json"
    loading = {"all_required_inference_keys_consumed": True, "all_source_tensors_finite": True,
               "native_tied_word_embeddings": True, "linear_modules": 300,
               "theoretical_parameter_payload_bytes": runner.quant.EXPECTED_PAYLOAD_BYTES}
    runner.write_json(loading_path, loading)
    case = next(case for case in runner.cases_for_study() if case["suite"] == suite)
    protocol = {"fingerprint": "a" * 64, "inputs_sha256": "b" * 64}
    directory = root / "outputs/study/episodes" / case["id"]
    directory.mkdir(parents=True)
    run_record = "outputs/study/runs/run_0001.json"
    runner.write_json(root / run_record, {"status": "complete", "protocol_fingerprint": protocol["fingerprint"]})
    obs = observation()
    for name, value in (("initial.png", 20), ("initial_wrist.png", 40), ("final.png", 20), ("rollout.gif", 20)):
        Image.fromarray(np.full((360, 360, 3), value, dtype=np.uint8)).save(directory / name)
    camera_hashes = {key: runner.array_sha(obs[key]) for key in runner.FEATURES[:2]}
    states = np.repeat(np.arange(steps + 1, dtype=np.float32)[:, None], 8, axis=1)
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, 0] = np.arange(steps, dtype=np.float32) / 1000
    actions[:, 6] = 1
    calls, names = [], [f"synthetic.linear_{i}" for i in range(300)]
    replans = list(range(0, steps, 7))
    for index, start in enumerate(replans):
        env_actions = np.zeros((1, 7, 7), dtype=np.float32)
        env_actions[0, :, 0] = np.arange(start, start + 7, dtype=np.float32) / 1000
        env_actions[..., 6] = 1
        noise, rng = np.zeros((1, 7, 7), dtype=np.float32), np.arange(32, dtype=np.uint8)
        npz_path = directory / f"call_{index:04d}.npz"
        np.savez_compressed(npz_path, model_actions=env_actions.copy(), environment_actions=env_actions,
                            noise=noise, rng_state=rng, input_robot_state=states[start])
        call = {"call_index": index, "status": "complete", "forward_returned": True, "action_head_nfe": 4,
                "quantized_module_calls": dict.fromkeys(names, 1), "task": obs["task"],
                "artifact_sha256": runner.file_sha(npz_path), "noise_sha256": runner.array_sha(noise),
                "rng_state_sha256": runner.array_sha(rng),
                "input_sha256": {**camera_hashes, "observation.state": runner.array_sha(states[start])},
                "preprocess_seconds": 0.01, "forward_seconds": 1.0, "postprocess_seconds": 0.01}
        call_path = npz_path.with_suffix(".json")
        runner.write_json(call_path, call)
        calls.append({"call_index": index, "record": call_path.name, "sha256": runner.file_sha(call_path)})
    model_seconds, prep_seconds = np.zeros(steps), np.zeros(steps)
    model_seconds[replans], prep_seconds[replans] = 1.0, 0.01
    np.savez_compressed(directory / "trajectory.npz", actions=actions, states=states[:-1], final_state=states[-1],
                        replan_steps=np.asarray(replans), model_seconds=model_seconds, preprocessing_seconds=prep_seconds,
                        control_seconds=model_seconds + prep_seconds + 0.05, simulator_seconds=np.full(steps, 0.02))
    flags = {"success": success, "terminated": success, "truncated": not success}
    rows = [{"step": index + 1, "action": actions[index].tolist(), "next_robot_state": states[index + 1].tolist(),
             "next_camera_sha256": camera_hashes,
             **(flags if index == steps - 1 else {key: False for key in flags})} for index in range(steps)]
    (directory / "executed_steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = {"variant": runner.VARIANT, "policy_revision": runner.base.POLICY_REVISION,
                "lerobot_revision": runner.base.REVISION, "inputs_sha256": protocol["inputs_sha256"],
                "actual_modules": modules, "loading": {"path": loading_path.relative_to(root).as_posix(), "sha256": runner.file_sha(loading_path)},
                "quantized_modules": names, "run_record": run_record, "call_log": calls,
                "inference_count": len(calls), "action_head_nfe": 4 * len(calls)}
    result = {**case, **flags, "status": "complete", "protocol_fingerprint": protocol["fingerprint"],
              "steps": steps, "horizon": runner.HORIZONS[suite], "task": obs["task"], "inference_calls": len(calls),
              "policy_inference_seconds": float(len(calls)), "preprocess_seconds": float(prep_seconds.sum()),
              "simulator_seconds": steps * 0.02, "wall_seconds": steps * 0.1 + len(calls),
              "initial_robot_state": states[0].tolist(), "final_robot_state": states[-1].tolist(),
              "initial_image_sha256": runner.array_sha(np.full((360, 360, 3), 20, dtype=np.uint8)),
              "scheduler_config": runner.asdict(runner.SchedulerConfig(**case["scheduler"], horizon=runner.HORIZONS[suite])),
              "environment": {"suite": suite, "task_id": 1, "horizon": runner.HORIZONS[suite], "image_size": 360,
                              "control_hz": 20, "settle_steps": 10, "control_mode": "relative", "hard_reset": True,
                              "images_flipped_hw_once": True, "object_state_exposed_to_policy": False},
              "model_info": metadata, "video": "rollout.gif"}
    runner.write_json(directory / "attempt.json", {"case": case, "protocol_fingerprint": protocol["fingerprint"], "run_record": run_record})
    runner.write_json(directory / "result.json", result)
    runner.write_json(directory / "calls.json", calls)
    runner.seal_episode(directory)
    return directory, case, protocol


def reseal(directory):
    (directory / "receipt.json").unlink()
    runner.seal_episode(directory)


@pytest.mark.parametrize("steps,success,suite", [(1, True, "libero_spatial"), (8, True, "libero_spatial"), (300, False, "libero_goal")])
def test_evidence_consumer_accepts_success_and_full_horizon_failure_separately(tmp_path, monkeypatch, steps, success, suite):
    directory, case, protocol = complete_evidence(tmp_path, monkeypatch, steps=steps, success=success, suite=suite)
    result = runner.verify_episode(directory, case, protocol)
    assert result["success"] is success and result["steps"] == steps
    assert result["inference_calls"] == (steps + 6) // 7


@pytest.mark.parametrize("bad", ["changed_bytes", "extra_file", "missing_receipt", "error", "attempt", "early_failure", "scheduler", "environment", "executed_action", "next_state", "stale_camera", "continued_terminal", "loading"])
def test_real_receipts_cannot_hide_corrupted_or_inconsistent_evidence(tmp_path, monkeypatch, bad):
    directory, case, protocol = complete_evidence(tmp_path, monkeypatch)
    result = runner.read_json(directory / "result.json")
    if bad == "changed_bytes":
        with (directory / "final.png").open("ab") as stream:
            stream.write(b"tampered")
    elif bad == "extra_file":
        (directory / "unaccounted.txt").write_text("extra")
    elif bad == "missing_receipt":
        (directory / "receipt.json").unlink()
    elif bad == "error":
        runner.write_json(directory / "error.json", {"status": "error"})
    elif bad == "attempt":
        attempt = runner.read_json(directory / "attempt.json")
        attempt["case"]["init_state_id"] = 43
        runner.write_json(directory / "attempt.json", attempt)
    elif bad in {"early_failure", "scheduler", "environment"}:
        if bad == "early_failure":
            result.update(success=False, terminated=False, truncated=True)
        elif bad == "scheduler":
            result["scheduler_config"]["fixed_interval"] = 10
        else:
            result["environment"]["control_hz"] = 30
        runner.write_json(directory / "result.json", result)
    elif bad == "loading":
        path = runner.ROOT / result["model_info"]["loading"]["path"]
        path.write_text(path.read_text() + " ")
    else:
        path = directory / "executed_steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if bad == "executed_action":
            rows[0]["action"][0] = 0.5
        elif bad == "next_state":
            rows[0]["next_robot_state"][0] = 99.0
        elif bad == "stale_camera":
            rows[6]["next_camera_sha256"][runner.FEATURES[1]] = "f" * 64
        else:
            rows[0]["terminated"] = True
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if bad not in {"changed_bytes", "extra_file", "missing_receipt"}:
        reseal(directory)
    with pytest.raises((ValueError, RuntimeError)):
        runner.verify_episode(directory, case, protocol)


@pytest.mark.parametrize("bad", ["nfe", "coverage", "instruction", "timing", "noise", "chunk_action", "state"])
def test_rehashed_call_chain_still_requires_native_execution_and_causal_action_data(tmp_path, monkeypatch, bad):
    directory, case, protocol = complete_evidence(tmp_path, monkeypatch)
    path = directory / "call_0001.json"
    call = runner.read_json(path)
    if bad == "nfe":
        call["action_head_nfe"] = 2
    elif bad == "coverage":
        call["quantized_module_calls"].pop(next(iter(call["quantized_module_calls"])))
    elif bad == "instruction":
        call["task"] = "a different task"
    elif bad == "timing":
        call["forward_seconds"] = -1
    else:
        npz_path = path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as values:
            data = {key: values[key].copy() for key in values.files}
        if bad == "noise":
            data["noise"][0, 0, 0] = 1  # Inner noise hash intentionally remains bound to the old draw.
        elif bad == "chunk_action":
            data["environment_actions"][0, 0, 0] = 0.9
        else:
            data["input_robot_state"][0] = 6  # Previous tick, not this replan's state.
        np.savez_compressed(npz_path, **data)
        call["artifact_sha256"] = runner.file_sha(npz_path)
    runner.write_json(path, call)
    calls = runner.read_json(directory / "calls.json")
    calls[1]["sha256"] = runner.file_sha(path)
    result = runner.read_json(directory / "result.json")
    result["model_info"]["call_log"] = calls
    runner.write_json(directory / "calls.json", calls)
    runner.write_json(directory / "result.json", result)
    reseal(directory)
    with pytest.raises(ValueError):
        runner.verify_episode(directory, case, protocol)
