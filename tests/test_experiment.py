import copy

import numpy as np
import pytest

from embodied_research.experiment import ObservationCondition, digest_json, expand_plan, run_episode


def observation(t):
    return {"task": "pick up the bowl", "observation.state": np.ones(8, dtype=np.float32) * t,
            "observation.images.image": np.full((3, 8, 8), t / 10, dtype=np.float32),
            "observation.images.image2": np.full((3, 8, 8), t / 10, dtype=np.float32)}


def test_delayed_packet_keeps_vision_and_proprioception_synchronized_and_causal():
    transform = ObservationCondition("delay4")
    for step in range(10):
        delivered = transform.apply(observation(step))
        expected = max(0, step - 4)
        np.testing.assert_equal(delivered["observation.state"], np.ones(8) * expected)
        np.testing.assert_allclose(delivered["observation.images.image"], expected / 10)
        delivered["observation.state"][:] = 99  # mutation cannot corrupt future queue contents


def test_observation_perturbation_does_not_modify_original_or_robot_state():
    original = observation(5)
    saved = copy.deepcopy(original)
    dim = ObservationCondition("dim").apply(original)
    np.testing.assert_array_equal(original["observation.images.image"], saved["observation.images.image"])
    np.testing.assert_array_equal(dim["observation.state"], saved["observation.state"])
    np.testing.assert_allclose(dim["observation.images.image"], .325)


def test_protocol_materializes_every_paired_condition_and_rejects_duplicates():
    plan = {"models": [{"name": "minerva", "methods": [{"name": "fixed1"}, {"name": "fixed8"}]}],
            "tasks": [{"suite": "libero_goal", "task_id": 3}], "initial_states": [10, 11],
            "conditions": ["clean", "delay4"]}
    cases = expand_plan(plan)
    assert len(cases) == 8
    assert len({row["id"] for row in cases}) == 8
    assert {row["init_state_id"] for row in cases} == {10, 11}
    plan["initial_states"].append(10)
    with pytest.raises(ValueError, match="Duplicate"):
        expand_plan(plan)


def test_configuration_digest_is_order_independent_but_sensitive_to_semantics():
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
    assert digest_json({"a": 1, "b": 2}) != digest_json({"a": 2, "b": 2})


class FakeSimulator:
    horizon = 5
    task = "instrumented runner test"

    def __init__(self, success_step=None):
        self.success_step = success_step

    def reset(self, seed, init_state_id):
        self.steps = 0
        return observation(0)

    def frame(self, camera="image"):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        success = self.steps == self.success_step
        # A deliberately high reward does not itself qualify as success.
        return observation(self.steps), 999.0, success, self.steps == self.horizon, {"is_success": success}

    def metadata(self):
        return {"fixture": True}


class FakePolicy:
    metadata = {"fixture": True}

    def reset(self, seed):
        self.inference_count = 0
        self.observed = []
        self.last_timing = {"inference_seconds": 0., "preprocess_seconds": 0.}

    def observe(self, packet):
        self.observed.append(float(packet["observation.state"][0]))

    def predict_chunk(self):
        self.inference_count += 1
        return np.zeros((3, 7), dtype=np.float32)

    def select_action(self, packet):
        self.observe(packet)
        return self.predict_chunk()[0]


@pytest.mark.parametrize("success_step,expected_steps,expected_success", [(None, 5, False), (2, 2, True)])
def test_runner_uses_official_success_and_full_horizon(tmp_path, success_step, expected_steps, expected_success):
    policy = FakePolicy()
    case = {"seed": 1000, "init_state_id": 0, "condition": "clean", "scheduler": None}
    result = run_episode(FakeSimulator(success_step), policy, case, tmp_path)
    assert result["steps"] == result["inference_calls"] == expected_steps
    assert result["success"] is expected_success
    assert not (tmp_path / "result.json").exists()  # Caller adds provenance and commits exactly once.
    assert (tmp_path / "trajectory.npz").exists()


def test_runner_updates_observation_history_on_noninference_steps(tmp_path):
    policy = FakePolicy()
    case = {"seed": 1000, "init_state_id": 0, "condition": "delay4",
            "scheduler": {"mode": "fixed", "fixed_interval": 3}}
    result = run_episode(FakeSimulator(), policy, case, tmp_path)
    assert result["inference_calls"] == 2
    assert policy.observed == [0., 0., 0., 0., 0.]
    trajectory = np.load(tmp_path / "trajectory.npz")
    np.testing.assert_array_equal(trajectory["replan_steps"], [0, 3])
