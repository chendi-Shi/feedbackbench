import numpy as np
import pytest

from embodied_research.simulator import LiberoSimulator, policy_observation, quaternion_to_axis_angle


def raw_observation():
    return {
        "pixels": {"image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
                   "image2": np.full((2, 3, 3), 127, dtype=np.uint8)},
        "robot_state": {"eef": {"pos": np.array([1, 2, 3.]), "quat": np.array([0., 0., 0., 1.])},
                        "gripper": {"qpos": np.array([0.01, -0.01])}},
        "object_state": object(),
    }


def test_orientation_proprioception_and_no_privileged_object_state():
    raw = raw_observation()
    observation = policy_observation(raw, "pick the cup")
    assert set(observation) == {"task", "observation.images.image", "observation.images.image2", "observation.state"}
    np.testing.assert_array_equal(observation["observation.images.image"][:, 0, 0], raw["pixels"]["image"][-1, -1] / np.float32(255))
    assert observation["observation.images.image"].flags.c_contiguous
    assert observation["observation.images.image"].dtype == np.float32
    np.testing.assert_allclose(observation["observation.state"], [1, 2, 3, 0, 0, 0, 0.01, -0.01])
    assert observation["observation.state"].dtype == np.float32
    raw["object_state"] = "poisoned unrelated truth"
    np.testing.assert_array_equal(policy_observation(raw, "pick the cup")["observation.state"], observation["observation.state"])


def test_axis_angle_handles_identity_half_turn_and_invalid_inputs():
    np.testing.assert_array_equal(quaternion_to_axis_angle([0, 0, 0, 1]), np.zeros(3))
    np.testing.assert_allclose(quaternion_to_axis_angle([0, 0, 1, 0]), [0, 0, np.pi], atol=1e-6)
    with pytest.raises(ValueError):
        quaternion_to_axis_angle([0, 0, np.nan, 1])


class FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._init_states = np.zeros((50, 4))
        self.task_description = "pick the cup"
        self.closed = False
        self.success = False

    def reset(self, seed):
        self.seen_init = self.init_state_id
        self.init_state_id += 1
        self.seed = seed
        return raw_observation(), {"is_success": False}

    def step(self, action):
        self.last_action = action
        return raw_observation(), float(self.success), self.success, False, {"is_success": self.success}

    def close(self):
        self.closed = True


def test_paired_initial_states_exact_action_convention_and_official_success():
    sim = LiberoSimulator("libero_spatial", 3, _env_factory=FakeEnv)
    assert sim.env.kwargs["hard_reset"] is True
    assert sim.env.kwargs["control_freq"] == 20
    assert sim.env.kwargs["num_steps_wait"] == 10
    sim.reset(seed=1000, init_state_id=7)
    assert sim.env.seen_init == 7
    action = np.array([0, 0.1, 0, 0, 0, 0, -0.7], dtype=np.float32)
    sim.step(action)
    np.testing.assert_array_equal(sim.env.last_action, action)
    sim.reset(seed=1000, init_state_id=7)
    assert sim.env.seen_init == 7
    sim.env.success = True
    _, _, terminated, truncated, info = sim.step(action)
    assert terminated and not truncated and info["is_success"]
    with pytest.raises(RuntimeError, match="Reset"):
        sim.step(action)
    sim.close()
    assert sim.env.closed


def test_exact_horizon_truncation_and_no_initial_state_wrapping():
    sim = LiberoSimulator("libero_object", 0, _env_factory=FakeEnv)
    with pytest.raises(ValueError, match="published initial"):
        sim.reset(init_state_id=50)
    sim.reset(init_state_id=0)
    for _ in range(279):
        _, _, terminated, truncated, _ = sim.step(np.zeros(7))
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = sim.step(np.zeros(7))
    assert truncated and not terminated


def test_protocol_metadata_and_frame_match_policy_orientation():
    with LiberoSimulator("libero_10", 0, _env_factory=FakeEnv) as sim:
        obs = sim.reset()
        assert sim.metadata()["horizon"] == 520
        assert sim.metadata()["object_state_exposed_to_policy"] is False
        np.testing.assert_allclose(sim.frame().transpose(2, 0, 1) / np.float32(255), obs["observation.images.image"])
