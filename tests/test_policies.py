"""Actual pretrained MINERVA parity and temporal-state regression tests."""

from pathlib import Path

import numpy as np
import pytest

from embodied_research.policies import IMAGE_KEYS, PolicyAdapter, verify_model_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter():
    import torch

    torch.set_num_threads(2)
    return PolicyAdapter("minerva", temporal_ensemble_coeff=None)


def observation(offset=0.0):
    image = np.linspace(0, 1, 3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)
    return {"task": "pick up the orange juice and place it in the basket",
            IMAGE_KEYS[0]: image, IMAGE_KEYS[1]: image[:, :, ::-1].copy(),
            "observation.state": np.array([.03 + offset, .1, .2, .1, -.2, .1, .04, .04], dtype=np.float32)}


def test_complete_public_weights_and_processors_verify():
    manifest = verify_model_files(ROOT / "models" / "minerva", "minerva")
    weight = next(record for record in manifest["files"] if record["file"] == "model.safetensors")
    assert weight["sha256"] == "f173def4c0b11030161486401dce52e63471d4180c248779678eaa6952c32da5"


def test_scheduled_first_action_matches_upstream_on_consecutive_frames(adapter):
    frames = [observation(i * .01) for i in range(3)]
    adapter.reset(seed=12)
    baseline = [adapter.select_action(frame) for frame in frames]
    adapter.reset(seed=12)
    for frame, expected in zip(frames, baseline, strict=True):
        adapter.observe(frame)
        chunk = adapter.predict_chunk()
        assert chunk.shape == (15, 7)
        np.testing.assert_allclose(chunk[0], expected, rtol=1e-5, atol=1e-6)


def test_external_ensemble_matches_upstream_across_full_chunk_window(adapter):
    from embodied_research.scheduling import ChunkScheduler, SchedulerConfig

    frames = [observation(i * .003) for i in range(24)]
    adapter.config.temporal_ensemble_coeff = 0.01
    try:
        adapter.reset(seed=22)
        baseline = [adapter.select_action(frame) for frame in frames]
        adapter.reset(seed=22)
        scheduler = ChunkScheduler(SchedulerConfig(
            mode="fixed", fixed_interval=1, temporal_ensemble=True, ensemble_coeff=0.01,
            horizon=len(frames),
        ))
        for frame, expected in zip(frames, baseline, strict=True):
            adapter.observe(frame)
            assert scheduler.should_replan(frame)
            scheduler.submit_chunk(adapter.predict_chunk())
            np.testing.assert_allclose(scheduler.next_action(), expected, rtol=1e-5, atol=2e-6)
    finally:
        adapter.config.temporal_ensemble_coeff = None
        adapter.reset()


def test_skipped_replans_preserve_consecutive_observation_history(adapter):
    import torch

    frames = [observation(i * .02) for i in range(5)]
    adapter.reset()
    for frame in frames:
        adapter.observe(frame)
    expected_history = [adapter._prepare_observation(frame)["observation.state"] for frame in frames[-2:]]
    actual_history = list(adapter.policy._queues["observation.state"])
    assert len(actual_history) == 2
    for actual, expected in zip(actual_history, expected_history, strict=True):
        torch.testing.assert_close(actual, expected)
    before = [value.clone() for value in actual_history]
    adapter.predict_chunk()
    adapter.predict_chunk()
    for actual, expected in zip(adapter.policy._queues["observation.state"], before, strict=True):
        torch.testing.assert_close(actual, expected)
    assert adapter.observation_count == 5


def test_reset_removes_prior_episode_state(adapter):
    adapter.reset()
    adapter.observe(observation(.1))
    first = adapter.predict_chunk()
    for i in range(3):
        adapter.observe(observation(i * .03))
        adapter.predict_chunk()
    adapter.reset()
    adapter.observe(observation(.1))
    np.testing.assert_array_equal(adapter.predict_chunk(), first)


def test_control_modes_cannot_silently_double_append_history(adapter):
    adapter.reset()
    adapter.observe(observation())
    with pytest.raises(RuntimeError, match="reset"):
        adapter.select_action(observation())
    adapter.reset()
    adapter.select_action(observation())
    with pytest.raises(RuntimeError, match="reset"):
        adapter.observe(observation())


def test_no_ground_truth_object_input_enters_policy(adapter):
    frame = observation()
    adapter.reset()
    expected = adapter.select_action(frame)
    frame["object_state"] = np.full(100, np.nan)
    frame["is_success"] = True
    frame["segmentation"] = np.ones((128, 128))
    adapter.reset()
    np.testing.assert_array_equal(adapter.select_action(frame), expected)


@pytest.mark.parametrize("bad", [np.zeros((128, 128, 3), dtype=np.float32),
                                  np.zeros((3, 128, 128), dtype=np.uint8),
                                  np.full((3, 128, 128), np.nan),
                                  np.full((3, 128, 128), 255.0)])
def test_reject_incorrect_camera_contract(adapter, bad):
    frame = observation()
    frame[IMAGE_KEYS[0]] = bad
    adapter.reset()
    with pytest.raises(ValueError):
        adapter.observe(frame)


def test_unknown_task_does_not_fallback_to_arbitrary_task(adapter):
    frame = observation()
    frame["task"] = "an unseen instruction"
    adapter.reset()
    with pytest.raises(KeyError, match="not found"):
        adapter.observe(frame)


def test_chunk_requires_observation(adapter):
    adapter.reset()
    with pytest.raises(RuntimeError, match="observe"):
        adapter.predict_chunk()
