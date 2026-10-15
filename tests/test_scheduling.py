import json
import math

import numpy as np
import pytest

from embodied_research.scheduling import ChunkScheduler, SchedulerConfig


def observation(image=0.0, gripper=0.0):
    state = np.zeros(8, dtype=np.float32)
    state[-2:] = gripper
    return {
        "observation.images.image": np.full((3, 32, 32), image, dtype=np.float32),
        "observation.state": state,
    }


def actions(value, length=20):
    return np.full((length, 7), value, dtype=np.float32)


def tick(scheduler, obs=None, value=0, length=20):
    requested = scheduler.should_replan(observation() if obs is None else obs)
    if requested:
        scheduler.submit_chunk(actions(value, length))
    return requested, scheduler.next_action()


@pytest.mark.parametrize("interval,expected", [(1, 33), (4, 9), (8, 5), (15, 3), (16, 3)])
def test_fixed_intervals_have_exact_inference_budget(interval, expected):
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=interval, horizon=33))
    for _ in range(33):
        tick(scheduler)
    audit = scheduler.diagnostics()
    assert audit["chunks_received"] == expected
    assert audit["plan_steps"] == list(range(0, 33, interval))
    assert sum(audit["executed_chunk_spans"]) == 33
    assert scheduler.done
    with pytest.raises(StopIteration):
        scheduler.should_replan({})
    with pytest.raises(StopIteration):
        scheduler.next_action()


def test_temporal_ensemble_aligns_offsets_and_favors_older_prediction():
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=1, ensemble_coeff=0.01))
    assert scheduler.should_replan({})
    first = np.repeat(np.arange(4, dtype=np.float32)[:, None], 7, axis=1)
    scheduler.submit_chunk(first)
    np.testing.assert_array_equal(scheduler.next_action(), np.zeros(7))
    assert scheduler.should_replan({})
    scheduler.submit_chunk(actions(10, 4))
    expected = (1 + math.exp(-0.01) * 10) / (1 + math.exp(-0.01))
    result = scheduler.next_action()
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert result[0] < 5.5  # Older prediction has higher weight than newer.


def test_expired_predictions_leave_ensemble_and_zero_actions_are_valid():
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=1, ensemble_coeff=0))
    tick(scheduler, value=100, length=1)
    _, result = tick(scheduler, value=0, length=1)
    np.testing.assert_array_equal(result, np.zeros(7))
    assert scheduler.diagnostics()["events"][-1]["active_predictions"] == 1


def test_ensemble_disabled_selects_latest_chunk_with_correct_offset():
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=2, temporal_ensemble=False))
    assert scheduler.should_replan({})
    scheduler.submit_chunk(actions(100))
    scheduler.next_action()
    tick(scheduler)
    assert scheduler.should_replan({})
    scheduler.submit_chunk(np.repeat(np.arange(20, dtype=np.float32)[:, None], 7, axis=1))
    np.testing.assert_array_equal(scheduler.next_action(), np.zeros(7))
    _, result = tick(scheduler)
    np.testing.assert_array_equal(result, np.ones(7))
    assert scheduler.diagnostics()["events"][-1]["used_predictions"] == 1


def test_adaptive_minimum_maximum_and_reference_reset():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive", min_interval=2, max_interval=4))
    assert tick(scheduler)[0]
    assert not tick(scheduler, observation(image=0.5))[0]  # Cannot react at age 1.
    assert tick(scheduler, observation(image=0.5))[0]
    assert not tick(scheduler, observation(image=0.5))[0]
    assert not tick(scheduler, observation(image=0.5))[0]  # Reference moved to 0.5.
    assert not tick(scheduler, observation(image=0.5))[0]
    assert tick(scheduler, observation(image=0.5))[0]
    audit = scheduler.diagnostics()
    assert audit["plan_steps"] == [0, 2, 6]
    assert audit["trigger_counts"] == {"initial": 1, "vision_change": 1, "max_interval": 1}


def test_gripper_signal_reads_only_last_two_state_dimensions():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive", use_vision=False))
    tick(scheduler)
    changed = observation()
    changed["observation.state"][:6] = 100
    assert not tick(scheduler, changed)[0]
    assert not tick(scheduler, changed)[0]
    changed["observation.state"][-1] = 0.03
    assert tick(scheduler, changed)[0]
    assert scheduler.diagnostics()["events"][-1]["reasons"] == ["gripper_change"]


@pytest.mark.parametrize("use_vision,use_gripper", [(False, True), (True, False), (False, False)])
def test_ablations_do_not_read_disabled_observations(use_vision, use_gripper):
    config = SchedulerConfig(mode="adaptive", use_vision=use_vision, use_gripper=use_gripper)
    scheduler = ChunkScheduler(config)
    obs = observation()
    if not use_vision:
        obs.pop("observation.images.image")
    if not use_gripper:
        obs.pop("observation.state")
    assert tick(scheduler, obs)[0]
    for _ in range(14):
        assert not tick(scheduler, obs)[0]
    assert tick(scheduler, obs)[0]


def test_chunk_exhaustion_forces_inference_before_minimum_interval():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive", min_interval=4))
    tick(scheduler, length=2)
    assert not tick(scheduler, length=2)[0]
    assert tick(scheduler, length=2)[0]
    assert scheduler.diagnostics()["events"][-1]["reasons"] == ["chunk_exhausted"]


def test_fixed_interval_longer_than_true_model_output_is_rejected():
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=16))
    assert scheduler.should_replan({})
    with pytest.raises(ValueError, match="exceeds"):
        scheduler.submit_chunk(actions(0, 15))
    assert scheduler.needs_chunk
    with pytest.raises(RuntimeError, match="must arrive"):
        scheduler.next_action()


def test_future_observations_cannot_change_previous_decisions_or_actions():
    def run(future):
        scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive"))
        outputs = []
        for index, obs in enumerate([observation()] * 6 + [future] * 4):
            outputs.append(tick(scheduler, obs, value=index)[1])
        return scheduler.diagnostics()["events"], outputs

    first_events, first_actions = run(observation())
    second_events, second_actions = run(observation(image=1, gripper=1))
    assert first_events[:6] == second_events[:6]
    np.testing.assert_array_equal(first_actions[:6], second_actions[:6])
    assert first_events[6]["requested"] is False
    assert second_events[6]["requested"] is True


def test_no_success_reward_or_true_object_pose_is_read():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive"))
    obs = observation()
    obs.update(success=object(), reward=object(), object_pose=object(), future_image=object())
    tick(scheduler, obs)
    for _ in range(3):
        assert not tick(scheduler, obs)[0]


def test_observation_chunk_and_audit_are_defensively_copied():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive", temporal_ensemble=False))
    obs = observation()
    assert scheduler.should_replan(obs)
    source = actions(0.5)
    scheduler.submit_chunk(source)
    source[:] = 900
    obs["observation.images.image"][:] = 1
    obs["observation.state"][:] = 1
    action = scheduler.next_action()
    np.testing.assert_array_equal(action, np.full(7, 0.5))
    action[:] = 999
    assert not tick(scheduler)[0]
    requested, result = tick(scheduler)
    assert not requested  # Original zero image/state remained the reference.
    np.testing.assert_array_equal(result, np.full(7, 0.5))
    audit = scheduler.diagnostics()
    audit["events"][0]["reasons"].append("mutated")
    audit["plan_steps"].append(999)
    assert scheduler.diagnostics()["events"][0]["reasons"] == ["initial"]
    assert scheduler.diagnostics()["plan_steps"] == [0]


def test_reset_removes_actions_clock_sensors_and_counters():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive"))
    tick(scheduler, observation(image=1), value=99)
    scheduler.reset()
    assert scheduler.step == 0
    assert scheduler.diagnostics()["events"] == []
    assert scheduler.diagnostics()["chunks_received"] == 0
    requested, result = tick(scheduler, value=0)
    assert requested
    np.testing.assert_array_equal(result, np.zeros(7))
    assert scheduler.diagnostics()["events"][0]["image_delta"] is None


def test_lifecycle_rejects_duplicate_ticks_unsolicited_or_missing_chunks():
    scheduler = ChunkScheduler()
    with pytest.raises(RuntimeError):
        scheduler.next_action()
    with pytest.raises(RuntimeError):
        scheduler.submit_chunk(actions(0))
    assert scheduler.should_replan({})
    with pytest.raises(RuntimeError):
        scheduler.should_replan({})
    with pytest.raises(RuntimeError):
        scheduler.next_action()
    scheduler.submit_chunk(actions(0))
    with pytest.raises(RuntimeError):
        scheduler.submit_chunk(actions(0))
    scheduler.next_action()


@pytest.mark.parametrize("bad", [np.zeros(7), np.zeros((0, 7)), np.zeros((2, 8)), actions(np.nan)])
def test_invalid_chunks_do_not_consume_pending_request(bad):
    scheduler = ChunkScheduler(SchedulerConfig(fixed_interval=1))
    scheduler.should_replan({})
    with pytest.raises(ValueError):
        scheduler.submit_chunk(bad)
    assert scheduler.diagnostics()["chunks_received"] == 0
    assert scheduler.needs_chunk


@pytest.mark.parametrize("field,value", [
    ("mode", "unknown"), ("horizon", 0), ("fixed_interval", True), ("min_interval", 17),
    ("vision_threshold", 0), ("gripper_threshold", float("nan")), ("ensemble_coeff", -1),
    ("ensemble_coeff", float("inf")), ("ensemble_coeff", True), ("ensemble_coeff", "0.01"),
    ("vision_threshold", "0.04"), ("use_vision", 1), ("state_key", ""),
])
def test_invalid_configurations(field, value):
    with pytest.raises(ValueError):
        SchedulerConfig(**{field: value})


@pytest.mark.parametrize("key,value", [
    ("observation.images.image", np.zeros((32, 32, 3), dtype=np.float32)),
    ("observation.images.image", np.zeros((3, 32, 32), dtype=np.uint8)),
    ("observation.images.image", np.full((3, 32, 32), np.nan, dtype=np.float32)),
    ("observation.images.image", np.full((3, 32, 32), 2, dtype=np.float32)),
    ("observation.state", np.zeros(7)),
    ("observation.state", np.full(8, np.inf)),
])
def test_invalid_observations_do_not_advance_or_prepare_tick(key, value):
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive"))
    obs = observation()
    obs[key] = value
    with pytest.raises(ValueError):
        scheduler.should_replan(obs)
    assert scheduler.step == 0
    assert scheduler.diagnostics()["inference_requests"] == 0
    assert tick(scheduler)[0]


def test_batched_observation_custom_keys_and_json_diagnostics():
    scheduler = ChunkScheduler(SchedulerConfig(mode="adaptive", front_image_key="front", state_key="state"))
    obs = {"front": np.zeros((1, 3, 31, 29), dtype=np.float32), "state": np.zeros((1, 8))}
    assert tick(scheduler, obs)[0]
    json.dumps(scheduler.diagnostics(), allow_nan=False)
    assert "events" not in scheduler.diagnostics(include_events=False)
