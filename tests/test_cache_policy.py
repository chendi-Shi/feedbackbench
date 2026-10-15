"""NumPy-only fake policy/sampler: no Torch, checkpoint, model or simulator load."""
import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_research.action_cache import (
    CacheProvenance, FrozenActionCache, ProjectionConfig, SparseTernaryProjection,
)
from embodied_research.cache_policy import CachePolicyAdapter
from embodied_research.cache_sampler import latent_time_contract


FEATURE = np.arange(8, dtype=np.float32)
CONTRACT = {"input_dim": 8, "version": 1, "layout": ["image", "language", "state"]}


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = "cpu"

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeAdapter:
    name = "smolvla"
    chunk_length = 50

    def __init__(self):
        self.config = SimpleNamespace(num_steps=10, max_action_dim=32,
                                      action_feature=SimpleNamespace(shape=(7,)))
        self.policy = SimpleNamespace()
        self.metadata = {"name": "smolvla", "revision": "fixture-revision", "model_sha256": "a" * 64,
                         "num_steps": 10}
        self.reset(seed=0)
        self.postprocess_calls = 0

    def reset(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.inference_count = self.observation_count = 0
        self.last_timing = {"preprocess_seconds": 0., "inference_seconds": 0., "postprocess_seconds": 0.}
        self._batch = None

    def observe(self, observation):
        self._batch = copy.deepcopy(observation)
        self.last_timing["preprocess_seconds"] = .125
        self.observation_count += 1

    def _actions_to_numpy(self, tensor):
        self.postprocess_calls += 1
        self.last_timing["postprocess_seconds"] = .2
        return (tensor.numpy() * np.float32(3) + np.float32(7)).astype(np.float32)

    def predict_chunk(self):
        raise AssertionError("wrapper must never call adapter.predict_chunk")


class FakeSampler:
    def __init__(self, adapter):
        self.adapter, self.policy = adapter, adapter.policy
        self._torch = SimpleNamespace(from_numpy=FakeTensor)
        self.model = SimpleNamespace(sample_noise=self.sample_noise)
        self.latent_contract = latent_time_contract()
        self.feature_contract = copy.deepcopy(CONTRACT)
        self.prepares, self.noise_draws = 0, 0
        self.samples, self.refinements = [], []

    def sample_noise(self, shape, device):
        assert shape == (1, 50, 32) and device == "cpu"
        self.noise_draws += 1
        return FakeTensor(self.adapter.rng.normal(size=shape).astype(np.float32))

    def prepare(self, batch):
        self.prepares += 1
        return SimpleNamespace(key_features=batch.get("feature", FEATURE).copy(),
                               feature_contract=copy.deepcopy(self.feature_contract),
                               state=FakeTensor(np.zeros((1, 32), dtype=np.float32)),
                               timings={"vision_prefix_seconds": .002, "vlm_seconds": .003,
                                        "input_preparation_seconds": .001, "key_features_seconds": .001,
                                        "prepare_seconds": .007})

    def sample(self, context, *, num_steps, noise, capture_step):
        self.samples.append({"num_steps": num_steps, "capture_step": capture_step,
                             "noise": noise.numpy().copy()})
        latent = noise.numpy()[0] * np.float32(.2) if capture_step is not None else None
        return SimpleNamespace(actions=FakeTensor(noise.numpy()[:, :, :7].copy()), nfe=num_steps,
                               latent=latent, latent_contract=latent_time_contract() if latent is not None else None,
                               timings={"action_head_seconds": .0001 * num_steps,
                                        "latent_export_seconds": .0001, "sampling_seconds": .002})

    def refine(self, context, latent, *, full_num_steps, start_step):
        self.refinements.append((full_num_steps, start_step, latent.copy()))
        return SimpleNamespace(actions=FakeTensor(latent[None, :, :7] + np.float32(1)), nfe=2,
                               latent=None, latent_contract=None,
                               timings={"action_head_seconds": .0002, "sampling_seconds": .0003})


def make_cache(mode="cache2", *, feature=FEATURE, contract=None, revision="fixture-revision",
               model_hash="a" * 64, flow_time=None, frozen=True, empty=False):
    provenance = CacheProvenance(revision, model_hash, contract or CONTRACT,
                                (.2 if mode == "cache2" else 0.) if flow_time is None else flow_time,
                                ("dev-success-1",))
    projection = SparseTernaryProjection(ProjectionConfig(8, output_dim=8, density=1., seed=4))
    shape = (50, 32) if mode == "cache2" else (50, 7)
    cache = FrozenActionCache(projection, shape, provenance, threshold=.95)
    if not empty:
        cache.add(feature, np.full(shape, .25, dtype=np.float32), case_id="dev-success-1")
    return cache.freeze() if frozen else cache


def wrapper(mode="full10", **kwargs):
    adapter = FakeAdapter()
    sampler = FakeSampler(adapter)
    policy = CachePolicyAdapter(adapter, mode, sampler=sampler, **kwargs)
    policy.reset(seed=13)
    return policy, adapter, sampler


def test_import_does_not_load_torch_or_model_stack():
    code = ("import embodied_research.cache_policy; import sys; "
            "assert not any(n.split('.')[0] in {'torch', 'lerobot', 'transformers'} for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("mode,expected_nfe,capture", [("full10", 10, 8), ("noise2", 2, None)])
def test_noise_baselines_use_exact_sampler_path_and_one_postprocess(mode, expected_nfe, capture):
    policy, adapter, sampler = wrapper(mode)
    policy.observe({"feature": FEATURE})
    result = policy.predict_chunk()
    assert result.shape == (50, 7) and result.dtype == np.float32
    assert policy.inference_count == adapter.inference_count == adapter.postprocess_calls == 1
    assert sampler.prepares == sampler.noise_draws == 1
    assert sampler.samples[0]["num_steps"] == expected_nfe and sampler.samples[0]["capture_step"] == capture
    np.testing.assert_array_equal(result, sampler.samples[0]["noise"][0, :, :7] * 3 + 7)
    record = policy.call_log[0]
    assert record["nfe"] == expected_nfe and record["hit"] is None and record["retrieved_case_id"] is None
    assert len(record["noise_sha256"]) == 64
    assert record["vlm_seconds"] == .005 and record["projection_lookup_seconds"] == 0
    assert policy.config is adapter.config and policy.chunk_length == 50
    assert policy.metadata["cache_policy"]["total_nfe"] == expected_nfe
    json.dumps(policy.metadata, allow_nan=False)


@pytest.mark.parametrize("mode,nfe,expected", [("cache2", 2, 10.75), ("cache0", 0, 7.75)])
def test_cache_hit_uses_current_vlm_and_correct_space_without_mutating_cache(mode, nfe, expected):
    cache = make_cache(mode)
    digest = cache.cache_sha256
    policy, adapter, sampler = wrapper(mode, cache=cache)
    policy.observe({"feature": FEATURE})
    result = policy.predict_chunk()
    np.testing.assert_array_equal(result, np.full((50, 7), expected, dtype=np.float32))
    assert sampler.prepares == sampler.noise_draws == adapter.postprocess_calls == 1
    assert not sampler.samples
    assert len(sampler.refinements) == int(mode == "cache2")
    if mode == "cache2":
        assert sampler.refinements[0][:2] == (10, 8)
    log = policy.call_log[0]
    assert log["hit"] is True and log["nfe"] == nfe and log["similarity"] == 1
    assert log["retrieved_case_id"] == "dev-success-1" and log["retrieved_index"] == 0
    assert cache.cache_sha256 == digest and cache.size == 1
    assert not policy.collected_samples


@pytest.mark.parametrize("mode", ["cache2", "cache0"])
def test_miss_runs_full_ten_captures_eight_but_never_admits(mode):
    cache = make_cache(mode)
    digest = cache.cache_sha256
    policy, _, sampler = wrapper(mode, cache=cache)
    policy.observe({"feature": -FEATURE})
    policy.predict_chunk()
    log = policy.call_log[0]
    assert log["hit"] is False and log["nfe"] == 10 and log["similarity"] == -1
    assert log["retrieved_case_id"] is None and log["nearest_case_id"] == "dev-success-1"
    assert sampler.samples[0]["capture_step"] == 8
    assert cache.cache_sha256 == digest and cache.size == 1 and not policy.collected_samples


def test_empty_frozen_cache_is_a_measured_miss_not_a_failed_query():
    policy, _, _ = wrapper("cache2", cache=make_cache(empty=True))
    policy.observe({"feature": FEATURE})
    policy.predict_chunk()
    assert policy.call_log[0]["hit"] is False
    assert policy.call_log[0]["similarity"] is None


@pytest.mark.parametrize("mode", ["cache2", "cache0"])
def test_hit_then_miss_reserves_same_noise_draws_as_full_baseline(mode):
    baseline, _, full_sampler = wrapper()
    cached, _, cache_sampler = wrapper(mode, cache=make_cache(mode))
    for feature in (FEATURE, -FEATURE, FEATURE, -FEATURE):
        baseline.observe({"feature": feature})
        cached.observe({"feature": feature})
        baseline.predict_chunk()
        cached.predict_chunk()
    assert [row["hit"] for row in cached.call_log] == [True, False, True, False]
    assert [row["noise_sha256"] for row in baseline.call_log] == [row["noise_sha256"] for row in cached.call_log]
    assert full_sampler.noise_draws == cache_sampler.noise_draws == 4
    for cache_call, full_call in zip(cache_sampler.samples, (full_sampler.samples[1], full_sampler.samples[3])):
        np.testing.assert_array_equal(cache_call["noise"], full_call["noise"])


def test_collection_preserves_normalized_final_and_intermediate_and_reset_replays_seed():
    policy, adapter, sampler = wrapper(collect=True)
    policy.observe({"feature": FEATURE})
    first = policy.predict_chunk()
    collected = policy.collected_samples[0]
    assert collected.call_index == 0
    assert collected.intermediate_latent.shape == (50, 32) and collected.final_normalized.shape == (50, 7)
    np.testing.assert_array_equal(first, collected.final_normalized * 3 + 7)
    np.testing.assert_array_equal(collected.intermediate_latent, sampler.samples[0]["noise"][0] * .2)
    assert collected.feature_contract == policy.feature_contract == CONTRACT
    assert collected.latent_contract == latent_time_contract()
    assert all(not array.flags.writeable for array in (collected.key_features, collected.intermediate_latent,
                                                      collected.final_normalized))
    # Returned JSON snapshots cannot alter evidence or future provenance checks.
    policy.call_log[0]["nfe"] = 999
    policy.feature_contract["version"] = 999
    policy.metadata["cache_policy"]["call_log"].clear()
    assert policy.call_log[0]["nfe"] == 10 and policy.feature_contract["version"] == 1
    policy.reset(seed=13)
    assert adapter.seed == 13 and policy.inference_count == 0 and policy.observation_count == 0
    assert policy.call_log == [] and not policy.collected_samples
    assert policy.metadata["cache_policy"]["preprocess_seconds_total"] == 0
    with pytest.raises(RuntimeError, match="observe"):
        policy.predict_chunk()
    policy.observe({"feature": FEATURE})
    np.testing.assert_array_equal(policy.predict_chunk(), first)


def test_observe_timing_survives_skipped_inference_ticks_without_double_counting():
    policy, adapter, _ = wrapper()
    policy.observe({})
    policy.predict_chunk()
    assert policy.last_timing["preprocess_seconds"] == .125
    for _ in range(3):
        policy.observe({})
        assert policy.last_timing == {"preprocess_seconds": .125, "inference_seconds": 0., "postprocess_seconds": 0.}
    policy.predict_chunk()
    first, second = policy.call_log
    assert first["preprocess_since_last_call_seconds"] == .125
    assert second["preprocess_since_last_call_seconds"] == .375
    assert second["end_to_end_seconds"] == second["prediction_seconds"] + .125
    assert second["amortized_end_to_end_seconds"] == second["prediction_seconds"] + .375
    assert policy.metadata["cache_policy"]["preprocess_seconds_total"] == .5
    assert adapter.inference_count == 2 and policy.observation_count == 4


@pytest.mark.parametrize("kwargs,match", [
    ({"frozen": False}, "frozen"), ({"revision": "other"}, "revision/hash"),
    ({"model_hash": "b" * 64}, "revision/hash"), ({"flow_time": .8}, "flow time"),
    ({"mode": "cache0"}, "shape/native"),
])
def test_incompatible_cache_is_rejected_before_inference(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wrapper("cache2", cache=make_cache(**kwargs))


def test_feature_contract_and_model_identity_are_checked_on_every_call():
    policy, adapter, sampler = wrapper("cache2", cache=make_cache())
    policy.observe({})
    policy.predict_chunk()
    sampler.feature_contract["layout"].reverse()
    with pytest.raises(ValueError, match="contract changed"):
        policy.predict_chunk()
    sampler.feature_contract = copy.deepcopy(CONTRACT)
    adapter.metadata["revision"] = "changed"
    with pytest.raises(ValueError, match="revision/hash"):
        policy.predict_chunk()
    assert policy.inference_count == 1


def test_stored_feature_contract_must_match_before_first_lookup():
    policy, _, sampler = wrapper("cache2", cache=make_cache(contract={**CONTRACT, "version": 99}))
    policy.observe({})
    with pytest.raises(ValueError, match="feature contract differs"):
        policy.predict_chunk()
    assert sampler.noise_draws == 0 and not sampler.samples


@pytest.mark.parametrize("mode,kwargs", [("unknown", {}), ("noise2", {"collect": True}),
                                        ("cache2", {}), ("full10", {"cache": make_cache()})])
def test_invalid_modes_and_collection_combinations_rejected(mode, kwargs):
    with pytest.raises(ValueError):
        wrapper(mode, **kwargs)


def test_frozen_episode_runner_accepts_wrapper_and_counts_every_preprocessed_tick(tmp_path):
    from embodied_research.experiment import run_episode

    class StubEnvironment:
        horizon = 21
        task = "fixture only"

        def packet(self):
            return {"observation.state": np.zeros(8, dtype=np.float32),
                    "observation.images.image": np.zeros((3, 2, 2), dtype=np.float32),
                    "observation.images.image2": np.zeros((3, 2, 2), dtype=np.float32),
                    "task": self.task}

        def reset(self, seed, init_state_id):
            self.steps = 0
            return self.packet()

        def frame(self, camera="image"):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def step(self, action):
            assert action.shape == (7,)
            self.steps += 1
            return self.packet(), 0., False, self.steps == self.horizon, {"is_success": False}

        def metadata(self):
            return {"fixture": True}

    policy, _, sampler = wrapper("cache2", cache=make_cache())
    case = {"seed": 1000, "init_state_id": 0, "condition": "clean",
            "scheduler": {"mode": "fixed", "fixed_interval": 10, "temporal_ensemble": False}}
    result = run_episode(StubEnvironment(), policy, case, tmp_path)
    assert result["steps"] == 21 and result["inference_calls"] == policy.inference_count == 3
    assert result["preprocess_seconds"] == 21 * .125
    assert policy.observation_count == 21 and sampler.prepares == 3
    assert result["model_info"]["cache_policy"]["total_nfe"] == 6
    assert [log["observation_count"] for log in policy.call_log] == [1, 11, 21]
    assert result["policy_inference_seconds"] == sum(row["inference_seconds"] for row in policy.call_log)
    json.dumps(result, allow_nan=False)
