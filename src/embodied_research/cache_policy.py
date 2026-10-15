"""Episode-compatible, read-only ActionCache inference on frozen SmolVLA.

This is an independent transfer of ActionCache, not its authors' implementation.
All four modes run the current observation's VLM. The simulator may pause during
inference: measured CPU timings do not establish real-time robot operation.

The caller owns fixed-10 execution, disabling temporal ensembling, and admitting
only successful development episodes into separate intermediate/final caches.
This class never inserts a cache entry, including on a miss.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from .action_cache import FrozenActionCache
from .cache_sampler import SmolVLACacheSampler, latent_time_contract


MODES = ("full10", "noise2", "cache2", "cache0")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _owned_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CollectedSample:
    """Unadmitted full10 prefill evidence; arrays are independent read-only copies.

    No episode outcome is accepted here. The runner must discard all samples
    from unsuccessful/non-development episodes before building a cache.
    """

    call_index: int
    key_features: np.ndarray
    intermediate_latent: np.ndarray
    final_normalized: np.ndarray
    feature_contract: dict[str, Any]
    latent_contract: dict[str, Any]


class CachePolicyAdapter:
    """Wrap an already-loaded PolicyAdapter; do not load a second model.

    ``sampler`` is injectable for lightweight tests. ``cache`` must be frozen,
    model-matched and of the mode's exact latent/time contract. ``collect=True``
    is accepted only for full10 prefill. ``reset(seed)`` delegates to the base
    adapter's torch.manual_seed and clears episode observations/logs/collection.
    """

    def __init__(self, adapter: Any, mode: str = "full10", *, sampler: Any = None,
                 cache: FrozenActionCache | None = None, collect: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not isinstance(collect, bool) or (collect and mode != "full10"):
            raise ValueError("collection is supported only in full10 development prefill")
        if getattr(adapter, "name", None) != "smolvla":
            raise ValueError("CachePolicyAdapter requires the SmolVLA PolicyAdapter")
        config = adapter.config
        if (adapter.chunk_length != 50 or config.max_action_dim != 32
                or tuple(config.action_feature.shape) != (7,)):
            raise ValueError("this experiment requires SmolVLA chunks (50,32), actions (50,7)")
        if config.num_steps != 10:
            raise ValueError("base adapter must retain the full ten-step configuration")
        if mode.startswith("cache"):
            if not isinstance(cache, FrozenActionCache):
                raise ValueError("cache modes require a FrozenActionCache")
        elif cache is not None:
            raise ValueError("full10/noise2 must not receive a cache")
        self.adapter, self.mode, self.cache, self.collect = adapter, mode, cache, collect
        self.sampler = sampler if sampler is not None else SmolVLACacheSampler(adapter.policy)
        if self.sampler.policy is not adapter.policy:
            raise ValueError("sampler must wrap the same policy instance as the adapter")
        if self.sampler.latent_contract != latent_time_contract():
            raise ValueError("sampler must expose the original ten-step, capture-eight contract")
        self._feature_contract: dict[str, Any] | None = None
        self._validate_cache()
        self._cache_identity = None if cache is None else {
            "cache_sha256": cache.cache_sha256,
            "provenance_sha256": cache.provenance_sha256,
            "entries": cache.size,
            "threshold": cache.threshold,
            "latent_shape": list(cache.latent_shape),
            "provenance": cache.provenance.to_dict(),
        }
        self._clear_episode()

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def config(self) -> Any:
        return self.adapter.config

    @property
    def chunk_length(self) -> int:
        return self.adapter.chunk_length

    @property
    def inference_count(self) -> int:
        return self.adapter.inference_count

    @property
    def observation_count(self) -> int:
        return self.adapter.observation_count

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Detached JSON records for completed calls, without tensor contents."""
        return _json_copy(self._call_log)

    @property
    def collected_samples(self) -> tuple[CollectedSample, ...]:
        return tuple(self._collected_samples)

    @property
    def feature_contract(self) -> dict[str, Any] | None:
        return _json_copy(self._feature_contract)

    @property
    def metadata(self) -> dict[str, Any]:
        result = _json_copy(self.adapter.metadata)
        result["num_steps_scope"] = "base configuration; executed NFE is in cache_policy.call_log"
        result["cache_policy"] = {
            "implementation": "independent ActionCache transfer to frozen SmolVLA",
            "mode": self.mode,
            "cache": self._cache_identity,
            "feature_contract": self._feature_contract,
            "latent_contract": latent_time_contract(),
            "test_time_cache_updates": False,
            "noise_schedule": "one full (1,50,32) noise draw per call, including discarded draws on hits",
            "collect_unadmitted_prefill": self.collect,
            "collected_samples": len(self._collected_samples),
            "observations": self.observation_count,
            "preprocess_seconds_total": self._preprocess_seconds_total,
            "preprocess_seconds_since_last_call": self._pending_preprocess_seconds,
            "completed_calls": len(self._call_log),
            "total_nfe": sum(item["nfe"] for item in self._call_log),
            "cache_hits": sum(item["hit"] is True for item in self._call_log),
            "call_log": self._call_log,
            "timing_scope": {
                "vlm_seconds": "vision prefix embedding plus VLM transformer forward",
                "action_head_seconds": "denoising evaluations plus Euler updates",
                "projection_lookup_seconds": "projection, cosine search and retrieved latent copy",
                "inference_seconds": "prepare + provenance validation + lookup + sampling; excludes postprocess",
                "end_to_end_seconds": "prediction call + latest observation preprocessing; not elapsed simulator time",
                "amortized_end_to_end_seconds": "prediction call + all observations preprocessed since previous call",
            },
            "execution_claim": "synchronous paused simulation; no real-time deployment claim",
        }
        return _json_copy(result)

    def _clear_episode(self) -> None:
        self._call_log: list[dict[str, Any]] = []
        self._collected_samples: list[CollectedSample] = []
        self._preprocess_seconds_total = self._pending_preprocess_seconds = 0.0
        self._observed = False
        self.last_timing = {"preprocess_seconds": 0.0, "inference_seconds": 0.0,
                            "postprocess_seconds": 0.0}

    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset(seed=seed)
        self._clear_episode()
        self._validate_cache()

    def observe(self, observation: dict) -> None:
        self.adapter.observe(observation)
        elapsed = float(self.adapter.last_timing["preprocess_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("adapter preprocessing time must be finite and nonnegative")
        self.last_timing = {"preprocess_seconds": elapsed, "inference_seconds": 0.0,
                            "postprocess_seconds": 0.0}
        self._preprocess_seconds_total += elapsed
        self._pending_preprocess_seconds += elapsed
        self._observed = True

    def _validate_cache(self, feature_contract: dict | None = None) -> None:
        if self.cache is None:
            return
        cache = self.cache
        if not cache.is_frozen:
            raise ValueError("cache must be frozen; test-time admission is forbidden")
        metadata = self.adapter.metadata
        provenance = cache.provenance.to_dict()
        if (provenance["model_revision"] != metadata["revision"]
                or provenance["model_sha256"] != metadata["model_sha256"]):
            raise ValueError("cache model revision/hash differs from the active policy")
        shape, native_time = ((50, 32), .2) if self.mode == "cache2" else ((50, 7), 0.)
        if cache.latent_shape != shape or not math.isclose(provenance["flow_time"], native_time,
                                                        abs_tol=1e-12, rel_tol=0):
            raise ValueError("cache latent shape/native flow time differs from the selected mode")
        if feature_contract is not None:
            if provenance["feature_contract"] != _json_copy(feature_contract):
                raise ValueError("cache feature contract differs from the current sampler output")
            if cache.projection.config.input_dim != feature_contract["input_dim"]:
                raise ValueError("cache projection dimension differs from sampler feature contract")

    def predict_chunk(self) -> np.ndarray:
        if not self._observed or self.adapter._batch is None:
            raise RuntimeError("observe() is required before predict_chunk()")
        start = time.perf_counter()
        context = self.sampler.prepare(self.adapter._batch)
        contract = _json_copy(context.feature_contract)
        feature = _owned_array(context.key_features, (contract["input_dim"],), "key features")
        if self._feature_contract is not None and contract != self._feature_contract:
            raise ValueError("sampler feature contract changed during the experiment")
        self._feature_contract = contract
        self._validate_cache(contract)
        # Reserve the same random draw at every control-decision index for all
        # modes. Otherwise a hit would shift every subsequent miss's noise.
        noise = self.sampler.model.sample_noise((1, 50, 32), context.state.device)
        noise_array = _owned_array(noise.detach().cpu().numpy(), (1, 50, 32), "sampling noise")
        noise_sha256 = hashlib.sha256(noise_array.astype("<f4", copy=False).tobytes()).hexdigest()
        lookup, lookup_seconds = None, 0.0
        if self.cache is not None:
            tick = time.perf_counter()
            lookup = self.cache.lookup(feature)
            lookup_seconds = time.perf_counter() - tick

        sampled = None
        if lookup is not None and lookup.hit and self.mode == "cache0":
            # Independent t=0 cache stores only the seven normalized output
            # channels. The already-reserved noise is discarded; no denoising occurs.
            final = _owned_array(lookup.latent, (50, 7), "cached final actions")
            normalized = self.sampler._torch.from_numpy(np.array(final, copy=True))[None]
            nfe, sample_timings = 0, {}
        else:
            if lookup is not None and lookup.hit:
                sampled = self.sampler.refine(context, lookup.latent, full_num_steps=10, start_step=8)
                expected_nfe = 2
            else:
                expected_nfe = 2 if self.mode == "noise2" else 10
                sampled = self.sampler.sample(context, num_steps=expected_nfe,
                                              noise=noise,
                                              capture_step=8 if expected_nfe == 10 else None)
            if type(sampled.nfe) is not int or sampled.nfe != expected_nfe:
                raise ValueError("sampler NFE differs from the requested inference path")
            normalized, nfe, sample_timings = sampled.actions, sampled.nfe, sampled.timings
            if expected_nfe == 10:
                _owned_array(sampled.latent, (50, 32), "captured intermediate latent")
                if sampled.latent_contract != latent_time_contract():
                    raise ValueError("captured latent has the wrong flow-time contract")
        inference_seconds = time.perf_counter() - start
        collected = None
        if self.collect:
            final = normalized[0].detach().cpu().numpy()
            collected = CollectedSample(
                call_index=len(self._call_log), key_features=feature,
                intermediate_latent=_owned_array(sampled.latent, (50, 32), "prefill latent"),
                final_normalized=_owned_array(final, (50, 7), "prefill final actions"),
                feature_contract=_json_copy(contract), latent_contract=_json_copy(sampled.latent_contract),
            )
        tick = time.perf_counter()
        robot_actions = self.adapter._actions_to_numpy(normalized)
        postprocess_seconds = time.perf_counter() - tick
        output = _owned_array(robot_actions, (1, 50, 7), "postprocessed robot actions")[0].copy()
        prediction_seconds = time.perf_counter() - start
        prepare = context.timings
        log = {
            "call_index": len(self._call_log), "observation_count": self.observation_count,
            "mode": self.mode, "nfe": nfe, "hit": None if lookup is None else bool(lookup.hit),
            "noise_sha256": noise_sha256,
            "similarity": None if lookup is None else lookup.similarity,
            "retrieved_index": lookup.index if lookup is not None and lookup.hit else None,
            "retrieved_case_id": lookup.case_id if lookup is not None and lookup.hit else None,
            "nearest_case_id": None if lookup is None else lookup.case_id,
            "preprocess_seconds": self.last_timing["preprocess_seconds"],
            "preprocess_since_last_call_seconds": self._pending_preprocess_seconds,
            "vlm_seconds": prepare["vision_prefix_seconds"] + prepare["vlm_seconds"],
            "vlm_transformer_seconds": prepare["vlm_seconds"],
            "vision_prefix_seconds": prepare["vision_prefix_seconds"],
            "input_preparation_seconds": prepare["input_preparation_seconds"],
            "key_features_seconds": prepare["key_features_seconds"],
            "prepare_seconds": prepare["prepare_seconds"],
            "action_head_seconds": sample_timings.get("action_head_seconds", 0.0),
            "latent_export_seconds": sample_timings.get("latent_export_seconds", 0.0),
            "sampling_seconds": sample_timings.get("sampling_seconds", 0.0),
            "projection_lookup_seconds": lookup_seconds,
            "postprocess_seconds": postprocess_seconds,
            "inference_seconds": inference_seconds,
            "prediction_seconds": prediction_seconds,
            "end_to_end_seconds": prediction_seconds + self.last_timing["preprocess_seconds"],
            "amortized_end_to_end_seconds": prediction_seconds + self._pending_preprocess_seconds,
        }
        for key, value in log.items():
            if key.endswith("_seconds") and (not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid measured timing: {key}")
        self._call_log.append(_json_copy(log))
        if collected is not None:
            self._collected_samples.append(collected)
        # Calling adapter.predict_chunk here would run the model a second time
        # and increment this counter again. Only this wrapper increments it.
        self.adapter.inference_count += 1
        self.last_timing["inference_seconds"] = inference_seconds
        self.last_timing["postprocess_seconds"] = postprocess_seconds
        self._pending_preprocess_seconds = 0.0
        return output
