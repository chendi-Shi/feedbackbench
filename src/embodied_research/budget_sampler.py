"""Shared-first-velocity Euler sampling and current-observation budget features.

The base policy, cache sampler, RNG state and time grids are unchanged. A probe
costs one denoising evaluation; sampling with it costs K-1 additional evaluations
on the chosen K-step grid. Only that first velocity is shared across grids.
Contexts are for serial use only. This module imports no model or Torch stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any

import numpy as np

from .cache_sampler import PreparedContext, SmolVLACacheSampler, euler_grid


BUDGETS = (2, 4, 10)
FEATURE_MODES = {
    "full": ("image_mean", "language_mean", "encoded_state", "raw_state", "noise_stats", "velocity_stats"),
    "no_probe": ("image_mean", "language_mean", "encoded_state", "raw_state", "noise_stats"),
    "proprio_probe": ("raw_state", "noise_stats", "velocity_stats"),
}


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def _context_metadata_sha256(context: PreparedContext) -> str:
    digest = hashlib.sha256(json.dumps(context.feature_contract, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for value in (context.key_features, context.state.detach().cpu().numpy(),
                  context.prefix_pad_masks.detach().cpu().numpy()):
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("budget features must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FirstVelocity:
    """One actual denoise evaluation bound to its context and exact input noise.

    Torch tensors are mutable despite a frozen dataclass. Their recorded digest
    is checked on every reuse; changing velocity invalidates this probe.
    """

    velocity: Any
    noise_sha256: str
    velocity_sha256: str
    context_metadata_sha256: str
    nfe: int
    timings: dict[str, float]
    _sampler: Any = field(repr=False)
    _context: PreparedContext = field(repr=False)


@dataclass(frozen=True)
class BudgetSample:
    actions: Any
    full_latent: Any
    nfe: int
    new_evaluations: int
    timings: dict[str, float]


@dataclass(frozen=True)
class BudgetFeatures:
    values: np.ndarray
    layout: dict[str, Any]
    feature_slices: dict[str, tuple[int, int]]

    def for_mode(self, mode: str) -> np.ndarray:
        """Return an owned vector; no hidden VLM features in proprio_probe."""
        if mode not in FEATURE_MODES:
            raise ValueError(f"unknown budget feature mode: {mode}")
        return _readonly(np.concatenate([
            self.values[slice(*self.feature_slices[name])] for name in FEATURE_MODES[mode]
        ]))


class BudgetSampler:
    """Reuse an existing batch-one CPU SmolVLACacheSampler and its one model.

    Adapter inference counters are owned by the caller. ``nfe`` is the logical
    path length, whereas ``new_evaluations`` counts calls made by this sample.
    Probe time is reported separately, never added to the sample wall timer.
    The caller must keep model weights and prefix KV values unchanged until all
    paired samples finish. Metadata digests and KV length checks detect ordinary
    input changes; this wrapper does not hash every model or KV tensor per call.
    """

    def __init__(self, sampler: SmolVLACacheSampler):
        if not isinstance(sampler, SmolVLACacheSampler):
            raise TypeError("BudgetSampler requires an existing SmolVLACacheSampler")
        self.sampler = sampler
        self.model = sampler.model
        self.config = sampler.config
        self._torch = sampler._torch
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.sampler._validate_configuration()
        if (self.config.chunk_size, self.config.max_action_dim, self.config.action_feature.shape[0]) != (50, 32, 7):
            raise ValueError("budget sampler requires the SmolVLA (50,32) internal / 7 output contract")

    def _validate_inputs(self, context: PreparedContext, noise: Any) -> None:
        self._validate_configuration()
        self.sampler._validate_context(context)
        self.sampler._require_tensor(noise, "noise", shape=(1, 50, 32))
        if noise.dtype != self._torch.float32:
            raise ValueError("noise must have dtype float32")
        if noise.device != context.state.device:
            raise ValueError("noise and context must share the same CPU device")

    def _validate_probe(self, context: PreparedContext, noise: Any, probe: FirstVelocity) -> None:
        self._validate_inputs(context, noise)
        if not isinstance(probe, FirstVelocity) or probe._sampler is not self or probe._context is not context:
            raise ValueError("probe must belong to this BudgetSampler and exact context")
        if type(probe.nfe) is not int or probe.nfe != 1 or probe.noise_sha256 != _tensor_sha256(noise):
            raise ValueError("probe does not match the exact noise")
        if probe.context_metadata_sha256 != _context_metadata_sha256(context):
            raise ValueError("context metadata changed after probe evaluation")
        self._validate_velocity(probe.velocity, noise)
        if probe.velocity_sha256 != _tensor_sha256(probe.velocity):
            raise ValueError("probe velocity changed after its evaluation")

    def _validate_velocity(self, velocity: Any, reference: Any) -> None:
        self.sampler._require_tensor(velocity, "velocity", shape=tuple(reference.shape))
        if velocity.dtype != reference.dtype:
            raise ValueError("velocity must match noise dtype float32")

    def _denoise(self, context: PreparedContext, x_t: Any, native_time: float) -> Any:
        torch = self._torch
        timestep = torch.tensor(native_time, dtype=torch.float32, device=x_t.device).expand(1)
        velocity = self.model.denoise_step(
            x_t=x_t, prefix_pad_masks=context.prefix_pad_masks,
            past_key_values=context.past_key_values, timestep=timestep,
        )
        self._validate_velocity(velocity, x_t)
        # Fail before another evaluation if the upstream cache was not restored.
        self.sampler._validate_context(context)
        return velocity

    def probe(self, context: PreparedContext, noise: Any) -> FirstVelocity:
        started = time.perf_counter()
        self._validate_inputs(context, noise)
        with self._torch.inference_mode():
            tick = time.perf_counter()
            velocity = self._denoise(context, noise.detach().clone(), 1.0).detach().clone()
            head_seconds = time.perf_counter() - tick
            noise_sha = _tensor_sha256(noise)
            velocity_sha = _tensor_sha256(velocity)
            context_sha = _context_metadata_sha256(context)
        return FirstVelocity(
            velocity=velocity, noise_sha256=noise_sha, velocity_sha256=velocity_sha, nfe=1,
            context_metadata_sha256=context_sha,
            timings={"action_head_seconds": head_seconds, "probe_seconds": time.perf_counter() - started},
            _sampler=self, _context=context,
        )

    def sample(self, context: PreparedContext, noise: Any, num_steps: int = 10,
               probe: FirstVelocity | None = None) -> BudgetSample:
        if type(num_steps) is not int or num_steps not in BUDGETS:
            raise ValueError("num_steps must be one of 2, 4, 10")
        started = time.perf_counter()
        self._validate_inputs(context, noise)
        if probe is not None:
            self._validate_probe(context, noise, probe)
        dt, times = euler_grid(num_steps)
        head_seconds = 0.0
        with self._torch.inference_mode():
            x_t = noise.detach().clone()
            for step, native_time in enumerate(times):
                if step == 0 and probe is not None:
                    velocity = probe.velocity
                else:
                    tick = time.perf_counter()
                    velocity = self._denoise(context, x_t, native_time)
                    head_seconds += time.perf_counter() - tick
                # Preserve the frozen sampler's operation order and all 32 channels.
                x_t = x_t + dt * velocity
            self.sampler._require_tensor(x_t, "generated full latent", shape=(1, 50, 32))
            self.sampler._validate_context(context)
            full_latent = x_t.detach().clone()
            actions = x_t[:, :, :7].detach().clone()
        return BudgetSample(
            actions=actions, full_latent=full_latent, nfe=num_steps,
            new_evaluations=num_steps - int(probe is not None),
            timings={"action_head_seconds": head_seconds,
                     "sampling_seconds": time.perf_counter() - started,
                     "reused_probe_seconds": 0.0 if probe is None else probe.timings["probe_seconds"]},
        )


def extract_budget_features(context: PreparedContext, raw_state: np.ndarray,
                            noise: Any, probe: FirstVelocity) -> BudgetFeatures:
    """Pool current features only; no teacher action or episode outcome is read.

    Canonical language padding was added AFTER VLM evaluation. Recover the actual
    language positions from the original prefix mask and its last valid (state)
    token, so a valid zero-valued language embedding still counts in the mean.
    Noise and velocity statistics cover the first ten executed actions / seven
    output channels, with per-channel population mean then population std.
    """
    if not isinstance(probe, FirstVelocity) or not isinstance(probe._sampler, BudgetSampler):
        raise ValueError("features require a genuine FirstVelocity probe")
    probe._sampler._validate_probe(context, noise, probe)
    if (not isinstance(raw_state, np.ndarray) or raw_state.dtype != np.float32
            or raw_state.shape != (8,) or not np.isfinite(raw_state).all()):
        raise ValueError("raw_state must be a finite float32 NumPy array with shape (8,)")
    contract = context.feature_contract
    sizes = {key: contract.get(key) for key in ("image_tokens", "language_slots", "state_tokens", "hidden_size", "input_dim")}
    if any(type(value) is not int or value < 1 for value in sizes.values()) or sizes["state_tokens"] != 1:
        raise ValueError("invalid prefix feature contract")
    images, language, width = sizes["image_tokens"], sizes["language_slots"], sizes["hidden_size"]
    if sizes["input_dim"] != (images + language + 1) * width:
        raise ValueError("prefix feature dimensions disagree")
    values = context.key_features
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32
            or values.shape != (sizes["input_dim"],) or not np.isfinite(values).all()):
        raise ValueError("invalid prefix key features")
    padding = context.prefix_pad_masks.detach().cpu().numpy()
    if padding.ndim != 2 or padding.shape[0] != 1 or not np.isin(padding, [0, 1]).all():
        raise ValueError("invalid prefix padding mask")
    mask = padding[0].astype(bool)
    positions = np.flatnonzero(mask)
    if not positions.size:
        raise ValueError("prefix has no valid state token")
    state_position = int(positions[-1])
    language_length = state_position - images
    if not 1 <= language_length <= language or not mask[:images].any():
        raise ValueError("invalid image/language/state positions")
    language_mask = mask[images:state_position]
    if not language_mask.any():
        raise ValueError("prefix has no valid language token")
    canonical = values.reshape(images + language + 1, width)
    if np.any(canonical[images + language_length:images + language] != 0):
        raise ValueError("canonical trailing language padding must be zero")
    if (np.any(canonical[:images][~mask[:images]] != 0)
            or np.any(canonical[images:images + language_length][~language_mask] != 0)):
        raise ValueError("masked canonical prefix features must be zero")

    def stats(tensor: Any) -> np.ndarray:
        array = tensor.detach().cpu().numpy()[0, :10, :7]
        return np.concatenate((array.mean(axis=0, dtype=np.float32), array.std(axis=0, dtype=np.float32)))

    blocks = {
        "image_mean": canonical[:images][mask[:images]].mean(axis=0, dtype=np.float32),
        "language_mean": canonical[images:images + language_length][language_mask].mean(axis=0, dtype=np.float32),
        "encoded_state": canonical[-1],
        "raw_state": raw_state,
        "noise_stats": stats(noise),
        "velocity_stats": stats(probe.velocity),
    }
    slices, offset = {}, 0
    for name, block in blocks.items():
        slices[name] = (offset, offset + block.size)
        offset += block.size
    layout = {
        "version": 1, "dtype": "float32", "input_dim": offset,
        "prefix_feature_contract": dict(contract),
        "blocks": {name: {"start": start, "stop": stop} for name, (start, stop) in slices.items()},
        "pooling": "mean of valid image tokens; mean of valid language tokens; single encoded-state token",
        "statistics": {"action_rows": [0, 10], "channels": [0, 7],
                       "order": ["per_channel_mean", "per_channel_population_std"], "ddof": 0},
        "modes": {mode: {"blocks": list(names),
                         "dimension": sum(slices[name][1] - slices[name][0] for name in names)}
                  for mode, names in FEATURE_MODES.items()},
    }
    return BudgetFeatures(_readonly(np.concatenate(list(blocks.values()))), layout, slices)
