"""Split SmolVLA inference for an independent ActionCache transfer, without hooks.

The policy weights, configuration, upstream modules, and action queue are never
modified. This wrapper supports the project's single-observation, single-robot
CPU SmolVLA contract. It does not implement retrieval or decide cache admission.

ActionCache (https://arxiv.org/html/2607.06370v2) uses paper time tau=0 for
noise and tau=1 for actions. SmolVLA uses t=1-tau. On its ten-step Euler grid,
the state after eight updates is native t=.2 / paper tau=.8. A cache hit starts
there and performs the *last two original updates*, not two updates from t=1.

The upstream reference is lerobot/policies/smolvla/modeling_smolvla.py and
policies/common/flow_matching.py (Hugging Face, Apache-2.0). We call its image
preparation, prefix embedding, VLM forward and denoise_step directly, preserving
the Euler arithmetic and time grid. Torch is imported only when constructing a
sampler; importing this module does not import Torch, LeRobot or Transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np


LANGUAGE_TOKENS = "observation.language.tokens"
LANGUAGE_MASK = "observation.language.attention_mask"


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def euler_grid(num_steps: int, *, start_step: int = 0) -> tuple[float, tuple[float, ...]]:
    """Return the exact upstream scalar dt and remaining native-time grid."""
    _positive_integer(num_steps, "num_steps")
    _positive_integer(start_step, "start_step", minimum=0)
    if start_step >= num_steps:
        raise ValueError("start_step must be smaller than num_steps")
    dt = -1.0 / num_steps
    return dt, tuple(1.0 + step * dt for step in range(start_step, num_steps))


def latent_time_contract(full_num_steps: int = 10, completed_steps: int = 8) -> dict[str, Any]:
    """JSON-compatible convention for a cached, still-normalized internal latent."""
    euler_grid(full_num_steps, start_step=completed_steps)
    if completed_steps == 0:
        raise ValueError("a cached intermediate latent must follow at least one update")
    return {
        "full_num_steps": full_num_steps,
        "completed_steps": completed_steps,
        "remaining_nfe": full_num_steps - completed_steps,
        "native_time": (full_num_steps - completed_steps) / full_num_steps,
        "paper_tau": completed_steps / full_num_steps,
        "time_relation": "smolvla_t = 1 - paper_tau",
        "space": "normalized full max_action_dim, including internal padded channels",
        "dtype": "float32",
    }


@dataclass(frozen=True)
class PreparedContext:
    """One observation's VLM result; reuse serially, never across samplers.

    key_features is an owned read-only 1D NumPy array. past_key_values is an
    internal upstream cache: denoise_step temporarily extends it then restores
    the prefix. Therefore concurrent sampling from one context is unsupported.
    """

    key_features: np.ndarray
    feature_contract: dict[str, Any]
    timings: dict[str, float]
    state: Any = field(repr=False)
    prefix_pad_masks: Any = field(repr=False)
    past_key_values: Any = field(repr=False)
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class SampleResult:
    """Normalized BxHxaction_dim actions and optional Hxmax_action_dim latent."""

    actions: Any
    latent: np.ndarray | None
    latent_contract: dict[str, Any] | None
    nfe: int
    timings: dict[str, float]


class SmolVLACacheSampler:
    """Use with ``prepare(adapter._batch)`` after ``adapter.observe(obs)``.

    Convert returned actions through adapter._actions_to_numpy(result.actions)
    exactly once. Features and latents are not robot-unit actions. This class
    intentionally does not increment adapter inference counters: the caller owns
    episode accounting and combines context/result timing records explicitly.
    """

    def __init__(self, policy: Any):
        import torch

        self._torch = torch
        self.policy = policy
        self.model = policy.model
        self.config = policy.config
        self._owner = object()
        self._feature_contract: dict[str, Any] | None = None
        self._validate_configuration()
        self.policy.eval()

    @property
    def latent_contract(self) -> dict[str, Any]:
        return latent_time_contract()

    def _validate_configuration(self) -> None:
        if self.config.n_obs_steps != 1:
            raise ValueError("cache sampler requires n_obs_steps=1; observation queues are not emulated")
        if self.config.adapt_to_pi_aloha:
            raise ValueError("Aloha action/state conversion is unsupported by this LIBERO sampler")
        if not self.config.use_cache:
            raise ValueError("cache sampler requires the upstream prefix KV cache")
        rtc = getattr(self.config, "rtc_config", None)
        if rtc is not None and rtc.enabled:
            raise ValueError("RTC guidance must be disabled for ActionCache sampling")
        for name in ("chunk_size", "max_action_dim", "tokenizer_max_length"):
            _positive_integer(getattr(self.config, name), name)
        dim = self.config.action_feature.shape[0]
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 < dim <= self.config.max_action_dim:
            raise ValueError("invalid original action dimension")

    def _require_tensor(self, value: Any, name: str, *, shape: tuple[int, ...] | None = None) -> None:
        torch = self._torch
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a Torch tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU; GPU timing is not supported")
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, received {tuple(value.shape)}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must be finite")

    def prepare(self, batch: dict[str, Any]) -> PreparedContext:
        """Prepare the current processed batch, running its VLM exactly once.

        Canonical key layout fixes language padding *after* the model forward.
        Adding tokens to the actual VLM input would change its cache and could
        break upstream parity. The encoded state always occupies the last key
        slot rather than shifting with each instruction's token length.
        """
        self._validate_configuration()
        torch = self._torch
        started = time.perf_counter()
        if not isinstance(batch, dict):
            raise ValueError("prepare requires the preprocessed adapter._batch dictionary")
        missing = [key for key in (*self.config.image_features, "observation.state", LANGUAGE_TOKENS,
                                   LANGUAGE_MASK) if key not in batch]
        if missing:
            raise ValueError(f"missing preprocessed observation features: {missing}")
        tokens, language_mask = batch[LANGUAGE_TOKENS], batch[LANGUAGE_MASK]
        self._require_tensor(tokens, "language tokens")
        if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= self.config.tokenizer_max_length:
            raise ValueError("single-batch language tokens must fit tokenizer_max_length")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("language tokens must be integer IDs")
        self._require_tensor(language_mask, "language attention mask", shape=tuple(tokens.shape))
        if not ((language_mask == 0) | (language_mask == 1)).all().item():
            raise ValueError("language attention mask must be binary")
        with torch.inference_mode():
            images, image_masks = self.policy.prepare_images(dict(batch))
            state = self.policy.prepare_state(dict(batch)).detach().clone()
            self._require_tensor(state, "normalized state")
            if state.ndim != 2 or state.shape[0] != 1:
                raise ValueError("cache sampler supports exactly one current robot state")
            for index, pixels in enumerate(images):
                self._require_tensor(pixels, f"prepared image {index}")
                if pixels.shape[0] != 1:
                    raise ValueError("cache sampler supports batch_size=1")
            input_seconds = time.perf_counter() - started
            tick = time.perf_counter()
            embeddings, padding, attention = self.model.embed_prefix(
                images, image_masks, tokens, language_mask, state=state,
            )
            # Identical prefix attention relation to upstream make_att_2d_masks.
            groups = torch.cumsum(attention, dim=1)
            attention_2d = (groups[:, None, :] <= groups[:, :, None]) & (
                padding[:, None, :] * padding[:, :, None]
            )
            positions = torch.cumsum(padding, dim=1) - 1
            prefix_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            output, kv = self.model.vlm_with_expert.forward(
                attention_mask=attention_2d, position_ids=positions, past_key_values=None,
                inputs_embeds=[embeddings, None], use_cache=self.config.use_cache,
            )
            vlm_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            features, contract = self._key_features(output[0], padding, attention, tokens.shape[1])
            if kv is None:
                raise ValueError("upstream VLM did not return a prefix KV cache")
            if self._feature_contract is not None and contract != self._feature_contract:
                raise ValueError("VLM feature layout changed; a frozen cache requires one fixed feature contract")
            self._feature_contract = dict(contract)
            key_seconds = time.perf_counter() - tick
        return PreparedContext(
            key_features=features, feature_contract=contract,
            timings={"input_preparation_seconds": input_seconds, "vision_prefix_seconds": prefix_seconds,
                     "vlm_seconds": vlm_seconds, "key_features_seconds": key_seconds,
                     "prepare_seconds": time.perf_counter() - started},
            state=state, prefix_pad_masks=padding.detach().clone(), past_key_values=kv, _owner=self._owner,
        )

    def _key_features(self, output: Any, padding: Any, attention: Any,
                      language_length: int) -> tuple[np.ndarray, dict[str, Any]]:
        torch = self._torch
        self._require_tensor(output, "VLM final prefix output")
        if output.ndim != 3 or output.shape[0] != 1 or tuple(padding.shape) != tuple(output.shape[:2]):
            raise ValueError("invalid VLM prefix output/padding layout")
        # SmolVLA's single state token alone starts an autoregressive prefix block.
        state_positions = torch.nonzero(attention[0], as_tuple=False).flatten()
        if state_positions.numel() != 1:
            raise ValueError("expected exactly one state token in the SmolVLA prefix")
        state_position = int(state_positions.item())
        image_tokens = state_position - language_length
        if image_tokens < 1 or not padding[0, state_position].item():
            raise ValueError("invalid image/language/state prefix layout")
        width = output.shape[-1]
        max_language = self.config.tokenizer_max_length
        canonical = torch.zeros((image_tokens + max_language + 1, width), dtype=torch.float32)
        valid_output = output[0].float().masked_fill(~padding[0, :, None].bool(), 0)
        canonical[:image_tokens] = valid_output[:image_tokens]
        canonical[image_tokens:image_tokens + language_length] = valid_output[image_tokens:state_position]
        canonical[-1] = valid_output[state_position]
        features = canonical.reshape(-1).numpy().copy()
        features.setflags(write=False)
        contract = {
            "version": 1, "source": "SmolVLA final normalized VLM prefix output",
            "layout": "image tokens, language padded after VLM to tokenizer_max_length, encoded state token",
            "image_feature_keys": list(self.config.image_features), "image_tokens": image_tokens,
            "language_slots": max_language, "state_tokens": 1, "hidden_size": width,
            "input_dim": int(features.size), "dtype": "float32", "padding": "masked outputs set to zero",
        }
        return features, contract

    def _validate_context(self, context: PreparedContext) -> None:
        self._validate_configuration()
        if not isinstance(context, PreparedContext) or context._owner is not self._owner:
            raise ValueError("context must be prepared by this sampler")
        expected = context.prefix_pad_masks.shape[1]
        get_length = getattr(context.past_key_values, "get_seq_length", None)
        if callable(get_length) and get_length() != expected:
            raise ValueError("prefix KV cache was not restored after denoising")

    def sample(self, context: PreparedContext, *, num_steps: int = 10, noise: Any = None,
               capture_step: int | None = None) -> SampleResult:
        """Sample from pure noise with an unmodified Euler grid.

        Request capture_step=8 with num_steps=10 to build the two-step cache.
        Fewer-step baselines use their own full [1,0] grid and no capture.
        Supplying the same noise is the strongest upstream parity comparison.
        """
        euler_grid(num_steps)
        if capture_step is not None:
            latent_time_contract(num_steps, capture_step)
        self._validate_context(context)
        shape = (1, self.config.chunk_size, self.config.max_action_dim)
        with self._torch.inference_mode():
            if noise is None:
                noise = self.model.sample_noise(shape, context.state.device)
            self._require_tensor(noise, "noise", shape=shape)
            if noise.dtype != self._torch.float32:
                raise ValueError("noise must be float32, matching the upstream sampler")
            return self._integrate(context, noise.detach().clone(), num_steps, 0, capture_step)

    def refine(self, context: PreparedContext, latent: np.ndarray, *, full_num_steps: int = 10,
               start_step: int = 8) -> SampleResult:
        """Continue a stored internal latent on the original full sampling grid."""
        latent_time_contract(full_num_steps, start_step)
        self._validate_context(context)
        expected = (self.config.chunk_size, self.config.max_action_dim)
        if (not isinstance(latent, np.ndarray) or latent.dtype != np.float32
                or latent.shape != expected or not np.isfinite(latent).all()):
            raise ValueError(f"latent must be a finite float32 NumPy array with shape {expected}")
        with self._torch.inference_mode():
            initial = self._torch.from_numpy(np.array(latent, dtype=np.float32, order="C", copy=True))[None]
            return self._integrate(context, initial, full_num_steps, start_step, None)

    def _integrate(self, context: PreparedContext, initial: Any, num_steps: int,
                   start_step: int, capture_step: int | None) -> SampleResult:
        torch = self._torch
        dt, times = euler_grid(num_steps, start_step=start_step)
        started = time.perf_counter()
        head_seconds = export_seconds = 0.0
        x_t, latent, contract = initial, None, None
        for step, native_time in enumerate(times, start=start_step):
            tick = time.perf_counter()
            timestep = torch.tensor(native_time, dtype=torch.float32, device=initial.device).expand(1)
            velocity = self.model.denoise_step(
                x_t=x_t, prefix_pad_masks=context.prefix_pad_masks,
                past_key_values=context.past_key_values, timestep=timestep,
            )
            if tuple(velocity.shape) != tuple(x_t.shape) or velocity.dtype != x_t.dtype:
                raise ValueError("denoising velocity must match the latent shape and dtype")
            x_t = x_t + dt * velocity
            head_seconds += time.perf_counter() - tick
            if step + 1 == capture_step:
                tick = time.perf_counter()
                latent = x_t[0].detach().cpu().numpy().astype(np.float32, copy=True)
                latent.setflags(write=False)
                contract = latent_time_contract(num_steps, capture_step)
                export_seconds += time.perf_counter() - tick
        self._require_tensor(x_t, "generated actions")
        if latent is not None and not np.isfinite(latent).all():
            raise ValueError("captured intermediate latent must be finite")
        self._validate_context(context)
        actions = x_t[:, :, :self.config.action_feature.shape[0]].detach().clone()
        return SampleResult(
            actions=actions, latent=latent, latent_contract=contract, nfe=len(times),
            timings={"action_head_seconds": head_seconds, "latent_export_seconds": export_seconds,
                     "sampling_seconds": time.perf_counter() - started},
        )
