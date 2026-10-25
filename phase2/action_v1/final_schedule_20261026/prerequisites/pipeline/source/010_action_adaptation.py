"""CPU action-expert LoRA and explicitly detached training-prefix caches.

This module never loads a policy, downloads data, or changes a vendor source.
``compute_full_loss`` uses the vendor training forward as the reference path.
The cached path is an optional optimization, to be checked against that path on
the actual checkpoint before use. Actions supplied here are already normalized
by the checkpoint's original processor and come from real demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RANK = 8
ALPHA = 8
TARGET_COUNT = 64
TRAINABLE_PARAMETERS = 552_960
CHUNK_SIZE = 50
ACTION_DIM = 7
INTERNAL_ACTION_DIM = 32
SCHEMA = "feedbackbench.action_expert_lora.v1"
_TARGET_RE = re.compile(r"model\.vlm_with_expert\.lm_expert\.layers\.\d+\.self_attn\.[qv]_proj$")


def expected_lora_shapes() -> dict[str, tuple[int, int]]:
    """Exact (out, in) shapes in the pinned 32-layer SmolVLA expert."""
    shapes = {}
    for layer in range(32):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn"
        shapes[f"{prefix}.q_proj"] = (960, 480)
        shapes[f"{prefix}.v_proj"] = (320, 480 if layer % 2 == 0 else 320)
    return shapes


def _fingerprint(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("base_fingerprint must be a lowercase SHA-256 fingerprint")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def _tensor(value, name, shape=None, *, dtype=torch.float32):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(f"{name} must be a CPU {dtype} tensor")
    if value.is_inference():
        raise ValueError(f"{name} is an inference-mode tensor; use a training no_grad prefix")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape must be {tuple(shape)}, got {tuple(value.shape)}")
    if dtype.is_floating_point and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains nonfinite values")
    return value


class LoRALinear(nn.Module):
    """Frozen FP32 linear plus alpha/rank * B(A(x)); dropout is exactly zero."""

    def __init__(self, base: nn.Linear, *, generator: torch.Generator):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("LoRA requires an nn.Linear base")
        _tensor(base.weight, "base weight")
        if base.bias is not None:
            _tensor(base.bias, "base bias")
        self.base_linear = base
        self.base_linear.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha, self.dropout = RANK, ALPHA, 0.0
        self.scaling = ALPHA / RANK
        self.lora_A = nn.Parameter(torch.empty(RANK, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, RANK, dtype=torch.float32))
        with torch.no_grad():
            self.lora_A.uniform_(-1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features),
                                 generator=generator)

    @property
    def weight(self):
        # The vendor attention implementation consults q/v_proj.weight.dtype.
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def forward(self, value):
        return self.base_linear(value) + F.linear(F.linear(value, self.lora_A), self.lora_B) * self.scaling


class ActionAdapter:
    """Registry returned by install_action_lora; owns no second policy copy."""

    def __init__(self, policy, *, base_fingerprint, seed, _targets):
        self.policy = policy
        self.base_fingerprint = _fingerprint(base_fingerprint)
        self.seed = _seed(seed)
        self._targets = dict(sorted(_targets.items()))

    def named_parameters(self):
        for name, module in self._targets.items():
            yield f"{name}.lora_A", module.lora_A
            yield f"{name}.lora_B", module.lora_B

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())

    def manifest(self):
        return {"schema": SCHEMA, "base_fingerprint": self.base_fingerprint, "seed": self.seed,
                "rank": RANK, "alpha": ALPHA, "dropout": 0.0, "dtype": "float32", "device": "cpu",
                "modules": [{"name": name, "weight_shape": list(module.weight.shape),
                             "A_shape": list(module.lora_A.shape), "B_shape": list(module.lora_B.shape)}
                            for name, module in self._targets.items()],
                "trainable_parameters": sum(parameter.numel() for parameter in self.parameters())}

    def save(self, path, metadata: dict[str, Any] | None = None):
        """Save only numeric A/B arrays and a UTF-8 manifest in a pickle-free NPZ."""
        manifest = self.manifest()
        manifest["metadata"] = {} if metadata is None else metadata
        arrays = {}
        for index, (name, parameter) in enumerate(self.named_parameters()):
            _tensor(parameter, name)
            arrays[f"tensor_{index:03d}"] = parameter.detach().numpy().copy()
        manifest["tensor_sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest()
                                     for name, value in arrays.items()}
        arrays["manifest_utf8"] = np.frombuffer(json.dumps(manifest, sort_keys=True, allow_nan=False,
                                                          ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                                dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def reload(self, path):
        """Validate every binding and array before restoring into this one model."""
        manifest, arrays = _read_adapter(path)
        expected = self.manifest()
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Adapter base fingerprint, module shapes, seed or configuration differs")
        names = {f"tensor_{index:03d}" for index, _ in enumerate(self.named_parameters())}
        if set(arrays) != names or set(manifest.get("tensor_sha256", {})) != names:
            raise ValueError("Adapter tensor inventory differs")
        restored = []
        for index, (_, parameter) in enumerate(self.named_parameters()):
            name = f"tensor_{index:03d}"
            array = arrays[name]
            if (array.dtype != np.float32 or array.shape != tuple(parameter.shape)
                    or not np.isfinite(array).all()
                    or hashlib.sha256(array.tobytes()).hexdigest() != manifest["tensor_sha256"][name]):
                raise ValueError(f"Adapter tensor shape, dtype, finiteness or checksum differs: {name}")
            restored.append((parameter, torch.from_numpy(array)))
        with torch.no_grad():
            for parameter, value in restored:
                parameter.copy_(value)
        return manifest


def _read_adapter(path):
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["manifest_utf8"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1 or len(encoded) > 1_000_000:
            raise ValueError("Invalid adapter manifest encoding")
        manifest = json.loads(encoded.tobytes().decode("utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown adapter schema")
        arrays = {key: archive[key].copy() for key in archive.files if key != "manifest_utf8"}
    return manifest, arrays


def install_action_lora(policy, *, base_fingerprint: str, seed: int = 0) -> ActionAdapter:
    """Freeze the whole policy, then install exactly 64 rank-8 q/v adapters."""
    _fingerprint(base_fingerprint)
    _seed(seed)
    expected = expected_lora_shapes()
    targets = {name: module for name, module in policy.named_modules() if _TARGET_RE.fullmatch(name)}
    if set(targets) != set(expected) or len(targets) != TARGET_COUNT:
        raise ValueError("Expected exactly the pinned 64 action-expert q/v matrices")
    for name, module in targets.items():
        if (not isinstance(module, nn.Linear) or tuple(module.weight.shape) != expected[name]
                or module.bias is not None):
            raise ValueError(f"Unexpected action-expert projection shape/type/bias: {name}")
    for name, parameter in policy.named_parameters():
        _tensor(parameter, name)
    if sum(RANK * sum(shape) for shape in expected.values()) != TRAINABLE_PARAMETERS:
        raise ValueError("The pinned adapter parameter count changed")
    # Validate before mutation; preserve global RNG state with a private generator.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    policy.requires_grad_(False)
    policy.zero_grad(set_to_none=True)
    policy.eval()
    installed = {}
    for name, base in sorted(targets.items()):
        parent_name, attribute = name.rsplit(".", 1)
        module = LoRALinear(base, generator=generator)
        setattr(policy.get_submodule(parent_name), attribute, module)
        installed[name] = module
    adapter = ActionAdapter(policy, base_fingerprint=base_fingerprint, seed=seed, _targets=installed)
    if sum(p.numel() for p in policy.parameters() if p.requires_grad) != TRAINABLE_PARAMETERS:
        raise RuntimeError("Unexpected trainable parameters after adapter installation")
    return adapter


def load_action_adapter(policy, path, *, base_fingerprint: str) -> ActionAdapter:
    """Install and load an adapter on a fresh, already loaded base policy."""
    manifest, _ = _read_adapter(path)
    if manifest.get("base_fingerprint") != _fingerprint(base_fingerprint):
        raise ValueError("Adapter base fingerprint differs")
    adapter = install_action_lora(policy, base_fingerprint=base_fingerprint, seed=manifest.get("seed"))
    adapter.reload(path)
    return adapter


def _lora_name(name):
    return name.endswith((".lora_A", ".lora_B"))


def frozen_base_checksum(policy) -> str:
    """Hash base parameters/buffers without copying the entire model at once.

    Wrapper names are canonicalized so a pre-install checksum can be compared
    with a post-update checksum. LoRA A/B tensors alone are excluded.
    """
    digest = hashlib.sha256()
    tensors = list(policy.named_parameters()) + list(policy.named_buffers())
    tensors = [(name.replace(".base_linear.", "."), value) for name, value in tensors if not _lora_name(name)]
    for name, value in sorted(tensors):
        if value.device.type != "cpu":
            raise ValueError("Base checksum requires CPU tensors")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        # uint8 view handles both FP32 and nonfloating buffers, without pickle.
        byte_view = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(byte_view).cast("B")
        for start in range(0, len(view), 1024 * 1024):
            digest.update(view[start:start + 1024 * 1024])
    return digest.hexdigest()


def gradient_audit(policy) -> dict[str, Any]:
    trainable, frozen, missing, nonfinite, unexpected = [], [], [], [], []
    squared_norm = 0.0
    frozen_count = 0
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
            if not _lora_name(name):
                unexpected.append(name)
            if parameter.grad is None:
                missing.append(name)
            elif not torch.isfinite(parameter.grad).all().item():
                nonfinite.append(name)
            else:
                squared_norm += float(parameter.grad.detach().double().square().sum())
        else:
            frozen_count += parameter.numel()
            if parameter.grad is not None:
                frozen.append(name)
    return {"trainable_parameter_count": sum(p.numel() for p in trainable),
            "trainable_tensor_count": len(trainable), "frozen_parameter_count": frozen_count,
            "trainable_gradients_present": bool(trainable) and not missing,
            "all_trainable_gradients_finite": not nonfinite and not missing,
            "any_trainable_gradient_nonzero": squared_norm > 0,
            "trainable_gradient_l2": math.sqrt(squared_norm), "missing_trainable_gradients": missing,
            "frozen_gradients_present": frozen, "nonfinite_gradients": nonfinite,
            "unexpected_trainable_parameters": unexpected,
            "ok": bool(trainable) and not (missing or frozen or nonfinite or unexpected) and squared_norm > 0}


def _contract(policy):
    config = policy.config
    if (config.chunk_size != CHUNK_SIZE or config.max_action_dim != INTERNAL_ACTION_DIM
            or config.action_feature.shape[0] != ACTION_DIM or not config.use_cache
            or config.adapt_to_pi_aloha or config.n_obs_steps != 1):
        raise ValueError("Expected the pinned SmolVLA chunk50/internal32/action7 single-observation contract")


def _prefix_guard(policy):
    modules = (("vlm", policy.model.vlm_with_expert.vlm), ("state_proj", policy.model.state_proj))
    guard = []
    for label, module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise ValueError("VLM and state_proj must remain frozen for prefix reuse")
            guard.append((f"{label}.{name}", id(parameter), parameter._version))
    return tuple(guard)


def _attention_mask(padding, groups):
    cumulative = torch.cumsum(groups, dim=1)
    return (cumulative[:, None, :] <= cumulative[:, :, None]) & (padding[:, None, :] & padding[:, :, None])


@dataclass(frozen=True)
class TrainingPrefix:
    """Owned detached K/V, never a live cache containing a previous loss graph."""

    pad_masks: torch.Tensor
    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    owner_id: int
    parameter_guard: tuple
    tensor_versions: tuple

    def new_cache(self):
        from transformers.cache_utils import DynamicCache
        if torch.is_inference_mode_enabled():
            raise ValueError("Training caches cannot be created under inference_mode")
        cache = DynamicCache()
        # A fresh object and owned clones per forward prevent both aliasing and
        # grad_fn retention after the vendor appends/crops suffix K/V.
        with torch.no_grad():
            for index, (key, value) in enumerate(self.key_values):
                cache.update(key.clone(), value.clone(), index)
        return cache

    def validate(self, policy):
        if self.owner_id != id(policy) or self.parameter_guard != _prefix_guard(policy):
            raise ValueError("Prefix belongs to another policy or changed frozen prefix parameters")
        tensors = [self.pad_masks, *(value for pair in self.key_values for value in pair)]
        if tuple(value._version for value in tensors) != self.tensor_versions:
            raise ValueError("Training prefix tensors were modified")
        for value in tensors:
            if value.requires_grad or value.grad_fn is not None or value.is_inference():
                raise ValueError("Training prefix retains autograd or inference-mode tensors")


def prepare_training_prefix(policy, processed_batch) -> TrainingPrefix:
    """Run frozen vision/language/state encoding once with no_grad, not inference_mode."""
    if torch.is_inference_mode_enabled():
        raise ValueError("prepare_training_prefix must run outside inference_mode")
    _contract(policy)
    guard = _prefix_guard(policy)
    policy.eval()
    with torch.no_grad():
        batch = dict(processed_batch)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embeddings, padding, attention = policy.model.embed_prefix(
            images, image_masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], state=state)
        padding = padding.bool()
        _, cache = policy.model.vlm_with_expert.forward(
            attention_mask=_attention_mask(padding, attention), position_ids=torch.cumsum(padding, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embeddings, None], use_cache=True)
        if cache is None or len(cache.layers) != policy.model.vlm_with_expert.num_vlm_layers:
            raise ValueError("Prefix cache layer count differs from the model")
        pairs = []
        for index, layer in enumerate(cache.layers):
            key, value = layer.keys, layer.values
            _tensor(key, f"prefix key {index}")
            _tensor(value, f"prefix value {index}", key.shape)
            if key.ndim != 4 or key.shape[0] != padding.shape[0] or key.shape[2] != padding.shape[1]:
                raise ValueError("Prefix cache tensor shape differs from its mask")
            pairs.append((key.detach().clone(), value.detach().clone()))
        padding = padding.detach().clone()
    tensors = [padding, *(value for pair in pairs for value in pair)]
    prefix = TrainingPrefix(padding, tuple(pairs), id(policy), guard, tuple(value._version for value in tensors))
    prefix.validate(policy)
    return prefix


def _flow_inputs(actions, noise, time, action_is_pad):
    _tensor(actions, "normalized actions")
    if actions.ndim != 3 or tuple(actions.shape[1:]) != (CHUNK_SIZE, ACTION_DIM) or not len(actions):
        raise ValueError("Normalized expert actions must have shape [B,50,7]")
    batch_size = len(actions)
    _tensor(noise, "noise", (batch_size, CHUNK_SIZE, INTERNAL_ACTION_DIM))
    _tensor(time, "time", (batch_size,))
    if torch.any((time < 0) | (time > 1)).item():
        raise ValueError("Flow-matching times must lie in [0,1]")
    if action_is_pad is None:
        action_is_pad = torch.zeros((batch_size, CHUNK_SIZE), dtype=torch.bool)
    _tensor(action_is_pad, "action_is_pad", (batch_size, CHUNK_SIZE), dtype=torch.bool)
    if action_is_pad.all().item():
        raise ValueError("A training batch must contain at least one valid action")
    return F.pad(actions, (0, INTERNAL_ACTION_DIM - ACTION_DIM)), ~action_is_pad


def compute_full_loss(policy, processed_batch, *, noise, time):
    """Authoritative unmodified policy.forward; no prefix-cache prerequisite."""
    _contract(policy)
    _, valid = _flow_inputs(processed_batch["action"], noise, time, processed_batch.get("action_is_pad"))
    loss, diagnostics = policy.forward(dict(processed_batch), noise=noise, time=time)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("Vendor training forward returned an invalid scalar loss")
    return loss, {**diagnostics, "route": "vendor_policy_forward", "valid_action_steps": int(valid.sum()),
                  "effective_action_dim": ACTION_DIM}


def _cached_velocity(policy, prefix, x_t, time):
    prefix.validate(policy)
    temporary_cache = prefix.new_cache()
    return policy.model.denoise_step(prefix.pad_masks, temporary_cache, x_t, time)


def flow_matching_loss(policy, prefix, normalized_actions, *, noise, time, action_is_pad=None):
    """One sampled (noise,time) expert evaluation, masking padded time/action dimensions."""
    _contract(policy)
    actions, valid = _flow_inputs(normalized_actions, noise, time, action_is_pad)
    if prefix.pad_masks.shape[0] != len(actions):
        raise ValueError("Prefix and action batch sizes differ")
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    target = noise - actions
    velocity = _cached_velocity(policy, prefix, x_t, time)
    _tensor(velocity, "expert velocity", actions.shape)
    element_loss = (velocity[..., :ACTION_DIM] - target[..., :ACTION_DIM]).square()
    loss = (element_loss * valid.unsqueeze(-1)).sum() / (valid.sum() * ACTION_DIM)
    return loss, {"route": "detached_prefix_cached_suffix", "valid_action_steps": int(valid.sum()),
                  "effective_action_dim": ACTION_DIM, "loss": float(loss.detach())}


def training_forward_parity(policy, processed_batch, *, noise, time, atol=1e-5, rtol=1e-4):
    """Compare both scalar loss and every predicted velocity with vendor training.

    This is diagnostic only: callers decide whether a cached path is accepted.
    It does not call backward or update an adapter.
    """
    captured = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda module, args, output: captured.append(output.detach().clone()))
    try:
        with torch.no_grad():
            full_loss, _ = compute_full_loss(policy, processed_batch, noise=noise, time=time)
            prefix = prepare_training_prefix(policy, processed_batch)
            cached_loss, _ = flow_matching_loss(policy, prefix, processed_batch["action"], noise=noise,
                                                time=time, action_is_pad=processed_batch.get("action_is_pad"))
    finally:
        hook.remove()
    if len(captured) != 2 or captured[0].shape != captured[1].shape:
        raise ValueError("Expected exactly one output projection in each parity forward")
    velocity_ok = torch.allclose(captured[0], captured[1], atol=atol, rtol=rtol)
    loss_ok = torch.allclose(full_loss, cached_loss, atol=atol, rtol=rtol)
    return {"ok": bool(velocity_ok and loss_ok), "full_loss": float(full_loss), "cached_loss": float(cached_loss),
            "loss_absolute_error": float((full_loss - cached_loss).abs()),
            "velocity_max_absolute_error": float((captured[0] - captured[1]).abs().max()),
            "velocity_allclose": bool(velocity_ok), "atol": atol, "rtol": rtol,
            "prefix_layers": len(prefix.key_values)}


def deterministic_actions(policy, prefix, *, noise, num_steps=10):
    """Normalized [B,50,7] actions from a supplied noise tensor; never changes RNG."""
    _contract(policy)
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    _tensor(noise, "noise", (prefix.pad_masks.shape[0], CHUNK_SIZE, INTERNAL_ACTION_DIM))
    with torch.no_grad():
        latent = noise.clone()
        for step in range(num_steps):
            time = torch.full((len(latent),), 1 - step / num_steps, dtype=torch.float32)
            latent = latent - _cached_velocity(policy, prefix, latent, time) / num_steps
        return latent[..., :ACTION_DIM].detach().clone()
