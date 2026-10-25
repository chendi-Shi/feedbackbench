"""Native flow-matching baseline that trains only SmolVLA's output projection.

All 32 x 480 weights and 32 biases are trainable (15,392 parameters). The
unmodified LIBERO loss supervises only the first seven output rows (3,367
parameters). Keep optimizer weight_decay=0 so unlabelled rows are not changed
solely by regularization. The pretrained head is preserved at installation;
seed is experiment metadata, never an instruction to reinitialize the head.

No model loading, downloading, prefix-cache optimization or vendor edits occur
here. Checkpoints contain only numeric head tensors and a UTF-8 JSON manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
import torch
from torch import nn

from .action_adaptation import compute_full_loss as _native_full_loss
from .action_adaptation import frozen_base_checksum


HEAD_PATH = "model.action_out_proj"
HEAD_NAMES = (HEAD_PATH + ".weight", HEAD_PATH + ".bias")
WEIGHT_SHAPE = (32, 480)
TRAINABLE_PARAMETERS = 15_392
EFFECTIVE_PARAMETERS = 3_367
ACTION_DIM = 7
SCHEMA = "feedbackbench.action_output_head.v1"
_MARKER = "_feedbackbench_output_head_binding"


def _fingerprint(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256 base fingerprint")
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    return value


def _tensor(value, name, shape=None):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.is_inference() or not torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be a finite, ordinary FP32 CPU tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")


def _head(policy):
    head = dict(policy.named_modules()).get(HEAD_PATH)
    if (not isinstance(head, nn.Linear) or tuple(head.weight.shape) != WEIGHT_SHAPE
            or head.bias is None or tuple(head.bias.shape) != (32,)):
        raise ValueError("Expected pinned action_out_proj Linear(480,32,bias=True)")
    _tensor(head.weight, HEAD_NAMES[0], WEIGHT_SHAPE)
    _tensor(head.bias, HEAD_NAMES[1], (32,))
    return head


def _tensor_digest(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        if value.device.type != "cpu":
            raise ValueError("Checksum requires CPU tensors")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(raw).cast("B")
        for offset in range(0, len(view), 1024 * 1024):
            result.update(view[offset:offset + 1024 * 1024])
    return result.hexdigest()


def head_checksum(policy):
    """Exact hash of the current output weight and bias only."""
    head = _head(policy)
    return _tensor_digest([(HEAD_NAMES[0], head.weight), (HEAD_NAMES[1], head.bias)])


def frozen_rest_checksum(policy):
    """Exact parameters-and-buffers hash excluding only the two trainable tensors."""
    _head(policy)
    items = list(policy.named_parameters()) + list(policy.named_buffers())
    names = {name for name, _ in policy.named_parameters()}
    if not set(HEAD_NAMES) <= names:
        raise ValueError("Output-head parameter names differ from the pinned model")
    return _tensor_digest([(name, value) for name, value in items if name not in HEAD_NAMES])


class OutputHeadAdapter:
    """Owns references to the existing projection, never a second base model."""

    def __init__(self, policy, *, base_fingerprint, seed, original_head_sha256, frozen_rest_sha256):
        self.policy = policy
        self.base_fingerprint = _fingerprint(base_fingerprint)
        self.seed = _seed(seed)
        self.original_head_sha256 = _fingerprint(original_head_sha256)
        self.frozen_rest_sha256 = _fingerprint(frozen_rest_sha256)

    def named_parameters(self):
        head = _head(self.policy)
        yield HEAD_NAMES[0], head.weight
        yield HEAD_NAMES[1], head.bias

    def parameters(self):
        return (value for _, value in self.named_parameters())

    def manifest(self):
        return {"schema": SCHEMA, "mode": "output_head", "base_fingerprint": self.base_fingerprint,
                "seed": self.seed, "head_path": HEAD_PATH, "weight_shape": list(WEIGHT_SHAPE),
                "bias_shape": [32], "dtype": "float32", "device": "cpu",
                "trainable_parameters": TRAINABLE_PARAMETERS, "effective_supervised_parameters": EFFECTIVE_PARAMETERS,
                "supervised_output_rows": list(range(ACTION_DIM)), "initialization": "original_pretrained_head",
                "original_head_sha256": self.original_head_sha256, "frozen_rest_sha256": self.frozen_rest_sha256,
                "training_route": "unmodified_policy_forward_flow_matching", "recommended_weight_decay": 0.0}

    def verify_frozen_rest(self):
        actual = frozen_rest_checksum(self.policy)
        if actual != self.frozen_rest_sha256:
            raise ValueError("Frozen base outside action_out_proj changed")
        return actual

    def save(self, path, metadata: dict[str, Any] | None = None):
        """Write a new immutable, pickle-free checkpoint; retain any failed file."""
        self.verify_frozen_rest()
        arrays = {name.rsplit(".", 1)[1]: value.detach().numpy().copy()
                  for name, value in self.named_parameters()}
        manifest = {**self.manifest(), "metadata": {} if metadata is None else metadata,
                    "saved_head_sha256": head_checksum(self.policy),
                    "tensor_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                      for name, value in arrays.items()}}
        encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        arrays["manifest_utf8"] = np.frombuffer(encoded, dtype=np.uint8)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return destination

    def reload(self, path):
        """Validate provenance, both arrays and frozen remainder before any copy."""
        manifest, arrays = _read_checkpoint(path)
        if any(manifest.get(key) != value for key, value in self.manifest().items()):
            raise ValueError("Output-head checkpoint base, original head, seed or configuration differs")
        self.verify_frozen_rest()
        restored = []
        for name, parameter in self.named_parameters():
            short = name.rsplit(".", 1)[1]
            array = arrays[short]
            if (array.dtype != np.float32 or array.shape != tuple(parameter.shape)
                    or not np.isfinite(array).all()
                    or hashlib.sha256(array.tobytes()).hexdigest() != manifest["tensor_sha256"][short]):
                raise ValueError(f"Invalid output-head tensor shape, dtype, finiteness or checksum: {short}")
            restored.append((name, parameter, torch.from_numpy(array)))
        expected_head_hash = _tensor_digest([(name, value) for name, _, value in restored])
        if expected_head_hash != manifest.get("saved_head_sha256"):
            raise ValueError("Saved output-head fingerprint differs")
        with torch.no_grad():
            for _, parameter, value in restored:
                parameter.copy_(value)
        return manifest


def _read_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 2 * 1024 * 1024:
            raise ValueError("Output-head checkpoint exceeds its 2 MiB expansion cap")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"weight", "bias", "manifest_utf8"} or len(archive.files) != 3:
            raise ValueError("Unexpected output-head checkpoint array inventory")
        encoded = archive["manifest_utf8"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1 or len(encoded) > 1_000_000:
            raise ValueError("Invalid output-head manifest encoding")
        manifest = json.loads(encoded.tobytes().decode("utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
                or set(manifest.get("tensor_sha256", {})) != {"weight", "bias"}):
            raise ValueError("Invalid output-head manifest schema or tensor inventory")
        arrays = {name: archive[name].copy() for name in ("weight", "bias")}
    return manifest, arrays


def install_output_head(policy, *, base_fingerprint, seed=0):
    """Freeze every tensor, then unfreeze exactly the original 15,392-parameter head."""
    _fingerprint(base_fingerprint)
    _seed(seed)
    head = _head(policy)
    if hasattr(policy, _MARKER) or any(name.endswith((".lora_A", ".lora_B")) for name, _ in policy.named_parameters()):
        raise ValueError("Output-head baseline requires a fresh policy without another adapter")
    if frozen_base_checksum(policy) != base_fingerprint:
        raise ValueError("Policy does not match the supplied original base fingerprint")
    original_head = head_checksum(policy)
    frozen_rest = frozen_rest_checksum(policy)
    policy.requires_grad_(False)
    for parameter in policy.parameters():
        parameter.grad = None
    head.requires_grad_(True)
    if sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad) != TRAINABLE_PARAMETERS:
        raise ValueError("Output-head trainable parameter count differs")
    policy.eval()
    setattr(policy, _MARKER, {"base_fingerprint": base_fingerprint, "original_head_sha256": original_head,
                              "frozen_rest_sha256": frozen_rest})
    return OutputHeadAdapter(policy, base_fingerprint=base_fingerprint, seed=seed,
                             original_head_sha256=original_head, frozen_rest_sha256=frozen_rest)


def load_output_head(policy, path, *, base_fingerprint):
    """Restore on a fresh original base, validating original-head provenance as well."""
    manifest, _ = _read_checkpoint(path)
    if manifest.get("base_fingerprint") != _fingerprint(base_fingerprint):
        raise ValueError("Output-head checkpoint targets another base fingerprint")
    adapter = install_output_head(policy, base_fingerprint=base_fingerprint, seed=manifest.get("seed"))
    adapter.reload(path)
    return adapter


def compute_full_loss(policy, processed_batch, *, noise, time):
    """Same authoritative flow-matching route as LoRA; no loss approximation."""
    return _native_full_loss(policy, processed_batch, noise=noise, time=time)


def gradient_audit(policy):
    """Detect unfreezing outside the head and loss leaking into padded action rows."""
    head = _head(policy)
    trainable, missing, nonfinite, frozen, unexpected = [], [], [], [], []
    squared_norm = 0.0
    frozen_count = 0
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
            if name not in HEAD_NAMES:
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
    count = sum(parameter.numel() for parameter in trainable)
    padded_zero = all(parameter.grad is not None and torch.count_nonzero(parameter.grad[ACTION_DIM:]).item() == 0
                      for parameter in (head.weight, head.bias))
    ok = (count == TRAINABLE_PARAMETERS and len(trainable) == 2 and squared_norm > 0 and padded_zero
          and not (missing or nonfinite or frozen or unexpected))
    return {"ok": ok, "trainable_parameter_count": count, "trainable_tensor_count": len(trainable),
            "effective_supervised_parameter_count": EFFECTIVE_PARAMETERS, "frozen_parameter_count": frozen_count,
            "trainable_gradients_present": bool(trainable) and not missing,
            "all_trainable_gradients_finite": not nonfinite and not missing,
            "any_trainable_gradient_nonzero": squared_norm > 0, "trainable_gradient_l2": math.sqrt(squared_norm),
            "missing_trainable_gradients": missing, "nonfinite_gradients": nonfinite,
            "frozen_gradients_present": frozen, "unexpected_trainable_parameters": unexpected,
            "padded_output_rows_gradient_zero": padded_zero}
