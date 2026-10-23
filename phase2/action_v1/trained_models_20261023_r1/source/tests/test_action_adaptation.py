"""Small real-tensor tests; no pretrained checkpoint or large model is loaded.

The parity fixture executes the pinned vendor's actual policy forward, flow
forward, denoise_step and mask function on a tiny attention implementation.
This checks the training wrapper against upstream semantics without claiming
that the real checkpoint's full transformer parity has been measured here.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from embodied_research.action_adaptation import (
    ActionAdapter,
    LoRALinear,
    compute_full_loss,
    deterministic_actions,
    expected_lora_shapes,
    flow_matching_loss,
    frozen_base_checksum,
    gradient_audit,
    install_action_lora,
    load_action_adapter,
    prepare_training_prefix,
    training_forward_parity,
)


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/MINERVA/src/lerobot/policies"


def vendor_function(path, name, *, class_name=None, namespace=None):
    """Compile exactly one local vendor function, skipping heavyweight imports."""
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = source.body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = copy.deepcopy(next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name))
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              function], type_ignores=[])
    values = {"torch": torch, "F": F, "Tensor": torch.Tensor,
              "ACTION": "action", "OBS_STATE": "observation.state",
              "OBS_LANGUAGE_TOKENS": "observation.language.tokens",
              "OBS_LANGUAGE_ATTENTION_MASK": "observation.language.attention_mask", **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), values)
    return values[name]


MAKE_MASK = vendor_function(VENDOR / "common/vla_utils.py", "make_att_2d_masks")
MODEL_SOURCE = VENDOR / "smolvla/modeling_smolvla.py"


class TinyBackbone(nn.Module):
    """A single attention layer with genuine prefix KV append/crop behavior."""

    def __init__(self):
        super().__init__()
        self.vlm = nn.ModuleDict({"language": nn.Embedding(8, 4), "image": nn.Linear(4, 4),
                                  "key": nn.Linear(4, 4, bias=False), "value": nn.Linear(4, 4, bias=False)})
        self.expert_q = nn.Linear(4, 4, bias=False)
        self.expert_k = nn.Linear(4, 4, bias=False)
        self.expert_v = nn.Linear(4, 4, bias=False)
        self.num_vlm_layers = 1
        self.seen_caches = []

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
        prefix, suffix = inputs_embeds
        if prefix is not None:
            prefix_k = self.vlm["key"](prefix).unsqueeze(1)
            prefix_v = self.vlm["value"](prefix).unsqueeze(1)
        if suffix is None:
            cache = DynamicCache()
            cache.update(prefix_k, prefix_v, 0)
            return [prefix, None], cache
        q = self.expert_q(suffix).unsqueeze(1)
        k = self.expert_k(suffix).unsqueeze(1)
        v = self.expert_v(suffix).unsqueeze(1)
        if past_key_values is None:
            k, v = torch.cat([prefix_k, k], dim=2), torch.cat([prefix_v, v], dim=2)
        else:
            self.seen_caches.append(past_key_values)
            k, v = past_key_values.update(k, v, 0)
        mask = attention_mask[:, -suffix.shape[1]:].unsqueeze(1)
        logits = q @ k.transpose(-1, -2) / 2
        weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim=-1)
        return [prefix, (weights @ v).squeeze(1)], past_key_values


class TinyFlow(nn.Module):
    forward = vendor_function(MODEL_SOURCE, "forward", class_name="VLAFlowMatching",
                              namespace={"make_att_2d_masks": MAKE_MASK})
    denoise_step = vendor_function(MODEL_SOURCE, "denoise_step", class_name="VLAFlowMatching",
                                   namespace={"make_att_2d_masks": MAKE_MASK})

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vlm_with_expert = TinyBackbone()
        self.state_proj = nn.Linear(32, 4)
        self.action_in_proj = nn.Linear(32, 4)
        self.time_proj = nn.Linear(1, 4)
        self.action_out_proj = nn.Linear(4, 32)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, *, state):
        image = self.vlm_with_expert.vlm["image"](images)
        language = self.vlm_with_expert.vlm["language"](lang_tokens)
        embeddings = torch.cat([image, language, self.state_proj(state).unsqueeze(1)], dim=1)
        padding = torch.cat([img_masks, lang_masks, torch.ones((len(state), 1), dtype=torch.bool)], dim=1)
        groups = torch.zeros_like(padding, dtype=torch.int32)
        groups[:, -1] = 1
        return embeddings, padding, groups

    def embed_suffix(self, actions, time):
        embeddings = torch.tanh(self.action_in_proj(actions) + self.time_proj(time[:, None])[:, None, :])
        mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        return embeddings, mask, torch.ones_like(mask, dtype=torch.int32)


class TinyPolicy(nn.Module):
    forward = vendor_function(MODEL_SOURCE, "forward", class_name="SmolVLAPolicy")

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32, action_feature=SimpleNamespace(shape=(7,)),
                                      use_cache=True, adapt_to_pi_aloha=False, n_obs_steps=1)
        self.model = TinyFlow(self.config)

    def prepare_images(self, batch):
        return batch["image"], batch["image_mask"]

    def prepare_state(self, batch):
        return F.pad(batch["observation.state"], (0, 24))

    def prepare_action(self, batch):
        return F.pad(batch["action"], (0, 25))


@pytest.fixture
def tiny():
    with torch.random.fork_rng():
        torch.manual_seed(301)
        policy = TinyPolicy().eval().requires_grad_(False)
        batch = {"image": torch.randn(2, 1, 4), "image_mask": torch.ones(2, 1, dtype=torch.bool),
                 "observation.state": torch.randn(2, 8),
                 "observation.language.tokens": torch.tensor([[1, 2, 0], [2, 3, 0]]),
                 "observation.language.attention_mask": torch.tensor([[True, True, False]] * 2),
                 "action": torch.randn(2, 50, 7), "action_is_pad": torch.zeros(2, 50, dtype=torch.bool)}
        batch["action_is_pad"][0, 35:] = True
        batch["action_is_pad"][1, 42:] = True
        noise, time = torch.randn(2, 50, 32), torch.tensor([.2, .8])
    return policy, batch, noise, time


def adapt_tiny(policy):
    checksum = frozen_base_checksum(policy)
    generator = torch.Generator().manual_seed(11)
    targets = {}
    for name in ("expert_q", "expert_v"):
        backbone = policy.model.vlm_with_expert
        wrapped = LoRALinear(getattr(backbone, name), generator=generator)
        setattr(backbone, name, wrapped)
        targets[f"model.vlm_with_expert.{name}"] = wrapped
    return ActionAdapter(policy, base_fingerprint=checksum, seed=11, _targets=targets)


def test_zero_initialized_lora_exact_parity_and_private_randomness():
    with torch.random.fork_rng():
        torch.manual_seed(41)
        base = nn.Linear(4, 3)
        inputs = torch.randn(5, 4)
        expected = base(inputs).detach()
        random_state = torch.random.get_rng_state().clone()
        layer = LoRALinear(base, generator=torch.Generator().manual_seed(51))
        assert torch.equal(torch.random.get_rng_state(), random_state)
        assert torch.equal(layer(inputs), expected)
        assert layer.lora_B.count_nonzero() == 0
        assert layer.lora_A.count_nonzero() > 0
        assert not layer.weight.requires_grad and not layer.bias.requires_grad


def test_lora_gradient_finite_difference_and_frozen_base():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(3, 2))
        base_hash = frozen_base_checksum(model)
        model[0] = LoRALinear(model[0], generator=torch.Generator().manual_seed(5))
        assert frozen_base_checksum(model) == base_hash
        x, target = torch.randn(6, 3), torch.randn(6, 2)
    optimizer = torch.optim.SGD([model[0].lora_A, model[0].lora_B], lr=.1)
    F.mse_loss(model(x), target).backward()
    assert model[0].lora_A.grad.count_nonzero() == 0
    assert model[0].lora_B.grad.count_nonzero() > 0
    assert gradient_audit(model)["ok"]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(model(x), target).backward()
    assert model[0].lora_A.grad.count_nonzero() > 0
    for parameter, index in ((model[0].lora_A, (2, 1)), (model[0].lora_B, (1, 2))):
        analytic = float(parameter.grad[index])
        epsilon = .002
        with torch.no_grad():
            original = parameter[index].clone()
            parameter[index] = original + epsilon
            plus = F.mse_loss(model(x), target).item()
            parameter[index] = original - epsilon
            minus = F.mse_loss(model(x), target).item()
            parameter[index] = original
        assert analytic == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-5, rel=.01)
    assert frozen_base_checksum(model) == base_hash


def test_exact_production_projection_inventory_and_trainable_count():
    shapes = expected_lora_shapes()
    assert len(shapes) == 64
    assert sum(8 * sum(shape) for shape in shapes.values()) == 552_960
    # Expanded scalar backing avoids allocating the large frozen matrices.
    policy = nn.Module()
    for name, (out_features, in_features) in shapes.items():
        parent = policy
        pieces = name.split(".")
        for piece in pieces[:-1]:
            if not hasattr(parent, piece):
                parent.add_module(piece, nn.Module())
            parent = getattr(parent, piece)
        linear = nn.Linear(in_features, out_features, bias=False, device="meta")
        linear.weight = nn.Parameter(torch.zeros(1).expand(out_features, in_features))
        parent.add_module(pieces[-1], linear)
    policy.register_parameter("frozen_extra", nn.Parameter(torch.ones(3)))
    adapter = install_action_lora(policy, base_fingerprint="a" * 64, seed=19)
    assert len(list(adapter.named_parameters())) == 128
    assert sum(value.numel() for value in adapter.parameters()) == 552_960
    assert sum(value.numel() for value in policy.parameters() if value.requires_grad) == 552_960
    assert not policy.frozen_extra.requires_grad
    assert all(name.endswith((".lora_A", ".lora_B")) for name, _ in adapter.named_parameters())
    with pytest.raises(ValueError, match="shape/type/bias"):
        install_action_lora(policy, base_fingerprint="a" * 64)


def test_malformed_layout_and_fingerprint_rejected_without_mutation(tiny):
    policy, *_ = tiny
    policy.model.action_out_proj.requires_grad_(True)
    original = frozen_base_checksum(policy)
    with pytest.raises(ValueError, match="64"):
        install_action_lora(policy, base_fingerprint="a" * 64)
    assert policy.model.action_out_proj.weight.requires_grad
    assert frozen_base_checksum(policy) == original
    with pytest.raises(ValueError, match="SHA-256"):
        install_action_lora(policy, base_fingerprint="not-a-fingerprint")


def test_upstream_full_and_cached_velocity_loss_and_gradient_parity(tiny):
    policy, batch, noise, time = tiny
    baseline, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    adapter = adapt_tiny(policy)
    zero, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    assert torch.equal(baseline, zero)
    assert training_forward_parity(policy, batch, noise=noise, time=time)["ok"]
    # Nonzero B verifies the cache path with actual adapted attention.
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(.025)
    prefix = prepare_training_prefix(policy, batch)
    full, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    full.backward()
    full_gradients = [p.grad.clone() for p in adapter.parameters()]
    policy.zero_grad(set_to_none=True)
    cached, diagnostics = flow_matching_loss(policy, prefix, batch["action"], noise=noise, time=time,
                                             action_is_pad=batch["action_is_pad"])
    cached.backward()
    assert diagnostics["valid_action_steps"] == 77
    assert torch.allclose(full, cached, atol=1e-6, rtol=1e-5)
    assert all(torch.allclose(p.grad, expected, atol=1e-7, rtol=1e-4)
               for p, expected in zip(adapter.parameters(), full_gradients, strict=True))
    assert gradient_audit(policy)["ok"]


def test_repeated_backward_uses_fresh_cache_without_retained_prefix_graph(tiny):
    policy, batch, noise, time = tiny
    adapter = adapt_tiny(policy)
    prefix = prepare_training_prefix(policy, batch)
    assert not prefix.pad_masks[0, 3]  # Padded language token stays masked.
    snapshot = [(key.clone(), value.clone()) for key, value in prefix.key_values]
    optimizer = torch.optim.SGD(list(adapter.parameters()), lr=.05)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = flow_matching_loss(policy, prefix, batch["action"], noise=noise, time=time,
                                     action_is_pad=batch["action_is_pad"])
        loss.backward()
        assert gradient_audit(policy)["ok"]
        optimizer.step()
    caches = policy.model.vlm_with_expert.seen_caches
    assert len(caches) == 3 and len({id(cache) for cache in caches}) == 3
    for (key, value), (original_key, original_value) in zip(prefix.key_values, snapshot, strict=True):
        assert torch.equal(key, original_key) and torch.equal(value, original_value)
        assert key.grad_fn is None and value.grad_fn is None
        assert not key.requires_grad and not value.requires_grad
        assert not key.is_inference() and not value.is_inference()
        assert all(cache.layers[0].keys.data_ptr() != key.data_ptr() for cache in caches)
    assert frozen_base_checksum(policy) == adapter.base_fingerprint


def test_padding_masks_tail_and_only_seven_output_dimensions_have_loss_gradient(tiny):
    policy, batch, noise, time = tiny
    adapt_tiny(policy)
    prefix = prepare_training_prefix(policy, batch)
    outputs = []

    def capture(module, args, output):
        output.retain_grad()
        outputs.append(output)

    hook = policy.model.action_out_proj.register_forward_hook(capture)
    loss, _ = flow_matching_loss(policy, prefix, batch["action"], noise=noise, time=time,
                                 action_is_pad=batch["action_is_pad"])
    loss.backward()
    hook.remove()
    assert outputs[0].grad[..., 7:].count_nonzero() == 0
    assert outputs[0].grad[batch["action_is_pad"]].count_nonzero() == 0
    assert outputs[0].grad[..., :7].count_nonzero() > 0
    altered_actions, altered_noise = batch["action"].clone(), noise.clone()
    altered_actions[batch["action_is_pad"]] = 100
    altered_noise[batch["action_is_pad"]] = -100
    altered, _ = flow_matching_loss(policy, prefix, altered_actions, noise=altered_noise, time=time,
                                    action_is_pad=batch["action_is_pad"])
    assert torch.equal(loss, altered)


def test_prefix_rejects_inference_mode_mutation_unfrozen_or_different_policy(tiny):
    policy, batch, *_ = tiny
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        prepare_training_prefix(policy, batch)
    prefix = prepare_training_prefix(policy, batch)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        prefix.new_cache()
    with pytest.raises(ValueError, match="another policy"):
        prefix.validate(copy.deepcopy(policy))
    with torch.no_grad():
        policy.model.state_proj.weight.add_(.01)
    with pytest.raises(ValueError, match="changed frozen"):
        prefix.validate(policy)
    prefix = prepare_training_prefix(policy, batch)
    prefix.key_values[0][0].add_(1)
    with pytest.raises(ValueError, match="tensors were modified"):
        prefix.validate(policy)
    policy.model.state_proj.requires_grad_(True)
    with pytest.raises(ValueError, match="remain frozen"):
        prepare_training_prefix(policy, batch)


@pytest.mark.parametrize("bad", ["shape", "all_pad", "time", "dtype", "nan", "inference"])
def test_loss_input_contract(tiny, bad):
    policy, batch, noise, time = tiny
    if bad == "shape":
        batch["action"] = batch["action"][:, :49]
    elif bad == "all_pad":
        batch["action_is_pad"][:] = True
    elif bad == "time":
        time[0] = 1.1
    elif bad == "dtype":
        noise = noise.double()
    elif bad == "nan":
        noise[0, 0, 0] = float("nan")
    else:
        with torch.inference_mode():
            noise = noise.clone()
    with pytest.raises(ValueError):
        compute_full_loss(policy, batch, noise=noise, time=time)


def test_save_reload_reproduces_updated_actions_and_safe_arrays(tiny, tmp_path):
    policy, batch, noise, time = tiny
    adapter = adapt_tiny(policy)
    prefix = prepare_training_prefix(policy, batch)
    before = deterministic_actions(policy, prefix, noise=noise, num_steps=2)
    optimizer = torch.optim.AdamW(list(adapter.parameters()), lr=.01, weight_decay=0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_full_loss(policy, batch, noise=noise, time=time)
        loss.backward()
        optimizer.step()
    random_state, original_noise = torch.random.get_rng_state().clone(), noise.clone()
    after = deterministic_actions(policy, prefix, noise=noise, num_steps=2)
    assert torch.equal(noise, original_noise)
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert not torch.equal(before, after)
    path = adapter.save(tmp_path / "adapter.npz", metadata={"purpose": "small tensor test"})
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind in "fu" for key in archive.files)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
    assert not torch.equal(after, deterministic_actions(policy, prefix, noise=noise, num_steps=2))
    manifest = adapter.reload(path)
    assert manifest["metadata"]["purpose"] == "small tensor test"
    assert torch.equal(after, deterministic_actions(policy, prefix, noise=noise, num_steps=2))
    assert frozen_base_checksum(policy) == adapter.base_fingerprint


@pytest.mark.parametrize("tamper", ["base", "shape", "checksum", "nonfinite", "object"])
def test_reload_rejects_corrupt_or_unbound_archive_before_any_copy(tiny, tmp_path, tamper):
    policy, *_ = tiny
    adapter = adapt_tiny(policy)
    path = adapter.save(tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    manifest = json.loads(arrays["manifest_utf8"].tobytes())
    if tamper == "base":
        manifest["base_fingerprint"] = "b" * 64
    elif tamper == "shape":
        manifest["modules"][0]["A_shape"] = [1, 1]
    elif tamper == "checksum":
        arrays["tensor_001"][0, 0] = 1
    elif tamper == "nonfinite":
        arrays["tensor_001"][0, 0] = np.nan
    else:
        arrays["tensor_001"] = np.array([object()], dtype=object)
    arrays["manifest_utf8"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    corrupted = tmp_path / "invalid.npz"
    np.savez_compressed(corrupted, **arrays)
    before = [value.detach().clone() for value in adapter.parameters()]
    with pytest.raises(ValueError):
        adapter.reload(corrupted)
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(adapter.parameters(), before, strict=True))
    with pytest.raises(ValueError, match="fingerprint"):
        load_action_adapter(policy, path, base_fingerprint="c" * 64)


def test_gradient_audit_rejects_unexpected_trainable_and_base_gradients(tiny):
    policy, batch, noise, time = tiny
    adapt_tiny(policy)
    assert not gradient_audit(policy)["ok"]
    loss, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    loss.backward()
    assert gradient_audit(policy)["ok"]
    policy.model.state_proj.weight.grad = torch.ones_like(policy.model.state_proj.weight)
    audit = gradient_audit(policy)
    assert not audit["ok"] and audit["frozen_gradients_present"]
    policy.model.state_proj.weight.grad = None
    policy.model.action_out_proj.weight.requires_grad_(True)
    assert gradient_audit(policy)["unexpected_trainable_parameters"]
