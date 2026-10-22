"""Small real-tensor output-head training; no pretrained model is loaded."""
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

from embodied_research.action_adaptation import frozen_base_checksum
from embodied_research.action_head_baseline import (
    EFFECTIVE_PARAMETERS, HEAD_NAMES, TRAINABLE_PARAMETERS, compute_full_loss,
    frozen_rest_checksum, gradient_audit, head_checksum, install_output_head, load_output_head,
)


def original_policy_forward():
    """Execute the unmodified pinned policy loss/masking on a small flow fixture."""
    source = Path(__file__).resolve().parents[1] / "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    policy = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SmolVLAPolicy")
    function = copy.deepcopy(next(node for node in policy.body if isinstance(node, ast.FunctionDef)
                                  and node.name == "forward"))
    function.decorator_list = []
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            function], type_ignores=[])
    namespace = {"torch": torch, "OBS_STATE": "observation.state", "ACTION": "action",
                 "OBS_LANGUAGE_TOKENS": "observation.language.tokens",
                 "OBS_LANGUAGE_ATTENTION_MASK": "observation.language.attention_mask"}
    exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), namespace)
    return namespace["forward"]


class SmallFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(32, 480)
        self.action_out_proj = nn.Linear(480, 32)

    def forward(self, images, masks, tokens, language_mask, state, actions, noise, time):
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions
        velocity = self.action_out_proj(self.features(x_t))
        return F.mse_loss(target, velocity, reduction="none")


class SmallPolicy(nn.Module):
    forward = original_policy_forward()

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(12)
            self.model = SmallFlow()
        self.register_buffer("frozen_buffer", torch.tensor([3.0]))
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32, action_feature=SimpleNamespace(shape=(7,)),
                                      use_cache=True, adapt_to_pi_aloha=False, n_obs_steps=1)

    def prepare_images(self, batch):
        return [], []

    def prepare_state(self, batch):
        return batch["observation.state"]

    def prepare_action(self, batch):
        return F.pad(batch["action"], (0, 25))


def training_case():
    policy = SmallPolicy()
    original = frozen_base_checksum(policy)
    adapter = install_output_head(policy, base_fingerprint=original, seed=0)
    generator = torch.Generator().manual_seed(7)
    batch = {"action": torch.randn(1, 50, 7, generator=generator),
             "action_is_pad": torch.arange(50)[None, :] >= 37,
             "observation.state": torch.zeros(1, 8), "observation.language.tokens": torch.zeros(1, 2, dtype=torch.long),
             "observation.language.attention_mask": torch.ones(1, 2, dtype=torch.bool)}
    noise = torch.randn(1, 50, 32, generator=generator)
    time = torch.tensor([.5])
    return policy, adapter, batch, noise, time


def test_install_preserves_pretrained_head_and_exposes_exact_parameter_counts():
    policy = SmallPolicy()
    original_base, original_head, original_rest = frozen_base_checksum(policy), head_checksum(policy), frozen_rest_checksum(policy)
    adapter = install_output_head(policy, base_fingerprint=original_base, seed=17)
    assert frozen_base_checksum(policy) == original_base
    assert head_checksum(policy) == original_head and frozen_rest_checksum(policy) == original_rest
    assert {name for name, parameter in policy.named_parameters() if parameter.requires_grad} == set(HEAD_NAMES)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == TRAINABLE_PARAMETERS == 15392
    assert adapter.manifest()["effective_supervised_parameters"] == EFFECTIVE_PARAMETERS == 3367
    assert adapter.manifest()["original_head_sha256"] == original_head
    assert adapter.manifest()["frozen_rest_sha256"] == original_rest
    with pytest.raises(ValueError, match="fresh policy"):
        install_output_head(policy, base_fingerprint=original_base)


def test_actual_native_loss_backprop_updates_only_supervised_head_rows():
    policy, adapter, batch, noise, time = training_case()
    initial_head = head_checksum(policy)
    initial_rest = frozen_rest_checksum(policy)
    original_weight = policy.model.action_out_proj.weight.detach().clone()
    original_bias = policy.model.action_out_proj.bias.detach().clone()
    loss, diagnostics = compute_full_loss(policy, batch, noise=noise, time=time)
    direct, _ = policy.forward(batch, noise=noise, time=time)
    assert torch.equal(loss, direct) and diagnostics["valid_action_steps"] == 37
    loss.backward()
    audit = gradient_audit(policy)
    assert audit["ok"] and audit["padded_output_rows_gradient_zero"]
    assert audit["trainable_parameter_count"] == 15392 and audit["effective_supervised_parameter_count"] == 3367
    torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0).step()
    assert head_checksum(policy) != initial_head
    assert frozen_rest_checksum(policy) == initial_rest
    assert torch.equal(policy.model.action_out_proj.weight[7:], original_weight[7:])
    assert torch.equal(policy.model.action_out_proj.bias[7:], original_bias[7:])


def test_pickle_free_save_reload_and_fresh_base_restore(tmp_path):
    policy, adapter, batch, noise, time = training_case()
    loss, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    loss.backward()
    torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0).step()
    expected_head = head_checksum(policy)
    checkpoint = adapter.save(tmp_path / "head.npz", metadata={"development_only": True})
    with np.load(checkpoint, allow_pickle=False) as content:
        assert set(content.files) == {"weight", "bias", "manifest_utf8"}
        assert content["weight"].dtype == np.float32 and content["manifest_utf8"].dtype == np.uint8
    with torch.no_grad():
        policy.model.action_out_proj.weight.zero_()
        policy.model.action_out_proj.bias.zero_()
    assert head_checksum(policy) != expected_head
    adapter.reload(checkpoint)
    assert head_checksum(policy) == expected_head
    fresh = SmallPolicy()
    loaded = load_output_head(fresh, checkpoint, base_fingerprint=frozen_base_checksum(fresh))
    assert head_checksum(fresh) == expected_head
    assert loaded.manifest() == adapter.manifest()
    with pytest.raises(FileExistsError):
        adapter.save(checkpoint)


@pytest.mark.parametrize("kind", ["base", "original_head", "rest", "shape", "dtype", "nan", "contents", "extra"])
def test_checkpoint_corruption_rejected_without_partial_tensor_copy(tmp_path, kind):
    policy, adapter, _, _, _ = training_case()
    checkpoint = adapter.save(tmp_path / "good.npz")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    manifest = json.loads(arrays["manifest_utf8"].tobytes().decode())
    if kind in ("base", "original_head", "rest"):
        field = {"base": "base_fingerprint", "original_head": "original_head_sha256", "rest": "frozen_rest_sha256"}[kind]
        manifest[field] = "0" * 64
    elif kind == "shape":
        arrays["bias"] = arrays["bias"][:31]
    elif kind == "dtype":
        arrays["bias"] = arrays["bias"].astype(np.float64)
    elif kind == "nan":
        arrays["bias"][0] = np.nan
    elif kind == "contents":
        arrays["bias"][0] += 1
    else:
        arrays["unexpected"] = np.zeros(1)
    arrays["manifest_utf8"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    corrupted = tmp_path / "corrupted.npz"
    np.savez(corrupted, **arrays)
    before = head_checksum(policy)
    with pytest.raises(ValueError):
        adapter.reload(corrupted)
    assert head_checksum(policy) == before


def test_changed_frozen_remainder_is_rejected_before_save_or_restore(tmp_path):
    policy, adapter, _, _, _ = training_case()
    checkpoint = adapter.save(tmp_path / "good.npz")
    with torch.no_grad():
        policy.frozen_buffer.add_(1)
    with pytest.raises(ValueError, match="Frozen base"):
        adapter.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="Frozen base"):
        adapter.reload(checkpoint)
    assert not (tmp_path / "bad.npz").exists()


def test_rejects_wrong_base_or_projection_contract_without_unfreezing():
    policy = SmallPolicy().requires_grad_(False)
    with pytest.raises(ValueError, match="supplied original base"):
        install_output_head(policy, base_fingerprint="0" * 64)
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    policy.model.action_out_proj = nn.Linear(480, 7)
    with pytest.raises(ValueError, match="Linear"):
        install_output_head(policy, base_fingerprint=frozen_base_checksum(policy))


def test_gradient_audit_rejects_outside_unfreezing_and_padded_row_loss():
    policy, _, batch, noise, time = training_case()
    loss, _ = compute_full_loss(policy, batch, noise=noise, time=time)
    loss.backward()
    policy.model.features.weight.requires_grad_(True)
    audit = gradient_audit(policy)
    assert not audit["ok"] and "model.features.weight" in audit["unexpected_trainable_parameters"]
    policy.model.features.weight.requires_grad_(False)
    policy.model.action_out_proj.bias.grad[7] = 1
    assert not gradient_audit(policy)["padded_output_rows_gradient_zero"]
    assert not gradient_audit(policy)["ok"]
