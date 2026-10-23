"""Worker orchestration checks with synthetic tensors; no pretrained model loads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "action_probe_under_test", Path(__file__).resolve().parents[1] / "scripts/probe_action_adaptation.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def healthy_sample():
    return {"at": "2026-10-22T00:00:00+00:00", "elapsed_seconds": 1,
            "rss_bytes": 1024, "available_bytes": 8 * probe.GIB, "disk_free_bytes": 10 * probe.GIB}


def example_files(tmp_path):
    path = tmp_path / "example.npz"
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    np.savez(path, image=pixels, image2=pixels.copy(), state=np.zeros(8, dtype=np.float32),
             actions=np.ones((50, 7), dtype=np.float32), action_is_pad=np.zeros(50, dtype=bool),
             task=np.asarray("pick up the bowl"))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"example_npz_sha256": probe.digest(path),
                                    "camera_orientation": "policy_ready_rot180_once"}))
    return path, metadata


@pytest.mark.parametrize("field,value,initial,reason", [
    ("available_bytes", probe.START_AVAILABLE - 1, True, "start_memory_threshold"),
    ("available_bytes", probe.RUNTIME_AVAILABLE - 1, False, "runtime_memory_reserve"),
    ("disk_free_bytes", probe.MIN_DISK - 1, False, "disk_reserve"),
    ("elapsed_seconds", probe.MAX_SECONDS, False, "probe_20_minute_limit"),
    ("at", probe.DEADLINE.isoformat(), False, "hard_deadline"),
])
def test_immutable_resource_boundaries(field, value, initial, reason):
    sample = healthy_sample()
    sample[field] = value
    assert probe.limit_reason(sample, initial=initial) == reason
    assert probe.limit_reason(healthy_sample(), initial=True) is None


def test_example_requires_source_hash_and_orientation_without_double_flip(tmp_path):
    path, metadata = example_files(tmp_path)
    example, _ = probe.load_example(path, metadata)
    with np.load(path) as raw:
        np.testing.assert_array_equal(example["image"], raw["image"])
    metadata.write_text(json.dumps({"example_npz_sha256": "0" * 64,
                                    "camera_orientation": "policy_ready_rot180_once"}))
    with pytest.raises(ValueError, match="metadata hash"):
        probe.load_example(path, metadata)
    metadata.write_text(json.dumps({"example_npz_sha256": probe.digest(path), "camera_orientation": "raw"}))
    with pytest.raises(ValueError, match="oriented"):
        probe.load_example(path, metadata)


def test_output_is_exclusive_and_confined(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    output = tmp_path / "outputs/action_adaptation/probe_v1/attempt_001"
    output.mkdir(parents=True)
    sentinel = output / "existing.txt"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError):
        probe.ProbeSession(output)
    assert sentinel.read_text() == "preserved"
    with pytest.raises(ValueError, match="below outputs"):
        probe.ProbeSession(tmp_path / "elsewhere")


def test_resource_failure_never_calls_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    sample = healthy_sample()
    sample["available_bytes"] = probe.START_AVAILABLE - 1
    monkeypatch.setattr(probe.ProbeSession, "sample", lambda self: sample)
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("Model path must not be entered"))
    output = tmp_path / "outputs/action_adaptation/probe_v1/attempt_001"
    assert probe.main(["--example", str(tmp_path / "absent.npz"), "--output", str(output)]) == 75
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "resource_stopped"
    assert status["owner"]["pid"] > 0
    assert (output / "resources.jsonl").is_file()


def test_unimplemented_mode_fails_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("Model path must not be entered"))
    output = tmp_path / "outputs/action_adaptation/probe_v1/attempt_001"
    assert probe.main(["--example", "absent.npz", "--output", str(output), "--mode", "output_head"]) == 1
    assert "not implemented" in json.loads((output / "status.json").read_text())["error"]


def test_full_worker_orchestration_uses_native_updates_and_real_save_reload(monkeypatch, tmp_path):
    from embodied_research.runtime import configure_platform
    configure_platform()
    import torch
    import embodied_research
    import embodied_research.policies as policies
    import embodied_research.runtime as runtime

    native_forward_calls = []

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(.1), requires_grad=False)

        def reset(self):
            pass

        def value(self):
            return self.base + ((self.A * self.B).sum() if hasattr(self, "A") else 0)

        def predict_action_chunk(self, batch, noise):
            return self.value().expand(1, 50, 7).clone()

        def forward(self, batch, noise=None, time=None):
            native_forward_calls.append((noise.clone(), time.clone()))
            loss = (self.value() - batch["action"].mean()).square()
            return loss, {"loss": float(loss.detach())}

    class Processor:
        def reset(self):
            pass

        def __call__(self, raw):
            result = dict(raw)
            for key in ("observation.state", "observation.images.image", "observation.images.image2"):
                result[key] = raw[key].unsqueeze(0)
            return result

    class TinyAdapter:
        def __init__(self, *args, **kwargs):
            self.policy = TinyPolicy()
            self.preprocess = Processor()
            self.metadata = {"test_only": True}

        def _prepare_observation(self, observation):
            return self.preprocess({key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                                    for key, value in observation.items()})

        def _actions_to_numpy(self, actions):
            return actions.detach().cpu().numpy().copy() * 2

    class TinyTrainable:
        def __init__(self, policy):
            self.policy = policy
            policy.A = torch.nn.Parameter(torch.ones(8))
            policy.B = torch.nn.Parameter(torch.zeros(8))

        def named_parameters(self):
            return [("test.q_proj.lora_A", self.policy.A), ("test.q_proj.lora_B", self.policy.B)]

        def parameters(self):
            return [self.policy.A, self.policy.B]

        def manifest(self):
            return {"test_only": True, "rank": 8}

        def save(self, path, metadata=None):
            np.savez(path, A=self.policy.A.detach().numpy(), B=self.policy.B.detach().numpy())

        def reload(self, path):
            with np.load(path, allow_pickle=False) as saved, torch.no_grad():
                self.policy.A.copy_(torch.from_numpy(saved["A"]))
                self.policy.B.copy_(torch.from_numpy(saved["B"]))

    def audit(policy):
        values = [policy.A.grad, policy.B.grad]
        finite = all(value is not None and torch.isfinite(value).all().item() for value in values)
        nonzero = any(value is not None and value.abs().sum().item() > 0 for value in values)
        return {"ok": finite and nonzero and policy.base.grad is None, "all_trainable_gradients_finite": finite,
                "any_trainable_gradient_nonzero": nonzero,
                "frozen_gradients_present": [] if policy.base.grad is None else ["base"]}

    learning = types.ModuleType("embodied_research.action_adaptation")
    learning.frozen_base_checksum = lambda policy: hashlib.sha256(policy.base.detach().numpy().tobytes()).hexdigest()
    learning.install_action_lora = lambda policy, **kwargs: TinyTrainable(policy)
    learning.training_forward_parity = lambda *args, **kwargs: {"ok": False, "test_only": "cache unsupported"}
    learning.compute_full_loss = lambda policy, batch, **kwargs: policy.forward(batch, **kwargs)
    learning.gradient_audit = audit
    monkeypatch.setitem(sys.modules, "embodied_research.action_adaptation", learning)
    monkeypatch.setattr(embodied_research, "action_adaptation", learning, raising=False)
    monkeypatch.setattr(policies, "PolicyAdapter", TinyAdapter)
    monkeypatch.setattr(runtime, "configure_runtime", lambda root: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: None)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe.ProbeSession, "sample", lambda self: healthy_sample())
    example, metadata = example_files(tmp_path)
    output = tmp_path / "outputs/action_adaptation/probe_v1/attempt_001"
    assert probe.main(["--example", str(example), "--metadata", str(metadata), "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["optimizer_steps"] == 3 and len(native_forward_calls) == 4
    assert report["cache_parity"]["ok"] is False and report["cache_used_for_optimizer_steps"] is False
    assert report["lora_b_changed"] and report["base_exactly_unchanged"]
    assert report["reload_parameters_exact"] and report["reload_actions_exact"]
    assert report["same_noise_action_max_abs_change"] > 0
    assert report["closed_loop_success_evidence"] is False
    assert all(torch.equal(native_forward_calls[0][0], noise) for noise, _ in native_forward_calls)
    second = json.loads((output / "step_002.json").read_text())
    assert second["gradient_groups"]["lora_A"]["gradient_l2"] > 0
    with np.load(output / "prediction_evidence.npz") as evidence:
        np.testing.assert_array_equal(evidence["before"], evidence["zero_initialized"])
        np.testing.assert_array_equal(evidence["after"], evidence["reloaded"])
        assert not np.array_equal(evidence["after"], evidence["erased"])
    assert json.loads((output / "status.json").read_text())["status"] == "passed"
