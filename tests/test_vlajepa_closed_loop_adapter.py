"""Opt-in adapter wiring checks with real tiny CPU kernels, not a VLA evaluation.

No checkpoint, tokenizer, LeRobot model or simulator is loaded.  The 300 distinct
2x2 layers exercise actual DynamicLinear hooks; a small tensor action head and
nonidentity tensor processors expose RNG, reset, observation and transform bugs.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FEEDBACKBENCH_INT8_KERNEL_TEST") != "1",
    reason="Explicit opt-in for bounded real Torch CPU adapter wiring checks",
)
PATH = Path(__file__).resolve().parents[1] / "scripts/run_vlajepa_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("vlajepa_closed_loop_adapter_tested", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def observation(index):
    return {
        runner.FEATURES[0]: np.full((3, 360, 360), np.float32((index + 1) / 8), dtype=np.float32),
        runner.FEATURES[1]: np.full((3, 360, 360), np.float32((index + 2) / 8), dtype=np.float32),
        runner.FEATURES[2]: np.arange(8, dtype=np.float32) + np.float32(index),
        "task": f"synthetic wiring instruction {index}",
    }


@pytest.fixture(scope="module")
def tiny_runtime():
    import torch
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)

    def original_context(device_type, dtype=None):
        assert device_type == "cpu"
        return nullcontext()

    official = SimpleNamespace(get_autocast_context=original_context)

    class TinyStep(torch.nn.Module):
        def forward(self, actions, condition, step):
            return actions + condition.reshape(1, 1, 1) * 0.01 + (step + 1) * 0.001

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyStep()
            self.actual_noise = []

        def predict_action(self, condition):
            actions = torch.randn(1, 7, 7, dtype=condition.dtype, device=condition.device)
            self.actual_noise.append(actions.detach().clone())
            for step in range(4):
                actions = self.model(actions, condition, step)
            return actions

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            weight = torch.tensor([[0.5, 0.25], [-0.25, 0.5]], dtype=torch.float32)
            bias = torch.tensor([0.1, -0.1], dtype=torch.float32)
            self.model.qwen = torch.nn.ModuleList([
                runner.quant.quantize_one_linear(
                    torch.nn.Linear(2, 2, device="meta"), weight.clone(), bias.clone(), torch
                ) for _ in range(300)
            ])
            self.model.action_model = TinyHead()
            self.reset_count = 0
            self.reset()
            self.requires_grad_(False).eval()

        def reset(self):
            self.reset_count += 1
            self.seen_batches, self.context_functions, self.queue = [], [], []
            self.model.action_model.actual_noise = []
            self.episode_calls = 0

        def predict_action_chunk(self, batch):
            assert torch.is_inference_mode_enabled()
            assert not torch.is_autocast_enabled("cpu")
            self.seen_batches.append({
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            })
            context_function = official.get_autocast_context
            self.context_functions.append(context_function)
            with context_function("cpu", torch.bfloat16):
                # Both current cameras and the current state influence the result.
                value = torch.stack((batch[runner.FEATURES[2]][0, 0],
                                     batch[runner.FEATURES[0]][0, 0, 0, 0]
                                     + batch[runner.FEATURES[1]][0, 0, 0, 0])).reshape(1, 2)
                encoded = torch.stack([layer(value) for layer in self.model.qwen]).mean()
            self.episode_calls += 1
            self.queue.append(batch["task"])
            # Exposes a missing policy.reset instead of silently tolerating it.
            return self.model.action_model.predict_action(encoded + self.episode_calls * 0.1)

    class TensorPreprocessor:
        def __init__(self):
            self.reset_count = 0
            self.reset()

        def reset(self):
            self.reset_count += 1
            self.calls = []

        def __call__(self, packet):
            assert set(packet) == {*runner.FEATURES, "task"}
            assert all(packet[key].dtype == torch.float32 for key in runner.FEATURES)
            self.calls.append(packet["task"])
            batch = {key: packet[key].unsqueeze(0) for key in runner.FEATURES}
            # An observable, nonidentity preprocessing operation.
            batch[runner.FEATURES[2]] = batch[runner.FEATURES[2]] * 0.5
            batch["task"] = packet["task"]
            return batch

    class TensorPostprocessor:
        def __init__(self):
            self.reset_count = 0
            self.reset()

        def reset(self):
            self.reset_count += 1
            self.inputs, self.outputs = [], []
            self.fail = False
            self.before_call = None

        def __call__(self, raw):
            assert torch.is_inference_mode_enabled()
            self.inputs.append(raw.detach().clone())
            if self.before_call is not None:
                self.before_call()
            if self.fail:
                raise RuntimeError("synthetic postprocessing failure")
            # A real tensor transform; applying it twice changes the six controls.
            commands = raw.clone()
            commands[..., :6] = commands[..., :6] * 0.125 + 0.25
            commands[..., 6] = torch.where(raw[..., 6] >= 0, -1.0, 1.0)
            self.outputs.append(commands.detach().clone())
            return commands

    try:
        with torch.random.fork_rng(devices=[]):
            policy = TinyPolicy()
            names = [name for name, module in policy.named_modules() if isinstance(module, DynamicLinear)]
            assert len(names) == len({id(module) for module in policy.model.qwen}) == 300
            yield SimpleNamespace(
                torch=torch, policy=policy, official=official, original_context=original_context,
                pre_type=TensorPreprocessor, post_type=TensorPostprocessor, names=names,
            )
    finally:
        torch.set_num_threads(original_threads)


@pytest.fixture
def harness(tiny_runtime, tmp_path):
    runtime = tiny_runtime
    pre, post = runtime.pre_type(), runtime.post_type()
    adapter = runner.ClosedLoopAdapter(
        runtime.policy, pre, post, runtime.official, {"quantized_modules": runtime.names}
    )
    adapter.destination = tmp_path
    return SimpleNamespace(**vars(runtime), pre=pre, post=post, adapter=adapter, destination=tmp_path)


def direct_reference(harness, observations, seed):
    """Uninstrumented calls are the oracle: no adapter or randn interception."""
    torch, policy = harness.torch, harness.policy
    pre, post = harness.pre_type(), harness.post_type()
    policy.reset()
    torch.manual_seed(seed)
    rows = []
    for obs in observations:
        batch = pre({**{key: torch.from_numpy(obs[key]) for key in runner.FEATURES}, "task": obs["task"]})
        before = torch.get_rng_state().clone()
        with torch.inference_mode():
            raw = policy.predict_action_chunk(batch)
            commands = post(raw)
        rows.append({
            "rng_state": before.numpy().copy(), "next_rng_state": torch.get_rng_state().numpy().copy(),
            "noise": policy.model.action_model.actual_noise[-1].numpy().copy(),
            "model_actions": raw.numpy().copy(), "environment_actions": commands.numpy().copy(),
        })
    assert policy.context_functions == [harness.original_context] * len(observations)
    return rows


def assert_no_instrumentation_leak(harness, original_randn):
    assert harness.torch.randn is original_randn
    assert harness.official.get_autocast_context is harness.original_context
    assert "predict_action" not in harness.policy.model.action_model.__dict__
    assert all(not layer._forward_pre_hooks for layer in harness.policy.model.qwen)
    assert not harness.policy.model.action_model.model._forward_hooks


def assert_call_evidence(harness, index, obs, expected, returned):
    stem = harness.adapter.destination / f"call_{index:04d}"
    record = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["status"] == "complete" and record["forward_returned"] is True
    assert record["returned_at"] and record["action_head_nfe"] == 4
    assert record["quantized_module_calls"] == dict.fromkeys(harness.names, 1)
    assert record["task"] == obs["task"]
    assert record["input_sha256"] == {key: runner.array_sha(obs[key]) for key in runner.FEATURES}
    assert record["artifact_sha256"] == runner.file_sha(stem.with_suffix(".npz"))
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
        assert set(saved.files) == {"model_actions", "environment_actions", "noise", "rng_state", "input_robot_state"}
        for key in ("model_actions", "environment_actions", "noise", "rng_state"):
            np.testing.assert_array_equal(saved[key], expected[key])
        np.testing.assert_array_equal(saved["input_robot_state"], obs[runner.FEATURES[2]])
        np.testing.assert_array_equal(returned, saved["environment_actions"][0])
        assert record["noise_sha256"] == runner.array_sha(saved["noise"])
        assert record["rng_state_sha256"] == runner.array_sha(saved["rng_state"])
    assert harness.adapter.call_log[index] == {
        "call_index": index, "record": stem.with_suffix(".json").name,
        "sha256": runner.file_sha(stem.with_suffix(".json")),
    }


def test_two_real_kernel_replans_preserve_rng_current_observation_and_single_postprocess(harness):
    observations, seed = [observation(0), observation(2)], 127
    expected = direct_reference(harness, observations, seed)
    original_randn = harness.torch.randn
    harness.adapter.reset(seed)
    for index, obs in enumerate(observations):
        harness.adapter.observe(obs)
        returned = harness.adapter.predict_chunk()
        assert_call_evidence(harness, index, obs, expected[index], returned)
        np.testing.assert_array_equal(harness.torch.get_rng_state().numpy(), expected[index]["next_rng_state"])
        assert_no_instrumentation_leak(harness, original_randn)
        assert len(harness.pre.calls) == len(harness.post.inputs) == len(harness.post.outputs) == index + 1
        np.testing.assert_array_equal(harness.post.inputs[-1].numpy(), expected[index]["model_actions"])
        np.testing.assert_array_equal(harness.post.outputs[-1].numpy()[0], returned)
        seen = harness.policy.seen_batches[index]
        for key in runner.FEATURES[:2]:
            np.testing.assert_array_equal(seen[key].numpy()[0], obs[key])
        np.testing.assert_array_equal(seen[runner.FEATURES[2]].numpy()[0], obs[runner.FEATURES[2]] * 0.5)
        assert seen["task"] == obs["task"]
    assert harness.policy.context_functions == [runner.quant.cpu_fp32_context] * 2
    assert harness.adapter.inference_count == 2 and harness.adapter.metadata["action_head_nfe"] == 8
    assert not np.array_equal(expected[0]["noise"], expected[1]["noise"])


def test_episode_reset_clears_policy_processor_adapter_state_and_reproduces_first_call(harness):
    seed, obs = 197, observation(1)
    harness.adapter.reset(seed)
    harness.adapter.observe(obs)
    first = harness.adapter.predict_chunk()
    with np.load(harness.destination / "call_0000.npz", allow_pickle=False) as saved:
        expected = {key: saved[key].copy() for key in ("model_actions", "environment_actions", "noise", "rng_state")}
    harness.adapter.observe(observation(3))
    harness.adapter.predict_chunk()
    assert harness.policy.episode_calls == len(harness.policy.queue) == 2
    before = (harness.policy.reset_count, harness.pre.reset_count, harness.post.reset_count)
    harness.adapter.destination = harness.destination / "next_episode"
    harness.adapter.destination.mkdir()
    harness.adapter.reset(seed)
    assert (harness.policy.reset_count, harness.pre.reset_count, harness.post.reset_count) == tuple(n + 1 for n in before)
    assert harness.policy.episode_calls == 0 and harness.policy.queue == []
    assert harness.pre.calls == harness.post.inputs == harness.post.outputs == []
    assert harness.adapter.observation is None and harness.adapter.call_log == []
    assert harness.adapter.inference_count == 0
    assert harness.adapter.last_timing == {"inference_seconds": 0.0, "preprocess_seconds": 0.0}
    harness.adapter.observe(obs)
    repeated = harness.adapter.predict_chunk()
    np.testing.assert_array_equal(repeated, first)
    assert_call_evidence(harness, 0, obs, expected, repeated)
    assert len(harness.pre.calls) == len(harness.post.inputs) == 1


def test_postprocess_failure_preserves_returned_forward_and_releases_all_hooks(harness):
    original_randn = harness.torch.randn
    harness.adapter.reset(211)
    harness.adapter.observe(observation(0))
    harness.post.fail = True

    def assert_return_record_already_exists():
        record = json.loads((harness.destination / "call_0000.json").read_text(encoding="utf-8"))
        assert record["status"] == "forward_returned" and record["forward_returned"] is True
        assert record["returned_at"] and record["forward_seconds"] >= 0

    harness.post.before_call = assert_return_record_already_exists
    with pytest.raises(RuntimeError, match="synthetic postprocessing failure"):
        harness.adapter.predict_chunk()
    assert_return_record_already_exists()
    assert_no_instrumentation_leak(harness, original_randn)
    assert harness.adapter.inference_count == 0 and harness.adapter.call_log == []
    assert len(harness.post.inputs) == 1 and harness.post.outputs == []
    assert not (harness.destination / "call_0000.npz").exists()
    with pytest.raises(ValueError, match="Call evidence already exists"):
        harness.adapter.predict_chunk()
