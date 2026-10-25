"""Small data/optimizer/worker tests; never load pretrained models or render."""
import importlib.util
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from embodied_research import action_adaptation as learning


SPEC = importlib.util.spec_from_file_location(
    "action_training_under_test", Path(__file__).resolve().parents[1] / "scripts/train_action_adaptation.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)
OPTIMIZER = {"type": "AdamW", "lr": .002, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.name, "sha256": train.digest(path)}


def protocol_fixture(root):
    sources = {}
    for name in train.REQUIRED_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n", encoding="utf-8")
        sources[name] = train.digest(path)
    models = {}
    for name in ("smolvla", "smolvlm_assets"):
        directory = root / "models" / name
        directory.mkdir(parents=True)
        json_file(directory / "config.json", {})
        json_file(directory / "download_manifest.json", {"files": [{"file": "config.json"}]})
        for path in directory.iterdir():
            models[path.relative_to(root).as_posix()] = train.digest(path)
    source = root / "demos.hdf5"
    with h5py.File(source, "w") as dataset:
        for index in range(2):
            demo = dataset.create_group(f"data/demo_{index}")
            demo.attrs["num_samples"] = 8
            demo["actions"] = np.arange(56, dtype=np.float32).reshape(8, 7)
            obs = demo.create_group("obs")
            pixels = np.arange(8 * 4 * 5 * 3, dtype=np.uint8).reshape(8, 4, 5, 3)
            obs["agentview_rgb"], obs["eye_in_hand_rgb"] = pixels, pixels + np.uint8(1)
            obs["ee_states"] = np.arange(48, dtype=np.float32).reshape(8, 6)
            obs["gripper_states"] = np.arange(16, dtype=np.float32).reshape(8, 2)
    demos = [{"id": f"d{index}", "split": split, "task": "task0", "language": "pick up the bowl",
              "path": "demos.hdf5", "sha256": train.digest(source), "episode_key": f"demo_{index}", "frame_count": 8}
             for index, split in enumerate(("train", "validation"))]
    manifest = json_file(root / "demos.json", {"demos": demos})
    alignment = json_file(root / "alignment.json", {"status": "passed", "action_offset": 1})
    schedule = json_file(root / "schedule.json", {"train": [{"demo_id": "d0", "observation_index": i} for i in range(3)],
                                                  "validation": [{"demo_id": "d1", "observation_index": 0,
                                                                  "noise_seed": 901, "time": .5}]})
    protocol = {"schema": train.SCHEMA, "frozen": True, "deadline_utc": train.DEADLINE.isoformat(),
                "flow_time_distribution": train.FLOW_DISTRIBUTION, "source_files": sources,
                "model": {"models_dir": "models", "files": models, "num_steps": 10},
                "data": {"manifest": manifest, "alignment_evidence": alignment, "action_offset": 1,
                         "image_transform": "rot180_once"}, "optimizer": OPTIMIZER.copy(),
                "resources": {"torch_threads": 2, "interop_threads": 1, "max_seconds": 28_800},
                "checkpoint_every": 1, "validation_every": 1,
                "runs": {"lora_11": {"mode": "lora", "initialization_seed": 11, "flow_seed": 21,
                                     "steps": 3, "schedule": schedule}},
                "evaluation": {"extra_prospective_fields_are_allowed": True}}
    path = root / "protocol.json"
    json_file(path, protocol)
    return protocol, path


def verify(root, protocol, path):
    json_file(path, protocol)
    return train.verify_protocol(path, train.digest(path), "lora_11", root=root)


def test_protocol_checks_all_inputs_once_and_whole_demo_alignment(tmp_path):
    protocol, path = protocol_fixture(tmp_path)
    plan = verify(tmp_path, protocol, path)
    sample = train.read_demo_example(plan, {"demo_id": "d0", "observation_index": 5})
    with h5py.File(tmp_path / "demos.hdf5", "r") as dataset:
        original = dataset["data/demo_0"]
        np.testing.assert_array_equal(sample["image"], original["obs/agentview_rgb"][5][::-1, ::-1])
        np.testing.assert_array_equal(sample["state"][:6], original["obs/ee_states"][5])
        np.testing.assert_array_equal(sample["actions"][:2], original["actions"][6:8])
        np.testing.assert_array_equal(sample["actions"][2:], np.repeat(original["actions"][-1:], 48, axis=0))
    assert (~sample["action_is_pad"]).sum() == 2
    source = next(iter(protocol["source_files"]))
    (tmp_path / source).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Input hash differs"):
        verify(tmp_path, protocol, path)


@pytest.mark.parametrize("invalid", ["split_overlap", "schedule_crosses_split", "alignment", "missing_model_binding",
                                       "wrong_steps", "wrong_time_distribution", "wrong_threads", "traversal",
                                       "duplicate_source_alias"])
def test_protocol_rejects_unregistered_or_leaking_inputs(tmp_path, invalid):
    protocol, path = protocol_fixture(tmp_path)
    if invalid in ("split_overlap", "duplicate_source_alias"):
        manifest = train.read_json(tmp_path / "demos.json")
        manifest["demos"][1]["episode_key"] = "demo_0"
        if invalid == "duplicate_source_alias":
            shutil.copyfile(tmp_path / "demos.hdf5", tmp_path / "copied.hdf5")
            manifest["demos"][1]["path"] = "copied.hdf5"
        protocol["data"]["manifest"] = json_file(tmp_path / "demos.json", manifest)
    elif invalid == "schedule_crosses_split":
        schedule = train.read_json(tmp_path / "schedule.json")
        schedule["train"][0]["demo_id"] = "d1"
        protocol["runs"]["lora_11"]["schedule"] = json_file(tmp_path / "schedule.json", schedule)
    elif invalid == "alignment":
        protocol["data"]["alignment_evidence"] = json_file(tmp_path / "alignment.json",
                                                             {"status": "passed", "action_offset": 0})
    elif invalid == "missing_model_binding":
        del protocol["model"]["files"]["models/smolvla/config.json"]
    elif invalid == "wrong_steps":
        protocol["runs"]["lora_11"]["steps"] = 4
    elif invalid == "wrong_time_distribution":
        protocol["flow_time_distribution"] = "uniform"
    elif invalid == "wrong_threads":
        protocol["resources"]["torch_threads"] = 8
    else:
        protocol["data"]["manifest"]["path"] = "../outside.json"
    with pytest.raises(ValueError):
        verify(tmp_path, protocol, path)


def tiny_adapter():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        policy = nn.Sequential(nn.Linear(3, 2))
    fingerprint = learning.frozen_base_checksum(policy)
    policy[0] = learning.LoRALinear(policy[0], generator=torch.Generator().manual_seed(3))
    adapter = learning.ActionAdapter(policy, base_fingerprint=fingerprint, seed=3, _targets={"0": policy[0]})
    return adapter


def tiny_update(adapter, optimizer, flow, step):
    noise, times = flow.sample()
    # All four saved streams affect this small objective, making resume equality
    # sensitive to forgotten Python/NumPy/global-Torch or private RNG state.
    target = times[0] + random.random() + np.random.random() + torch.rand(())
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(adapter.policy(noise[0, :4, :3]), target.expand(4, 2))
    loss.backward()
    optimizer.step()
    return {"step": step, "loss": float(loss.detach()), "time": float(times[0]),
            "noise": train.json_digest(noise[0, 0].tolist())}


def test_checkpoint_resume_matches_uninterrupted_parameters_optimizer_rng_and_losses(tmp_path):
    torch.manual_seed(63)
    np.random.seed(64)
    random.seed(65)
    adapter, flow = tiny_adapter(), train.FlowRandom(66)
    optimizer = train.make_optimizer(adapter, OPTIMIZER)
    history = {"training": [tiny_update(adapter, optimizer, flow, step) for step in (1, 2)], "validation": []}
    binding = {"protocol_sha256": "a" * 64, "run_id": "tiny_11"}
    checkpoint = train.save_checkpoint(tmp_path, step=2, binding=binding, adapter=adapter, optimizer=optimizer,
                                       flow=flow, history=history, frozen_checksum=adapter.base_fingerprint)
    continued = [tiny_update(adapter, optimizer, flow, step) for step in (3, 4, 5)]
    expected_parameters = [value.detach().clone() for value in adapter.parameters()]
    expected_optimizer, _ = train.optimizer_arrays(optimizer, adapter)
    expected_rng_arrays, expected_rng_metadata = train.capture_rng(flow)
    restored, other_flow = tiny_adapter(), train.FlowRandom(999)
    other_optimizer = train.make_optimizer(restored, OPTIMIZER)
    step, restored_history = train.restore_checkpoint(checkpoint, binding=binding, adapter=restored,
                                                      optimizer=other_optimizer, flow=other_flow,
                                                      frozen_checksum=restored.base_fingerprint)
    assert step == 2 and restored_history == history
    resumed = [tiny_update(restored, other_optimizer, other_flow, step) for step in (3, 4, 5)]
    assert continued == resumed
    assert all(torch.equal(left, right) for left, right in zip(restored.parameters(), expected_parameters, strict=True))
    actual_optimizer, _ = train.optimizer_arrays(other_optimizer, restored)
    assert all(np.array_equal(actual_optimizer[name], value) for name, value in expected_optimizer.items())
    actual_arrays, actual_metadata = train.capture_rng(other_flow)
    assert all(np.array_equal(actual_arrays[name], value) for name, value in expected_rng_arrays.items())
    assert train.canonical(actual_metadata) == train.canonical(expected_rng_metadata)
    assert learning.frozen_base_checksum(restored.policy) == restored.base_fingerprint


def test_checkpoint_is_unique_hash_bound_numeric_and_retains_partial_failure(tmp_path, monkeypatch):
    adapter, flow = tiny_adapter(), train.FlowRandom(4)
    optimizer = train.make_optimizer(adapter, OPTIMIZER)
    binding = {"protocol_sha256": "a" * 64, "run_id": "tiny_11"}
    arguments = dict(step=0, binding=binding, adapter=adapter, optimizer=optimizer, flow=flow,
                     history={"training": [], "validation": []}, frozen_checksum=adapter.base_fingerprint)
    checkpoint = train.save_checkpoint(tmp_path, **arguments)
    receipt_before = train.digest(checkpoint / "receipt.json")
    for name in ("optimizer.npz", "rng.npz"):
        arrays = train.load_numeric(checkpoint / name)
        assert all(array.dtype.kind in "biuf" for array in arrays.values())
    with pytest.raises(ValueError, match="overwrite"):
        train.save_checkpoint(tmp_path, **arguments)
    assert train.digest(checkpoint / "receipt.json") == receipt_before
    with pytest.raises(ValueError, match="binding differs"):
        train.inspect_checkpoint(checkpoint, {**binding, "run_id": "different"})
    (checkpoint / "rng.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash differs"):
        train.inspect_checkpoint(checkpoint, binding)

    def fail_save(*args, **kwargs):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(adapter, "save", fail_save)
    with pytest.raises(OSError, match="interruption"):
        train.save_checkpoint(tmp_path, **{**arguments, "step": 1})
    assert not (tmp_path / "checkpoints/step_000001").exists()
    failures = list((tmp_path / "checkpoints").glob(".s000001_*/failure.json"))
    assert len(failures) == 1 and "interruption" in failures[0].read_text()


def test_fixed_validation_preserves_every_training_rng_and_does_not_choose_checkpoint(monkeypatch):
    flow = train.FlowRandom(21)
    model = SimpleNamespace(policy=nn.Linear(1, 1))
    before_arrays, before_meta = train.capture_rng(flow)
    plan = {"schedule": {"validation": [{"demo_id": "v", "observation_index": 0, "noise_seed": 55, "time": .5}]}}
    monkeypatch.setattr(train, "read_demo_example", lambda *args: {})
    monkeypatch.setattr(train, "preprocess_example", lambda *args: {})

    def loss(*args, noise, time):
        flow.sample()
        random.random()
        np.random.random()
        torch.rand(())
        return noise.square().mean(), {"valid_action_steps": 20}

    session = SimpleNamespace(checkpoint=lambda *args, **kwargs: None)
    first = train.validate_fixed(model, plan, flow, loss, session, step=0)
    second = train.validate_fixed(model, plan, flow, loss, session, step=10)
    assert first["samples"] == second["samples"]
    assert first["selection_use"].startswith("diagnostic only")
    arrays, metadata = train.capture_rng(flow)
    assert all(np.array_equal(value, arrays[key]) for key, value in before_arrays.items())
    assert train.canonical(metadata) == train.canonical(before_meta)


def test_beta_time_and_noise_refresh_without_global_rng_use():
    flow, duplicate = train.FlowRandom(17), train.FlowRandom(17)
    global_state = torch.get_rng_state().clone()
    samples = [flow.sample() for _ in range(3)]
    for expected_noise, expected_time in samples:
        noise, times = duplicate.sample()
        assert torch.equal(noise, expected_noise) and torch.equal(times, expected_time)
        assert .001 <= times[0] <= 1
    assert not torch.equal(samples[0][0], samples[1][0])
    assert not torch.equal(samples[0][1], samples[1][1])
    assert torch.equal(global_state, torch.get_rng_state())


@pytest.mark.parametrize("field,value,initial,reason", [
    ("available_bytes", train.START_AVAILABLE - 1, True, "start_memory_threshold"),
    ("available_bytes", train.RUNTIME_AVAILABLE - 1, False, "runtime_memory_reserve"),
    ("elapsed_seconds", 28_800, False, "protocol_runtime_limit"),
    ("at", train.DEADLINE.isoformat(), False, "hard_deadline"),
])
def test_resource_limits_are_fixed_and_not_probe_twenty_minutes(field, value, initial, reason):
    sample = {"at": "2026-10-22T00:00:00+00:00", "elapsed_seconds": 1800,
              "available_bytes": 8 * train.GIB, "disk_free_bytes": 8 * train.GIB}
    assert train.resource_reason(sample, 28_800) is None
    sample[field] = value
    assert train.resource_reason(sample, 28_800, initial=initial) == reason


@pytest.mark.parametrize("reason,expected", [("runtime_memory_reserve", 75), ("hard_deadline", 3), ("ordinary", 1)])
def test_only_runtime_reserve_is_retryable_exit_75(tmp_path, monkeypatch, reason, expected):
    monkeypatch.setattr(train, "verify_protocol", lambda *args: {})
    outcomes = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def begin(self):
            if reason == "ordinary":
                raise ValueError("ordinary")
            raise train.ResourceStop(reason)

        def finish(self, status, **details):
            outcomes.append((status, details))

    monkeypatch.setattr(train, "TrainingSession", Session)
    assert train.main(["--protocol", str(tmp_path / "p.json"), "--protocol-sha256", "a" * 64,
                       "--run-id", "r", "--output", str(tmp_path / "out")]) == expected
    assert outcomes and outcomes[0][0] == ("failed" if expected == 1 else "stopped")


def test_worker_fixed_final_step_report_and_native_loss_wiring(tmp_path, monkeypatch):
    """Run the complete loop with a tiny LoRA policy, including real AdamW/checkpoints."""
    from embodied_research import policies, runtime
    protocol, protocol_path = protocol_fixture(tmp_path)
    plan = verify(tmp_path, protocol, protocol_path)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(32, 7)
            self.config = SimpleNamespace(chunk_size=50, max_action_dim=32, action_feature=SimpleNamespace(shape=(7,)),
                                          use_cache=True, adapt_to_pi_aloha=False, n_obs_steps=1)

        def forward(self, batch, noise, time):
            actions = F.pad(batch["action"], (0, 25))
            velocity = self.projection(time[:, None, None] * noise + (1 - time[:, None, None]) * actions)
            errors = (velocity - (noise - actions)[..., :7]).square()
            valid = (~batch["action_is_pad"]).unsqueeze(-1)
            return (errors * valid).sum() / (valid.sum() * 7), {}

    class Model:
        def __init__(self, *args, **kwargs):
            self.policy = Policy().requires_grad_(False)

    def install(policy, *, base_fingerprint, seed):
        policy.projection = learning.LoRALinear(policy.projection, generator=torch.Generator().manual_seed(seed))
        return learning.ActionAdapter(policy, base_fingerprint=base_fingerprint, seed=seed,
                                       _targets={"projection": policy.projection})

    def preprocess(model, example, **kwargs):
        return {"action": torch.from_numpy(example["actions"]).unsqueeze(0),
                "action_is_pad": torch.from_numpy(example["action_is_pad"]).unsqueeze(0)}

    monkeypatch.setattr(runtime, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(policies, "PolicyAdapter", Model)
    monkeypatch.setattr(learning, "install_action_lora", install)
    monkeypatch.setattr(train, "preprocess_example", preprocess)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    output = tmp_path / "run"
    output.mkdir()
    attempt = output / "attempt"
    attempt.mkdir()
    session = SimpleNamespace(output=output, attempt=attempt, peak_rss=100, minimum_available=1000,
                              prior_seconds=0, started=train.time.monotonic(), checkpoint=lambda *args, **kwargs: None)
    report = train.run_training(plan, session)
    assert report["status"] == "complete" and report["completed_steps"] == 3
    assert report["final_checkpoint"]["path"] == "checkpoints/step_000003"
    assert report["protocol_sha256"] == train.digest(protocol_path)
    assert report["frozen_parameters_unchanged"]
    assert [record["step"] for record in report["history"]["training"]] == [1, 2, 3]
    assert len({record["noise_sha256"] for record in report["history"]["training"]}) == 3
    assert "original policy.forward" in report["training_path"]
    checkpoint = output / report["final_checkpoint"]["path"]
    state, history = train.inspect_checkpoint(checkpoint, plan["binding"])
    assert state["completed_steps"] == 3 and history == report["history"]
    assert train.digest(checkpoint / "adapter.npz") == report["final_checkpoint"]["sha256"]
    assert train.read_json(output / "report.json") == report
    with pytest.raises(ValueError, match="report already exists"):
        train.run_training(plan, session, resume=checkpoint)
    # A final checkpoint survives a missing report. Recovery in another attempt
    # seals the same parameters and history without any further optimizer update.
    recovery_output = tmp_path / "recovered"
    recovery_output.mkdir()
    recovery_attempt = recovery_output / "attempt"
    recovery_attempt.mkdir()
    resumed_session = SimpleNamespace(output=recovery_output, attempt=recovery_attempt, peak_rss=100,
                                      minimum_available=1000, prior_seconds=0, started=train.time.monotonic(),
                                      checkpoint=lambda *args, **kwargs: None)
    recovered_report = train.run_training(plan, resumed_session, resume=checkpoint)
    assert recovered_report["history"] == report["history"]
    assert recovered_report["final_checkpoint"]["sha256"] == report["final_checkpoint"]["sha256"]


def test_resume_accepts_new_sibling_attempt_ignores_partial_and_rejects_rollback(tmp_path):
    adapter, flow = tiny_adapter(), train.FlowRandom(2)
    optimizer = train.make_optimizer(adapter, OPTIMIZER)
    binding = {"protocol_sha256": "a" * 64, "run_id": "tiny_11"}
    first = tmp_path / "training/tiny_11/attempt_001"
    first.mkdir(parents=True)
    checkpoint = train.save_checkpoint(first, step=0, binding=binding, adapter=adapter, optimizer=optimizer,
                                       flow=flow, history={"training": [], "validation": []},
                                       frozen_checksum=adapter.base_fingerprint)
    next_output = first.with_name("attempt_002")
    train.validate_resume_path(next_output, checkpoint, binding)
    partial = checkpoint.with_name("step_000999.partial_incomplete")
    partial.mkdir()
    (partial / "receipt.json").write_text("invalid", encoding="utf-8")
    train.validate_resume_path(next_output, checkpoint, binding)
    record = tiny_update(adapter, optimizer, flow, 1)
    later = train.save_checkpoint(first, step=1, binding=binding, adapter=adapter, optimizer=optimizer,
                                  flow=flow, history={"training": [record], "validation": []},
                                  frozen_checksum=adapter.base_fingerprint)
    with pytest.raises(ValueError, match="latest complete"):
        train.validate_resume_path(next_output, checkpoint, binding)
    train.validate_resume_path(next_output, later, binding)
    with pytest.raises(ValueError, match="same run"):
        train.validate_resume_path(tmp_path / "other_run/attempt_002", later, binding)


def test_interrupted_resource_journal_retains_last_complete_runtime(tmp_path):
    path = tmp_path / "resources.jsonl"
    path.write_bytes(b'{"attempt_elapsed_seconds": 10.5}\n{"attempt_elapsed_seconds": 11.0}\n{"attempt_')
    assert train.last_attempt_seconds(path) == 11.0


def test_resume_refuses_changed_runtime_before_restoring_parameters(tmp_path, monkeypatch):
    adapter, flow = tiny_adapter(), train.FlowRandom(2)
    optimizer = train.make_optimizer(adapter, OPTIMIZER)
    binding = {"protocol_sha256": "a" * 64, "run_id": "tiny_11"}
    checkpoint = train.save_checkpoint(tmp_path, step=0, binding=binding, adapter=adapter, optimizer=optimizer,
                                       flow=flow, history={"training": [], "validation": []},
                                       frozen_checksum=adapter.base_fingerprint)
    environment = train.runtime_environment()
    monkeypatch.setattr(train, "runtime_environment", lambda: {**environment, "torch_threads": 999})
    before = [value.detach().clone() for value in adapter.parameters()]
    with pytest.raises(ValueError, match="runtime environment differs"):
        train.restore_checkpoint(checkpoint, binding=binding, adapter=adapter, optimizer=optimizer,
                                 flow=flow, frozen_checksum=adapter.base_fingerprint)
    assert all(torch.equal(left, right) for left, right in zip(adapter.parameters(), before, strict=True))
