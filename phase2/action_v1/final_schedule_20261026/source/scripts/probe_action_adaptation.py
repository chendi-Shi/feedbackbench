"""One supervised CPU action-expert LoRA probe; never launches another process.

The external run_action_adaptation_probe supervisor enforces hard stop boundaries
while a Torch call is blocked. This worker samples its own resources and checks
the same immutable limits between stages. A decreasing single-example training
loss is not evidence of better LIBERO success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
START_AVAILABLE = int(4.25 * GIB)
RUNTIME_AVAILABLE = int(.75 * GIB)
MIN_DISK = 4 * GIB
MAX_SECONDS = 20 * 60
DEADLINE = datetime(2026, 10, 30, 6, 8, tzinfo=timezone.utc)
SAMPLE_SECONDS = .5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    """Atomic status, bounded retries for Windows sharing violations only."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    delays = (.05, .1, .2, .4, .5)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


class ResourceStop(RuntimeError):
    pass


def limit_reason(sample, *, initial=False):
    if datetime.fromisoformat(sample["at"]) >= DEADLINE:
        return "hard_deadline"
    if sample["elapsed_seconds"] >= MAX_SECONDS:
        return "probe_20_minute_limit"
    if sample["available_bytes"] < (START_AVAILABLE if initial else RUNTIME_AVAILABLE):
        return "start_memory_threshold" if initial else "runtime_memory_reserve"
    if sample["disk_free_bytes"] < MIN_DISK:
        return "disk_reserve"
    return None


class ProbeSession:
    def __init__(self, output):
        import psutil

        self.output = Path(output).resolve()
        allowed = (ROOT / "outputs/action_adaptation").resolve()
        require(self.output.is_relative_to(allowed) and self.output != allowed,
                "Probe output must be a new directory below outputs/action_adaptation")
        self.output.mkdir(parents=True, exist_ok=False)
        self.process = psutil.Process()
        self.psutil = psutil
        self.started = time.monotonic()
        self.done = threading.Event()
        self.lock = threading.RLock()
        self.stop_reason = None
        self.samples = 0
        self.minimum_available = None
        self.peak_rss = 0
        self.state = {"schema": "feedbackbench.action_adaptation_probe.v1", "status": "starting",
                      "stage": "preflight", "started_at": utc_now(),
                      "owner": {"pid": self.process.pid, "created": self.process.create_time()},
                      "limits": {"start_available_bytes": START_AVAILABLE,
                                 "runtime_available_bytes": RUNTIME_AVAILABLE, "min_disk_bytes": MIN_DISK,
                                 "max_seconds": MAX_SECONDS, "deadline": DEADLINE.isoformat(),
                                 "sample_seconds": SAMPLE_SECONDS},
                      "native_training_path": "policy.forward", "worker_launches_children": False}
        self.log = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.resource_log = (self.output / "resources.jsonl").open("x", encoding="utf-8")
        self.thread = None

    def sample(self):
        return {"at": utc_now(), "elapsed_seconds": time.monotonic() - self.started,
                "rss_bytes": self.process.memory_info().rss,
                "available_bytes": self.psutil.virtual_memory().available,
                "disk_free_bytes": shutil.disk_usage(self.output).free}

    def record_sample(self):
        sample = self.sample()
        with self.lock:
            self.samples += 1
            self.peak_rss = max(self.peak_rss, sample["rss_bytes"])
            self.minimum_available = (sample["available_bytes"] if self.minimum_available is None
                                      else min(self.minimum_available, sample["available_bytes"]))
            self.resource_log.write(json.dumps(sample, allow_nan=False) + "\n")
            self.resource_log.flush()
            reason = limit_reason(sample)
            if reason:
                self.stop_reason = self.stop_reason or reason
        return sample

    def _monitor(self):
        while not self.done.wait(SAMPLE_SECONDS):
            try:
                self.record_sample()
            except BaseException as error:
                with self.lock:
                    self.stop_reason = self.stop_reason or f"resource_sampling_failed: {error!r}"
                return

    def begin(self):
        sample = self.record_sample()
        reason = limit_reason(sample, initial=True)
        if reason:
            self.stop_reason = reason
            raise ResourceStop(reason)
        self.checkpoint("resource_preflight_passed")
        self.thread = threading.Thread(target=self._monitor, daemon=True, name="action-probe-resource-sampler")
        self.thread.start()

    def checkpoint(self, stage, **details):
        sample = self.record_sample()
        with self.lock:
            if (self.output / "STOP_REQUEST").exists():
                self.stop_reason = self.stop_reason or "external_stop_request"
            if self.stop_reason:
                raise ResourceStop(self.stop_reason)
            self.state.update(status="running", stage=stage, updated_at=utc_now(), last_resource_sample=sample,
                              **details)
            event = {"at": utc_now(), "stage": stage, **details}
            self.log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self.log.flush()
            write_json(self.output / "status.json", self.state)

    def finish(self, status, **details):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            self.state.update(status=status, ended_at=utc_now(), elapsed_seconds=time.monotonic() - self.started,
                              resource_summary={"samples": self.samples, "peak_rss_bytes": self.peak_rss,
                                                "minimum_available_bytes": self.minimum_available}, **details)
            try:
                write_json(self.output / "status.json", self.state)
            except BaseException as error:
                diagnostic = self.output / ("persistence_failure_" + uuid4().hex + ".json")
                with diagnostic.open("x", encoding="utf-8") as stream:
                    json.dump({"status": "failed", "error": repr(error), "last_state": self.state}, stream)
                raise
            finally:
                self.log.close()
                self.resource_log.close()


def load_example(path, metadata_path):
    import numpy as np

    path, metadata_path = Path(path).resolve(), Path(metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("example_npz_sha256") == digest(path), "Example differs from source metadata hash")
    require(metadata.get("camera_orientation") == "policy_ready_rot180_once",
            "Expected explicitly oriented policy-ready images; probe never flips them")
    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 32 * 1024 ** 2,
                "Single-example NPZ expands beyond 32 MiB")
    with np.load(path, allow_pickle=False) as archive:
        expected = {"image", "image2", "state", "actions", "action_is_pad", "task"}
        require(expected <= set(archive.files), "Example is missing required observation/target fields")
        example = {key: archive[key].copy() for key in expected}
    for key in ("image", "image2"):
        pixels = example[key]
        require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
                and min(pixels.shape[:2]) >= 32, f"{key} must be uint8 HWC RGB")
    for key, shape in (("state", (8,)), ("actions", (50, 7))):
        require(example[key].dtype == np.float32 and example[key].shape == shape
                and np.isfinite(example[key]).all(), f"Expected finite float32 {key} {shape}")
    require(example["action_is_pad"].dtype == np.bool_ and example["action_is_pad"].shape == (50,)
            and (~example["action_is_pad"]).any(), "Expected a bool padding mask with valid targets")
    require(example["task"].shape == () and example["task"].dtype.kind == "U"
            and str(example["task"].item()).strip(), "Expected nonempty scalar Unicode task")
    example["task"] = str(example["task"].item())
    return example, metadata


def preprocess_example(adapter, example, torch):
    """Use checkpoint normalization including action targets; never rotate twice."""
    import numpy as np

    observation = {"task": example["task"], "observation.state": example["state"].copy()}
    for name in ("image", "image2"):
        observation[f"observation.images.{name}"] = (
            np.ascontiguousarray(example[name].transpose(2, 0, 1), dtype=np.float32) / np.float32(255))
    raw = {key: (torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value)
           for key, value in observation.items()}
    # The original to_batch action processor only expands a 1D action, not HxA.
    raw["action"] = torch.from_numpy(example["actions"].copy()).unsqueeze(0)
    raw["action_is_pad"] = torch.from_numpy(example["action_is_pad"].copy()).unsqueeze(0)
    adapter.preprocess.reset()
    batch = adapter.preprocess(raw)
    adapter.preprocess.reset()
    reference = adapter._prepare_observation(observation)
    parity = {key: bool(torch.equal(value, batch[key])) for key, value in reference.items()
              if isinstance(value, torch.Tensor)}
    require(parity and all(parity.values()), "Training observation preprocessing differs from original adapter")
    require(tuple(batch["action"].shape) == (1, 50, 7), "Normalized actions have wrong batch/horizon shape")
    require(tuple(batch["action_is_pad"].shape) == (1, 50), "Padding mask lost its batch dimension")
    require(torch.isfinite(batch["action"]).all().item(), "Normalized action target is nonfinite")
    return batch, {"observation_exact_parity": parity, "normalizer": "checkpoint original preprocessor",
                   "action_shape": list(batch["action"].shape), "image_flips_in_probe": 0,
                   "action_target": "demonstration actions; not model predictions"}


def parameter_checksum(named_parameters):
    checksum = hashlib.sha256()
    for name, parameter in named_parameters:
        checksum.update(name.encode())
        checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def gradient_groups(named_parameters):
    groups = {}
    for suffix in ("lora_A", "lora_B"):
        parameters = [(name, value) for name, value in named_parameters if name.endswith("." + suffix)]
        values = [value.grad for _, value in parameters]
        require(parameters, f"No {suffix} tensors installed")
        finite = all(value is not None and math.isfinite(float(value.detach().double().square().sum()))
                     for value in values)
        total = sum(float(value.detach().double().square().sum()) for value in values if value is not None)
        groups[suffix] = {"tensors": len(values), "all_present_finite": finite, "gradient_l2": math.sqrt(total)}
    return groups


def predict_native(adapter, batch, noise, torch):
    adapter.policy.reset()
    adapter.policy.eval()
    with torch.inference_mode():
        normalized = adapter.policy.predict_action_chunk(dict(batch), noise=noise.clone()).detach().clone()
        robot = adapter._actions_to_numpy(normalized).copy()
    require(torch.isfinite(normalized).all().item(), "Native prediction is nonfinite")
    return normalized, robot


def run_probe(args, session):
    from embodied_research.runtime import configure_runtime
    configure_runtime(ROOT)
    import numpy as np
    import torch
    from embodied_research.policies import PolicyAdapter
    from embodied_research import action_adaptation as learning

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    example, source = load_example(args.example, args.metadata)
    write_json(session.output / "input_binding.json", {"example": str(args.example.resolve()),
               "example_sha256": digest(args.example), "metadata": str(args.metadata.resolve()),
               "metadata_sha256": digest(args.metadata), "source_metadata": source})
    session.checkpoint("loading_model")
    adapter = PolicyAdapter("smolvla", num_steps=10, n_action_steps=1)
    require(all(value.device.type == "cpu" and value.dtype == torch.float32 for value in adapter.policy.parameters()),
            "Probe requires the complete FP32 CPU model")
    require(not any(value.requires_grad for value in adapter.policy.parameters()), "Base must initially be frozen")
    session.checkpoint("model_loaded", policy=adapter.metadata)
    batch, preprocessing = preprocess_example(adapter, example, torch)
    write_json(session.output / "preprocessing.json", preprocessing)
    noise = torch.randn((1, 50, 32), generator=torch.Generator(device="cpu").manual_seed(args.seed), dtype=torch.float32)
    time_tensor = torch.tensor([.5], dtype=torch.float32)
    session.checkpoint("hashing_frozen_base")
    base_before = learning.frozen_base_checksum(adapter.policy)
    session.checkpoint("baseline_native_prediction")
    before, before_robot = predict_native(adapter, batch, noise, torch)
    trainable = learning.install_action_lora(adapter.policy, base_fingerprint=base_before, seed=args.seed)
    named = list(trainable.named_parameters())
    initial_parameters = parameter_checksum(named)
    initial_b = parameter_checksum([(name, value) for name, value in named if name.endswith(".lora_B")])
    write_json(session.output / "adapter_manifest.json", trainable.manifest())
    session.checkpoint("zero_initialized_native_prediction")
    zero_initialized, _ = predict_native(adapter, batch, noise, torch)
    zero_init_equal = bool(torch.equal(before, zero_initialized))
    require(zero_init_equal, "Zero-B LoRA installation changed native predictions")
    session.checkpoint("training_cache_parity")
    try:
        cache_parity = learning.training_forward_parity(adapter.policy, dict(batch), noise=noise.clone(),
                                                        time=time_tensor.clone(), atol=1e-5, rtol=1e-4)
    except (ResourceStop, MemoryError):
        raise
    except Exception as error:
        cache_parity = {"ok": False, "error": repr(error), "traceback": traceback.format_exc()}
    write_json(session.output / "cache_parity.json", cache_parity)
    # Cache is a separately audited optimization. All probe updates use upstream.
    session.checkpoint("native_training", cache_parity_ok=bool(cache_parity.get("ok", False)))
    optimizer = torch.optim.AdamW(list(trainable.parameters()), lr=1e-4, weight_decay=0.0)
    records = []
    for index in range(args.steps):
        session.checkpoint("native_training_step", optimizer_step=index + 1)
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        loss, diagnostics = learning.compute_full_loss(adapter.policy, dict(batch), noise=noise.clone(),
                                                       time=time_tensor.clone())
        require(loss.ndim == 0 and torch.isfinite(loss).item(), "Native scalar loss is nonfinite")
        forward_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - tick
        gradients = learning.gradient_audit(adapter.policy)
        groups = gradient_groups(named)
        require(gradients["ok"] and gradients["all_trainable_gradients_finite"]
                and gradients["any_trainable_gradient_nonzero"] and not gradients["frozen_gradients_present"],
                "Finite nonzero trainable gradients / frozen base gradient guard failed")
        require(all(row["all_present_finite"] for row in groups.values()), "Missing/nonfinite LoRA gradient")
        require(groups["lora_B"]["gradient_l2"] > 0, "LoRA B gradient is zero")
        if index >= 1:
            require(groups["lora_A"]["gradient_l2"] > 0, "LoRA A gradient is zero after first update")
        tick = time.perf_counter()
        optimizer.step()
        require(all(torch.isfinite(value).all().item() for _, value in named), "Optimizer produced nonfinite adapters")
        record = {"step": index + 1, "loss": float(loss.detach()), "loss_diagnostics": diagnostics,
                  "gradients": gradients, "gradient_groups": groups, "forward_seconds": forward_seconds,
                  "backward_seconds": backward_seconds, "optimizer_seconds": time.perf_counter() - tick,
                  "same_noise_seed": args.seed, "native_flow_time": .5}
        records.append(record)
        write_json(session.output / f"step_{index + 1:03d}.json", record)
        session.checkpoint("optimizer_step_complete", completed_steps=index + 1)
    optimizer.zero_grad(set_to_none=True)
    final_parameters = parameter_checksum(named)
    final_b = parameter_checksum([(name, value) for name, value in named if name.endswith(".lora_B")])
    require(final_parameters != initial_parameters and final_b != initial_b, "LoRA parameters/B did not change")
    session.checkpoint("checking_frozen_base_after_training")
    base_after = learning.frozen_base_checksum(adapter.policy)
    require(base_before == base_after, "Frozen base changed during training")
    with torch.no_grad():
        final_loss, _ = learning.compute_full_loss(adapter.policy, dict(batch), noise=noise.clone(), time=time_tensor.clone())
    require(torch.isfinite(final_loss).item(), "Post-update native loss is nonfinite")
    session.checkpoint("trained_native_prediction")
    after, after_robot = predict_native(adapter, batch, noise, torch)
    action_delta = float((after - before).abs().max())
    require(action_delta > 0, "Training did not change same-noise native actions")
    checkpoint = session.output / "action_adapter.npz"
    session.checkpoint("saving_adapter")
    trainable.save(checkpoint, metadata={"probe": "action_adaptation/probe_v1", "seed": args.seed,
                                        "steps": args.steps, "example_sha256": digest(args.example),
                                        "training_path": "original policy.forward"})
    # Genuine disk reload on the same frozen base: remove trained adapter values
    # first. No second complete model is held in memory.
    with torch.no_grad():
        for _, value in named:
            value.zero_()
    session.checkpoint("zeroed_adapter_prediction")
    erased, _ = predict_native(adapter, batch, noise, torch)
    require(not torch.equal(after, erased), "Reload challenge failed: zeroing adapters did not change predictions")
    trainable.reload(checkpoint)
    session.checkpoint("reloaded_adapter_prediction")
    reloaded, reloaded_robot = predict_native(adapter, batch, noise, torch)
    reload_parameters_equal = parameter_checksum(named) == final_parameters
    reload_actions_equal = bool(torch.equal(after, reloaded)) and bool(np.array_equal(after_robot, reloaded_robot))
    require(reload_parameters_equal and reload_actions_equal, "Saved adapter reload does not reproduce trained actions")
    base_reloaded = learning.frozen_base_checksum(adapter.policy)
    require(base_reloaded == base_before, "Frozen base changed during checkpoint reload")
    np.savez_compressed(session.output / "prediction_evidence.npz", noise=noise.numpy(), flow_time=time_tensor.numpy(),
                        before=before.numpy(), zero_initialized=zero_initialized.numpy(), after=after.numpy(),
                        erased=erased.numpy(), reloaded=reloaded.numpy(), before_robot=before_robot,
                        after_robot=after_robot, reloaded_robot=reloaded_robot,
                        targets_robot=example["actions"], action_is_pad=example["action_is_pad"])
    result = {"status": "passed", "mode": "lora", "rank": 8, "seed": args.seed,
              "optimizer_steps": len(records), "training_path": "original policy.forward",
              "losses_before_each_update": [item["loss"] for item in records], "loss_after_last_update": float(final_loss),
              "loss_interpretation": "single-example fixed-noise optimization diagnostic, not task success",
              "cache_parity": cache_parity, "cache_used_for_optimizer_steps": False,
              "base_sha256_before": base_before, "base_sha256_after_training": base_after,
              "base_sha256_after_reload": base_reloaded, "base_exactly_unchanged": True,
              "zero_initialized_predictions_exact": zero_init_equal, "same_noise_action_max_abs_change": action_delta,
              "lora_b_changed": initial_b != final_b, "reload_parameters_exact": reload_parameters_equal,
              "reload_actions_exact": reload_actions_equal, "checkpoint_sha256": digest(checkpoint),
              "checkpoint": checkpoint.name, "model": adapter.metadata,
              "closed_loop_success_evidence": False,
              "limitations": ["One expert observation/chunk is a feasibility probe, not a trained benchmark policy.",
                              "Dataset time alignment remains the source metadata's explicit probe assumption.",
                              "No claim of better robot task success; no benchmark episode is run here."]}
    session.checkpoint("probe_checks_complete")
    write_json(session.output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/action_adaptation/probe_v1/attempt_001")
    parser.add_argument("--mode", choices=("lora", "output_head"), default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.metadata = args.metadata or args.example.parent / "metadata.json"
    session = ProbeSession(args.output)
    try:
        require(args.mode == "lora", "output_head is not implemented; no model will be loaded")
        require(args.rank == 8 and args.steps == 3, "This probe fixes rank=8 and exactly three optimizer steps")
        require(0 <= args.seed < 2**32, "Seed must fit an unsigned 32-bit integer")
        session.begin()
        result = run_probe(args, session)
        session.finish("passed", report="report.json", completed_steps=result["optimizer_steps"])
        return 0
    except BaseException as error:
        try:
            session.finish("resource_stopped" if isinstance(error, ResourceStop) else "failed",
                           error=repr(error), traceback=traceback.format_exc())
        except BaseException as persistence_error:
            print(json.dumps({"error": repr(error), "persistence_error": repr(persistence_error)}), flush=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({"status": "failed", "error": repr(error), "output": str(session.output)}), flush=True)
        return 75 if isinstance(error, ResourceStop) else 1


if __name__ == "__main__":
    raise SystemExit(main())
