"""Protocol-bound, single-model CPU action adaptation with exact safe resume.

The parent supervisor owns process termination during blocked numerical calls.
This worker never launches children, chooses a checkpoint by validation/test
performance, changes a sampling plan, or downloads models/data. Heavy imports
occur only after source/data verification and the resource preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading
import time
import traceback
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "feedbackbench.action_adaptation_train.v1"
CHECKPOINT_SCHEMA = "feedbackbench.action_adaptation_checkpoint.v1"
FLOW_DISTRIBUTION = "beta_1.5_1_scaled_0.999_plus_0.001"
DEADLINE = datetime(2026, 10, 30, 6, 8, tzinfo=timezone.utc)
GIB = 1024 ** 3
START_AVAILABLE = int(4.25 * GIB)
RUNTIME_AVAILABLE = int(.75 * GIB)
MIN_DISK = 4 * GIB
REQUIRED_SOURCES = {
    "scripts/train_action_adaptation.py", "src/embodied_research/action_adaptation.py",
    "src/embodied_research/action_head_baseline.py", "src/embodied_research/policies.py",
    "src/embodied_research/runtime.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def json_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    # Keep below legacy Windows MAX_PATH even under nested campaign attempts.
    temporary = path.with_name("." + uuid4().hex[:12] + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt, delay in enumerate((.05, .1, .2, .4, .5, 0)):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(delay)


def relative_file(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "Expected a relative file path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve() and path.is_file(),
            f"Missing/outside-root file: {name}")
    return path


def checked_file(root, record):
    path = relative_file(root, record["path"])
    require(re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) is not None,
            "Expected lowercase SHA256")
    require(digest(path) == record["sha256"], f"Input hash differs: {record['path']}")
    return path


def positive_int(value, label):
    require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def seed_value(value):
    require(type(value) is int and 0 <= value < 2**63, "Invalid seed")
    return value


def verify_protocol(protocol_path, expected_sha256, run_id, *, root=ROOT):
    """Verify all bound files once on every start/resume, before model import."""
    protocol_path, root = Path(protocol_path).resolve(), Path(root).resolve()
    require(digest(protocol_path) == expected_sha256, "Frozen protocol SHA256 differs")
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA and protocol.get("frozen") is True, "Protocol is not frozen v1")
    require(datetime.fromisoformat(protocol["deadline_utc"].replace("Z", "+00:00")) == DEADLINE,
            "Protocol cannot extend the hard deadline")
    require(protocol.get("flow_time_distribution") == FLOW_DISTRIBUTION, "Unexpected flow-time distribution")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) is not None, "Invalid run_id")
    run = protocol["runs"][run_id]
    require(run["mode"] in ("lora", "output_head"), "Unknown trainable module mode")
    seed_value(run["initialization_seed"])
    seed_value(run["flow_seed"])
    positive_int(run["steps"], "steps")
    positive_int(protocol["checkpoint_every"], "checkpoint_every")
    positive_int(protocol["validation_every"], "validation_every")
    resources = protocol["resources"]
    require(resources["torch_threads"] == 2 and resources["interop_threads"] == 1, "Expected 2/1 CPU threads")
    positive_int(resources["max_seconds"], "max_seconds")
    sources = protocol["source_files"]
    require(REQUIRED_SOURCES <= set(sources), "Protocol is missing a required source binding")
    inventory = {}
    for category, records in (("source", sources), ("model", protocol["model"]["files"])):
        require(isinstance(records, dict) and records, f"Missing {category} file inventory")
        for path, checksum in records.items():
            checked_file(root, {"path": path, "sha256": checksum})
            inventory[path] = checksum
    models_dir = (root / protocol["model"]["models_dir"]).resolve()
    require(models_dir.is_relative_to(root) and models_dir.is_dir(), "Invalid model directory")
    # Every file declared by the checkpoint manifests and their manifests must
    # also be bound by the new protocol; mutable local metadata is insufficient.
    for name in ("smolvla", "smolvlm_assets"):
        manifest_path = models_dir / name / "download_manifest.json"
        model_manifest = read_json(manifest_path)
        for path in [manifest_path, *(models_dir / name / item["file"] for item in model_manifest["files"])]:
            relative = path.relative_to(root).as_posix()
            require(relative in protocol["model"]["files"], f"Unbound model input: {relative}")
    positive_int(protocol["model"]["num_steps"], "prediction integration steps")
    optimizer = protocol["optimizer"]
    require(optimizer["type"] == "AdamW", "Only fixed AdamW is supported")
    for name in ("lr", "eps", "weight_decay"):
        value = optimizer[name]
        require(type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if name == "weight_decay" else value > 0), f"Invalid optimizer {name}")
    require(isinstance(optimizer["betas"], list) and len(optimizer["betas"]) == 2
            and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1
                    for value in optimizer["betas"]), "Invalid AdamW betas")
    data = protocol["data"]
    require(type(data["action_offset"]) is int and data["action_offset"] in (0, 1), "Invalid action alignment")
    require(data["image_transform"] == "rot180_once", "Unexpected image transform")
    alignment = read_json(checked_file(root, data["alignment_evidence"]))
    require(alignment.get("status") == "passed" and alignment.get("action_offset") == data["action_offset"],
            "Training requires passed replay evidence for the exact action offset")
    manifest = read_json(checked_file(root, data["manifest"]))
    demos, episode_splits, source_hashes = {}, {}, {}
    for demo in manifest["demos"]:
        require(isinstance(demo["id"], str) and demo["id"] and demo["id"] not in demos, "Duplicate/invalid demo id")
        require(demo["split"] in ("train", "validation"), "Training worker only accepts train/validation demos")
        require(isinstance(demo["task"], str) and demo["task"] and isinstance(demo["language"], str)
                and demo["language"].strip(), "Missing task/language")
        require(re.fullmatch(r"demo_\d+", demo["episode_key"]) is not None, "Invalid demo key")
        positive_int(demo["frame_count"], "frame_count")
        path = relative_file(root, demo["path"])
        # Copies/symlinks of the same source file cannot evade whole-demo splits.
        key = (demo["sha256"], demo["episode_key"])
        require(key not in episode_splits, "A whole demonstration occurs more than once or crosses splits")
        episode_splits[key] = demo["split"]
        if path not in source_hashes:
            checked_file(root, demo)
            source_hashes[path] = demo["sha256"]
        require(source_hashes[path] == demo["sha256"], "Conflicting source hashes")
        demos[demo["id"]] = demo
        inventory[demo["path"]] = demo["sha256"]
    require({demo["split"] for demo in demos.values()} == {"train", "validation"}, "Both whole-demo splits required")
    schedule = read_json(checked_file(root, run["schedule"]))
    require(len(schedule["train"]) == run["steps"] and schedule["validation"], "Sampling schedule size differs")
    for split in ("train", "validation"):
        for item in schedule[split]:
            demo = demos[item["demo_id"]]
            frame = item["observation_index"]
            require(demo["split"] == split, "Sampling schedule crosses whole-demo split")
            require(type(frame) is int and 0 <= frame < demo["frame_count"] - data["action_offset"],
                    "Observation has no aligned real action")
            if split == "validation":
                seed_value(item["noise_seed"])
                require(type(item["time"]) in (int, float) and math.isfinite(item["time"])
                        and .001 <= item["time"] <= 1, "Invalid fixed validation flow time")
    binding = {"protocol_sha256": expected_sha256, "run_id": run_id,
               "schedule_sha256": run["schedule"]["sha256"], "data_manifest_sha256": data["manifest"]["sha256"],
               "alignment_sha256": data["alignment_evidence"]["sha256"], "input_files": inventory}
    return {"protocol": protocol, "run": run, "demos": demos, "schedule": schedule,
            "binding": binding, "binding_sha256": json_digest(binding), "root": root}


def read_demo_example(plan, item):
    """Read one observation and its aligned human-action chunk, never object state."""
    import h5py
    import numpy as np

    demo = plan["demos"][item["demo_id"]]
    frame = item["observation_index"]
    offset = plan["protocol"]["data"]["action_offset"]
    with h5py.File(plan["root"] / demo["path"], "r") as dataset:
        episode = dataset[f"data/{demo['episode_key']}"]
        actions, obs = episode["actions"], episode["obs"]
        count = demo["frame_count"]
        require(actions.shape == (count, 7) and int(episode.attrs["num_samples"]) == count,
                "Demonstration action count changed")
        for name, tail in (("ee_states", (6,)), ("gripper_states", (2,))):
            require(obs[name].shape == (count, *tail), f"Unexpected {name} shape")
        example = {"task": demo["language"]}
        for name, source in (("image", "agentview_rgb"), ("image2", "eye_in_hand_rgb")):
            pixels = obs[source][frame]
            require(obs[source].shape[0] == count and pixels.dtype == np.uint8 and pixels.ndim == 3
                    and pixels.shape[-1] == 3, "Expected uint8 RGB HWC demonstration images")
            example[name] = np.ascontiguousarray(pixels[::-1, ::-1])
        example["state"] = np.concatenate((obs["ee_states"][frame], obs["gripper_states"][frame])).astype(np.float32)
        start = frame + offset
        valid = min(50, count - start)
        require(valid > 0, "No valid aligned action")
        chunk = np.repeat(np.asarray(actions[-1:], dtype=np.float32), 50, axis=0)
        chunk[:valid] = actions[start:start + valid]
        example["actions"] = chunk
        example["action_is_pad"] = np.arange(50) >= valid
    require(np.isfinite(example["state"]).all() and np.isfinite(example["actions"]).all(), "Nonfinite demonstration")
    return example


def preprocess_example(adapter, example, *, verify_observation=False):
    import numpy as np
    import torch

    observation = {"task": example["task"], "observation.state": example["state"].copy()}
    for name in ("image", "image2"):
        observation[f"observation.images.{name}"] = (
            np.ascontiguousarray(example[name].transpose(2, 0, 1), dtype=np.float32) / np.float32(255))
    raw = {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
           for key, value in observation.items()}
    raw["action"] = torch.from_numpy(example["actions"].copy()).unsqueeze(0)
    raw["action_is_pad"] = torch.from_numpy(example["action_is_pad"].copy()).unsqueeze(0)
    adapter.preprocess.reset()
    batch = adapter.preprocess(raw)
    if verify_observation:
        adapter.preprocess.reset()
        reference = adapter._prepare_observation(observation)
        require(all(torch.equal(value, batch[key]) for key, value in reference.items()
                    if isinstance(value, torch.Tensor)), "Training observation differs from original preprocessing")
    require(batch["action"].shape == (1, 50, 7) and batch["action_is_pad"].shape == (1, 50),
            "Original processor changed action/mask dimensions")
    require(batch["action"].dtype == torch.float32 and torch.isfinite(batch["action"]).all().item()
            and batch["action_is_pad"].dtype == torch.bool, "Invalid normalized action/mask")
    return batch


class FlowRandom:
    """Separate reproducible streams for new standard-normal noise and Beta time."""

    def __init__(self, seed):
        import numpy as np
        import torch
        self.noise = torch.Generator(device="cpu").manual_seed(seed_value(seed))
        self.time = np.random.Generator(np.random.PCG64(seed))

    def sample(self):
        import torch
        time_tensor = torch.tensor([self.time.beta(1.5, 1.0) * .999 + .001], dtype=torch.float32)
        noise = torch.randn((1, 50, 32), generator=self.noise, dtype=torch.float32)
        return noise, time_tensor


def capture_rng(flow):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    arrays = {"torch_cpu": torch.get_rng_state().numpy().copy(), "flow_noise": flow.noise.get_state().numpy().copy(),
              "numpy_global_keys": numpy_state[1].copy()}
    metadata = {"flow_time": flow.time.bit_generator.state, "python": random.getstate(),
                "numpy_global": [numpy_state[0], numpy_state[2], numpy_state[3], numpy_state[4]]}
    return arrays, metadata


def restore_rng(flow, arrays, metadata):
    import numpy as np
    import torch
    require(set(arrays) == {"torch_cpu", "flow_noise", "numpy_global_keys"}, "RNG inventory differs")
    for name in ("torch_cpu", "flow_noise"):
        require(arrays[name].dtype == np.uint8 and arrays[name].ndim == 1, "Invalid Torch RNG state")
    require(arrays["numpy_global_keys"].dtype == np.uint32 and arrays["numpy_global_keys"].shape == (624,),
            "Invalid NumPy RNG state")
    require(metadata["flow_time"].get("bit_generator") == "PCG64", "Unexpected time RNG algorithm")
    torch.set_rng_state(torch.from_numpy(arrays["torch_cpu"].copy()))
    flow.noise.set_state(torch.from_numpy(arrays["flow_noise"].copy()))
    flow.time.bit_generator.state = metadata["flow_time"]
    state = metadata["numpy_global"]
    np.random.set_state((state[0], arrays["numpy_global_keys"].copy(), state[1], state[2], state[3]))
    python_state = metadata["python"]
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))


def make_optimizer(adapter, config):
    import torch
    return torch.optim.AdamW(list(adapter.parameters()), lr=config["lr"], betas=tuple(config["betas"]),
                             eps=config["eps"], weight_decay=config["weight_decay"], amsgrad=False,
                             foreach=False, fused=False)


def optimizer_arrays(optimizer, adapter):
    import torch
    named = list(adapter.named_parameters())
    require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == len(named)
            and all(left is right for left, (_, right) in zip(optimizer.param_groups[0]["params"], named, strict=True)),
            "Optimizer parameter order differs from adapter")
    arrays, present = {}, []
    for index, (_, parameter) in enumerate(named):
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected AdamW state fields")
        present.append(index)
        for name, value in state.items():
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32
                    and torch.isfinite(value).all().item(), "Optimizer state must be finite CPU FP32")
            arrays[f"p{index:03d}_{name}"] = value.detach().numpy().copy()
    group = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    return arrays, {"parameter_names": [name for name, _ in named], "present": present, "group": group}


def restore_optimizer(optimizer, adapter, arrays, metadata, completed_steps):
    import numpy as np
    import torch
    _, expected = optimizer_arrays(optimizer, adapter)
    require(metadata["parameter_names"] == expected["parameter_names"]
            and canonical(metadata["group"]) == canonical(expected["group"]), "Optimizer identity/configuration differs")
    named = list(adapter.named_parameters())
    require(metadata["present"] == (list(range(len(named))) if completed_steps else []),
            "Optimizer state is incomplete for this step")
    expected_keys = {f"p{index:03d}_{name}" for index in metadata["present"]
                     for name in ("step", "exp_avg", "exp_avg_sq")}
    require(set(arrays) == expected_keys, "Optimizer array inventory differs")
    restored = []
    for index in metadata["present"]:
        parameter = named[index][1]
        state = {}
        for name in ("step", "exp_avg", "exp_avg_sq"):
            array = arrays[f"p{index:03d}_{name}"]
            require(array.dtype == np.float32 and array.shape == (() if name == "step" else tuple(parameter.shape))
                    and np.isfinite(array).all(), "Invalid optimizer tensor")
            if name == "step":
                require(float(array) == completed_steps, "Optimizer step counter differs")
            if name == "exp_avg_sq":
                require((array >= 0).all(), "Invalid second moment")
            state[name] = torch.from_numpy(array.copy())
        restored.append((parameter, state))
    optimizer.state.clear()
    for parameter, state in restored:
        optimizer.state[parameter] = state


def load_numeric(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    require(all(value.dtype.kind in "biuf" for value in result.values()), "Checkpoint arrays must be numeric")
    return result


def runtime_environment():
    from importlib.metadata import version
    import torch
    return {"python": sys.version, "platform": sys.platform,
            "packages": {name: version(name) for name in ("torch", "numpy", "h5py", "transformers")},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def save_checkpoint(output, *, step, binding, adapter, optimizer, flow, history, frozen_checksum):
    """One immutable step directory; incomplete directories remain as evidence."""
    import numpy as np
    parent = Path(output) / "checkpoints"
    parent.mkdir(exist_ok=True)
    final = parent / f"step_{step:06d}"
    require(not final.exists(), "Completed step checkpoint already exists; refusing overwrite")
    temporary = parent / f".s{step:06d}_{uuid4().hex[:12]}"
    temporary.mkdir()
    try:
        adapter.save(temporary / "adapter.npz", metadata={"binding_sha256": json_digest(binding), "step": step,
                                                          "protocol_sha256": binding["protocol_sha256"],
                                                          "run_id": binding["run_id"]})
        optimizer_values, optimizer_metadata = optimizer_arrays(optimizer, adapter)
        rng_values, rng_metadata = capture_rng(flow)
        np.savez_compressed(temporary / "optimizer.npz", **optimizer_values)
        np.savez_compressed(temporary / "rng.npz", **rng_values)
        state = {"completed_steps": step, "optimizer": optimizer_metadata, "rng": rng_metadata,
                 "frozen_checksum": frozen_checksum, "base_fingerprint": adapter.base_fingerprint,
                 "runtime_environment": runtime_environment()}
        write_json(temporary / "state.json", state)
        write_json(temporary / "history.json", history)
        files = {name: digest(temporary / name)
                 for name in ("adapter.npz", "optimizer.npz", "rng.npz", "state.json", "history.json")}
        receipt = {"schema": CHECKPOINT_SCHEMA, "completed_steps": step, "binding_sha256": json_digest(binding),
                   "binding": binding, "files": files, "created_at": datetime.now(timezone.utc).isoformat(),
                   "checkpoint_selection": "fixed interval or final step; never selected by validation/test"}
        receipt["receipt_sha256"] = json_digest(receipt)
        write_json(temporary / "receipt.json", receipt)
        # Never replace an existing complete checkpoint, including a concurrent writer.
        require(not final.exists(), "Checkpoint appeared during save")
        os.rename(temporary, final)
        return final
    except BaseException as error:
        if temporary.exists():
            write_json(temporary / "failure.json", {"error": repr(error), "step": step})
        raise


def inspect_checkpoint(path, binding):
    path = Path(path).resolve()
    receipt = read_json(path / "receipt.json")
    checksum = receipt.pop("receipt_sha256", None)
    require(checksum == json_digest(receipt) and receipt.get("schema") == CHECKPOINT_SCHEMA,
            "Checkpoint receipt is invalid")
    require(receipt["binding_sha256"] == json_digest(binding) and receipt["binding"] == binding,
            "Checkpoint protocol/run/input binding differs")
    require(set(receipt["files"]) == {"adapter.npz", "optimizer.npz", "rng.npz", "state.json", "history.json"},
            "Checkpoint file inventory differs")
    require({item.name for item in path.iterdir()} == set(receipt["files"]) | {"receipt.json"},
            "Checkpoint contains missing or unbound artifacts")
    for name, checksum in receipt["files"].items():
        require(digest(path / name) == checksum, f"Checkpoint file hash differs: {name}")
    state, history = read_json(path / "state.json"), read_json(path / "history.json")
    require(type(state["completed_steps"]) is int and state["completed_steps"] >= 0
            and state["completed_steps"] == receipt["completed_steps"], "Checkpoint step differs")
    require([record["step"] for record in history["training"]] == list(range(1, state["completed_steps"] + 1)),
            "Checkpoint training history is incomplete")
    return state, history


def restore_checkpoint(path, *, binding, adapter, optimizer, flow, frozen_checksum):
    state, history = inspect_checkpoint(path, binding)
    require(state["frozen_checksum"] == frozen_checksum and state["base_fingerprint"] == adapter.base_fingerprint,
            "Checkpoint base model binding differs")
    require(state["runtime_environment"] == runtime_environment(), "Exact resume runtime environment differs")
    optimizer_values, rng_values = load_numeric(Path(path) / "optimizer.npz"), load_numeric(Path(path) / "rng.npz")
    # The receipt binds every artifact before any state is copied into the model.
    adapter.reload(Path(path) / "adapter.npz")
    restore_optimizer(optimizer, adapter, optimizer_values, state["optimizer"], state["completed_steps"])
    restore_rng(flow, rng_values, state["rng"])
    return state["completed_steps"], history


def validate_resume_path(output, checkpoint, binding):
    """Accept the latest sealed step from this output or a sibling run attempt."""
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    previous_output = checkpoint.parent.parent
    require(checkpoint.parent.name == "checkpoints" and re.fullmatch(r"step_\d{6}", checkpoint.name),
            "Invalid resume checkpoint directory")
    if output == previous_output:
        outputs = [output]
    else:
        require(output.parent == previous_output.parent and re.fullmatch(r"attempt_\d+", output.name)
                and re.fullmatch(r"attempt_\d+", previous_output.name), "Resume must stay within the same run")
        outputs = [path for path in output.parent.glob("attempt_*") if re.fullmatch(r"attempt_\d+", path.name)]
    candidates = []
    for directory in outputs:
        for path in (directory / "checkpoints").glob("step_*"):
            if re.fullmatch(r"step_\d{6}", path.name):
                state, _ = inspect_checkpoint(path, binding)
                candidates.append((state["completed_steps"], str(path), path))
    require(candidates and checkpoint == max(candidates)[2], "Resume must use latest complete step, never roll back")


def last_attempt_seconds(resource_log):
    with resource_log.open("rb") as stream:
        stream.seek(max(0, resource_log.stat().st_size - 4096))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return float(json.loads(line)["attempt_elapsed_seconds"])
        except (ValueError, KeyError):
            # A killed worker may leave the final line incomplete. Earlier
            # complete samples and all checkpoint receipts remain untouched.
            continue
    return 0.0


class ResourceStop(RuntimeError):
    pass


def resource_reason(sample, max_seconds, *, initial=False):
    if datetime.fromisoformat(sample["at"]) >= DEADLINE:
        return "hard_deadline"
    if sample["elapsed_seconds"] >= max_seconds:
        return "protocol_runtime_limit"
    if sample["available_bytes"] < (START_AVAILABLE if initial else RUNTIME_AVAILABLE):
        return "start_memory_threshold" if initial else "runtime_memory_reserve"
    if sample["disk_free_bytes"] < MIN_DISK:
        return "disk_reserve"
    return None


class TrainingSession:
    def __init__(self, output, plan, *, resume):
        import psutil
        self.output = Path(output).resolve()
        allowed = (ROOT / "outputs/action_adaptation").resolve()
        require(self.output.is_relative_to(allowed) and self.output != allowed, "Output is outside action_adaptation")
        if resume and self.output == Path(resume).resolve().parent.parent:
            require(self.output.is_dir(), "Resume output does not exist")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
        self.prior_seconds = 0.0
        previous_outputs = [self.output]
        if resume and re.fullmatch(r"attempt_\d+", self.output.name):
            previous_outputs = [path for path in self.output.parent.glob("attempt_*")
                                if re.fullmatch(r"attempt_\d+", path.name)]
        for resource_log in (log for directory in previous_outputs
                             for log in (directory / "attempts").glob("*/resources.jsonl")):
            # Use the last flushed resource sample even after a killed worker;
            # resuming does not grant another full protocol runtime allowance.
            self.prior_seconds += last_attempt_seconds(resource_log)
        self.attempt = self.output / "attempts" / uuid4().hex
        self.attempt.mkdir(parents=True)
        self.started, self.done, self.lock = time.monotonic(), threading.Event(), threading.RLock()
        self.process, self.psutil = psutil.Process(), psutil
        self.max_seconds = plan["protocol"]["resources"]["max_seconds"]
        self.stop_reason, self.thread = None, None
        self.peak_rss, self.minimum_available = 0, None
        self.log = (self.attempt / "events.jsonl").open("x", encoding="utf-8")
        self.resources = (self.attempt / "resources.jsonl").open("x", encoding="utf-8")
        self.state = {"schema": SCHEMA, "status": "starting", "stage": "preflight", "binding": plan["binding"],
                      "attempt": self.attempt.name, "resume_from": str(resume) if resume else None,
                      "owner": {"pid": self.process.pid, "created": self.process.create_time()},
                      "started_at": datetime.now(timezone.utc).isoformat(), "worker_launches_children": False,
                      "limits": {"deadline_utc": DEADLINE.isoformat(), "max_seconds": self.max_seconds,
                                 "start_available_bytes": START_AVAILABLE, "runtime_available_bytes": RUNTIME_AVAILABLE,
                                 "min_disk_bytes": MIN_DISK}}

    def sample(self):
        elapsed = time.monotonic() - self.started
        return {"at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": self.prior_seconds + elapsed,
                "attempt_elapsed_seconds": elapsed,
                "rss_bytes": self.process.memory_info().rss, "available_bytes": self.psutil.virtual_memory().available,
                "disk_free_bytes": shutil.disk_usage(self.output).free}

    def record_resource(self):
        sample = self.sample()
        with self.lock:
            self.resources.write(canonical(sample) + "\n")
            self.resources.flush()
            self.peak_rss = max(self.peak_rss, sample["rss_bytes"])
            self.minimum_available = (sample["available_bytes"] if self.minimum_available is None
                                      else min(self.minimum_available, sample["available_bytes"]))
            self.stop_reason = self.stop_reason or resource_reason(sample, self.max_seconds)
        return sample

    def _monitor(self):
        while not self.done.wait(.5):
            try:
                self.record_resource()
            except BaseException as error:
                self.stop_reason = self.stop_reason or f"resource_sampling_failed: {error!r}"
                return

    def begin(self):
        sample = self.record_resource()
        reason = resource_reason(sample, self.max_seconds, initial=True)
        if reason:
            raise ResourceStop(reason)
        self.checkpoint("resource_preflight_passed")
        self.thread = threading.Thread(target=self._monitor, daemon=True, name="action-training-resource-sampler")
        self.thread.start()

    def checkpoint(self, stage, **details):
        sample = self.record_resource()
        with self.lock:
            if (self.output / "STOP_REQUEST").exists():
                self.stop_reason = self.stop_reason or "external_stop_request"
            if self.stop_reason:
                raise ResourceStop(self.stop_reason)
            self.state.update(status="running", stage=stage, updated_at=sample["at"], last_resource_sample=sample,
                              **details)
            self.log.write(canonical({"stage": stage, "at": sample["at"], **details}) + "\n")
            self.log.flush()
            write_json(self.attempt / "status.json", self.state)
            write_json(self.output / "status.json", self.state)

    def finish(self, status, **details):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            self.state.update(status=status, ended_at=datetime.now(timezone.utc).isoformat(),
                              elapsed_seconds=self.prior_seconds + time.monotonic() - self.started,
                              attempt_elapsed_seconds=time.monotonic() - self.started, peak_rss_bytes=self.peak_rss,
                              minimum_available_bytes=self.minimum_available, **details)
            try:
                write_json(self.attempt / "status.json", self.state)
                write_json(self.output / "status.json", self.state)
            finally:
                self.log.close()
                self.resources.close()


def validate_fixed(adapter, plan, flow, loss_function, session, *, step):
    import torch
    saved_arrays, saved_metadata = capture_rng(flow)
    records = []
    try:
        adapter.policy.eval()
        with torch.no_grad():
            for index, item in enumerate(plan["schedule"]["validation"]):
                session.checkpoint("validation", completed_steps=step, validation_index=index)
                batch = preprocess_example(adapter, read_demo_example(plan, item))
                noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(item["noise_seed"]))
                times = torch.tensor([item["time"]], dtype=torch.float32)
                loss, diagnostics = loss_function(adapter.policy, batch, noise=noise, time=times)
                records.append({**item, "loss": float(loss), "valid_action_steps": diagnostics["valid_action_steps"]})
    finally:
        restore_rng(flow, saved_arrays, saved_metadata)
    denominator = sum(item["valid_action_steps"] for item in records)
    return {"step": step, "samples": records,
            "mean_loss_per_valid_action": sum(item["loss"] * item["valid_action_steps"] for item in records) / denominator,
            "selection_use": "diagnostic only; no checkpoint selection or early stopping"}


def run_training(plan, session, *, resume=None):
    require(not (session.output / "report.json").exists(), "A completed run report already exists")
    from embodied_research.runtime import configure_runtime
    configure_runtime(ROOT)
    import numpy as np
    import torch
    from embodied_research import action_adaptation as learning
    from embodied_research.policies import PolicyAdapter

    protocol, run = plan["protocol"], plan["run"]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(run["initialization_seed"])
    np.random.seed(run["initialization_seed"] % 2**32)
    random.seed(run["initialization_seed"])
    session.checkpoint("loading_model")
    model = PolicyAdapter("smolvla", models_dir=plan["root"] / protocol["model"]["models_dir"],
                          num_steps=protocol["model"]["num_steps"])
    model.policy.eval()
    session.checkpoint("hashing_base")
    base_hash = learning.frozen_base_checksum(model.policy)
    if run["mode"] == "lora":
        trainable = learning.install_action_lora(model.policy, base_fingerprint=base_hash,
                                                seed=run["initialization_seed"])
        audit, frozen_hash = learning.gradient_audit, learning.frozen_base_checksum
    else:
        from embodied_research import action_head_baseline as head
        trainable = head.install_output_head(model.policy, base_fingerprint=base_hash,
                                             seed=run["initialization_seed"])
        audit, frozen_hash = head.gradient_audit, head.frozen_rest_checksum
    frozen_before = frozen_hash(model.policy)
    optimizer, flow = make_optimizer(trainable, protocol["optimizer"]), FlowRandom(run["flow_seed"])
    completed, history = 0, {"training": [], "validation": []}
    if resume:
        completed, history = restore_checkpoint(resume, binding=plan["binding"], adapter=trainable,
                                                optimizer=optimizer, flow=flow, frozen_checksum=frozen_before)
        require(completed <= run["steps"], "Checkpoint exceeds the protocol's final step")
        session.checkpoint("resumed_exact_state", completed_steps=completed)
        if completed == run["steps"]:
            # Recover a worker interrupted after sealing the last checkpoint but
            # before publishing its report; do not take an extra optimizer step.
            checkpoint = session.output / "checkpoints" / Path(resume).name
            if checkpoint.resolve() != Path(resume).resolve():
                checkpoint.parent.mkdir(exist_ok=True)
                temporary = checkpoint.with_name(f".s{completed:06d}_{uuid4().hex[:12]}")
                shutil.copytree(resume, temporary)
                inspect_checkpoint(temporary, plan["binding"])
                require(not checkpoint.exists(), "Final checkpoint already exists")
                os.rename(temporary, checkpoint)
    else:
        history["validation"].append(validate_fixed(model, plan, flow, learning.compute_full_loss, session, step=0))
        checkpoint = save_checkpoint(session.output, step=0, binding=plan["binding"], adapter=trainable,
                                     optimizer=optimizer, flow=flow, history=history, frozen_checksum=frozen_before)
        session.checkpoint("checkpoint_saved", completed_steps=0, checkpoint=str(checkpoint.relative_to(session.output)))
    for index in range(completed, run["steps"]):
        step, item = index + 1, plan["schedule"]["train"][index]
        session.checkpoint("training_step", completed_steps=index, next_step=step, sample=item)
        started = time.perf_counter()
        example = read_demo_example(plan, item)
        batch = preprocess_example(model, example, verify_observation=(index == completed))
        noise, times = flow.sample()
        optimizer.zero_grad(set_to_none=True)
        model.policy.eval()
        tick = time.perf_counter()
        loss, diagnostics = learning.compute_full_loss(model.policy, batch, noise=noise, time=times)
        forward_seconds = time.perf_counter() - tick
        session.checkpoint("backward", completed_steps=index, next_step=step)
        tick = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - tick
        gradients = audit(model.policy)
        require(gradients["ok"], "Training gradient audit failed")
        session.checkpoint("optimizer_update", completed_steps=index, next_step=step)
        optimizer.step()
        require(all(torch.isfinite(value).all().item() for value in trainable.parameters()), "Nonfinite adapter update")
        record = {"step": step, **item, "loss": float(loss.detach()), "diagnostics": diagnostics,
                  "flow_time": float(times[0]), "noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                  "gradient_audit": gradients, "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                  "step_seconds": time.perf_counter() - started}
        history["training"].append(record)
        write_json(session.attempt / f"step_{step:06d}.json", record)
        optimizer.zero_grad(set_to_none=True)
        # Drop the completed graph before validation/checkpoint bookkeeping.
        del loss, batch, example, noise, times
        if step % protocol["validation_every"] == 0 or step == run["steps"]:
            validation = validate_fixed(model, plan, flow, learning.compute_full_loss, session, step=step)
            history["validation"].append(validation)
            write_json(session.attempt / f"validation_{step:06d}.json", validation)
        if step % protocol["checkpoint_every"] == 0 or step == run["steps"]:
            session.checkpoint("checking_frozen_base", completed_steps=step)
            require(frozen_hash(model.policy) == frozen_before, "Frozen model parameters or buffers changed")
            checkpoint = save_checkpoint(session.output, step=step, binding=plan["binding"], adapter=trainable,
                                         optimizer=optimizer, flow=flow, history=history, frozen_checksum=frozen_before)
            session.checkpoint("checkpoint_saved", completed_steps=step,
                               checkpoint=str(checkpoint.relative_to(session.output)))
    relative_checkpoint = str(checkpoint.relative_to(session.output)).replace("\\", "/")
    result = {"schema": SCHEMA, "status": "complete", "run_id": plan["binding"]["run_id"], "mode": run["mode"],
              "protocol_sha256": plan["binding"]["protocol_sha256"],
              "completed_steps": run["steps"], "binding_sha256": plan["binding_sha256"],
              "final_checkpoint": {"path": relative_checkpoint, "adapter": "adapter.npz",
                                   "sha256": digest(checkpoint / "adapter.npz")},
              "adapter_manifest": trainable.manifest(),
              "trainable_parameters": sum(parameter.numel() for parameter in trainable.parameters()),
              "base_fingerprint": base_hash, "frozen_parameters_unchanged": True,
              "runtime_environment": runtime_environment(),
              "training_path": "original policy.forward; no prefix caching", "policy_mode": "eval with adapter gradients",
              "flow_time_distribution": FLOW_DISTRIBUTION, "history": history,
              "validation_summary": history["validation"][-1],
              "resource_summary": {"peak_rss_bytes_this_attempt": session.peak_rss,
                                   "minimum_available_bytes_this_attempt": session.minimum_available,
                                   "elapsed_seconds_all_attempts": session.prior_seconds + time.monotonic() - session.started},
              "checkpoint_selection": "last protocol step; validation diagnostic only", "closed_loop_success_claim": False}
    write_json(session.output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    session = None
    try:
        plan = verify_protocol(args.protocol, args.protocol_sha256, args.run_id)
        if args.resume:
            validate_resume_path(args.output, args.resume, plan["binding"])
        session = TrainingSession(args.output, plan, resume=args.resume)
        session.begin()
        # Environment limits precede Torch imports; no shell or child process.
        os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
        result = run_training(plan, session, resume=args.resume)
        session.finish("complete", final_checkpoint=result["final_checkpoint"], completed_steps=result["completed_steps"])
        return 0
    except BaseException as error:
        if session is not None:
            session.finish("stopped" if isinstance(error, ResourceStop) else "failed", error=repr(error),
                           traceback=traceback.format_exc())
        else:
            # Input failures cannot claim a valid run directory; retain a distinct
            # small launch-failure receipt next to the requested output instead.
            parent = args.output.resolve().parent
            allowed = (ROOT / "outputs/action_adaptation").resolve()
            if parent.is_relative_to(allowed):
                parent.mkdir(parents=True, exist_ok=True)
                write_json(parent / f"{args.output.name}.preflight_failure_{uuid4().hex}.json",
                           {"status": "failed", "run_id": args.run_id, "error": repr(error),
                            "traceback": traceback.format_exc()})
        print(f"Training failed: {error}", file=sys.stderr)
        if isinstance(error, ResourceStop):
            return 75 if str(error) == "runtime_memory_reserve" else 3
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform
    configure_platform()
    raise SystemExit(main())
