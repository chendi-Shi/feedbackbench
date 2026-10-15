"""Paired, resumable experiments on frozen policies and published initial states."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np

from .scheduling import ChunkScheduler, SchedulerConfig


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_fingerprint(root):
    root = Path(root)
    algorithm_files = ("experiment.py", "policies.py", "runtime.py", "simulator.py", "scheduling.py")
    return {name: hashlib.sha256((root / "src/embodied_research" / name).read_bytes()).hexdigest()
            for name in algorithm_files}


def versions():
    result = {}
    for name in ("torch", "torchvision", "numpy", "mujoco", "robosuite", "hf-libero", "lerobot",
                 "transformers", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def artifact_provenance(root, model_names):
    """Bind each run to verified weights/assets and an unmodified upstream tree."""
    from .policies import verify_model_files

    upstream = root / "vendor/MINERVA"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    expected = "64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e"
    if commit != expected:
        raise ValueError(f"Unexpected upstream checkout: {commit}")
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Upstream tracked files changed; reproduce the pinned clean checkout first")
    names = set(model_names)
    if "smolvla" in names:
        names.add("smolvlm_assets")
    models = {}
    for name in sorted(names):
        manifest = verify_model_files(root / "models" / name, name)
        models[name] = {"revision": manifest["revision"], "files": {
            item["file"]: item["sha256"] for item in manifest["files"]}}
    assets = json.loads((root / "assets/libero/asset_manifest.json").read_text(encoding="utf-8"))
    asset_root = (root / "assets/libero").resolve()
    for item in assets["files"]:
        path = (asset_root / item["path"]).resolve()
        if not path.is_relative_to(asset_root) or not path.is_file():
            raise ValueError(f"Invalid asset path: {item['path']}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Asset size changed: {item['path']}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Asset content changed: {item['path']}")
    return {"upstream_commit": commit, "models": models, "assets_revision": assets["revision"],
            "assets_manifest_sha256": digest_json(assets)}


def episode_id(case):
    parts = [case[key] for key in ("model", "suite", "task_id", "init_state_id", "condition", "method")]
    label = "__".join(str(part) for part in parts)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Unsafe episode identifier")
    return label


def expand_plan(plan):
    """Materialize the entire evaluation denominator before the first rollout."""
    cases = []
    seen = set()
    for model in plan["models"]:
        for task_index, task in enumerate(plan["tasks"]):
            for initial_index, initial in enumerate(plan["initial_states"]):
                for condition_index, condition in enumerate(plan["conditions"]):
                    methods = model["methods"]
                    if plan.get("rotate_method_order", True):
                        offset = (task_index + initial_index + condition_index) % len(methods)
                        methods = methods[offset:] + methods[:offset]
                    for method in methods:
                        case = {
                            "model": model["name"], "suite": task["suite"], "task_id": task["task_id"],
                            "init_state_id": initial, "seed": int(plan.get("seed", 1000)),
                            "condition": condition, "method": method["name"],
                            "num_steps": int(method.get("num_steps", model.get("num_steps", 10))),
                            "scheduler": method.get("scheduler"),
                            "record_video": bool(task.get("record_video", False)
                                                 and initial == plan["initial_states"][0]),
                        }
                        case["id"] = episode_id(case)
                        if case["id"] in seen:
                            raise ValueError(f"Duplicate case: {case['id']}")
                        seen.add(case["id"])
                        cases.append(case)
    if not cases:
        raise ValueError("Empty experiment")
    return cases


class ObservationCondition:
    """Controlled sensor changes, independent of action/reward/success.

    delay4 delays BOTH camera and proprioceptive packets by four simulator ticks.
    Initial frames are repeated until the queue fills. This is a prescribed
    observation delay; it is not a real-time simulation of CPU inference stalls.
    """

    def __init__(self, name):
        if name not in ("clean", "delay4", "dim"):
            raise ValueError(f"Unknown sensor condition {name}")
        self.name = name
        self.history = deque(maxlen=5)

    @staticmethod
    def copy(observation):
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()}

    def apply(self, observation):
        current = self.copy(observation)
        if self.name == "delay4":
            self.history.append(current)
            return self.copy(self.history[0])
        if self.name == "dim":
            for key in ("observation.images.image", "observation.images.image2"):
                current[key] = current[key] * np.float32(0.65)
        return current


def run_episode(simulator, policy, case, destination):
    """Run the official full horizon (or official success terminal), then save."""
    from PIL import Image

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    observation = simulator.reset(seed=case["seed"], init_state_id=case["init_state_id"])
    policy.reset(seed=case["seed"])
    condition = ObservationCondition(case["condition"])
    scheduler = None
    if case["scheduler"] is not None:
        scheduler = ChunkScheduler(SchedulerConfig(**{**case["scheduler"], "horizon": simulator.horizon}))
    initial_state = observation["observation.state"].copy()
    first_frame = simulator.frame()
    Image.fromarray(first_frame).save(destination / "initial.png")
    Image.fromarray(simulator.frame("image2")).save(destination / "initial_wrist.png")
    frames = [Image.fromarray(first_frame)] if case.get("record_video") else None
    actions, states, model_seconds, prep_seconds, control_seconds, simulator_seconds = [], [], [], [], [], []
    replan_steps = []
    terminated = truncated = success = False
    for step in range(simulator.horizon):
        incoming = condition.apply(observation)
        control_start = time.perf_counter()
        if scheduler is None:
            previous_count = policy.inference_count
            action = policy.select_action(incoming)
            replanned = policy.inference_count > previous_count
        else:
            policy.observe(incoming)
            replanned = scheduler.should_replan(incoming)
            if replanned:
                scheduler.submit_chunk(policy.predict_chunk())
            action = scheduler.next_action()
        control_seconds.append(time.perf_counter() - control_start)
        timing = policy.last_timing
        model_seconds.append(timing["inference_seconds"] if replanned else 0.0)
        prep_seconds.append(timing["preprocess_seconds"])
        if replanned:
            replan_steps.append(step)
        actions.append(action.copy())
        states.append(observation["observation.state"].copy())
        env_start = time.perf_counter()
        observation, reward, terminated, truncated, info = simulator.step(action)
        simulator_seconds.append(time.perf_counter() - env_start)
        # Upstream LIBERO terminates on the official task predicate, not on a
        # surrogate coordinate distance. Read the published success flag.
        success = bool(info["is_success"])
        if frames is not None and ((step + 1) % 4 == 0 or terminated or truncated):
            frames.append(Image.fromarray(simulator.frame()))
        if (step + 1) % 50 == 0:
            print(f"    step={step + 1}/{simulator.horizon} calls={len(replan_steps)} "
                  f"elapsed={time.perf_counter() - begin:.1f}s", flush=True)
        if terminated or truncated:
            break
    rollout_wall = time.perf_counter() - begin
    Image.fromarray(simulator.frame()).save(destination / "final.png")
    if frames:
        frames[0].save(destination / "rollout.gif", save_all=True, append_images=frames[1:],
                       duration=200, loop=0, optimize=False)
    np.savez_compressed(destination / "trajectory.npz", actions=np.asarray(actions), states=np.asarray(states),
                        final_state=observation["observation.state"], model_seconds=np.asarray(model_seconds),
                        preprocessing_seconds=np.asarray(prep_seconds), control_seconds=np.asarray(control_seconds),
                        simulator_seconds=np.asarray(simulator_seconds), replan_steps=np.asarray(replan_steps))
    result = {
        **case, "status": "complete", "task": simulator.task, "success": success,
        "steps": simulator.steps, "horizon": simulator.horizon,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "inference_calls": len(replan_steps), "policy_inference_seconds": float(sum(model_seconds)),
        "preprocess_seconds": float(sum(prep_seconds)),
        "scheduler_seconds": float(max(0.0, sum(control_seconds) - sum(model_seconds) - sum(prep_seconds))),
        "scheduler_seconds_scope": "controller overhead: action postprocessing, queues, ensembling, scheduling and bookkeeping",
        "simulator_seconds": float(sum(simulator_seconds)), "wall_seconds": rollout_wall,
        "control_ms_p50": float(np.percentile(control_seconds, 50) * 1000),
        "control_ms_p95": float(np.percentile(control_seconds, 95) * 1000),
        "initial_robot_state": initial_state.tolist(),
        "initial_image_sha256": hashlib.sha256(first_frame.tobytes()).hexdigest(),
        "final_robot_state": observation["observation.state"].tolist(),
        "model_info": policy.metadata, "environment": simulator.metadata(),
        "scheduler_config": None if scheduler is None else asdict(scheduler.config),
        "decisions": None if scheduler is None else scheduler.diagnostics(),
        "video": "rollout.gif" if frames else None, "finished_at": utc_now(),
        "wall_time_scope": "reset + control + physics + camera rendering; excludes model load and GIF encoding",
    }
    return result


def run_plan(plan_path, output_dir, *, root=None, limit=None):
    """Resume exact matching plans only; a failed rollout remains visible."""
    from .runtime import compatibility_audit, configure_runtime

    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    output = Path(output_dir).resolve()
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    cases = expand_plan(plan)
    configure_runtime(root)
    import torch
    from .policies import PolicyAdapter
    from .simulator import LiberoSimulator

    torch.set_num_threads(int(plan.get("threads", 2)))
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    provenance = {"plan": plan, "sources": source_fingerprint(root), "runtime_versions": versions(),
                  "artifacts": artifact_provenance(root, [model["name"] for model in plan["models"]])}
    protocol = {**provenance, "cases": cases, "fingerprint": digest_json(provenance),
                "question": plan.get("question"), "hypothesis": plan.get("hypothesis"),
                "split": plan.get("split"), "limitations": plan.get("limitations", [])}
    protocol_file = output / "protocol.json"
    if protocol_file.exists():
        existing = json.loads(protocol_file.read_text(encoding="utf-8"))
        if existing["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Existing run has different source/configuration/dependencies; use a new output folder")
    else:
        write_json(protocol_file, {**protocol, "created_at": utc_now()})
    simulator = policy = None
    simulator_key = policy_key = None
    warmed_policy = False
    complete = 0
    try:
        for index, case in enumerate(cases):
            if limit is not None and complete >= limit:
                break
            directory = output / "episodes" / case["id"]
            result_path = directory / "result.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("status") != "complete":
                    raise RuntimeError(f"Prior failed case requires investigation: {case['id']}")
                if previous.get("protocol_fingerprint") != protocol["fingerprint"]:
                    raise ValueError(f"Episode provenance mismatch: {case['id']}")
                complete += 1
                continue
            write_json(output / "progress.json", {"completed": complete, "total": len(cases),
                                                   "current": case, "updated_at": utc_now()})
            print(f"[{index + 1}/{len(cases)}] {case['id']}", flush=True)
            try:
                next_policy_key = (case["model"], case["num_steps"])
                if next_policy_key != policy_key:
                    if policy is not None:
                        del policy
                    policy = PolicyAdapter(case["model"], num_steps=case["num_steps"])
                    policy_key = next_policy_key
                    warmed_policy = False
                next_simulator_key = (case["suite"], case["task_id"])
                if next_simulator_key != simulator_key:
                    if simulator is not None:
                        simulator.close()
                    simulator = LiberoSimulator(case["suite"], case["task_id"], project_root=root,
                                                image_size=int(plan.get("image_size", 360)))
                    simulator_key = next_simulator_key
                if not warmed_policy:
                    warmup_observation = simulator.reset(seed=case["seed"], init_state_id=case["init_state_id"])
                    policy.reset(seed=case["seed"])
                    for _ in range(int(plan.get("warmup_calls", 2))):
                        policy.select_action(warmup_observation)
                    warmed_policy = True
                    # run_episode resets both simulator and policy/RNG after warmup.
                result = run_episode(simulator, policy, case, directory)
                result["protocol_fingerprint"] = protocol["fingerprint"]
                write_json(result_path, result)
                complete += 1
                print(f"  success={result['success']} steps={result['steps']} "
                      f"calls={result['inference_calls']} wall={result['wall_seconds']:.1f}s", flush=True)
            except Exception as error:
                write_json(result_path, {**case, "status": "error", "error": str(error),
                                        "traceback": traceback.format_exc(), "finished_at": utc_now(),
                                        "protocol_fingerprint": protocol["fingerprint"]})
                raise
    finally:
        if simulator is not None:
            simulator.close()
        write_json(output / "runtime_audit.json", compatibility_audit())
        write_json(output / "progress.json", {"completed": complete, "total": len(cases),
                                               "finished": complete == len(cases), "updated_at": utc_now()})
    return {"completed": complete, "total": len(cases), "output": str(output)}
