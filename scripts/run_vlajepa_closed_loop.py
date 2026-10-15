"""Frozen, four-task CPU INT8 VLA-JEPA closed-loop feasibility diagnostic."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from embodied_research.experiment import (  # noqa: E402
    artifact_provenance, digest_json, episode_id, run_episode, utc_now, versions, write_json,
)
from run_cache_study import seal_episode, verify_complete  # noqa: E402
from run_flow_confirmation import file_sha, initial_state_manifest, read_json  # noqa: E402
from embodied_research.scheduling import SchedulerConfig  # noqa: E402

SPEC = importlib.util.spec_from_file_location("closed_loop_int8_loader", ROOT / "scripts/probe_vlajepa_int8.py")
quant = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quant)
base = quant.base
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (280, 280, 300, 520), strict=True))
VARIANT = "cpu_qwen_dynamic_int8_bf16_embedding_fp32_action"
MIN_FREE_GIB, MAX_RSS_GIB, MIN_RUNTIME_FREE_GIB = 5.0, 5.5, 0.5
SOURCES = (
    "scripts/run_vlajepa_closed_loop.py", "scripts/probe_vlajepa_int8.py", "scripts/probe_vlajepa.py",
    "scripts/run.py", "scripts/run_cache_study.py", "scripts/run_flow_confirmation.py",
    "src/embodied_research/__init__.py", "src/embodied_research/experiment.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
    "src/embodied_research/simulator.py", "src/embodied_research/scheduling.py",
    "docs/VLAJEPA_CLOSED_LOOP_PROTOCOL.md",
)
FEATURES = ("observation.images.image", "observation.images.image2", "observation.state")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def project_file(name):
    require(isinstance(name, str) and not Path(name).is_absolute() and not Path(name).drive
            and "\\" not in name and ":" not in name and ".." not in Path(name).parts,
            "Unsafe project-relative evidence path")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "Evidence file missing or outside project")
    return path


def cases_for_study():
    cases = []
    for suite in SUITES:
        case = {"model": "vlajepa_int8", "suite": suite, "task_id": 1, "init_state_id": 41,
                "seed": 1000, "condition": "clean", "method": "native4_chunk7", "num_steps": 4,
                "scheduler": {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False},
                "record_video": True, "split": "closed_loop_feasibility"}
        case["id"] = episode_id(case)
        cases.append(case)
    return cases


def study_spec():
    return {"name": "VLA-JEPA dynamic INT8 four-task closed-loop feasibility",
            "variant": VARIANT, "cases": cases_for_study(), "horizons": HORIZONS,
            "execution": {"threads": 2, "interop_threads": 1, "image_size": 360,
                          "chunk_size": 7, "flow_steps": 4, "warmup_calls": 0,
                          "preprocessor": "pinned serialized pipeline; no extra image orientation transform",
                          "postprocessor": "pinned clip, pre-snap, unnormalize, binarize; no extra transform"},
            "memory_guard": {"minimum_available_gib": MIN_FREE_GIB, "maximum_rss_gib": MAX_RSS_GIB,
                             "minimum_runtime_available_gib": MIN_RUNTIME_FREE_GIB, "sampling_seconds": 1},
            "limitations": ["four known tasks, one previously used initial state each; no held-out generalization claim",
                            "changed numerics, no original BF16 comparison or quantization quality attribution",
                            "different checkpoint, action interval and source from SmolVLA; not a matched comparison",
                            "simulation pauses during inference; no real robot or real-time control claim",
                            "all successes, task failures, incomplete and runtime errors retained separately"]}


def verify_protocol(output, root=ROOT):
    output, root = Path(output), Path(root)
    protocol = read_json(output / "protocol.json")
    body = {key: value for key, value in protocol.items() if key != "fingerprint"}
    require(digest_json(body) == protocol["fingerprint"], "Protocol fingerprint changed")
    require(protocol["spec"] == study_spec(), "Declared closed-loop design changed")
    require(set(protocol["sources"]) == set(SOURCES), "Source inventory changed")
    for name, checksum in protocol["sources"].items():
        require(file_sha(root / name) == checksum, f"Frozen source changed: {name}")
    require(file_sha(output / "inputs.json") == protocol["inputs_sha256"], "Pinned inputs snapshot changed")
    require(protocol["runtime_versions"] == versions(), "Installed runtime versions changed")
    return protocol


def freeze_protocol(output, root=ROOT):
    output, root = Path(output), Path(root)
    if (output / "protocol.json").exists():
        return verify_protocol(output, root)
    require(not output.exists(), "Existing unbound output directory retained; choose a new output")
    manifest = base.verify_manifest()
    dependencies = base.dependency_audit()
    require(not dependencies["failures"], "Pinned dependency audit failed")
    artifacts = artifact_provenance(root, [])
    body = {"schema": "feedbackbench.vlajepa_closed_loop_protocol.v1", "spec": study_spec(),
            "sources": {name: file_sha(root / name) for name in SOURCES}, "runtime_versions": versions(),
            "inputs_sha256": manifest["sha256"], "dependencies": dependencies,
            "simulation_artifacts": artifacts, "initial_states": initial_state_manifest(),
            "frozen_at": utc_now()}
    output.mkdir(parents=True)
    (output / "inputs.json").write_bytes((root / manifest["path"]).read_bytes())
    protocol = {**body, "fingerprint": digest_json(body)}
    write_json(output / "protocol.json", protocol)
    return verify_protocol(output, root)


def validate_observation(observation):
    require(set(observation) == {*FEATURES, "task"}, "Unexpected policy input; only cameras, robot state and task allowed")
    require(isinstance(observation["task"], str) and observation["task"].strip(), "Missing task instruction")
    for key in FEATURES:
        value = observation[key]
        require(isinstance(value, np.ndarray) and value.dtype == np.float32 and np.isfinite(value).all(),
                f"Invalid float32 observation: {key}")
        if key == "observation.state":
            require(value.shape == (8,), "Expected eight robot proprioception values")
        else:
            require(value.shape == (3, 360, 360) and value.min() >= 0 and value.max() <= 1,
                    "Expected pre-oriented CHW RGB [0,1] at frozen resolution")


def validate_action_chunk(value, *, environment=False):
    require(value.shape == (1, 7, 7) and value.dtype == np.float32 and np.isfinite(value).all(),
            "Expected finite float32 action chunk (1,7,7)")
    if environment:
        require(np.isin(value[..., 6], [-1.0, 1.0]).all(), "Official postprocessed gripper must be binary")


class ClosedLoopAdapter:
    """A seven-action scheduler adapter; all transforms remain in official pipelines."""

    def __init__(self, policy, preprocess, postprocess, official, identity):
        self.policy, self.preprocess, self.postprocess = policy, preprocess, postprocess
        self.official, self.identity = official, identity
        self.destination = None
        self.observation = None
        self.call_log = []
        self.inference_count = 0
        self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

    def reset(self, seed=1000):
        import torch

        self.policy.reset()
        self.preprocess.reset()
        self.postprocess.reset()
        torch.manual_seed(seed)
        self.observation, self.call_log, self.inference_count = None, [], 0
        self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

    def observe(self, observation):
        validate_observation(observation)
        self.observation = observation
        self.last_timing = {"inference_seconds": 0.0, "preprocess_seconds": 0.0}

    def predict_chunk(self):
        import torch
        from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

        require(self.destination is not None and self.observation is not None, "Missing episode destination or observation")
        call_index = self.inference_count
        stem = self.destination / f"call_{call_index:04d}"
        require(not stem.with_suffix(".npz").exists() and not stem.with_suffix(".json").exists(), "Call evidence already exists")
        obs = self.observation
        begin = time.perf_counter()
        batch = self.preprocess({**{key: torch.from_numpy(obs[key]) for key in FEATURES}, "task": obs["task"]})
        prep_seconds = time.perf_counter() - begin
        rng = torch.get_rng_state().numpy().copy()
        modules, noises, nfe = {}, [], []

        def check_input(name):
            def hook(module, inputs):
                require(len(inputs) == 1 and inputs[0].dtype == torch.float32 and inputs[0].device.type == "cpu",
                        "Dynamic INT8 activation must remain CPU float32")
                modules[name] = modules.get(name, 0) + 1
            return hook

        original_head = self.policy.model.action_model.predict_action
        original_randn = torch.randn

        def capture_noise(*args, **kwargs):
            value = original_randn(*args, **kwargs)
            require(tuple(value.shape) == (1, 7, 7) and value.dtype == torch.float32, "Unexpected actual action noise")
            noises.append(value.detach().cpu().numpy().copy())
            return value

        def observed_head(*args, **kwargs):
            # Observe the official RNG result, without replacing it or reseeding.
            with patch.object(torch, "randn", capture_noise):
                return original_head(*args, **kwargs)

        attempt = {"call_index": call_index, "status": "forward_started", "started_at": utc_now(),
                   "input_sha256": {key: array_sha(obs[key]) for key in FEATURES}, "task": obs["task"],
                   "rng_state_sha256": array_sha(rng)}
        write_json(stem.with_suffix(".json"), attempt)
        begin = time.perf_counter()
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.official, "get_autocast_context", quant.cpu_fp32_context))
            stack.enter_context(patch.object(self.policy.model.action_model, "predict_action", observed_head))
            for name, module in self.policy.named_modules():
                if isinstance(module, DynamicLinear):
                    handle = module.register_forward_pre_hook(check_input(name))
                    stack.callback(handle.remove)
            handle = self.policy.model.action_model.model.register_forward_hook(lambda *unused: nfe.append(1))
            stack.callback(handle.remove)
            with torch.inference_mode():
                normalized = self.policy.predict_action_chunk(batch)
        forward_seconds = time.perf_counter() - begin
        attempt.update(status="forward_returned", forward_returned=True, returned_at=utc_now(), forward_seconds=forward_seconds)
        write_json(stem.with_suffix(".json"), attempt)
        require(set(modules) == set(self.identity["quantized_modules"]), "Not every expected quantized module executed")
        require(len(noises) == 1 and len(nfe) == 4, "Actual noise draws or action-head NFE differ")
        raw = normalized.detach().cpu().numpy().copy()
        validate_action_chunk(raw)
        begin = time.perf_counter()
        with torch.inference_mode():
            commands = self.postprocess(normalized).detach().cpu().numpy().copy()
        post_seconds = time.perf_counter() - begin
        validate_action_chunk(commands, environment=True)
        with stem.with_suffix(".npz").open("xb") as stream:
            np.savez_compressed(stream, model_actions=raw, environment_actions=commands,
                                noise=noises[0], rng_state=rng, input_robot_state=obs["observation.state"])
        record = {**attempt, "status": "complete", "forward_returned": True, "finished_at": utc_now(),
                  "artifact_sha256": file_sha(stem.with_suffix(".npz")), "noise_sha256": array_sha(noises[0]),
                  "quantized_module_calls": modules, "action_head_nfe": len(nfe),
                  "preprocess_seconds": prep_seconds, "forward_seconds": forward_seconds,
                  "postprocess_seconds": post_seconds,
                  "forward_scope": "predict_action_chunk including internal Qwen input construction; excludes external processors"}
        write_json(stem.with_suffix(".json"), record)
        self.call_log.append({"call_index": call_index, "record": stem.with_suffix(".json").name,
                              "sha256": file_sha(stem.with_suffix(".json"))})
        self.inference_count += 1
        self.last_timing = {"inference_seconds": forward_seconds, "preprocess_seconds": prep_seconds}
        return commands[0]

    @property
    def metadata(self):
        return {**self.identity, "variant": VARIANT, "call_log": self.call_log,
                "inference_count": self.inference_count, "action_head_nfe": self.inference_count * 4}


class RecordedSimulator:
    """Keep executed-step evidence even if the process memory guard exits mid-run."""

    def __init__(self, simulator, destination):
        self.simulator = simulator
        self.path = Path(destination) / "executed_steps.jsonl"
        require(not self.path.exists(), "Executed-step log already exists")

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.simulator.step(action)
        row = {"step": self.simulator.steps, "action": np.asarray(action).tolist(),
               "next_robot_state": observation["observation.state"].tolist(),
               "next_camera_sha256": {key: array_sha(observation[key]) for key in FEATURES[:2]},
               "terminated": bool(terminated), "truncated": bool(truncated), "success": bool(info["is_success"])}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        return observation, reward, terminated, truncated, info


def verify_episode(directory, case, protocol):
    from PIL import Image

    result = verify_complete(directory, case, protocol["fingerprint"])
    if result is None:
        return None
    directory = Path(directory)
    require(not (directory / "error.json").exists(), "A runtime error cannot be resealed as a completed episode")
    attempt = read_json(directory / "attempt.json")
    require(attempt["case"] == case and attempt["protocol_fingerprint"] == protocol["fingerprint"], "Attempt identity differs")
    steps = result["steps"]
    require(type(steps) is int and 0 < steps <= HORIZONS[case["suite"]], "Invalid episode length")
    require(result["horizon"] == HORIZONS[case["suite"]], "Horizon changed")
    require(all(type(result[key]) is bool for key in ("success", "terminated", "truncated")), "Invalid terminal flags")
    require(result["terminated"] or result["truncated"], "Episode did not reach an official terminal or horizon")
    require(result["success"] == result["terminated"] and result["truncated"] == (not result["success"]),
            "Official success/terminal/horizon flags disagree")
    require(result["success"] or steps == result["horizon"], "Unsuccessful episode ended before full horizon")
    expected_replans = list(range(0, steps, 7))
    metadata = result["model_info"]
    require(attempt["run_record"] == metadata["run_record"], "Run identity differs from attempt")
    require(metadata["variant"] == VARIANT and len(metadata["quantized_modules"]) == 300, "Model identity differs")
    require(len(set(metadata["quantized_modules"])) == 300, "Duplicate module identity")
    require(metadata["policy_revision"] == base.POLICY_REVISION and metadata["lerobot_revision"] == base.REVISION
            and metadata["inputs_sha256"] == protocol["inputs_sha256"], "Checkpoint/source revision differs")
    loading_path = project_file(metadata["loading"]["path"])
    require(file_sha(loading_path) == metadata["loading"]["sha256"], "Loading audit changed")
    loading = read_json(loading_path)
    require(loading["all_required_inference_keys_consumed"] is True and loading["all_source_tensors_finite"] is True
            and loading["native_tied_word_embeddings"] is True and loading["linear_modules"] == 300
            and loading["theoretical_parameter_payload_bytes"] == quant.EXPECTED_PAYLOAD_BYTES,
            "Incomplete quantized loading audit")
    for module in metadata["actual_modules"].values():
        path = project_file(module["path"])
        require(path.is_relative_to(base.SOURCE) and file_sha(path) == module["sha256"], "Actual official module changed")
    require(set(metadata["actual_modules"]) == {"lerobot", "environment", "policy"}, "Runtime module audit incomplete")
    environment = result["environment"]
    expected_environment = {"suite": case["suite"], "task_id": 1, "horizon": HORIZONS[case["suite"]],
                            "image_size": 360, "control_hz": 20, "settle_steps": 10, "control_mode": "relative",
                            "hard_reset": True, "images_flipped_hw_once": True, "object_state_exposed_to_policy": False}
    require(all(environment.get(key) == value for key, value in expected_environment.items()), "Environment contract differs")
    require(result["scheduler_config"] == asdict(SchedulerConfig(**{**case["scheduler"], "horizon": result["horizon"]})),
            "Unexpected scheduler configuration")
    calls = read_json(directory / "calls.json")
    require(len(calls) == result["inference_calls"] == len(expected_replans), "Fixed-seven planning count differs")
    require(calls == metadata["call_log"] and metadata["action_head_nfe"] == 4 * len(calls), "Call metadata differs")
    require(metadata["inference_count"] == len(calls), "Adapter inference count differs")
    rows = [json.loads(line) for line in (directory / "executed_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == steps, "Executed-step log differs")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        actions, states = trajectory["actions"], trajectory["states"]
        require(actions.shape == (steps, 7) and states.shape == (steps, 8), "Trajectory dimensions differ")
        require(actions.dtype == np.float32 and states.dtype == np.float32, "Trajectory dtype differs")
        require(all(np.isfinite(trajectory[key]).all() for key in trajectory.files), "Nonfinite trajectory")
        for key in ("model_seconds", "preprocessing_seconds", "control_seconds", "simulator_seconds"):
            require(trajectory[key].shape == (steps,) and (trajectory[key] >= 0).all(), "Invalid per-step timing")
        expected_model_times = np.zeros(steps, dtype=np.float64)
        expected_preprocess_times = np.zeros(steps, dtype=np.float64)
        require(np.array_equal(trajectory["replan_steps"], expected_replans), "Planning indices differ")
        require(np.array_equal(states[0], np.asarray(result["initial_robot_state"], dtype=np.float32))
                and np.array_equal(trajectory["final_state"], np.asarray(result["final_robot_state"], dtype=np.float32)),
                "Initial/final robot state summaries differ")
        for name in ("initial.png", "initial_wrist.png", "final.png"):
            with Image.open(directory / name) as picture:
                require(picture.mode == "RGB" and picture.size == (360, 360), "Camera evidence dimensions differ")
                if name == "initial.png":
                    require(array_sha(np.asarray(picture)) == result["initial_image_sha256"], "Initial camera summary differs")
        for index, row in enumerate(rows):
            require(row["step"] == index + 1 and np.array_equal(np.asarray(row["action"], dtype=np.float32), actions[index]),
                    "Executed action log differs")
            next_state = states[index + 1] if index + 1 < steps else trajectory["final_state"]
            require(np.array_equal(np.asarray(row["next_robot_state"], dtype=np.float32), next_state), "Executed robot state log differs")
            if index < steps - 1:
                require(not row["terminated"] and not row["truncated"] and not row["success"], "Continued after terminal")
        require(all(rows[-1][key] == result[key] for key in ("success", "terminated", "truncated")), "Final flags differ")
        forward_total = 0.0
        for index, reference in enumerate(calls):
            require(reference["record"] == f"call_{index:04d}.json" and reference["call_index"] == index, "Call order differs")
            call_path = directory / reference["record"]
            require(file_sha(call_path) == reference["sha256"], "Call record changed")
            call = read_json(call_path)
            require(call["status"] == "complete" and call["forward_returned"] is True and call["action_head_nfe"] == 4,
                    "Incomplete real forward or wrong NFE")
            require(set(call["quantized_module_calls"]) == set(metadata["quantized_modules"])
                    and all(type(n) is int and n > 0 for n in call["quantized_module_calls"].values()), "Qwen execution audit differs")
            require(call["task"] == result["task"], "Instruction changed")
            require(all(type(call[key]) in (float, int) and math.isfinite(call[key]) and call[key] >= 0
                        for key in ("preprocess_seconds", "forward_seconds", "postprocess_seconds")), "Invalid call timing")
            forward_total += call["forward_seconds"]
            expected_model_times[expected_replans[index]] = call["forward_seconds"]
            expected_preprocess_times[expected_replans[index]] = call["preprocess_seconds"]
            data_path = call_path.with_suffix(".npz")
            require(file_sha(data_path) == call["artifact_sha256"], "Action evidence changed")
            with np.load(data_path, allow_pickle=False) as chunk:
                require(set(chunk.files) == {"model_actions", "environment_actions", "noise", "rng_state", "input_robot_state"},
                        "Unexpected chunk artifact inventory")
                validate_action_chunk(chunk["model_actions"])
                validate_action_chunk(chunk["environment_actions"], environment=True)
                validate_action_chunk(chunk["noise"])
                require(array_sha(chunk["noise"]) == call["noise_sha256"] and array_sha(chunk["rng_state"]) == call["rng_state_sha256"],
                        "Actual noise/RNG evidence differs")
                start = expected_replans[index]
                end = min(start + 7, steps)
                require(np.array_equal(actions[start:end], chunk["environment_actions"][0, :end-start]), "Executed chunk differs")
                require(np.array_equal(states[start], chunk["input_robot_state"])
                        and array_sha(states[start]) == call["input_sha256"]["observation.state"], "Replan used stale robot state")
                if index == 0:
                    for name, key in (("initial.png", FEATURES[0]), ("initial_wrist.png", FEATURES[1])):
                        with Image.open(directory / name) as picture:
                            image = np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
                        require(array_sha(image) == call["input_sha256"][key], "First policy camera differs from saved image")
                else:
                    require(all(call["input_sha256"][key] == rows[start-1]["next_camera_sha256"][key]
                                for key in FEATURES[:2]), "Replan used a camera packet different from the latest environment output")
        require(math.isclose(forward_total, result["policy_inference_seconds"], rel_tol=1e-9, abs_tol=1e-8), "Inference totals differ")
        require(np.allclose(trajectory["model_seconds"], expected_model_times, rtol=0, atol=1e-10)
                and np.allclose(trajectory["preprocessing_seconds"], expected_preprocess_times, rtol=0, atol=1e-10),
                "Per-step model/processor timings differ from calls")
        for vector, total in (("preprocessing_seconds", "preprocess_seconds"), ("simulator_seconds", "simulator_seconds")):
            require(math.isclose(float(trajectory[vector].sum()), result[total], rel_tol=1e-9, abs_tol=1e-8), "Timing summary differs")
    return result


def verify_study(output, *, partial=False):
    output = Path(output)
    protocol = verify_protocol(output)
    planned = protocol["spec"]["cases"]
    require(all(path.name in {case["id"] for case in planned} for path in (output / "episodes").glob("*")), "Unplanned episode")
    results, statuses = [], {}
    for case in planned:
        directory = output / "episodes" / case["id"]
        if not directory.exists():
            statuses[case["id"]] = "missing"
        elif not (directory / "receipt.json").exists():
            statuses[case["id"]] = "error" if (directory / "error.json").exists() else "unsealed"
        else:
            results.append(verify_episode(directory, case, protocol))
            statuses[case["id"]] = "complete"
    complete = len(results) == 4
    require(partial or complete, "Closed-loop study incomplete; partial must be explicit")
    return {"protocol_fingerprint": protocol["fingerprint"], "complete": complete,
            "planned": 4, "completed": len(results), "successes": sum(row["success"] for row in results),
            "task_failures": sum(not row["success"] for row in results), "statuses": statuses, "results": results,
            "limitations": protocol["spec"]["limitations"]}


def write_report(output):
    output = Path(output)
    study = verify_study(output, partial=True)
    study["run_audits"] = []
    for path in sorted((output / "runs").glob("run_*.json")):
        run_record = read_json(path)
        memory = run_record.get("memory", {})
        study["run_audits"].append({"path": path.relative_to(output).as_posix(), "sha256": file_sha(path),
            "status": run_record["status"], "weights_loaded": run_record.get("weights_loaded", False),
            "memory": {key: value for key, value in memory.items() if key != "samples"},
            "load_quantize_seconds": run_record.get("load_quantize_seconds"),
            "imports_seconds": run_record.get("imports_seconds")})
    write_json(output / "report.json", study)
    rows = study["results"]
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ("id", "suite", "task", "success", "steps", "horizon", "inference_calls", "policy_inference_seconds", "wall_seconds")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# VLA-JEPA INT8 四任务闭环诊断", "", f"计划 **4** 回合，已完成 **{len(rows)}**，官方成功 **{study['successes']}**，完整运行但任务失败 **{study['task_failures']}**。",
             "运行完成与任务成功分别计数。缺失、未封存及错误见 report.json；本报告不填补或排除这些计划回合。", "",
             "固定四个已知任务、各 init41，一次量化实现；没有原生 BF16 对照，不能估计量化损失或泛化能力，也不是与 SmolVLA 的匹配比较。", "",
             "| Suite | 官方成功 | 步数 / 时限 | 规划次数 | 动作头 NFE | 推理总秒数 | 回合墙钟秒数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['suite']} | {row['success']} | {row['steps']} / {row['horizon']} | {row['inference_calls']} | {row['model_info']['action_head_nfe']} | {row['policy_inference_seconds']:.3f} | {row['wall_seconds']:.3f} |")
    lines += ["", "推理时仿真暂停；20 Hz 为模拟控制频率，不是模型推理速度。墙钟包含 reset、控制、物理和渲染，排除模型加载与 GIF 编码。共享主机计时不代表稳定部署吞吐。", "",
              "## 运行与内存记录", ""]
    for audit in study["run_audits"]:
        memory = audit["memory"]
        lines.append(f"- [运行原始记录]({audit['path']})：`{audit['status']}`；每秒采样峰值 RSS `{memory.get('peak_rss_bytes')}` bytes，最低系统可用内存 `{memory.get('minimum_available_bytes')}` bytes。采样不能保证捕获瞬时峰值。")
    lines += ["", "## 实际调用延迟分布", "", "| Suite | 前向 p50 秒 | 前向 p95 秒 |", "|---|---:|---:|"]
    for row in rows:
        durations = [read_json(output / "episodes" / row["id"] / item["record"])["forward_seconds"]
                     for item in row["model_info"]["call_log"]]
        lines.append(f"| {row['suite']} | {np.percentile(durations, 50):.3f} | {np.percentile(durations, 95):.3f} |")
    for row in rows:
        path = f"episodes/{row['id']}"
        lines += [f"## {row['suite']}", "", row["task"], "",
                  f"[原始结果]({path}/result.json) · [逐调用记录]({path}/calls.json) · [轨迹]({path}/trajectory.npz)", "",
                  f"![实际闭环录像]({path}/rollout.gif)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {key: value for key, value in study.items() if key != "results"}


def portable_error(error):
    text = str(error)
    for path, token in ((ROOT, "<project>"), (Path.home(), "<user>"), (Path(sys.prefix), "<environment>")):
        text = text.replace(str(path), token).replace(str(path).replace("\\", "/"), token)
    return text


def portable_value(value):
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    return portable_error(value) if isinstance(value, str) else value


def run(output, *, freeze_only=False, limit=None):
    import psutil

    output = Path(output).resolve()
    require(output.is_relative_to(ROOT), "Keep diagnostic evidence inside this project")
    require(limit is None or type(limit) is int and limit > 0, "limit must be positive")
    protocol = freeze_protocol(output)
    if freeze_only:
        return {"status": "frozen", "fingerprint": protocol["fingerprint"]}
    study = verify_study(output, partial=True)
    require(all(status in ("complete", "missing") for status in study["statuses"].values()), "Partial/error episode retained; automatic retry forbidden")
    if study["complete"]:
        return write_report(output)
    lock = output / "active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_at": utc_now()}, stream)
    runs = output / "runs"
    runs.mkdir(exist_ok=True)
    index = len(list(runs.glob("run_*.json"))) + 1
    run_path = runs / f"run_{index:04d}.json"
    record = {"status": "preflight", "started_at": utc_now(), "protocol_fingerprint": protocol["fingerprint"],
              "available_bytes": psutil.virtual_memory().available, "minimum_available_bytes": MIN_FREE_GIB * 1024**3}
    guard = simulator = None
    current_directory = None
    try:
        write_json(run_path, record)
        if record["available_bytes"] < MIN_FREE_GIB * 1024**3:
            record.update(status="blocked_preflight", weights_loaded=False)
            return {"status": "blocked_preflight", "available_gib": record["available_bytes"] / 1024**3}
        guard = base.MemoryGuard(run_path, MAX_RSS_GIB, MIN_RUNTIME_FREE_GIB)
        guard.start()
        faulthandler.dump_traceback_later(180, repeat=True)
        require(base.verify_manifest()["sha256"] == protocol["inputs_sha256"], "Checkpoint/source inputs changed")
        require(artifact_provenance(ROOT, []) == protocol["simulation_artifacts"], "Simulation assets changed")
        require(initial_state_manifest() == protocol["initial_states"], "Published initial states changed")
        from embodied_research.runtime import configure_runtime, configure_libero_paths, compatibility_audit
        configure_runtime(ROOT)
        base.isolate_official_source()
        configure_libero_paths(ROOT)
        print("Importing the isolated official model and LIBERO runtime", flush=True)
        started = time.perf_counter()
        import torch
        import lerobot
        import lerobot.envs.libero as official_env
        import lerobot.policies.vla_jepa.modeling_vla_jepa as official
        from lerobot.processor import load_pretrained_policy_processors
        from embodied_research.simulator import LiberoSimulator

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        source_modules = {}
        for name, module in (("lerobot", lerobot), ("environment", official_env), ("policy", official)):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(base.SOURCE), "Mixed LeRobot sources detected")
            source_modules[name] = {"path": base.relative(path), "sha256": file_sha(path)}
        record.update(status="building_meta", imports_seconds=time.perf_counter() - started, actual_modules=source_modules)
        write_json(run_path, record)
        print("Constructing meta policy and checking the complete checkpoint schema", flush=True)
        policy, construction = base.build_meta_policy()
        require(policy.config.n_action_steps == 7 and policy.config.n_obs_steps == 1, "Unexpected native execution contract")
        checkpoint = base.ASSETS / "policy/model.safetensors"
        schema = base.validate_schema(base.read_safetensors_header(checkpoint),
                                      {key: tuple(value.shape) for key, value in policy.state_dict().items()})
        record["pre_materialization_available_bytes"] = psutil.virtual_memory().available
        if record["pre_materialization_available_bytes"] < MIN_FREE_GIB * 1024**3:
            record.update(status="blocked_preflight", weights_loaded=False, reason="Below closed-loop budget after imports")
            return {"status": "blocked_preflight", "stage": "after_imports"}
        record["status"] = "loading_quantized_checkpoint"
        write_json(run_path, record)
        started = time.perf_counter()
        loading = quant.load_quantized_tensors(policy, schema, checkpoint)
        record.update(weights_loaded=True, load_quantize_seconds=time.perf_counter() - started,
                      rss_after_load_bytes=psutil.Process().memory_info().rss, construction=construction)
        loading_path = runs / f"loading_{index:04d}.json"
        write_json(loading_path, loading)
        from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
        identity = {"policy_revision": base.POLICY_REVISION, "lerobot_revision": base.REVISION,
                    "inputs_sha256": protocol["inputs_sha256"], "actual_modules": source_modules,
                    "loading": {"path": base.relative(loading_path), "sha256": file_sha(loading_path)},
                    "quantized_modules": [name for name, module in policy.named_modules() if isinstance(module, DynamicLinear)],
                    "run_record": base.relative(run_path)}
        require(len(identity["quantized_modules"]) == 300, "Quantized inventory differs")
        preprocess, postprocess = load_pretrained_policy_processors(
            base.ASSETS / "policy", preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        adapter = ClosedLoopAdapter(policy, preprocess, postprocess, official, identity)
        record["status"] = "running_episodes"
        write_json(run_path, record)
        completed = 0
        for ordinal, case in enumerate(protocol["spec"]["cases"], 1):
            directory = output / "episodes" / case["id"]
            if directory.exists():
                continue
            if limit is not None and completed >= limit:
                break
            directory.mkdir(parents=True)
            current_directory = directory
            write_json(directory / "attempt.json", {"case": case, "protocol_fingerprint": protocol["fingerprint"],
                                                     "run_record": base.relative(run_path), "started_at": utc_now()})
            write_json(output / "progress.json", {"current": case, "ordinal": ordinal, "planned": 4, "updated_at": utc_now()})
            print(f"[{ordinal}/4] {case['id']} (VLA-JEPA INT8 closed loop)", flush=True)
            simulator = LiberoSimulator(case["suite"], case["task_id"], project_root=ROOT, image_size=360)
            require(type(simulator.env) is official_env.LiberoEnv, "Unexpected environment implementation")
            adapter.destination = directory
            result = run_episode(RecordedSimulator(simulator, directory), adapter, case, directory)
            result["protocol_fingerprint"] = protocol["fingerprint"]
            write_json(directory / "calls.json", adapter.call_log)
            write_json(directory / "result.json", result)
            seal_episode(directory)
            verify_episode(directory, case, protocol)
            print(f"  success={result['success']} steps={result['steps']} calls={result['inference_calls']} wall={result['wall_seconds']:.1f}s", flush=True)
            simulator.close()
            simulator = None
            current_directory = None
            completed += 1
        record["status"] = "complete" if verify_study(output, partial=True)["complete"] else "stopped_after_requested_limit"
        record["compatibility"] = portable_value(compatibility_audit())
        summary = write_report(output)
        write_json(output / "progress.json", summary)
        return summary
    except Exception as error:
        record.update(status="error", error=portable_error(error), traceback=portable_error(traceback.format_exc()))
        if current_directory is not None:
            write_json(current_directory / "error.json", record)
        raise
    finally:
        if simulator is not None:
            simulator.close()
        if guard is not None:
            faulthandler.cancel_dump_traceback_later()
            record["memory"] = guard.finish()
        record["finished_at"] = utc_now()
        write_json(run_path, record)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/frontier/vlajepa_closed_loop")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.validate_only:
        result = verify_study(args.output, partial=args.partial)
        result.pop("results")
    elif args.report:
        result = write_report(args.output)
    else:
        result = run(args.output, freeze_only=args.freeze_only, limit=args.limit)
        if "completed" in result:
            result = write_report(args.output)  # Include the finalized memory/run audit after the guard finishes.
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
