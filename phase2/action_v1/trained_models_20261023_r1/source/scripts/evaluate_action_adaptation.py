"""One prospective, complete native-SmolVLA LIBERO evaluation case.

This worker never trains, retries, launches another process, or changes a
completed case. The external campaign supervisor owns hard process/time limits.
Only RGB observations, 8D proprioception and the frozen task instruction reach
the policy. Brightness changes observations, never simulator state or dynamics.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from uuid import uuid4
import zipfile

from action_campaign_common import (ROOT, deadline_reached, digest, now, read, resolve, seal,
                                    verify_protocol, write_json)


GIB = 1024 ** 3
START_AVAILABLE = int(4.25 * GIB)
RUNTIME_AVAILABLE = int(.75 * GIB)
MIN_DISK = 4 * GIB


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def noise_seed(case, decision):
    """Pair method and brightness conditions; diverged states are not identical."""
    key = f"action-eval-v1:{case['task_uid']}:{case['initial_state_id']}:{decision}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def normalized_language(value):
    return " ".join(value.replace("_", " ").lower().split())


def conditioned_observation(observation, condition):
    import numpy as np

    require(condition in ("clean", "brightness075"), "Unregistered observation condition")
    # Whitelist policy fields: object state / success metadata cannot leak here.
    result = {"task": observation["task"],
              "observation.state": np.asarray(observation["observation.state"], dtype=np.float32).copy()}
    require(result["observation.state"].shape == (8,) and np.isfinite(result["observation.state"]).all(),
            "Invalid 8D proprioception")
    for camera in ("image", "image2"):
        key = f"observation.images.{camera}"
        image = np.asarray(observation[key])
        require(image.ndim == 3 and image.shape[0] == 3 and image.dtype == np.float32
                and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1,
                "Expected policy-ready FP32 CHW RGB in [0,1]")
        result[key] = image.copy() if condition == "clean" else image * np.float32(.75)
    return result


def select_case(protocol, case_id):
    evaluation = protocol["evaluation"]
    require(evaluation["num_steps"] == evaluation["replan_steps"] == 10
            and evaluation["image_size"] == 360 and evaluation["arena_bytes"] == 268435456,
            "Evaluation dynamics/integration contract differs from registered study")
    cases = [case for case in evaluation["cases"] if case["case_id"] == case_id]
    require(len(cases) == 1, "Case is missing or ambiguous in frozen protocol")
    case = cases[0]
    methods = [method for method in evaluation["methods"] if method["method_id"] == case["method_id"]]
    require(len(methods) == 1 and methods[0]["family"] in ("frozen", "lora", "output_head"),
            "Case method is missing/ambiguous/unsupported")
    permitted_inits = (0,) if protocol.get("development_only") is True else (5, 15, 25, 35, 45)
    require(case["task_id"] == 0 and case["condition"] in ("clean", "brightness075")
            and type(case["initial_state_id"]) is int and case["initial_state_id"] in permitted_inits,
            "Unexpected prospective task/condition/init")
    require(bool(str(case["language"]).strip()), "Case instruction is empty")
    method = methods[0]
    require((method["family"] == "frozen" and method["training_run_id"] is None)
            or (method["family"] != "frozen" and method["training_run_id"] in protocol["runs"]
                and protocol["runs"][method["training_run_id"]]["mode"] == method["family"]),
            "Method selects a missing or mismatched training run")
    return case, method


def _adapter_manifest(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 16 * 1024 * 1024,
                "Adapter checkpoint exceeds the small-checkpoint expansion limit")
    with np.load(path, allow_pickle=False) as checkpoint:
        encoded = checkpoint["manifest_utf8"]
        require(encoded.dtype == np.uint8 and encoded.ndim == 1 and len(encoded) <= 1_000_000,
                "Invalid adapter manifest encoding")
        return json.loads(encoded.tobytes().decode("utf-8"))


def verify_trained_manifest(path, expected_sha, protocol, protocol_sha):
    """Check every completed run before any policy/model loading, including frozen baseline cases."""
    require(digest(path) == expected_sha, "Frozen trained manifest changed")
    manifest = read(path)
    require(manifest["protocol_sha256"] == protocol_sha, "Trained manifest targets another protocol")
    require(set(manifest["runs"]) == set(protocol["runs"]) and bool(manifest["runs"]),
            "All registered training runs must complete before evaluation")
    for run_id, record in manifest["runs"].items():
        run = protocol["runs"][run_id]
        require(record["mode"] == run["mode"] and record["seed"] == run["initialization_seed"],
                f"Training mode/seed differs: {run_id}")
        adapter_path, receipt_path = resolve(record["adapter"]["path"]), resolve(record["receipt"]["path"])
        require(digest(adapter_path) == record["adapter"]["sha256"]
                and digest(receipt_path) == record["receipt"]["sha256"], f"Training checkpoint changed: {run_id}")
        receipt = read(receipt_path)
        payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        require(receipt.get("schema") == "feedbackbench.action_adaptation_checkpoint.v1"
                and receipt.get("receipt_sha256") == canonical_sha(payload), "Checkpoint receipt self-hash differs")
        binding = receipt["binding"]
        require(receipt["binding_sha256"] == canonical_sha(binding)
                and binding["protocol_sha256"] == protocol_sha and binding["run_id"] == run_id,
                "Checkpoint protocol/run binding differs")
        require(binding["schedule_sha256"] == run["schedule"]["sha256"]
                and binding["data_manifest_sha256"] == protocol["data"]["manifest"]["sha256"]
                and binding["alignment_sha256"] == protocol["data"]["alignment_evidence"]["sha256"],
                "Checkpoint training input bindings differ")
        require(receipt["completed_steps"] == run["steps"] and receipt["completed_steps"] > 0,
                "Checkpoint is not the registered final step")
        require(set(receipt["files"]) == {"adapter.npz", "optimizer.npz", "rng.npz", "state.json", "history.json"},
                "Checkpoint receipt file inventory differs")
        for filename, expected in receipt["files"].items():
            require(digest(resolve(filename, receipt_path.parent)) == expected,
                    f"Sealed checkpoint file changed: {run_id}/{filename}")
        require(adapter_path == resolve("adapter.npz", receipt_path.parent)
                and receipt["files"]["adapter.npz"] == record["adapter"]["sha256"],
                "Selected adapter is outside its final checkpoint receipt")
        saved = _adapter_manifest(adapter_path)
        expected_schema = ("feedbackbench.action_expert_lora.v1" if run["mode"] == "lora"
                           else "feedbackbench.action_output_head.v1")
        require(run["mode"] in ("lora", "output_head") and saved["schema"] == expected_schema
                and saved["base_fingerprint"] == record["base_fingerprint"] and saved["seed"] == record["seed"],
                "Adapter mode/base/seed differs from final trained manifest")
        metadata = saved["metadata"]
        require(metadata["protocol_sha256"] == protocol_sha and metadata["run_id"] == run_id
                and metadata["binding_sha256"] == receipt["binding_sha256"]
                and metadata["step"] == receipt["completed_steps"], "Adapter metadata binding differs")
    require(len({record["base_fingerprint"] for record in manifest["runs"].values()}) == 1,
            "Training methods use different original base policies")
    return manifest


class ResourceStop(RuntimeError):
    pass


class EvaluationSession:
    def __init__(self, output, max_seconds):
        import psutil

        require(type(max_seconds) is int and 0 < max_seconds <= 7200, "Evaluation wall limit must not exceed two hours")
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.psutil = psutil
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self.done = threading.Event()
        self.lock = threading.RLock()
        self.reason = None
        self.minimum_available = None
        self.peak_rss = 0
        self.sample_count = 0
        self.resource_stream = (output / "resources.jsonl").open("x", encoding="utf-8")
        self.state = {"status": "starting", "owner": {"pid": self.process.pid, "created": self.process.create_time()},
                      "started_at": now(), "stage": "preflight", "max_seconds": max_seconds}
        self.thread = None

    def sample(self):
        value = {"at": now(), "elapsed_seconds": time.monotonic() - self.started,
                 "available_bytes": self.psutil.virtual_memory().available,
                 "rss_bytes": self.process.memory_info().rss, "disk_free_bytes": shutil.disk_usage(self.output).free}
        with self.lock:
            self.minimum_available = (value["available_bytes"] if self.minimum_available is None
                                      else min(value["available_bytes"], self.minimum_available))
            self.peak_rss = max(self.peak_rss, value["rss_bytes"])
            self.sample_count += 1
            self.resource_stream.write(json.dumps(value) + "\n")
            self.resource_stream.flush()
            reason = ("hard_deadline" if deadline_reached() else "evaluation_time_limit"
                      if value["elapsed_seconds"] >= self.max_seconds else "runtime_memory_reserve"
                      if value["available_bytes"] < RUNTIME_AVAILABLE else "disk_reserve"
                      if value["disk_free_bytes"] < MIN_DISK else None)
            self.reason = self.reason or reason
        return value

    def begin(self):
        sample = self.sample()
        if sample["available_bytes"] < START_AVAILABLE:
            self.reason = "start_memory_threshold"
        self.checkpoint("resource_preflight_passed")
        self.thread = threading.Thread(target=self.monitor, daemon=True, name="action-evaluation-resource-sampler")
        self.thread.start()

    def monitor(self):
        while not self.done.wait(.5):
            try:
                self.sample()
            except BaseException as error:
                self.reason = self.reason or f"resource_sampling_failure: {error!r}"
                return

    def checkpoint(self, stage, **details):
        sample = self.sample()
        if (self.output / "STOP_REQUEST").exists():
            self.reason = self.reason or "external_stop_request"
        if self.reason:
            raise ResourceStop(self.reason)
        with self.lock:
            self.state.update(status="running", stage=stage, updated_at=now(), last_resource_sample=sample, **details)
            write_json(self.output / "status.json", self.state)

    def finish(self, status, **details):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            self.state.update(status=status, ended_at=now(), elapsed_seconds=time.monotonic() - self.started,
                              resource_summary={"peak_rss_bytes": self.peak_rss,
                                                "minimum_available_bytes": self.minimum_available,
                                                "samples": self.sample_count}, **details)
            try:
                write_json(self.output / "status.json", self.state)
            except BaseException as error:
                with (self.output / ("persistence_failure_" + uuid4().hex[:12] + ".json")).open("x", encoding="utf-8") as stream:
                    json.dump({"error": repr(error), "state": self.state}, stream)
                raise
            finally:
                self.resource_stream.close()


def save_trajectory(path, actions, states, next_states, rewards, terminations, truncations):
    import numpy as np

    # Avoid appending a UUID to an already long Windows case filename.
    temporary = path.with_name("." + uuid4().hex[:12] + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
                            states=np.asarray(states, dtype=np.float32).reshape(-1, 8),
                            next_states=np.asarray(next_states, dtype=np.float32).reshape(-1, 8),
                            rewards=np.asarray(rewards, dtype=np.float32),
                            terminated=np.asarray(terminations, dtype=np.bool_),
                            truncated=np.asarray(truncations, dtype=np.bool_))
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


def save_observation_images(output, prefix, observation):
    import numpy as np
    from PIL import Image

    for camera in ("image", "image2"):
        frame = observation[f"observation.images.{camera}"].transpose(1, 2, 0)
        Image.fromarray(np.rint(frame * 255).astype(np.uint8)).save(output / f"{prefix}_{camera}.png")


def native_chunk(adapter, observation, seed, torch):
    """Count actual velocity projections in the original ten-step inference route."""
    adapter.observe(observation)
    noise = torch.randn((1, 50, 32), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    counts = []
    handle = adapter.policy.model.action_out_proj.register_forward_hook(lambda module, inputs, output: counts.append(1))
    tick = time.perf_counter()
    try:
        with torch.inference_mode():
            normalized = adapter.policy.predict_action_chunk(dict(adapter._batch), noise=noise)
    finally:
        handle.remove()
    inference_seconds = time.perf_counter() - tick
    require(len(counts) == 10, f"Native inference used {len(counts)} velocity evaluations instead of ten")
    actions = adapter._actions_to_numpy(normalized)[0, :10]
    require(actions.shape == (10, 7), "Native action chunk has wrong shape")
    return actions, {"inference_seconds": inference_seconds, "actual_nfe": len(counts),
                     "preprocess_seconds": adapter.last_timing["preprocess_seconds"],
                     "postprocess_seconds": adapter.last_timing["postprocess_seconds"],
                     "noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest()}


def execute_episode(args, protocol, case, method, trained, session):
    from embodied_research.runtime import configure_runtime
    configure_runtime(ROOT)
    import numpy as np
    import torch
    from embodied_research.policies import PolicyAdapter
    from embodied_research.simulator import LiberoSimulator, SUITE_HORIZONS
    from embodied_research.action_adaptation import frozen_base_checksum, load_action_adapter
    from embodied_research.action_head_baseline import load_output_head
    from budget_startup_memory import reclaim_own_startup_pages
    from libero_arena_budget import arena_budget
    from robosuite.utils.binding_utils import MjSim

    require(protocol["resources"]["torch_threads"] == 2 and protocol["resources"]["interop_threads"] == 1,
            "Registered CPU thread limits changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    actions, states, next_states, rewards, terminations, truncations, calls = [], [], [], [], [], [], []
    arena = None
    session.checkpoint("loading_frozen_base")
    # common.verify_protocol binds both download manifests to protocol.model;
    # PolicyAdapter's default verify=True hashes their complete file inventory.
    models = resolve(protocol["model"]["models_dir"])
    adapter = PolicyAdapter("smolvla", models_dir=models, num_steps=10, n_action_steps=1)
    expected_base = next(iter(trained["runs"].values()))["base_fingerprint"]
    require(frozen_base_checksum(adapter.policy) == expected_base, "Evaluation base differs from all trained runs")
    require(not (getattr(adapter.config, "rtc_config", None) and adapter.config.rtc_config.enabled),
            "RTC inference must remain disabled")
    selection = None
    if method["family"] == "frozen":
        require(method["training_run_id"] is None, "Frozen baseline cannot select a trained adapter")
    else:
        run_id = method["training_run_id"]
        selection = trained["runs"][run_id]
        require(selection["mode"] == method["family"], "Evaluation selected the wrong adapter family")
        checkpoint = resolve(selection["adapter"]["path"])
        require(digest(checkpoint) == selection["adapter"]["sha256"], "Selected checkpoint changed before loading")
        loader = load_action_adapter if method["family"] == "lora" else load_output_head
        loader(adapter.policy, checkpoint, base_fingerprint=expected_base)
    adapter.policy.requires_grad_(False)
    adapter.reset(seed=1000)
    session.checkpoint("model_loaded")
    startup = reclaim_own_startup_pages()
    write_json(session.output / "startup_memory.json", startup)
    require(not startup["supported"] or startup["success"], "Registered own startup memory reclaim failed")
    metadata = {"case": case, "method": method, "policy": adapter.metadata,
                "protocol_sha256": args.protocol_sha256, "trained_manifest_sha256": args.trained_manifest_sha256,
                "selected_training_run": selection, "base_fingerprint": expected_base,
                "observation_condition": case["condition"], "policy_fields": ["two RGB cameras", "8D robot state", "task"],
                "environment_reset_seed": 1000, "noise_rule": "sha256(action-eval-v1:task_uid:init:decision)[:4] little endian",
                "prospective_evaluation": True, "pretraining_overlap": "unknown; no unseen-task claim"}
    write_json(session.output / "metadata.json", metadata)
    success = terminated = truncated = False
    try:
        with (session.output / "calls.jsonl").open("x", encoding="utf-8") as stream, ExitStack() as stack:
            arena = stack.enter_context(arena_budget(MjSim, memory_bytes=protocol["evaluation"]["arena_bytes"]))
            simulator = stack.enter_context(LiberoSimulator(case["suite"], case["task_id"], image_size=360))
            require(simulator.horizon == SUITE_HORIZONS[case["suite"]], "Official full horizon changed")
            observation = conditioned_observation(simulator.reset(seed=1000, init_state_id=case["initial_state_id"]),
                                                  case["condition"])
            require(normalized_language(simulator.task) == normalized_language(case["language"]),
                    "Frozen case language differs from the official simulator task")
            metadata["simulator"] = simulator.metadata()
            write_json(session.output / "metadata.json", metadata)
            save_observation_images(session.output, "initial", observation)
            while not (terminated or truncated):
                decision = len(calls)
                session.checkpoint("predicting", steps=len(actions), decisions=decision)
                seed = noise_seed(case, decision)
                tick = time.perf_counter()
                chunk, timings = native_chunk(adapter, observation, seed, torch)
                call = {"decision": decision, "start_step": len(actions), "noise_seed": seed,
                        "chunk_sha256": hashlib.sha256(chunk.tobytes()).hexdigest(), **timings}
                try:
                    for action in chunk:
                        session.checkpoint("executing", steps=len(actions), decisions=decision)
                        state_before = observation["observation.state"].copy()
                        raw, reward, terminated, truncated, info = simulator.step(action)
                        require("is_success" in info, "Official simulator success predicate is missing")
                        observation = conditioned_observation(raw, case["condition"])
                        actions.append(action.copy())
                        states.append(state_before)
                        next_states.append(observation["observation.state"].copy())
                        rewards.append(reward)
                        terminations.append(terminated)
                        truncations.append(truncated)
                        success = bool(info["is_success"])
                        if terminated or truncated:
                            # This irreversible boundary is recorded before any
                            # further resource check or artifact bookkeeping.
                            write_json(session.output / "terminal_observed.json", {
                                "case_id": case["case_id"], "protocol_sha256": args.protocol_sha256,
                                "trained_manifest_sha256": args.trained_manifest_sha256,
                                "observed_at": now(), "steps": len(actions), "terminated": bool(terminated),
                                "truncated": bool(truncated), "official_success": success,
                                "completed_episode_may_be_retried": False})
                            break
                finally:
                    call.update(executed_steps=len(actions) - call["start_step"],
                                decision_wall_seconds=time.perf_counter() - tick)
                    calls.append(call)
                    stream.write(json.dumps(call, allow_nan=False) + "\n")
                    stream.flush()
                save_trajectory(session.output / "partial_actions.npz", actions, states, next_states, rewards,
                                terminations, truncations)
                write_json(session.output / "arena.json", arena)
                if not (terminated or truncated):
                    session.checkpoint("decision_complete", steps=len(actions), decisions=len(calls), official_success=success)
            save_observation_images(session.output, "final", observation)
        write_json(session.output / "arena.json", arena)
        require(not arena["native_warnings"], "Native renderer/simulator warning invalidated the episode")
        require(terminated or (truncated and len(actions) == SUITE_HORIZONS[case["suite"]]),
                "Episode ended before official termination or its full horizon")
        save_trajectory(session.output / "actions.npz", actions, states, next_states, rewards, terminations, truncations)
        return {"status": "complete", "official_success": success, "terminated": terminated, "truncated": truncated,
                "steps": len(actions), "horizon": SUITE_HORIZONS[case["suite"]], "decision_count": len(calls),
                "elapsed_seconds": time.monotonic() - session.started,
                "inference_seconds": sum(item["inference_seconds"] for item in calls),
                "NFE": sum(item["actual_nfe"] for item in calls), "case": case,
                "protocol_sha256": args.protocol_sha256, "trained_manifest_sha256": args.trained_manifest_sha256,
                "completed_at": now(), "renderer_warnings": arena["native_warnings"],
                "failure_is_final": not success, "completed_episode_may_be_retried": False}
    except BaseException:
        save_trajectory(session.output / "partial_actions.npz", actions, states, next_states, rewards, terminations, truncations)
        if arena is not None:
            write_json(session.output / "arena.json", arena)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--trained-manifest", required=True, type=Path)
    parser.add_argument("--trained-manifest-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    protocol = verify_protocol(args.protocol, args.protocol_sha256)
    case, method = select_case(protocol, args.case_id)
    campaign = args.protocol.resolve().parent
    require(campaign.is_relative_to((ROOT / "outputs/action_adaptation").resolve()), "Campaign must be independently owned")
    output = args.output.resolve()
    require(output.is_relative_to(campaign / "evaluation") and output != campaign / "evaluation",
            "Output must be a new directory below the campaign evaluation directory")
    session = EvaluationSession(output, protocol["resources"]["evaluation_max_seconds"])
    result = None
    try:
        session.begin()
        session.checkpoint("verifying_all_training_checkpoints")
        trained = verify_trained_manifest(args.trained_manifest, args.trained_manifest_sha256, protocol,
                                          args.protocol_sha256)
        result = execute_episode(args, protocol, case, method, trained, session)
        # Once an official episode ends, a late low-memory/time observation must
        # never convert a completed failure into a retryable partial attempt.
        write_json(output / "report.json", result)
        session.finish("complete", official_success=result["official_success"], steps=result["steps"])
        receipt = seal(output, {"protocol_sha256": args.protocol_sha256,
                                "trained_manifest_sha256": args.trained_manifest_sha256, "case_id": args.case_id})
        print(json.dumps({"status": "complete", "case_id": args.case_id, "official_success": result["official_success"],
                          "receipt_sha256": receipt}))
        return 0
    except BaseException as error:
        if (output / "receipt.json").exists():
            # Never edit evidence after publication of its receipt, even if the
            # final receipt digest or stdout operation subsequently failed.
            print(json.dumps({"status": "sealed_requires_verification", "case_id": args.case_id,
                              "error": repr(error)}), flush=True)
            return 1
        terminal_observed = (output / "terminal_observed.json").exists() or (output / "report.json").exists() or result is not None
        retryable_resource_stop = isinstance(error, ResourceStop) and not terminal_observed
        try:
            session.finish("needs_attention" if terminal_observed else "resource_stopped" if retryable_resource_stop else "failed",
                           error=repr(error), traceback=traceback.format_exc())
        except BaseException as persistence_error:
            print(json.dumps({"error": repr(error), "persistence_error": repr(persistence_error)}), flush=True)
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            raise
        print(json.dumps({"status": "failed", "case_id": args.case_id, "error": repr(error)}), flush=True)
        return 75 if retryable_resource_stop else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform
    configure_platform()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    raise SystemExit(main())
