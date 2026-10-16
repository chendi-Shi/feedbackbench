"""Recover one interrupted trajectory using exact replay and a recomputed boundary."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_vlajepa_long_horizon as long_stage

DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_recovery"
SOURCE_OUTPUT = ROOT / "outputs/frontier/vlajepa_long_horizon"
REPLAY_STEPS, PREFIX_STEPS, BOUNDARY_CALL = 392, 398, 56
SOURCES = ("scripts/run_vlajepa_recovery.py", "docs/VLAJEPA_RECOVERY_PROTOCOL.md", "scripts/audit_vlajepa_interrupted.py")
ARRAYS = {"model_actions", "environment_actions", "noise", "rng_state", "input_robot_state"}


def require(value, message):
    if not value:
        raise ValueError(message)


def separate_output(output):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT.resolve()), "Recovery evidence must remain inside the project")
    for name in ("vlajepa_closed_loop", "vlajepa_paged_closed_loop", "vlajepa_compact_closed_loop",
                 "vlajepa_isolated_closed_loop", "vlajepa_long_horizon"):
        previous = (ROOT / "outputs/frontier" / name).resolve()
        require(not output.is_relative_to(previous) and not previous.is_relative_to(output), "Previous studies must remain separate")
    return output


def phase_for_step(step):
    require(type(step) is int and 1 <= step <= 520, "Invalid recovery step")
    return "saved_action_replay" if step <= REPLAY_STEPS else "recomputed_prefix_overlap" if step <= PREFIX_STEPS else "new_continuation"


def recovery_counts(steps, actual_calls):
    require(type(steps) is int and PREFIX_STEPS < steps <= 520, "Recovery must execute beyond the saved prefix")
    require(type(actual_calls) is int and actual_calls == math.ceil(steps / 7) - BOUNDARY_CALL, "Recovery actual forward count differs")
    return {"independent_benchmark_samples": 0, "replayed_steps": REPLAY_STEPS, "recomputed_overlap_steps": 6,
            "new_execution_steps": steps - PREFIX_STEPS, "boundary_recomputed_forwards": 1,
            "new_policy_forwards": actual_calls - 1, "actual_model_forwards": actual_calls,
            "historical_completed_forwards": 57, "historical_chunks_replayed_without_inference": 56}


def exact_observation(runner, observation, expected):
    import numpy as np
    actual = observation["observation.state"]
    state = np.asarray(expected["next_robot_state"], dtype=np.float32)
    runner.require(actual.dtype == np.float32 and actual.shape == state.shape == (8,)
                   and np.array_equal(actual, state), "Recovered robot observation differs")
    runner.require(all(runner.array_sha(observation[key]) == expected["next_camera_sha256"][key]
                       for key in runner.FEATURES[:2]), "Recovered camera pixels differ")


def exact_step(runner, observation, terminated, truncated, info, expected):
    exact_observation(runner, observation, expected)
    runner.require(all(type(expected[key]) is bool for key in ("terminated", "truncated", "success"))
                   and bool(terminated) == expected["terminated"] and bool(truncated) == expected["truncated"]
                   and bool(info["is_success"]) == expected["success"], "Recovered terminal/success flags differ")


def compare_boundary(runner, directory, source_directory):
    import numpy as np
    current = runner.read_json(Path(directory) / "call_0000.json")
    original = runner.read_json(Path(source_directory) / "call_0056.json")
    runner.require(current["call_index"] == 0 and original["call_index"] == 56, "Boundary call indices differ")
    for record in (current, original):
        runner.require(record["status"] == "complete" and record["forward_returned"] is True
                       and record["action_head_nfe"] == 4 and len(record["quantized_module_calls"]) == 300
                       and all(type(value) is int and value > 0 for value in record["quantized_module_calls"].values()),
                       "Boundary did not complete all required modules and four NFE")
    for key in ("input_sha256", "task", "rng_state_sha256", "noise_sha256", "quantized_module_calls", "action_head_nfe"):
        runner.require(current[key] == original[key], "Boundary recomputation differs: " + key)
    new_path, old_path = Path(directory) / "call_0000.npz", Path(source_directory) / "call_0056.npz"
    runner.require(runner.file_sha(new_path) == current["artifact_sha256"]
                   and runner.file_sha(old_path) == original["artifact_sha256"], "Boundary NPZ identity differs")
    with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
        runner.require(set(new.files) == set(old.files) == ARRAYS, "Boundary array inventory differs")
        for name in ARRAYS:
            runner.require(new[name].dtype == old[name].dtype and new[name].shape == old[name].shape
                           and np.array_equal(new[name], old[name]), "Boundary array differs: " + name)
    return {"status": "exact_recomputed_boundary", "source_call": 56, "actual_call": 0,
            "arrays_equal": sorted(ARRAYS), "module_call_counts_equal": True, "action_head_nfe": 4,
            "source_record_sha256": runner.file_sha(Path(source_directory) / "call_0056.json"),
            "actual_record_sha256": runner.file_sha(Path(directory) / "call_0000.json")}


def source_inventory(runner):
    return {runner.base.relative(path): {"sha256": runner.file_sha(path), "bytes": path.stat().st_size}
            for path in sorted(SOURCE_OUTPUT.rglob("*")) if path.is_file()}


def verify_source_inventory(runner, source):
    runner.require(source_inventory(runner) == source["files"], "Original interrupted evidence changed after recovery freeze")


def configured_runner():
    import audit_vlajepa_interrupted as interrupted
    runner = long_stage.configured_runner()
    prefix = interrupted.audit(SOURCE_OUTPUT)
    runner.require(prefix["saved_executed_steps"] == PREFIX_STEPS and prefix["complete_forward_calls"] == 57
                   and prefix["prediction_slots_without_persisted_execution"] == 1, "Recovery requires the fixed 398-step/57-call prefix")
    source = {"study": runner.base.relative(SOURCE_OUTPUT), "episode_id": prefix["episode_id"],
              "protocol_fingerprint": prefix["protocol_fingerprint"], "files": source_inventory(runner),
              "saved_steps": PREFIX_STEPS, "complete_forward_calls": 57}
    prior_cases, prior_spec = runner.cases_for_study, runner.study_spec
    runner.SOURCES = (*runner.SOURCES, *SOURCES)

    def cases():
        rows = prior_cases()
        runner.require(len(rows) == 1 and rows[0]["suite"] == "libero_10", "Recovery case differs")
        rows[0]["model"], rows[0]["split"] = "vlajepa_recovery", "interrupted_trajectory_recovery"
        rows[0]["id"] = runner.episode_id(rows[0])
        return rows

    def spec():
        value = prior_spec()
        value["name"] = "VLA-JEPA recovery of one interrupted trajectory; zero new independent benchmark samples"
        value["cases"] = cases()
        value["recovery"] = {"source": source, "replay_steps": REPLAY_STEPS, "overlap_steps": 6,
            "boundary_source_call": BOUNDARY_CALL, "first_new_execution_step": 399,
            "rng": "restore saved BEFORE state and recompute the entire final forward; never infer AFTER",
            "independent_benchmark_episodes": 0, "original_attempt_remains_unsealed": True,
            "video": "frames explicitly labelled saved-action replay, recomputed overlap, or new continuation"}
        value["limitations"].extend(["Recovery of an existing interrupted trajectory, not a new independent evaluation sample",
            "Exact observable robot states and both camera hashes do not prove equality of unrecorded full MuJoCo internal state",
            "Prefix replay and one repeated boundary forward are included in recovery wall time; original timing is not synthesized"])
        return value

    runner.cases_for_study, runner.study_spec = cases, spec
    runner.verify_episode = lambda directory, case, protocol: verify_episode(runner, directory, case, protocol, source)
    runner.verify_study = lambda output, partial=False: verify_study(runner, output, partial=partial)
    runner.write_report = lambda output: write_report(runner, output)
    runner.run_episode = lambda simulator, policy, case, destination: run_recovery_episode(runner, source, simulator, policy, case, destination)
    return runner


def restore_rng(before):
    import torch
    torch.set_rng_state(torch.from_numpy(before.copy()))


def current_rng():
    import torch
    return torch.get_rng_state().numpy().copy()


def run_recovery_episode(runner, source, simulator, policy, case, destination):
    import numpy as np
    from PIL import Image, ImageDraw
    from embodied_research.scheduling import ChunkScheduler, SchedulerConfig
    verify_source_inventory(runner, source)
    directory = Path(destination)
    original = SOURCE_OUTPUT / "episodes" / source["episode_id"]
    saved = [json.loads(line) for line in (original / "executed_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    started = time.perf_counter()
    observation = simulator.reset(seed=case["seed"], init_state_id=case["init_state_id"])
    policy.reset(seed=case["seed"])
    runner.require(case["condition"] == "clean", "Recovery only supports the frozen clean condition")
    first = runner.read_json(original / "call_0000.json")
    runner.require(all(runner.array_sha(observation[key]) == first["input_sha256"][key] for key in runner.FEATURES),
                   "Recovery reset observation differs from original call zero")
    initial_state = observation["observation.state"].copy()
    initial_frame = simulator.frame()
    initial_image_sha256 = hashlib.sha256(initial_frame.tobytes()).hexdigest()
    Image.fromarray(initial_frame).save(directory / "initial.png")
    Image.fromarray(simulator.frame("image2")).save(directory / "initial_wrist.png")
    scheduler = ChunkScheduler(SchedulerConfig(**{**case["scheduler"], "horizon": simulator.horizon}))
    frames = [] if case.get("record_video") else None

    def frame(step):
        picture = Image.fromarray(simulator.frame()).copy()
        label = "SAVED ACTION REPLAY" if step <= REPLAY_STEPS else "RECOMPUTED OVERLAP" if step <= PREFIX_STEPS else "NEW CONTINUATION"
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, picture.width, 24), fill="black")
        draw.text((4, 5), f"{label} | step {step}", fill="white")
        return picture

    if frames is not None:
        frames.append(frame(0))
    checks_path = directory / "recovery_checks.jsonl"
    runner.require(not checks_path.exists(), "Previous recovery checks retained")
    with checks_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": 0, "phase": "initial", "input_sha256": first["input_sha256"], "exact": True}) + "\n")
    actions, states, model_times, prep_times, control_times, sim_times, replans = [], [], [], [], [], [], []
    boundary = None
    replay_finished = overlap_finished = None
    terminated = truncated = success = False
    for step in range(simulator.horizon):
        control_started = time.perf_counter()
        policy.observe(observation)
        planned = scheduler.should_replan(observation)
        actual_forward = planned and step >= REPLAY_STEPS
        if planned:
            if step < REPLAY_STEPS:
                with np.load(original / f"call_{step // 7:04d}.npz", allow_pickle=False) as saved_chunk:
                    chunk = saved_chunk["environment_actions"][0].copy()
            else:
                if step == REPLAY_STEPS:
                    with np.load(original / "call_0056.npz", allow_pickle=False) as saved_chunk:
                        before = saved_chunk["rng_state"].copy()
                    restore_rng(before)
                    runner.require(np.array_equal(current_rng(), before), "Restored boundary RNG differs")
                chunk = policy.predict_chunk()
                if step == REPLAY_STEPS:
                    boundary = compare_boundary(runner, directory, original)
                    after = current_rng()
                    with (directory / "boundary_rng.npz").open("xb") as stream:
                        np.savez_compressed(stream, before=before, after=after)
                    boundary.update(before_sha256=runner.array_sha(before), after_sha256=runner.array_sha(after),
                                    rng_artifact_sha256=runner.file_sha(directory / "boundary_rng.npz"))
                    runner.write_json(directory / "boundary.json", boundary)
                replans.append(step)
            scheduler.submit_chunk(chunk)
        action = scheduler.next_action()
        if step < PREFIX_STEPS:
            runner.require(np.array_equal(action, np.asarray(saved[step]["action"], dtype=np.float32)), "Recovered action differs from saved prefix")
        control_times.append(time.perf_counter() - control_started)
        model_times.append(policy.last_timing["inference_seconds"] if actual_forward else 0.0)
        prep_times.append(policy.last_timing["preprocess_seconds"] if actual_forward else 0.0)
        actions.append(action.copy())
        states.append(observation["observation.state"].copy())
        sim_started = time.perf_counter()
        observation, reward, terminated, truncated, info = simulator.step(action)
        sim_times.append(time.perf_counter() - sim_started)
        success = bool(info["is_success"])
        if step < PREFIX_STEPS:
            exact_step(runner, observation, terminated, truncated, info, saved[step])
            with checks_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step + 1, "phase": phase_for_step(step + 1),
                    "source_row_sha256": runner.digest_json(saved[step]), "exact": True}) + "\n")
                stream.flush()
        if step + 1 == REPLAY_STEPS:
            replay_finished = time.perf_counter()
        if step + 1 == PREFIX_STEPS:
            overlap_finished = time.perf_counter()
        if frames is not None and ((step + 1) % 4 == 0 or terminated or truncated):
            frames.append(frame(step + 1))
        if (step + 1) % 50 == 0:
            print(f"recovery step={step + 1}/520 actual_forwards={policy.inference_count} phase={phase_for_step(step + 1)}", flush=True)
        if terminated or truncated:
            break
    rollout_finished = time.perf_counter()
    runner.require(boundary is not None and overlap_finished is not None, "Recovery ended before the verified overlap")
    counts = recovery_counts(simulator.steps, policy.inference_count)
    Image.fromarray(simulator.frame()).save(directory / "final.png")
    Image.fromarray(simulator.frame("image2")).save(directory / "final_wrist.png")
    if frames:
        frames[0].save(directory / "rollout.gif", save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=False)
    np.savez_compressed(directory / "trajectory.npz", actions=np.asarray(actions), states=np.asarray(states),
        final_state=observation["observation.state"], model_seconds=np.asarray(model_times), preprocessing_seconds=np.asarray(prep_times),
        control_seconds=np.asarray(control_times), simulator_seconds=np.asarray(sim_times), replan_steps=np.asarray(replans))
    runner.write_json(directory / "recovery.json", {"source": source, "counts": counts,
        "prefix_checks_sha256": runner.file_sha(checks_path), "boundary_sha256": runner.file_sha(directory / "boundary.json"),
        "video_labels": {"through_step_392": "SAVED ACTION REPLAY", "steps_393_to_398": "RECOMPUTED OVERLAP", "from_step_399": "NEW CONTINUATION"}})
    verify_source_inventory(runner, source)
    return {**case, "status": "complete", "task": simulator.task, "success": success,
        "steps": simulator.steps, "horizon": simulator.horizon, "terminated": bool(terminated), "truncated": bool(truncated),
        "inference_calls": policy.inference_count, "policy_inference_seconds": float(sum(model_times)),
        "preprocess_seconds": float(sum(prep_times)), "simulator_seconds": float(sum(sim_times)),
        "scheduler_seconds": float(max(0.0, sum(control_times) - sum(model_times) - sum(prep_times))),
        "wall_seconds": rollout_finished - started, "wall_time_scope": "fresh reset, replay, boundary recomputation, overlap and continuation; excludes model loading and GIF encoding",
        "recovery_timing": {"reset_and_replay_seconds": replay_finished - started,
            "boundary_and_overlap_seconds": overlap_finished - replay_finished,
            "new_continuation_seconds": rollout_finished - overlap_finished},
        "initial_robot_state": initial_state.tolist(), "initial_image_sha256": initial_image_sha256,
        "final_robot_state": observation["observation.state"].tolist(), "model_info": policy.metadata,
        "environment": simulator.metadata(), "scheduler_config": asdict(scheduler.config),
        "decisions": {"logical_scheduler": scheduler.diagnostics(), "actual_model_calls": policy.inference_count,
                      "saved_chunks_without_new_inference": BOUNDARY_CALL},
        "recovery_counts": counts, "independent_benchmark_episodes": 0, "video": "rollout.gif" if frames else None,
        "finished_at": runner.utc_now()}


def verify_episode(runner, directory, case, protocol, source):
    import numpy as np
    from PIL import Image
    from embodied_research.scheduling import SchedulerConfig
    directory = Path(directory)
    verify_source_inventory(runner, source)
    result = runner.verify_complete(directory, case, protocol["fingerprint"])
    runner.require(result is not None and not (directory / "error.json").exists(), "Recovery has no clean sealed terminal result")
    steps = result["steps"]
    counts = recovery_counts(steps, result["inference_calls"])
    runner.require(result["horizon"] == 520 and result["independent_benchmark_episodes"] == 0
                   and result["recovery_counts"] == counts, "Recovery scope/counts differ")
    runner.require(all(type(result[key]) is bool for key in ("success", "terminated", "truncated"))
                   and result["success"] == result["terminated"] and result["truncated"] == (not result["success"])
                   and (result["success"] or steps == 520), "Recovery did not reach an official terminal or full horizon")
    attempt = runner.read_json(directory / "attempt.json")
    metadata = result["model_info"]
    runner.require(attempt["case"] == case and attempt["protocol_fingerprint"] == protocol["fingerprint"]
                   and attempt["run_record"] == metadata["run_record"], "Recovery attempt/run identity differs")
    runner.require(metadata["variant"] == runner.VARIANT and len(set(metadata["quantized_modules"])) == len(metadata["quantized_modules"]) == 300
                   and metadata["policy_revision"] == runner.base.POLICY_REVISION
                   and metadata["lerobot_revision"] == runner.base.REVISION
                   and metadata["inputs_sha256"] == protocol["inputs_sha256"], "Recovery model identity differs")
    run = runner.read_json(runner.project_file(metadata["run_record"]))
    runner.require(run["protocol_fingerprint"] == protocol["fingerprint"] and run["weights_loaded"] is True,
                   "Recovery model load is not bound to this protocol")
    loading_path = runner.project_file(metadata["loading"]["path"])
    runner.require(runner.file_sha(loading_path) == metadata["loading"]["sha256"], "Recovery loading audit changed")
    loading = runner.read_json(loading_path)
    runner.require(loading["all_required_inference_keys_consumed"] is True and loading["all_source_tensors_finite"] is True
                   and loading["native_tied_word_embeddings"] is True and loading["linear_modules"] == 300
                   and loading["theoretical_parameter_payload_bytes"] == runner.quant.EXPECTED_PAYLOAD_BYTES, "Recovery loading coverage differs")
    long_stage.compact.paged.validate_paging(loading["paging"])
    paging = loading["paging"]
    runner.require(paging["embedding_shape"] == [151936, 2048] and paging["embedding_dtype"] == "torch.bfloat16"
                   and paging["tied_parameter_identity"] is True and paging["checkpoint_sha256"] == runner.base.POLICY_SHA256,
                   "Recovery paged/tied parameter identity differs")
    runner.require(set(metadata["actual_modules"]) == {"lerobot", "environment", "policy"}, "Recovery runtime module inventory differs")
    for module in metadata["actual_modules"].values():
        path = runner.project_file(module["path"])
        runner.require(path.is_relative_to(runner.base.SOURCE) and runner.file_sha(path) == module["sha256"], "Recovery runtime source changed")
    environment = {"suite": "libero_10", "task_id": 1, "horizon": 520, "image_size": 360, "control_hz": 20,
                   "settle_steps": 10, "control_mode": "relative", "hard_reset": True,
                   "images_flipped_hw_once": True, "object_state_exposed_to_policy": False}
    runner.require(all(result["environment"].get(key) == value for key, value in environment.items()), "Recovery environment contract differs")
    runner.require(result["scheduler_config"] == asdict(SchedulerConfig(**{**case["scheduler"], "horizon": 520})), "Recovery scheduling contract differs")
    timing = result["recovery_timing"]
    runner.require(set(timing) == {"reset_and_replay_seconds", "boundary_and_overlap_seconds", "new_continuation_seconds"}
                   and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in timing.values())
                   and math.isfinite(result["wall_seconds"]) and result["wall_seconds"] >= 0
                   and math.isclose(sum(timing.values()), result["wall_seconds"], rel_tol=1e-9, abs_tol=1e-8),
                   "Recovery timing segments differ from wall time")
    decisions = result["decisions"]
    logical = decisions["logical_scheduler"]
    runner.require(decisions["actual_model_calls"] == result["inference_calls"]
                   and decisions["saved_chunks_without_new_inference"] == BOUNDARY_CALL
                   and logical["plan_steps"] == list(range(0, steps, 7)) and logical["steps"] == steps
                   and logical["inference_requests"] == logical["chunks_received"] == math.ceil(steps / 7),
                   "Logical replay scheduling was confused with actual model inference")
    original = SOURCE_OUTPUT / "episodes" / source["episode_id"]
    saved = [json.loads(line) for line in (original / "executed_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [json.loads(line) for line in (directory / "executed_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    runner.require(len(rows) == steps and rows[:PREFIX_STEPS] == saved, "Recovery's 398 observable/action/flag rows differ from the source")
    checks = [json.loads(line) for line in (directory / "recovery_checks.jsonl").read_text(encoding="utf-8").splitlines()]
    first = runner.read_json(original / "call_0000.json")
    runner.require(len(checks) == PREFIX_STEPS + 1
                   and checks[0] == {"step": 0, "phase": "initial", "input_sha256": first["input_sha256"], "exact": True}, "Initial/prefix replay proof differs")
    for index, row in enumerate(saved, 1):
        runner.require(checks[index] == {"step": index, "phase": phase_for_step(index),
            "source_row_sha256": runner.digest_json(row), "exact": True}, "Replay proof sequence or binding differs")
    recovery = runner.read_json(directory / "recovery.json")
    runner.require(recovery["source"] == source and recovery["counts"] == counts
                   and recovery["prefix_checks_sha256"] == runner.file_sha(directory / "recovery_checks.jsonl")
                   and recovery["boundary_sha256"] == runner.file_sha(directory / "boundary.json"), "Recovery source/check binding differs")
    runner.require(recovery["video_labels"] == {"through_step_392": "SAVED ACTION REPLAY", "steps_393_to_398": "RECOMPUTED OVERLAP", "from_step_399": "NEW CONTINUATION"}
                   and result["video"] == "rollout.gif" and (directory / "rollout.gif").is_file(),
                   "Recovery requires its labelled replay/continuation recording")
    boundary = compare_boundary(runner, directory, original)
    recorded_boundary = runner.read_json(directory / "boundary.json")
    runner.require(all(recorded_boundary.get(key) == value for key, value in boundary.items()), "Recorded boundary comparison differs")
    runner.require(recorded_boundary["rng_artifact_sha256"] == runner.file_sha(directory / "boundary_rng.npz"), "Boundary RNG artifact changed")
    with np.load(directory / "boundary_rng.npz", allow_pickle=False) as rng, np.load(original / "call_0056.npz", allow_pickle=False) as old:
        runner.require(set(rng.files) == {"before", "after"} and rng["after"].dtype == np.uint8
                       and rng["after"].shape == old["rng_state"].shape and np.array_equal(rng["before"], old["rng_state"])
                       and not np.array_equal(rng["before"], rng["after"]), "Boundary RNG before/actual after differ")
        for name in ("before", "after"):
            runner.require(runner.array_sha(rng[name]) == recorded_boundary[name + "_sha256"], "Boundary RNG hash differs")
        if result["inference_calls"] > 1:
            with np.load(directory / "call_0001.npz", allow_pickle=False) as following:
                runner.require(np.array_equal(rng["after"], following["rng_state"]), "Next forward did not continue the actual boundary RNG")
    calls = runner.read_json(directory / "calls.json")
    runner.require(calls == metadata["call_log"] and len(calls) == result["inference_calls"] == metadata["inference_count"]
                   and metadata["action_head_nfe"] == 4 * len(calls), "Actual recovery call inventory/count differs")
    runner.require({path.name for path in directory.glob("call_*.json")} == {f"call_{i:04d}.json" for i in range(len(calls))}
                   and {path.name for path in directory.glob("call_*.npz")} == {f"call_{i:04d}.npz" for i in range(len(calls))}, "Unexpected or missing real call evidence")
    expected_replans = list(range(REPLAY_STEPS, steps, 7))
    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        runner.require(set(trajectory.files) == {"actions", "states", "final_state", "model_seconds", "preprocessing_seconds", "control_seconds", "simulator_seconds", "replan_steps"}, "Recovery trajectory inventory differs")
        for key in trajectory.files:
            runner.require(np.isfinite(trajectory[key]).all(), "Nonfinite recovery trajectory")
        actions, states = trajectory["actions"], trajectory["states"]
        runner.require(actions.dtype == states.dtype == np.float32 and actions.shape == (steps, 7) and states.shape == (steps, 8)
                       and np.array_equal(trajectory["replan_steps"], expected_replans), "Recovery trajectory/replan dimensions differ")
        runner.require(np.array_equal(states[0], np.asarray(result["initial_robot_state"], dtype=np.float32))
                       and np.array_equal(trajectory["final_state"], np.asarray(result["final_robot_state"], dtype=np.float32))
                       and runner.array_sha(states[0]) == first["input_sha256"]["observation.state"], "Recovery initial/final state differs")
        for index, row in enumerate(rows):
            next_state = states[index + 1] if index < steps - 1 else trajectory["final_state"]
            runner.require(type(row["step"]) is int and row["step"] == index + 1
                           and np.array_equal(actions[index], np.asarray(row["action"], dtype=np.float32))
                           and np.array_equal(next_state, np.asarray(row["next_robot_state"], dtype=np.float32)), "Recovery step/action/state differs")
            runner.require(all(type(row[key]) is bool for key in ("success", "terminated", "truncated")), "Recovery step flags must be booleans")
            runner.require(index == steps - 1 or not any(row[key] for key in ("success", "terminated", "truncated")), "Recovery continued after a terminal")
        runner.require(all(rows[-1][key] == result[key] for key in ("success", "terminated", "truncated")), "Recovery final flags differ")
        model_times, prep_times = np.zeros(steps), np.zeros(steps)
        for index, reference in enumerate(calls):
            path = directory / f"call_{index:04d}.json"
            runner.require(reference["call_index"] == index and reference["record"] == path.name
                           and runner.file_sha(path) == reference["sha256"], "Actual recovery call binding differs")
            call = runner.read_json(path)
            runner.require(call["call_index"] == index and call["status"] == "complete" and call["forward_returned"] is True
                           and call["action_head_nfe"] == 4 and set(call["quantized_module_calls"]) == set(metadata["quantized_modules"])
                           and all(type(value) is int and value > 0 for value in call["quantized_module_calls"].values()), "Actual recovery forward/module proof differs")
            runner.require(call["task"] == result["task"], "Recovery language instruction differs")
            for key in ("forward_seconds", "preprocess_seconds", "postprocess_seconds"):
                runner.require(type(call[key]) in (int, float) and math.isfinite(call[key]) and call[key] >= 0, "Invalid recovery call timing")
            data_path = path.with_suffix(".npz")
            runner.require(runner.file_sha(data_path) == call["artifact_sha256"], "Actual recovery NPZ changed")
            start, end = expected_replans[index], min(expected_replans[index] + 7, steps)
            with np.load(data_path, allow_pickle=False) as chunk:
                runner.require(set(chunk.files) == ARRAYS, "Actual recovery call array inventory differs")
                for name in ("model_actions", "environment_actions", "noise"):
                    runner.validate_action_chunk(chunk[name], environment=name == "environment_actions")
                runner.require(chunk["rng_state"].dtype == np.uint8 and chunk["rng_state"].ndim == 1 and chunk["rng_state"].size > 0
                               and runner.array_sha(chunk["rng_state"]) == call["rng_state_sha256"]
                               and runner.array_sha(chunk["noise"]) == call["noise_sha256"], "Recovery actual RNG/noise differs")
                runner.require(chunk["input_robot_state"].dtype == np.float32 and chunk["input_robot_state"].shape == (8,)
                               and np.array_equal(states[start], chunk["input_robot_state"])
                               and runner.array_sha(states[start]) == call["input_sha256"]["observation.state"], "Recovery forward used stale state")
                runner.require(np.array_equal(actions[start:end], chunk["environment_actions"][0, :end-start]), "Recovery action queue differs")
                runner.require(all(call["input_sha256"][key] == rows[start - 1]["next_camera_sha256"][key] for key in runner.FEATURES[:2]), "Recovery forward used stale cameras")
            model_times[start], prep_times[start] = call["forward_seconds"], call["preprocess_seconds"]
        for name, expected in (("model_seconds", model_times), ("preprocessing_seconds", prep_times)):
            runner.require(trajectory[name].shape == (steps,) and np.allclose(trajectory[name], expected, rtol=0, atol=1e-10), "Recovery per-step inference timing differs")
        for name, total in (("model_seconds", "policy_inference_seconds"), ("preprocessing_seconds", "preprocess_seconds"), ("simulator_seconds", "simulator_seconds")):
            runner.require(trajectory[name].shape == (steps,) and (trajectory[name] >= 0).all()
                           and math.isclose(float(trajectory[name].sum()), result[total], rel_tol=1e-9, abs_tol=1e-8), "Recovery timing total differs")
        runner.require(trajectory["control_seconds"].shape == (steps,) and (trajectory["control_seconds"] >= 0).all(), "Recovery control timing differs")
    for filename, expected in (("initial.png", first["input_sha256"][runner.FEATURES[0]]),
                               ("initial_wrist.png", first["input_sha256"][runner.FEATURES[1]]),
                               ("final.png", rows[-1]["next_camera_sha256"][runner.FEATURES[0]]),
                               ("final_wrist.png", rows[-1]["next_camera_sha256"][runner.FEATURES[1]])):
        with Image.open(directory / filename) as picture:
            runner.require(picture.mode == "RGB" and picture.size == (360, 360), "Recovery image dimensions differ")
            if filename == "initial.png":
                runner.require(hashlib.sha256(np.asarray(picture).tobytes()).hexdigest() == result["initial_image_sha256"],
                               "Recovery initial raw pixels differ")
            pixels = np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        runner.require(runner.array_sha(pixels) == expected, "Recovery saved image pixels differ")
    arena = [json.loads(line) for line in (directory / "arena_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    runner.require(len(arena) == steps + 1, "Recovery arena trace is incomplete")
    for index, row in enumerate(arena):
        long_stage.compact.validate_arena_row(row)
        runner.require(row["step"] == index, "Recovery arena step order differs")
    initialization = runner.read_json(directory / "arena_initialization.json")
    runner.require(initialization["memory_bytes"] == 268435456 and initialization["native_warnings"] == []
                   and bool(initialization["compilations"]), "Recovery arena initialization differs")
    for compilation in initialization["compilations"]:
        long_stage.compact.validate_compilation(compilation)
    verify_source_inventory(runner, source)
    return result


def verify_study(runner, output, *, partial=False):
    output = Path(output)
    protocol = runner.verify_protocol(output)
    cases = protocol["spec"]["cases"]
    runner.require(len(cases) == 1 and cases == runner.cases_for_study(), "Recovery requires exactly its own one case")
    case = cases[0]
    runner.require(all(path.name == case["id"] for path in (output / "episodes").glob("*")), "Unplanned recovery episode")
    directory = output / "episodes" / case["id"]
    results = []
    if not directory.exists():
        status = "missing"
    elif not (directory / "receipt.json").exists():
        status = "error" if (directory / "error.json").exists() else "unsealed"
    else:
        results.append(runner.verify_episode(directory, case, protocol))
        status = "complete"
    complete = status == "complete"
    runner.require(partial or complete, "Recovery incomplete; partial must be explicit")
    return {"protocol_fingerprint": protocol["fingerprint"], "complete": complete, "planned": 1, "completed": len(results),
        "successes": sum(row["success"] for row in results), "task_failures": sum(not row["success"] for row in results),
        "statuses": {case["id"]: status}, "results": results, "limitations": protocol["spec"]["limitations"],
        "independent_benchmark_episodes": 0, "evaluation_scope": "recovery_of_existing_interrupted_trajectory",
        "recovery_counts": results[0]["recovery_counts"] if results else None}


def write_report(runner, output):
    output = Path(output)
    study = runner.verify_study(output, partial=True)
    study["run_audits"] = []
    for path in sorted((output / "runs").glob("run_*.json")):
        record = runner.read_json(path)
        study["run_audits"].append({"path": path.relative_to(output).as_posix(), "sha256": runner.file_sha(path),
            "status": record["status"], "weights_loaded": record.get("weights_loaded", False),
            "memory": {key: value for key, value in record.get("memory", {}).items() if key != "samples"},
            "load_quantize_seconds": record.get("load_quantize_seconds"), "imports_seconds": record.get("imports_seconds")})
    long_stage.compact.reliable_write_json(runner.write_json, output / "report.json", study)
    lines = ["# Interrupted VLA-JEPA trajectory recovery", "", "Independent new benchmark episodes: **0**.",
             f"Planned recovery: **1**; completed: **{study['completed']}**; official task successes: **{study['successes']}**; full-horizon task failures: **{study['task_failures']}**.",
             "The original interrupted attempt remains unsealed. This recovery does not form a new four-of-four experiment.", ""]
    for row in study["results"]:
        lines += [f"- Logical trajectory: {row['steps']} / 520 steps.",
                  "- Saved-action replay: 392 steps; recomputed overlap checked against the original: 6 steps.",
                  f"- Newly executed continuation: {row['recovery_counts']['new_execution_steps']} steps; boundary forward recomputed once; subsequent new forwards: {row['recovery_counts']['new_policy_forwards']}.",
                  f"- Recovery wall time: {row['wall_seconds']:.3f}s, including reset/replay/recomputation/continuation; model loading and GIF encoding excluded.",
                  f"- Timing segments: `{json.dumps(row['recovery_timing'], sort_keys=True)}`.", "",
                  f"[Result](episodes/{row['id']}/result.json) · [Recovery proof](episodes/{row['id']}/recovery.json) · [Boundary](episodes/{row['id']}/boundary.json)", "",
                  f"![Labelled replay and continuation](episodes/{row['id']}/rollout.gif)", ""]
    lines += ["## Run and loading evidence", ""]
    for audit in study["run_audits"]:
        lines.append(f"- [Run]({audit['path']}): `{audit['status']}`; loading {audit['load_quantize_seconds']}s; memory `{json.dumps(audit['memory'], sort_keys=True)}`.")
    lines += ["", "## Limitations", "", *["- " + item for item in study["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {key: value for key, value in study.items() if key != "results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    output, runner = separate_output(args.output), configured_runner()
    if args.command == "freeze":
        result = runner.run(output, freeze_only=True)
    elif args.command == "run":
        result = long_stage.run(runner, output)
    elif args.command == "report":
        result = runner.write_report(output)
    else:
        result = runner.verify_study(output, partial=args.partial)
        result.pop("results")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
