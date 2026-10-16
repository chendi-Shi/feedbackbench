"""Read-only audit of a saved interrupted single-case VLA-JEPA action prefix.

No model/environment is imported or executed. A passing prefix audit never
changes episode completion, creates a receipt, or passes strict validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vlajepa_long_horizon as long_stage
from run_vlajepa_compact import validate_arena_row, validate_compilation


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Evidence must remain inside project")
    return path.relative_to(ROOT).as_posix()


def identity(path):
    path = Path(path)
    return {"path": relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def check_prefix_actions(rows, chunk, call_index):
    """Match only durable executed rows; a final chunk can be partly consumed."""
    start = call_index * 7
    end = min(start + 7, len(rows))
    require(start < len(rows), "Saved complete call has no executed prefix")
    actual = np.asarray([row["action"] for row in rows[start:end]], dtype=np.float32)
    require(actual.shape == (end - start, 7) and np.isfinite(actual).all(), "Invalid executed actions")
    require(np.array_equal(actual, chunk[0, :end-start]), "Saved actions differ from predicted chunk prefix")
    return end - start


def audit(study):
    require("torch" not in sys.modules, "Torch was unexpectedly imported")
    study = Path(study).resolve()
    relative(study)
    runner = long_stage.configured_runner()
    protocol = runner.verify_protocol(study)
    require(len(protocol["spec"]["cases"]) == 1, "Audit requires exactly one planned case")
    case = protocol["spec"]["cases"][0]
    directory = study / "episodes" / case["id"]
    require(directory.is_dir(), "No interrupted episode directory")
    require(not (directory / "receipt.json").exists(), "Sealed episodes require the strict validator")
    require(not (directory / "result.json").exists(), "This prefix auditor does not assess terminal results")
    attempt = read(directory / "attempt.json")
    require(attempt["case"] == case and attempt["protocol_fingerprint"] == protocol["fingerprint"],
            "Attempt identity differs")
    run_path = runner.project_file(attempt["run_record"])
    require(run_path.parent == study / "runs", "Run record outside this study")
    run = read(run_path)
    require(run["protocol_fingerprint"] == protocol["fingerprint"] and run["weights_loaded"] is True,
            "Run protocol/loading identity differs")
    for module in run["actual_modules"].values():
        path = runner.project_file(module["path"])
        require(path.is_relative_to(runner.base.SOURCE) and runner.file_sha(path) == module["sha256"],
                "Official runtime source identity differs")
    stop_path = run_path.with_name(run_path.stem + ".memory_stop.json")
    stop = read(stop_path)
    require(stop["status"] == "stopped_memory_guard" and stop["action"] == "exit own process only",
            "Missing own-process memory stop")
    require(type(stop["pid"]) is int and stop["pid"] > 0, "Invalid stop PID")
    start_time = datetime.fromisoformat(run["started_at"])
    stop_time = datetime.fromisoformat(stop["utc"])
    require(0 < stop["process_create_time"] <= start_time.timestamp() <= stop_time.timestamp(),
            "Stop process/run timestamps disagree")
    guard = protocol["spec"]["memory_guard"]
    limits = stop["limits"]
    require(limits["max_rss_bytes"] == guard["maximum_rss_gib"] * 2**30
            and limits["min_available_bytes"] == guard["minimum_runtime_available_gib"] * 2**30,
            "Stop limits differ from frozen protocol")
    triggers = []
    if stop["rss_bytes"] > limits["max_rss_bytes"]:
        triggers.append("own_rss_limit")
    if stop["available_bytes"] < limits["min_available_bytes"]:
        triggers.append("system_available_floor")
    require(triggers, "Stop snapshot did not cross a frozen limit")

    rows = [json.loads(line) for line in (directory / "executed_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    require(0 < len(rows) < runner.HORIZONS[case["suite"]], "Not a nonterminal partial action prefix")
    for index, row in enumerate(rows):
        require(type(row["step"]) is int and row["step"] == index + 1, "Executed-step sequence differs")
        require(all(type(row[key]) is bool and row[key] is False for key in ("success", "terminated", "truncated")),
                "The saved prefix contains a terminal event")
        state = np.asarray(row["next_robot_state"], dtype=np.float32)
        require(state.shape == (8,) and np.isfinite(state).all(), "Invalid saved robot state")
    call_paths = sorted(directory.glob("call_*.json"))
    require([p.name for p in call_paths] == [f"call_{i:04d}.json" for i in range(len(call_paths))],
            "Call inventory is not contiguous")
    require(len(call_paths) == math.ceil(len(rows) / 7), "Complete call count does not match fixed-seven prefix")
    require({p.stem for p in directory.glob("call_*.npz")} == {p.stem for p in call_paths}, "Call NPZ inventory differs")
    expected_modules = None
    instruction = None
    total_seconds = 0.0
    matched_actions = 0
    for index, path in enumerate(call_paths):
        call = read(path)
        require(call["call_index"] == index and call["status"] == "complete"
                and call["forward_returned"] is True and call["action_head_nfe"] == 4, "Incomplete/wrong forward")
        modules = call["quantized_module_calls"]
        require(len(modules) == 300 and all(type(n) is int and n > 0 for n in modules.values()),
                "Missing 300-module execution coverage")
        if expected_modules is None:
            expected_modules = set(modules)
            instruction = call["task"]
        require(set(modules) == expected_modules and call["task"] == instruction, "Module/instruction identity changed")
        require(datetime.fromisoformat(call["finished_at"]) <= stop_time, "Call finished after process stop")
        for key in ("preprocess_seconds", "forward_seconds", "postprocess_seconds"):
            require(type(call[key]) in (float, int) and math.isfinite(call[key]) and call[key] >= 0, "Invalid call timing")
        total_seconds += call["forward_seconds"]
        npz = path.with_suffix(".npz")
        require(runner.file_sha(npz) == call["artifact_sha256"], "Call NPZ hash differs")
        with np.load(npz, allow_pickle=False) as chunk:
            require(set(chunk.files) == {"model_actions", "environment_actions", "noise", "rng_state", "input_robot_state"},
                    "Unexpected call array inventory")
            runner.validate_action_chunk(chunk["model_actions"])
            runner.validate_action_chunk(chunk["environment_actions"], environment=True)
            runner.validate_action_chunk(chunk["noise"])
            require(chunk["rng_state"].dtype == np.uint8 and chunk["rng_state"].ndim == 1
                    and chunk["rng_state"].size > 0, "Invalid saved RNG state")
            require(runner.array_sha(chunk["rng_state"]) == call["rng_state_sha256"]
                    and runner.array_sha(chunk["noise"]) == call["noise_sha256"], "Saved noise/RNG hash differs")
            state = chunk["input_robot_state"]
            require(state.dtype == np.float32 and state.shape == (8,) and np.isfinite(state).all()
                    and runner.array_sha(state) == call["input_sha256"]["observation.state"], "Input state hash differs")
            matched_actions += check_prefix_actions(rows, chunk["environment_actions"], index)
            if index == 0:
                for name, key in zip(("initial.png", "initial_wrist.png"), runner.FEATURES[:2], strict=True):
                    with Image.open(directory / name) as picture:
                        require(picture.mode == "RGB" and picture.size == (360, 360), "Initial image format differs")
                        image = np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
                    require(runner.array_sha(image) == call["input_sha256"][key], "Initial camera hash differs")
            else:
                latest = rows[index * 7 - 1]
                require(np.array_equal(state, np.asarray(latest["next_robot_state"], dtype=np.float32)),
                        "Replan state differs from the latest saved environment output")
                require(all(call["input_sha256"][key] == latest["next_camera_sha256"][key]
                            for key in runner.FEATURES[:2]), "Replan camera differs from latest output")
    require(matched_actions == len(rows), "Some executed actions were not matched")
    arena_path = directory / "arena_steps.jsonl"
    arena_rows = [json.loads(line) for line in arena_path.read_text(encoding="utf-8").splitlines()]
    require(len(arena_rows) == len(rows) + 1, "Missing reset/per-step arena record")
    for index, row in enumerate(arena_rows):
        require(row["step"] == index, "Arena step sequence differs")
        validate_arena_row(row)
    arena = read(directory / "arena_initialization.json")
    require(arena["memory_bytes"] == 268435456 and arena["native_warnings"] == []
            and bool(arena["compilations"]), "Arena initialization budget/warnings differ")
    for compilation in arena["compilations"]:
        validate_compilation(compilation)
    require("torch" not in sys.modules, "Torch was unexpectedly imported")
    return {
        "schema": "feedbackbench.vlajepa_interrupted_prefix_audit.v1",
        "status": "verified_saved_nonterminal_prefix",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "audit_source": identity(__file__), "protocol": identity(study / "protocol.json"),
        "protocol_fingerprint": protocol["fingerprint"], "frozen_source_files_verified": len(protocol["sources"]),
        "episode_id": case["id"], "planned": 1, "completed": 0, "episode_status": "unsealed",
        "official_task_outcome": "not_assessed_due_to_runtime_interruption",
        "strict_validation_passed": False, "receipt_created": False,
        "model_calls_executed_by_this_audit": 0, "torch_imported": False,
        "saved_executed_steps": len(rows), "complete_forward_calls": len(call_paths),
        "verified_action_head_nfe": len(call_paths) * 4, "complete_forward_seconds": total_seconds,
        "quantized_modules_executed_each_call": len(expected_modules),
        "executed_action_prefix_matches": True, "latest_replan_state_and_camera_hashes_match": True,
        "initial_camera_pixels_match": True, "saved_rng_and_noise_hashes_match": True,
        "complete_call_npz_hashes_match": True, "arena_rows": len(arena_rows),
        "arena_warnings": 0, "maximum_recorded_arena_use_bytes": max(row["maxuse_arena_bytes"] for row in arena_rows),
        "predicted_action_slots": 7 * len(call_paths),
        "prediction_slots_without_persisted_execution": 7 * len(call_paths) - len(rows),
        "last_saved_step": rows[-1], "stop": {**stop, "triggered_guards": triggers},
        "stop_pid_association": "Stop filename, protocol-run directory and creation/start/stop timestamps checked; run JSON has no independent PID binding",
        "evidence": [identity(study / "protocol.json"), identity(run_path), identity(stop_path)]
                    + [identity(path) for path in sorted(directory.iterdir()) if path.is_file()],
        "limitations": [
            "Only durable saved prefix verified; an in-flight action before process exit may not have been logged",
            "No final trajectory/result/receipt; no completed episode, task failure, or success-rate denominator inferred",
            "RNG/noise hashes checked without re-executing RNG or loading any model",
            "Later camera checks compare saved hashes; those camera pixels are not independently rerendered",
            "The initial input robot state is checked against its saved hash, not independently simulated",
            "300-module sets are consistent across calls; no complete result metadata exists for an independent runtime-module inventory",
            "Stop memory is one snapshot, not a whole-run peak/minimum",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/frontier/vlajepa_long_horizon")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vlajepa-long-horizon-audit-20261016.json")
    args = parser.parse_args()
    relative(args.output)
    require(not args.output.exists(), "Audit output already exists; use a new path")
    result = audit(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "saved_executed_steps": result["saved_executed_steps"],
                      "complete_forward_calls": result["complete_forward_calls"],
                      "completed": 0, "strict_validation_passed": False,
                      "output": relative(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
