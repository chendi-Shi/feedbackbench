"""Exact four-trajectory replay for a separate 256 MiB MuJoCo arena diagnostic.

No robot policy is loaded. The source trajectories remain scored only in their
original study; this diagnostic tests environment allocation equivalence.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from complete_vlajepa_stage import replace_with_retry

OUTPUT = ROOT / "artifacts/frontier/libero_arena_replay_20261016.json"
PLAN = OUTPUT.with_suffix(".plan.json")
EVIDENCE = OUTPUT.with_suffix("")
STUDY = ROOT / "outputs/frontier/flow_confirmation"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARENA_BYTES = 268435456


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = module("arena_probe_native_base", "scripts/probe_vlajepa.py")
flow = module("arena_probe_frozen_flow", "scripts/run_flow_confirmation.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def binding(path):
    return {"path": relative(path), "sha256": sha(path)}


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if exclusive:
        with path.open("xb") as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    paths = dict.fromkeys(("scripts/probe_libero_arena.py", "scripts/libero_arena_budget.py",
                          "scripts/probe_vlajepa.py", "scripts/run.py", "scripts/complete_vlajepa_stage.py",
                          "scripts/run_flow_confirmation.py", *flow.SOURCES,
                          "frontier/vendor/lerobot/src/lerobot/envs/libero.py"))
    return {name: sha(ROOT / name) for name in paths}


def reference_inputs():
    protocol = read(STUDY / "protocol.json")
    flow.verify_protocol(protocol, ROOT)
    require(flow.initial_state_manifest() == protocol["initial_states"], "Published initial states changed")
    selected, references = [], {"flow_protocol": binding(STUDY / "protocol.json")}
    for suite in SUITES:
        matches = [case for case in protocol["spec"]["cases"] if case["suite"] == suite
                   and case["task_id"] == 1 and case["init_state_id"] == 41 and case["method"] == "full10"]
        require(len(matches) == 1, "Expected exactly one original full10/init41 case")
        case = matches[0]
        require(case["seed"] == 1000, "Original seed differs")
        directory = STUDY / "episodes" / case["id"]
        result = flow.verify_complete(directory, case, protocol["fingerprint"])
        require(result is not None, "Original episode missing")
        receipt = read(directory / "receipt.json")
        references[suite] = {"case": case, "receipt": binding(directory / "receipt.json"),
                             "files": {name: {"path": relative(directory / name), "sha256": checksum}
                                       for name, checksum in receipt["files"].items()}}
        selected.append((case, directory, result))
    return selected, references


def memory_snapshot(process):
    import psutil
    return {"rss_bytes": process.memory_info().rss, "available_bytes": psutil.virtual_memory().available}


def data_snapshot(simulator, step, process):
    data, model = simulator.env._env.sim.data._data, simulator.env._env.sim.model._model
    warnings = [{"index": index, "number": int(item.number), "lastinfo": int(item.lastinfo)}
                for index, item in enumerate(data.warning)]
    return {"step": step, "utc": base.utc_now(), "arena_bytes": int(data.narena),
            "model_nconmax": int(model.nconmax), "ncon": int(data.ncon), "nefc": int(data.nefc),
            "maxuse_arena_bytes": int(data.maxuse_arena), "maxuse_stack_bytes": int(data.maxuse_stack),
            "warning_count": sum(item["number"] for item in warnings), "warnings": warnings,
            **memory_snapshot(process)}


def validate_capacity(row):
    require(row["arena_bytes"] == ARENA_BYTES, "Compiled arena differs from 256 MiB")
    require(row["model_nconmax"] == 5000, "Original contact cap changed")
    require(row["warning_count"] == 0, "MuJoCo warning encountered; replay cannot establish equivalence")
    require(row["maxuse_arena_bytes"] < ARENA_BYTES, "Arena capacity reached")


def main():
    import numpy as np
    import psutil
    from PIL import Image
    from libero_arena_budget import arena_budget

    require(not OUTPUT.exists() and not PLAN.exists() and not EVIDENCE.exists(),
            "Existing replay attempt is immutable; automatic retry forbidden")
    selected, references = reference_inputs()
    plan = {"schema": "feedbackbench.libero_arena_replay_plan.v1", "frozen_at": base.utc_now(),
            "sources": source_hashes(), "references": references,
            "cases": [{"suite": case["suite"], "task_id": 1, "init_state_id": 41, "seed": 1000,
                       "reference_id": case["id"], "steps": result["steps"]} for case, _, result in selected],
            "configuration": {"arena_bytes": ARENA_BYTES, "nconmax": 5000, "image_size": 360,
                              "hard_reset": True, "settle_steps": 10, "threads": 2, "interop_threads": 1},
            "comparison": "Exact float32 robot states, uint8 initial two cameras/final camera, and all terminal flags; zero tolerance",
            "memory_guard": {"maximum_rss_gib": 5.5, "minimum_runtime_available_gib": 0.5, "sampling_seconds": 1},
            "scope": "Replay all saved actions from four sealed trajectories; no policy inference; no new benchmark denominator",
            "limitations": ["Known saved trajectories only; not an all-input or all-trajectory physics proof",
                            "Memory/latency on a shared host are descriptive; no controlled speedup claim"]}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    write_json(PLAN, plan, exclusive=True)
    EVIDENCE.mkdir()
    record = {"schema": "feedbackbench.libero_arena_replay.v1", "started_at": base.utc_now(),
              "status": "starting", "complete": False, "planned": 4, "completed": 0,
              "plan": binding(PLAN), "cases": [], "failures": [], "new_policy_inference_calls": 0,
              "excluded_from_benchmark_counts": True, "scope": plan["scope"]}
    write_json(OUTPUT, record, exclusive=True)
    process = psutil.Process()
    guard = base.MemoryGuard(OUTPUT, 5.5, 0.5)
    guard.start()
    simulator = None
    current = None
    try:
        from embodied_research.runtime import configure_runtime, configure_libero_paths
        configure_runtime(ROOT)
        base.isolate_official_source()
        configure_libero_paths(ROOT)
        from embodied_research.simulator import LiberoSimulator
        import lerobot.envs.libero as official_env
        import robosuite.utils.binding_utils as bindings
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(Path(official_env.__file__).resolve().is_relative_to(base.SOURCE), "Mixed official environment source")
        for ordinal, (case, source, expected) in enumerate(selected, 1):
            current = EVIDENCE / case["suite"]
            current.mkdir()
            write_json(current / "attempt.json", {"case": case, "reference": references[case["suite"]],
                       "plan": record["plan"], "started_at": base.utc_now()}, exclusive=True)
            record.update(status="running", current_suite=case["suite"])
            write_json(OUTPUT, record)
            print(f"[{ordinal}/4] Replay {case['suite']}: {expected['steps']} sealed actions", flush=True)
            with np.load(source / "trajectory.npz", allow_pickle=False) as saved:
                actions = saved["actions"].copy()
                expected_states = np.concatenate((saved["states"], saved["final_state"][None]), axis=0)
            require(actions.shape == (expected["steps"], 7) and actions.dtype == np.float32,
                    "Original action shape/dtype differs")
            states, flags, capacity = [], [], []
            started = time.perf_counter()
            with arena_budget(bindings.MjSim) as arena_audit:
                simulator = LiberoSimulator(case["suite"], 1, project_root=ROOT, image_size=360)
                require(type(simulator.env) is official_env.LiberoEnv, "Unexpected environment implementation")
                observation = simulator.reset(seed=1000, init_state_id=41)
                require(not arena_audit["native_warnings"], "Native warning occurred during initialization")
                require(np.array_equal(observation["observation.state"], expected_states[0]), "Initial robot state differs")
                states.append(observation["observation.state"].copy())
                for name, camera in (("initial.png", "image"), ("initial_wrist.png", "image2")):
                    actual = simulator.frame(camera).copy()
                    Image.fromarray(actual).save(current / name)
                    with Image.open(source / name) as image:
                        require(np.array_equal(actual, np.asarray(image)), "Initial camera pixels differ: " + name)
                row = data_snapshot(simulator, 0, process)
                capacity.append(row)
                with (current / "capacity.jsonl").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                validate_capacity(row)
                with (current / "executed_steps.jsonl").open("x", encoding="utf-8") as executed, \
                     (current / "capacity.jsonl").open("a", encoding="utf-8") as memory:
                    for index, action in enumerate(actions):
                        observation, reward, terminated, truncated, info = simulator.step(action)
                        state = observation["observation.state"].copy()
                        actual_flags = [bool(terminated), bool(truncated), bool(info["is_success"])]
                        expected_flags = ([expected["terminated"], expected["truncated"], expected["success"]]
                                          if index == len(actions) - 1 else [False, False, False])
                        row = data_snapshot(simulator, index + 1, process)
                        step_record = {"step": index + 1, "action": action.tolist(), "robot_state": state.tolist(),
                                       "terminated": actual_flags[0], "truncated": actual_flags[1], "success": actual_flags[2]}
                        executed.write(json.dumps(step_record, allow_nan=False) + "\n")
                        executed.flush()
                        memory.write(json.dumps(row, allow_nan=False) + "\n")
                        memory.flush()
                        states.append(state); flags.append(actual_flags); capacity.append(row)
                        validate_capacity(row)
                        require(not arena_audit["native_warnings"], "Native warning occurred during replay")
                        require(np.array_equal(state, expected_states[index + 1]), f"Robot state differs after saved action {index}")
                        require(actual_flags == expected_flags, f"Terminal flags differ after saved action {index}")
                        if (index + 1) % 50 == 0:
                            print(f"  step={index + 1}/{len(actions)} exact; max arena={row['maxuse_arena_bytes']}", flush=True)
                final = simulator.frame().copy()
                Image.fromarray(final).save(current / "final.png")
                with Image.open(source / "final.png") as image:
                    require(np.array_equal(final, np.asarray(image)), "Final camera pixels differ")
                with (current / "trajectory.npz").open("xb") as stream:
                    np.savez_compressed(stream, states=np.asarray(states), actions=actions,
                                        flags=np.asarray(flags, dtype=np.bool_))
                write_json(current / "arena_audit.json", arena_audit, exclusive=True)
                require(arena_audit["compilations"]
                        and all(item["status"] == "compiled" and item["nconmax_preserved"] is True
                                and item["outside_size_opening_tag_bytes_unchanged"] is True
                                and item["new_size"].get("nconmax") == "5000"
                                for item in arena_audit["compilations"]), "Compiled XML audit differs")
                simulator.close()
                simulator = None
            gc.collect()
            evidence = {"schema": "feedbackbench.libero_arena_case_evidence.v1", "plan": record["plan"],
                        "reference": references[case["suite"]],
                        "files": {path.name: binding(path) for path in sorted(current.iterdir()) if path.is_file()}}
            write_json(current / "evidence.json", evidence, exclusive=True)
            record["cases"].append({"suite": case["suite"], "steps": len(actions), "max_state_abs_diff": 0.0,
                                    "initial_images_exact": True, "final_image_exact": True,
                                    "terminal_flags_exact": True, "warning_count": 0,
                                    "maxuse_arena_bytes": max(row["maxuse_arena_bytes"] for row in capacity),
                                    "arena_bytes": ARENA_BYTES, "evidence": binding(current / "evidence.json"),
                                    "elapsed_seconds": time.perf_counter() - started})
            record["completed"] = len(record["cases"])
            write_json(OUTPUT, record)
            print(f"  Passed exact {case['suite']} replay", flush=True)
            current = None
        _, after_references = reference_inputs()
        require(after_references == references, "Original evidence changed during replay")
        require(source_hashes() == plan["sources"], "Diagnostic sources changed during replay")
        record.update(status="passed_exact_replay", complete=True, completed=4)
        return 0
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                   "suite": record.get("current_suite"), "utc": base.utc_now()}
        record.update(status="failed", complete=False)
        record["failures"].append(failure)
        if current is not None:
            write_json(current / "error.json", failure, exclusive=True)
        return 1
    finally:
        if simulator is not None:
            simulator.close()
        record["memory"] = guard.finish()
        record["finished_at"] = base.utc_now()
        write_json(OUTPUT, record)
        print(json.dumps({"status": record["status"], "complete": record["complete"],
                          "completed": record["completed"], "output": relative(OUTPUT)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
