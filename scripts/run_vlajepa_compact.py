"""Paged VLA-JEPA with a measured, guarded 256 MiB MuJoCo arena allocation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vlajepa_paged as paged

ARENA_BYTES = 268435456
DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_compact_closed_loop"
REPLAY = ROOT / "artifacts/frontier/libero_arena_replay_20261016.json"
SOURCES = ("scripts/run_vlajepa_compact.py", "scripts/libero_arena_budget.py",
           "scripts/probe_libero_arena.py", "docs/VLAJEPA_COMPACT_PROTOCOL.md")


def separate_output(output):
    output = paged.separate_output(output)
    previous = paged.DEFAULT_OUTPUT.resolve()
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("The interrupted paged-only experiment must remain separate")
    return output


def validate_arena_row(row):
    for key in ("step", "arena_bytes", "maxuse_arena_bytes", "ncon", "nefc"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError("Invalid arena observation: " + key)
    if row["arena_bytes"] != ARENA_BYTES or not 0 < row["maxuse_arena_bytes"] < ARENA_BYTES:
        raise ValueError("Unexpected or exhausted simulation arena")
    warnings = row.get("warnings")
    if (row["ncon"] > 5000 or not isinstance(warnings, list) or len(warnings) != 8
            or any(type(value) is not int or value != 0 for value in warnings)):
        raise ValueError("MuJoCo warning/contact limit reached")


def validate_compilation(row):
    from libero_arena_budget import CAPACITY_ATTRIBUTES
    old, new = row["old_size"], row["new_size"]
    expected = {key: value for key, value in old.items() if key not in CAPACITY_ATTRIBUTES}
    expected["memory"] = str(ARENA_BYTES)
    if (row.get("status") != "compiled" or row.get("memory_budget_bytes") != ARENA_BYTES
            or row.get("outside_size_opening_tag_bytes_unchanged") is not True
            or row.get("nconmax_preserved") is not True or old.get("nconmax") != "5000" or new != expected
            or set(row["changed_attributes"]) - CAPACITY_ATTRIBUTES):
        raise ValueError("Compiled scene changed beyond the declared arena capacity")
    for key in ("input_sha256", "output_sha256", "outside_size_opening_tag_sha256"):
        checksum = row.get(key, "")
        if len(checksum) != 64 or any(value not in "0123456789abcdef" for value in checksum):
            raise ValueError("Missing compiled XML identity")
    memory = row["compiled_memory"]
    if memory["model_arena_bytes"] != ARENA_BYTES:
        raise ValueError("Compiled model arena differs")
    validate_arena_row({"step": 0, "arena_bytes": memory["data_arena_bytes"],
                        "maxuse_arena_bytes": memory["maximum_used_arena_bytes"],
                        "ncon": 0, "nefc": 0, "warnings": memory["warning_counts"]})


def reliable_write_json(writer, path, value):
    """Retry only transient Windows file replacement, never simulation/inference."""
    for attempt in range(7):
        try:
            return writer(path, value)
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * 2**attempt)


def verify_replay(runner):
    proof = runner.read_json(REPLAY)
    runner.require(proof.get("status") == "passed_exact_replay" and proof.get("complete") is True
                   and proof.get("planned") == proof.get("completed") == 4, "Four exact arena replays are required")
    plan_path = runner.project_file(proof["plan"]["path"])
    runner.require(runner.file_sha(plan_path) == proof["plan"]["sha256"], "Arena replay plan changed")
    plan = runner.read_json(plan_path)
    required_sources = {"scripts/probe_libero_arena.py", "scripts/libero_arena_budget.py",
                        "src/embodied_research/simulator.py"}
    runner.require(required_sources <= set(plan["sources"]), "Arena replay source inventory incomplete")
    runner.require(plan["configuration"] == {"arena_bytes": ARENA_BYTES, "nconmax": 5000, "image_size": 360,
                    "hard_reset": True, "settle_steps": 10, "threads": 2, "interop_threads": 1}, "Arena replay configuration differs")
    runner.require(len(plan["cases"]) == 4 and {item["suite"] for item in plan["cases"]} == set(runner.SUITES),
                   "Arena replay plan task coverage differs")
    for item in plan["cases"]:
        runner.require(item["task_id"] == 1 and item["init_state_id"] == 41 and item["seed"] == 1000
                       and type(item["steps"]) is int and item["steps"] > 0, "Arena replay plan case differs")
    for name, checksum in plan["sources"].items():
        runner.require(runner.file_sha(runner.project_file(name)) == checksum, "Arena replay source changed: " + name)
    def check_binding(binding):
        path = runner.project_file(binding["path"])
        runner.require(runner.file_sha(path) == binding["sha256"], "Arena replay bound file changed: " + binding["path"])
        return path

    check_binding(plan["references"]["flow_protocol"])
    runner.require(len(proof["cases"]) == 4 and {case["suite"] for case in proof["cases"]} == set(runner.SUITES),
                   "Arena replay task coverage differs")
    for case in proof["cases"]:
        expected = next(item for item in plan["cases"] if item["suite"] == case["suite"])
        runner.require(case["steps"] == expected["steps"], "Arena replay truncated the original trajectory")
        runner.require(case["max_state_abs_diff"] == 0.0 and case["initial_images_exact"] is True
                       and case["final_image_exact"] is True and case["terminal_flags_exact"] is True
                       and case["warning_count"] == 0 and case["arena_bytes"] == ARENA_BYTES
                       and 0 < case["maxuse_arena_bytes"] < ARENA_BYTES and case["steps"] > 0,
                       "Arena replay is not exact/warning-free")
        evidence = case["evidence"]
        evidence_path = check_binding(evidence)
        record = runner.read_json(evidence_path)
        reference = plan["references"][case["suite"]]
        runner.require(record["reference"] == reference and record["plan"] == proof["plan"], "Arena replay evidence identity differs")
        check_binding(reference["receipt"])
        for item in reference["files"].values():
            check_binding(item)
        for item in record["files"].values():
            check_binding(item)
        # Recompute comparisons from retained arrays/pixels instead of trusting summary flags.
        import numpy as np
        from PIL import Image
        with np.load(check_binding(reference["files"]["trajectory.npz"]), allow_pickle=False) as before, \
                np.load(check_binding(record["files"]["trajectory.npz"]), allow_pickle=False) as after:
            expected_states = np.concatenate((before["states"], before["final_state"][None]), axis=0)
            runner.require(np.array_equal(expected_states, after["states"])
                           and np.array_equal(before["actions"], after["actions"])
                           and len(after["actions"]) == case["steps"], "Recomputed arena replay differs")
            terminal = runner.read_json(check_binding(reference["files"]["result.json"]))
            expected_flags = np.zeros((case["steps"], 3), dtype=np.bool_)
            expected_flags[-1] = [terminal["terminated"], terminal["truncated"], terminal["success"]]
            runner.require(np.array_equal(expected_flags, after["flags"]), "Recomputed terminal flags differ")
        for name in ("initial.png", "initial_wrist.png", "final.png"):
            with Image.open(check_binding(reference["files"][name])) as before, Image.open(check_binding(record["files"][name])) as after:
                runner.require(np.array_equal(np.asarray(before), np.asarray(after)), "Recomputed camera pixels differ")
        audit = runner.read_json(check_binding(record["files"]["arena_audit.json"]))
        runner.require(audit["native_warnings"] == [] and bool(audit["compilations"]), "Replay native warning/compilation differs")
        for compilation in audit["compilations"]:
            validate_compilation(compilation)
        capacity = [json.loads(line) for line in check_binding(record["files"]["capacity.jsonl"]).read_text(encoding="utf-8").splitlines()]
        runner.require(len(capacity) == case["steps"] + 1, "Missing replay capacity steps")
        for index, row in enumerate(capacity):
            runner.require(row["step"] == index and row["model_nconmax"] == 5000, "Replay step/contact cap differs")
            validate_arena_row({**row, "warnings": [item["number"] for item in row["warnings"]]})
    return {"path": runner.base.relative(REPLAY), "sha256": runner.file_sha(REPLAY), "plan_sha256": runner.file_sha(plan_path)}


def configured_runner():
    runner = paged.configured_runner()
    proof = verify_replay(runner)
    prior_cases, prior_spec, prior_verify = runner.cases_for_study, runner.study_spec, runner.verify_episode
    runner.SOURCES = (*runner.SOURCES, *SOURCES)

    def cases():
        rows = prior_cases()
        for row in rows:
            row["model"] = "vlajepa_compact"
            row["split"] = "compact_closed_loop_feasibility"
            row["id"] = runner.episode_id(row)
        return rows

    def spec():
        result = prior_spec()
        result["name"] = "VLA-JEPA paged embedding and 256 MiB MuJoCo arena four-task feasibility"
        result["arena"] = {"bytes": ARENA_BYTES, "nconmax": 5000,
                           "change": "replace legacy njmax/nstack allocation only; retain nconmax and all other XML bytes",
                           "warnings": "reject native warnings and check data.warning after reset and every control step"}
        result["arena_replay"] = proof
        result["limitations"].append("Independent successor to the interrupted paged-only attempt; all earlier failures retained")
        return result

    def verify(directory, case, protocol):
        result = prior_verify(directory, case, protocol)
        if result is None:
            return None
        rows = [json.loads(line) for line in (Path(directory) / "arena_steps.jsonl").read_text(encoding="utf-8").splitlines()]
        runner.require(len(rows) == result["steps"] + 1, "Missing reset/control-step arena audit")
        for index, row in enumerate(rows):
            validate_arena_row(row)
            runner.require(row["step"] == index, "Arena step order differs")
        initialization = runner.read_json(Path(directory) / "arena_initialization.json")
        runner.require(initialization["memory_bytes"] == ARENA_BYTES and initialization["native_warnings"] == [],
                       "Arena initialization warning/budget differs")
        runner.require(bool(initialization["compilations"]), "No capacity-controlled compilation was recorded")
        for compilation in initialization["compilations"]:
            validate_compilation(compilation)
        return result

    runner.cases_for_study, runner.study_spec, runner.verify_episode = cases, spec, verify
    return runner


def run(runner, output, *, limit=None):
    from libero_arena_budget import arena_budget
    original_load = runner.quant.load_quantized_tensors
    original_recorded = runner.RecordedSimulator
    original_write = runner.write_json
    compilation = None

    with ExitStack() as stack:
        def load(policy, schema, checkpoint):
            nonlocal compilation
            from robosuite.utils.binding_utils import MjSim
            compilation = stack.enter_context(arena_budget(MjSim, memory_bytes=ARENA_BYTES))
            return original_load(policy, schema, checkpoint)

        class ArenaRecordedSimulator(original_recorded):
            def __init__(self, simulator, destination):
                super().__init__(simulator, destination)
                self.destination = Path(destination)
                self.arena_path = self.destination / "arena_steps.jsonl"
                runner.require(not self.arena_path.exists(), "Previous arena trace retained")
                self.first_compilation = len(compilation["compilations"])

            def record_arena(self):
                runner.require(compilation["native_warnings"] == [], "Native warning occurred during simulation")
                data = self.simulator.env._env.sim.data._data
                row = {"step": int(self.simulator.steps), "arena_bytes": int(data.narena),
                       "maxuse_arena_bytes": int(data.maxuse_arena), "ncon": int(data.ncon), "nefc": int(data.nefc),
                       "warnings": [int(value.number) for value in data.warning]}
                with self.arena_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                validate_arena_row(row)

            def reset(self, *, seed=1000, init_state_id=0):
                observation = self.simulator.reset(seed=seed, init_state_id=init_state_id)
                runner.require(compilation["native_warnings"] == [], "Native warning occurred during initialization")
                for item in compilation["compilations"][self.first_compilation:]:
                    validate_compilation(item)
                self.record_arena()
                runner.write_json(self.destination / "arena_initialization.json",
                                  {"memory_bytes": ARENA_BYTES, "native_warnings": compilation["native_warnings"],
                                   "compilations": compilation["compilations"][self.first_compilation:]})
                return observation

            def step(self, action):
                result = super().step(action)
                self.record_arena()
                return result

        stack.enter_context(patch.object(runner.quant, "load_quantized_tensors", load))
        stack.enter_context(patch.object(runner, "RecordedSimulator", ArenaRecordedSimulator))
        writer = lambda path, value: reliable_write_json(original_write, path, value)
        stack.enter_context(patch.object(runner, "write_json", writer))
        stack.enter_context(patch.dict(runner.run_episode.__globals__, {"write_json": writer}))
        return runner.run(output, limit=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = separate_output(args.output)
    runner = configured_runner()
    if args.command == "freeze":
        result = runner.run(output, freeze_only=True)
    elif args.command == "validate":
        result = runner.verify_study(output, partial=args.partial)
        result.pop("results")
    elif args.command == "report":
        result = runner.write_report(output)
    else:
        result = run(runner, output, limit=args.limit)
        if "completed" in result:
            result = runner.write_report(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
