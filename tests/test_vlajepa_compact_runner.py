"""Compact-runner contracts with synthetic evidence only; no Torch or LIBERO."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compact_runner_tested", ROOT / "scripts/run_vlajepa_compact.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
import libero_arena_budget as arena

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arena_row(step=0):
    return {"step": step, "arena_bytes": entry.ARENA_BYTES,
            "maxuse_arena_bytes": 10_000_000, "ncon": 2220, "nefc": 6672, "warnings": [0] * 8}


def compilation_row():
    _, row = arena.rewrite_xml('<mujoco><size njmax="5000" nconmax="5000"/><worldbody/></mujoco>')
    row.update(status="compiled", compile_index=0,
               compiled_memory={"model_arena_bytes": entry.ARENA_BYTES, "data_arena_bytes": entry.ARENA_BYTES,
                                "maximum_used_arena_bytes": 10_000_000,
                                "warning_counts": [0] * 8, "warning_lastinfo": [0] * 8})
    return row


def test_import_does_not_load_model_or_simulation_dependencies():
    result = subprocess.run([sys.executable, "-c",
                             "import runpy,sys;runpy.run_path(sys.argv[1]);"
                             "assert not ({'torch','mujoco','robosuite','libero'} & sys.modules.keys())",
                             str(ROOT / "scripts/run_vlajepa_compact.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suffix", ["", "outputs", "outputs/frontier",
                                     "outputs/frontier/vlajepa_closed_loop",
                                     "outputs/frontier/vlajepa_closed_loop/child",
                                     "outputs/frontier/vlajepa_paged_closed_loop",
                                     "outputs/frontier/vlajepa_paged_closed_loop/child"])
def test_output_rejects_both_previous_studies_and_ancestors(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(entry.paged, "ROOT", tmp_path)
    monkeypatch.setattr(entry.paged, "DEFAULT_OUTPUT", tmp_path / "outputs/frontier/vlajepa_paged_closed_loop")
    with pytest.raises(ValueError, match="separate"):
        entry.separate_output(tmp_path / suffix)


def test_output_accepts_new_sibling_but_rejects_external_path(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(entry.paged, "ROOT", root)
    monkeypatch.setattr(entry.paged, "DEFAULT_OUTPUT", root / "outputs/frontier/vlajepa_paged_closed_loop")
    target = root / "outputs/frontier/vlajepa_compact_closed_loop"
    assert entry.separate_output(target) == target.resolve()
    with pytest.raises(ValueError):
        entry.separate_output(tmp_path / "outside")


def test_valid_capacity_and_compilation_audit():
    entry.validate_arena_row(arena_row())
    entry.validate_compilation(compilation_row())


@pytest.mark.parametrize(("key", "value"), [
    ("step", True), ("step", -1), ("arena_bytes", entry.ARENA_BYTES + 64),
    ("maxuse_arena_bytes", 0), ("maxuse_arena_bytes", entry.ARENA_BYTES),
    ("ncon", 5001), ("nefc", -1), ("warnings", []), ("warnings", [0] * 7),
    ("warnings", [False] * 8), ("warnings", [0.0] * 8), ("warnings", [0] * 7 + [1]),
])
def test_bad_arena_observations_are_rejected(key, value):
    row = arena_row()
    row[key] = value
    with pytest.raises(ValueError):
        entry.validate_arena_row(row)


@pytest.mark.parametrize("bad", ["failed", "contact_limit", "non_capacity_attribute", "changed_outside_size",
                                "bad_hash", "model_arena", "native_warning", "unexpected_attribute"])
def test_compilation_audit_rejects_uncontrolled_scene_changes(bad):
    row = compilation_row()
    if bad == "failed":
        row["status"] = "failed"
    elif bad == "contact_limit":
        row["new_size"]["nconmax"] = "4000"
    elif bad == "non_capacity_attribute":
        row["changed_attributes"].append("nconmax")
    elif bad == "changed_outside_size":
        row["outside_size_opening_tag_bytes_unchanged"] = False
    elif bad == "bad_hash":
        row["output_sha256"] = "g" * 64
    elif bad == "model_arena":
        row["compiled_memory"]["model_arena_bytes"] += 64
    elif bad == "native_warning":
        row["compiled_memory"]["warning_counts"][1] = 1
    else:
        row["new_size"]["nuser_geom"] = "1"
    with pytest.raises(ValueError):
        entry.validate_compilation(row)


@pytest.fixture
def replay(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    def binding(path):
        return {"path": Path(path).relative_to(tmp_path).as_posix(), "sha256": sha(path)}

    sources = {}
    for name in ("scripts/probe_libero_arena.py", "scripts/libero_arena_budget.py", "scripts/probe_vlajepa.py",
                 "scripts/run.py", "scripts/complete_vlajepa_stage.py", "scripts/run_flow_confirmation.py",
                 "src/embodied_research/simulator.py", "frontier/vendor/lerobot/src/lerobot/envs/libero.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source " + name)
        sources[name] = sha(path)
    cases = [{"suite": suite, "task_id": 1, "init_state_id": 41, "seed": 1000,
              "reference_id": "source-" + suite, "steps": steps}
             for suite, steps in zip(SUITES, (70, 120, 268, 520))]
    references = {}
    for case in cases:
        path = tmp_path / "references" / case["suite"] / "trajectory.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        states = np.arange((case["steps"] + 1) * 8, dtype=np.float32).reshape(-1, 8)
        np.savez(path, states=states[:-1], final_state=states[-1],
                 actions=np.zeros((case["steps"], 7), dtype=np.float32))
        write_json(path.with_name("result.json"), {"terminated": True, "truncated": False, "success": True})
        for name in ("initial.png", "initial_wrist.png", "final.png"):
            Image.fromarray(np.zeros((360, 360, 3), dtype=np.uint8)).save(path.with_name(name))
        files = {file.name: binding(file) for file in path.parent.iterdir()}
        receipt = path.with_name("receipt.json")
        write_json(receipt, {"files": {name: item["sha256"] for name, item in files.items()}})
        references[case["suite"]] = {"case": case, "receipt": binding(receipt),
                                      "files": files}
    protocol = tmp_path / "references/protocol.json"
    write_json(protocol, {"synthetic": True})
    references["flow_protocol"] = binding(protocol)
    plan = {"schema": "feedbackbench.libero_arena_replay_plan.v1", "sources": sources, "cases": cases,
            "references": references,
            "configuration": {"arena_bytes": entry.ARENA_BYTES, "nconmax": 5000, "image_size": 360,
                              "hard_reset": True, "settle_steps": 10, "threads": 2, "interop_threads": 1}}
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    proof = {"status": "passed_exact_replay", "complete": True, "planned": 4, "completed": 4,
             "plan": binding(plan_path), "cases": [], "failures": [],
             "new_policy_inference_calls": 0, "excluded_from_benchmark_counts": True}
    for case in cases:
        directory = tmp_path / "replay" / case["suite"]
        directory.mkdir(parents=True)
        trajectory = directory / "trajectory.npz"
        with np.load(tmp_path / references[case["suite"]]["files"]["trajectory.npz"]["path"]) as original:
            flags = np.zeros((case["steps"], 3), dtype=np.bool_)
            flags[-1] = [True, False, True]
            np.savez(trajectory, states=np.concatenate((original["states"], original["final_state"][None])),
                     actions=original["actions"], flags=flags)
        for name in ("initial.png", "initial_wrist.png", "final.png"):
            original = tmp_path / references[case["suite"]]["files"][name]["path"]
            (directory / name).write_bytes(original.read_bytes())
        audit_path = directory / "arena_audit.json"
        write_json(audit_path, {"memory_budget_bytes": entry.ARENA_BYTES, "native_warnings": [],
                                "compilations": [compilation_row()]})
        capacity = []
        for step in range(case["steps"] + 1):
            row = arena_row(step)
            row.update(model_nconmax=5000, warning_count=0,
                       warnings=[{"index": index, "number": 0, "lastinfo": 0} for index in range(8)])
            capacity.append(json.dumps(row))
        (directory / "capacity.jsonl").write_text("\n".join(capacity), encoding="utf-8")
        evidence = {"schema": "feedbackbench.libero_arena_case_evidence.v1", "plan": proof["plan"],
                    "reference": references[case["suite"]],
                    "files": {file.name: binding(file) for file in directory.iterdir()}}
        evidence_path = directory / "evidence.json"
        write_json(evidence_path, evidence)
        proof["cases"].append({"suite": case["suite"], "steps": case["steps"], "max_state_abs_diff": 0.0,
                                "initial_images_exact": True, "final_image_exact": True, "terminal_flags_exact": True,
                                "warning_count": 0, "arena_bytes": entry.ARENA_BYTES,
                                "maxuse_arena_bytes": 10_000_000, "evidence": binding(evidence_path)})
    proof_path = tmp_path / "proof.json"
    write_json(proof_path, proof)
    monkeypatch.setattr(entry, "REPLAY", proof_path)
    runner = SimpleNamespace(read_json=lambda path: json.loads(Path(path).read_text()), require=require,
                             project_file=lambda path: tmp_path / path, file_sha=sha, SUITES=SUITES,
                             base=SimpleNamespace(relative=lambda path: Path(path).relative_to(tmp_path).as_posix()))
    return SimpleNamespace(root=tmp_path, runner=runner, proof=proof, plan=plan, proof_path=proof_path,
                           plan_path=plan_path, binding=binding)


def test_complete_four_suite_replay_is_bound_to_current_files(replay):
    proof = entry.verify_replay(replay.runner)
    assert proof["sha256"] == sha(replay.proof_path)
    assert proof["plan_sha256"] == sha(replay.plan_path)


@pytest.mark.parametrize("bad", ["partial", "wrong_suite", "nonexact", "warning", "wrong_arena", "short_replay",
                                "source_changed", "plan_changed", "evidence_changed", "trajectory_changed"])
def test_replay_gate_rejects_incomplete_or_changed_evidence(replay, bad):
    if bad == "partial":
        replay.proof["complete"] = False
    elif bad == "wrong_suite":
        replay.proof["cases"][0]["suite"] = "libero_10"
    elif bad == "nonexact":
        replay.proof["cases"][0]["max_state_abs_diff"] = 0.001
    elif bad == "warning":
        replay.proof["cases"][0]["warning_count"] = 1
    elif bad == "wrong_arena":
        replay.proof["cases"][0]["arena_bytes"] += 64
    elif bad == "short_replay":
        replay.proof["cases"][0]["steps"] -= 1
    elif bad == "source_changed":
        (replay.root / next(iter(replay.plan["sources"]))).write_text("changed")
    elif bad == "plan_changed":
        replay.plan_path.write_text("changed")
    elif bad == "evidence_changed":
        (replay.root / replay.proof["cases"][0]["evidence"]["path"]).write_text("changed")
    else:
        (replay.root / "replay/libero_spatial/trajectory.npz").write_bytes(b"changed")
    write_json(replay.proof_path, replay.proof)
    with pytest.raises(ValueError):
        entry.verify_replay(replay.runner)


@pytest.mark.parametrize("bad", ["arena", "contact", "seed", "initial", "empty_sources"])
def test_replay_plan_must_match_the_actual_compact_environment_contract(replay, bad):
    if bad == "arena":
        replay.plan["configuration"]["arena_bytes"] += 64
    elif bad == "contact":
        replay.plan["configuration"]["nconmax"] = 100
    elif bad == "seed":
        replay.plan["cases"][0]["seed"] = 5
    elif bad == "initial":
        replay.plan["cases"][0]["init_state_id"] = 5
    else:
        replay.plan["sources"] = {}
    write_json(replay.plan_path, replay.plan)
    replay.proof["plan"] = replay.binding(replay.plan_path)
    write_json(replay.proof_path, replay.proof)
    with pytest.raises(ValueError):
        entry.verify_replay(replay.runner)


def test_writer_retries_only_permission_errors_with_finite_backoff(monkeypatch, tmp_path):
    sleeps, calls = [], []
    monkeypatch.setattr(entry.time, "sleep", sleeps.append)
    target, payload = tmp_path / "evidence.json", {"status": "complete"}

    def writer(path, value):
        calls.append((path, value))
        if len(calls) < 3:
            raise PermissionError("simulated sharing violation")
        return "persisted"

    assert entry.reliable_write_json(writer, target, payload) == "persisted"
    assert calls == [(target, payload)] * 3 and sleeps == [0.05, 0.1]
    calls.clear()

    def permanent(path, value):
        calls.append(None)
        raise PermissionError("permanent")

    with pytest.raises(PermissionError):
        entry.reliable_write_json(permanent, target, payload)
    assert len(calls) == 7
    calls.clear()

    def invalid(path, value):
        calls.append(None)
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        entry.reliable_write_json(invalid, target, payload)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["complete", "run_error", "swallowed_reset_warning", "step_warning"])
def test_run_scopes_patches_and_blocks_native_warnings_even_when_reset_swallows_them(tmp_path, monkeypatch, mode):
    events = []
    audit = {"compilations": [], "native_warnings": []}
    fake_mjsim = object()
    bindings = ModuleType("robosuite.utils.binding_utils")
    bindings.MjSim = fake_mjsim
    monkeypatch.setitem(sys.modules, "robosuite.utils.binding_utils", bindings)

    @contextmanager
    def controlled_budget(cls, memory_bytes):
        assert cls is fake_mjsim and memory_bytes == entry.ARENA_BYTES
        events.append("enter")
        try:
            yield audit
        finally:
            events.append("exit")

    monkeypatch.setattr(arena, "arena_budget", controlled_budget)
    data = SimpleNamespace(narena=entry.ARENA_BYTES, maxuse_arena=10_000_000, ncon=100, nefc=300,
                           warning=[SimpleNamespace(number=0) for _ in range(8)])

    class Simulator:
        steps = 0
        env = SimpleNamespace(_env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(_data=data))))

        def reset(self, **kwargs):
            assert kwargs == {"seed": 1000, "init_state_id": 41}
            audit["compilations"].append(compilation_row())
            if mode == "swallowed_reset_warning":
                audit["native_warnings"].append({"message": "caught and swallowed inside reset"})
            return {"synthetic": "observation"}

        def step(self, action):
            self.steps += 1
            if mode == "step_warning":
                audit["native_warnings"].append({"message": "runtime warning"})
            return {"synthetic": "next"}

    class Recorded:
        def __init__(self, simulator, destination):
            self.simulator = simulator

        def step(self, action):
            return self.simulator.step(action)

    simulator = Simulator()
    original_load = lambda *args: "weights loaded"
    globals_before = {"write_json": write_json}
    exec("def run_episode():\n    pass\n", globals_before)
    runner = SimpleNamespace(quant=SimpleNamespace(load_quantized_tensors=original_load),
                             RecordedSimulator=Recorded, require=require, write_json=write_json,
                             run_episode=globals_before["run_episode"])

    def run(output, limit=None):
        assert output == tmp_path and limit == 1
        assert runner.run_episode.__globals__["write_json"] is runner.write_json
        assert runner.quant.load_quantized_tensors(None, None, None) == "weights loaded"
        wrapper = runner.RecordedSimulator(simulator, tmp_path)
        wrapper.reset(seed=1000, init_state_id=41)
        wrapper.step([0] * 7)
        if mode == "run_error":
            raise RuntimeError("controlled rollout error")
        return {"completed": 1}

    runner.run = run
    if mode == "complete":
        assert entry.run(runner, tmp_path, limit=1) == {"completed": 1}
        rows = [json.loads(line) for line in (tmp_path / "arena_steps.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == [0, 1]
        initialization = json.loads((tmp_path / "arena_initialization.json").read_text())
        assert initialization["native_warnings"] == []
        assert len(initialization["compilations"]) == 1
    else:
        with pytest.raises((ValueError, RuntimeError)):
            entry.run(runner, tmp_path, limit=1)
        if mode == "swallowed_reset_warning":
            assert simulator.steps == 0
    assert events == ["enter", "exit"]
    assert runner.quant.load_quantized_tensors is original_load
    assert runner.RecordedSimulator is Recorded
    assert runner.write_json is write_json
    assert runner.run_episode.__globals__["write_json"] is write_json
