"""Exact saved-action diagnostic for the frozen prospective flow study.

No policy is loaded or rerun. The old replay helper is used through an isolated
module and a narrowly scoped provenance-validator replacement. A recording
simulator additionally checks every robot state exactly and retains those states
for independent, model-free audit. This diagnostic never changes the 16-case
confirmation denominator and does not establish a cause for host-side delays.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module("flow_replay_frozen_runner", "run_flow_confirmation.py")
base = _module("flow_replay_legacy_helper", "replay_episode.py")
SOURCE_FILES = ("scripts/replay_flow_episode.py", "scripts/replay_episode.py", "scripts/run_flow_confirmation.py")
EXPORTS = ("attempt.json", "wrapper_source.py", "base_helper_source.py", "replayed_trajectory.npz",
           "initial.png", "initial_wrist.png", "final.png", "base/replay.gif", "base/replay_audit.json")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _safe(root, name):
    _require(isinstance(name, str) and not Path(name).is_absolute() and not Path(name).drive
             and ":" not in name and "\\" not in name, "Unsafe replay-relative path")
    path = (Path(root) / name).resolve()
    _require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Missing or out-of-root replay evidence")
    return path


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _exclusive_bytes(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)


def _source_inputs(original, root=ROOT):
    root, original = Path(root).resolve(), Path(original).resolve()
    _require(original.is_relative_to(root) and original.parent.name == "episodes", "Original must be a project episode")
    study_dir = original.parent.parent
    _require(not (study_dir / "active.lock").exists(), "Wait for the scored study process to finish before replay")
    study = runner.verify_study(study_dir, partial=False, root=root)
    _require(study["complete"] and len(study["records"]) == 16, "Replay requires all 16 sealed confirmation cases")
    _require(original.name in study["records"], "Original episode is not in the frozen confirmation protocol")
    row = study["records"][original.name]
    _require(row["case"]["id"] == original.name and row["case"]["split"] == "confirmation", "Replay case identity differs")
    protocol_path = study_dir / "protocol.json"
    binding = {"study": _relative(root, study_dir), "episode": _relative(root, original),
               "episode_id": original.name, "protocol_sha256": runner.file_sha(protocol_path),
               "protocol_fingerprint": study["protocol"]["fingerprint"],
               "receipt_sha256": row["receipt_sha256"], "result_sha256": row["result_sha256"],
               "files_sha256": row["artifact_sha256"]}
    return study, row, binding


def _verify_runtime(root, protocol):
    """Read metadata and hash inputs; never instantiate or load a robot policy."""
    actual = runner.versions()
    _require(actual == protocol["runtime_versions"], "Replay runtime versions differ from the scored study")
    _require(runner.artifact_provenance(Path(root), ["smolvla"]) == protocol["artifacts"],
             "Replay upstream/model/asset identities differ from the scored study")
    _require(runner.initial_state_manifest() == protocol["initial_states"],
             "Replay published initial-state inputs differ from the scored study")
    return actual


class RecordingSimulator:
    """Exact state/action checks around the original simulator, without a policy."""

    def __init__(self, simulator, expected_actions, expected_states):
        self.simulator = simulator
        self.expected_actions = expected_actions
        self.expected_states = expected_states
        self.states, self.actions, self.flags, self.step_seconds = [], [], [], []
        self.images = {}

    def metadata(self):
        return self.simulator.metadata()

    def frame(self, camera="image"):
        return self.simulator.frame(camera)

    def reset(self, *, seed, init_state_id):
        _require(not self.states, "Diagnostic simulator must reset exactly once")
        observation = self.simulator.reset(seed=seed, init_state_id=init_state_id)
        state = np.asarray(observation["observation.state"])
        _require(np.array_equal(state, self.expected_states[0]), "Replay initial robot state is not exactly equal")
        self.states.append(state.copy())
        self.images = {"initial.png": self.frame().copy(), "initial_wrist.png": self.frame("image2").copy()}
        return observation

    def step(self, action):
        index = len(self.actions)
        _require(index < len(self.expected_actions) and np.array_equal(action, self.expected_actions[index]),
                 "Replay submitted action differs from saved original")
        started = time.perf_counter()
        returned = self.simulator.step(action)
        elapsed = time.perf_counter() - started
        observation, _, terminated, truncated, info = returned
        state = np.asarray(observation["observation.state"])
        _require(np.array_equal(state, self.expected_states[index + 1]),
                 f"Replay robot state is not exactly equal after action {index}")
        self.actions.append(np.asarray(action).copy())
        self.states.append(state.copy())
        self.step_seconds.append(elapsed)
        self.flags.append((bool(terminated), bool(truncated), bool(info["is_success"])))
        self.images["final.png"] = self.frame().copy()
        return returned

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.simulator.close()


def _runtime_factory(root):
    from embodied_research.runtime import configure_runtime
    configure_runtime(root)
    from embodied_research.simulator import LiberoSimulator

    def construct(*args, **kwargs):
        return LiberoSimulator(*args, project_root=root, **kwargs)

    return construct


def _portable_error(text, root):
    for path, token in ((str(root), "<project>"), (sys.prefix, "<environment>"),
                        (sys.base_prefix, "<python>"), (str(Path.home()), "<user>")):
        text = text.replace(path, token).replace(path.replace("\\", "/"), token)
    return text


def replay(original, output, *, root=ROOT, simulator_factory=None):
    root, original, output = Path(root).resolve(), Path(original).resolve(), Path(output).resolve()
    base.validate_output(original, output)
    _require(output.is_relative_to(root / "outputs/replays"), "Diagnostic output must be inside project outputs/replays")
    study, row, binding = _source_inputs(original, root)
    runtime_versions = _verify_runtime(root, study["protocol"])
    sources = {name: runner.file_sha(root / name) for name in SOURCE_FILES}
    with np.load(original / "trajectory.npz", allow_pickle=False) as saved:
        actions = saved["actions"].copy()
        states = np.concatenate((saved["states"], saved["final_state"][None]), axis=0)
    output.mkdir(parents=True)
    _exclusive_bytes(output / "wrapper_source.py", (root / SOURCE_FILES[0]).read_bytes())
    _exclusive_bytes(output / "base_helper_source.py", (root / SOURCE_FILES[1]).read_bytes())
    attempt = {"schema": "feedbackbench.flow_replay_attempt.v1", "started_at": runner.utc_now(),
               "original": binding, "sources_sha256": sources, "runtime_versions": runtime_versions,
               "scope": "Saved-action diagnostic only; no policy inference; excluded from the16 scored cases."}
    runner._exclusive_json(output / "attempt.json", attempt)
    recorder = None
    audit = {"schema": "feedbackbench.flow_replay_audit.v1", "status": "running",
             "original": binding, "sources_sha256": sources, "runtime_versions": runtime_versions,
             "kind": "diagnostic saved-action replay", "excluded_from_benchmark_counts": True,
             "new_policy_inference_calls": 0, "causal_attribution": "No cause of timing variation is established by this diagnostic."}
    try:
        factory = simulator_factory or _runtime_factory(root)

        def recording_factory(*args, **kwargs):
            nonlocal recorder
            _require(recorder is None, "Only one diagnostic simulator is permitted")
            recorder = RecordingSimulator(factory(*args, **kwargs), actions, states)
            return recorder

        def provenance_override(replay_root, replay_original, result):
            _require(Path(replay_root).resolve() == root and Path(replay_original).resolve() == original,
                     "Base helper requested another episode")
            _require(result == row["record"], "Original result changed after confirmation verification")
            path = original.parent.parent / "protocol.json"
            _require(runner.file_sha(path) == binding["protocol_sha256"], "Protocol changed before replay")
            return path, path.read_bytes()

        started = time.perf_counter()
        with patch.object(base, "validate_provenance", provenance_override):
            base_audit = base.replay_episode(original, output / "base", root=root,
                                             simulator_factory=recording_factory)
        elapsed = time.perf_counter() - started
        _require(base_audit["max_state_absolute_difference"] == 0.0, "Replay was close but not identical")
        _, _, after_binding = _source_inputs(original, root)
        _require(after_binding == binding, "Original sealed evidence changed during diagnostic")
        _require(sources == {name: runner.file_sha(root / name) for name in SOURCE_FILES},
                 "Diagnostic implementation changed during replay")
        with (output / "replayed_trajectory.npz").open("xb") as stream:
            np.savez_compressed(stream, states=np.asarray(recorder.states), actions=np.asarray(recorder.actions),
                                flags=np.asarray(recorder.flags, dtype=np.bool_),
                                simulator_seconds=np.asarray(recorder.step_seconds, dtype=np.float64))
        for name, image in recorder.images.items():
            Image.fromarray(image).save(output / name)
        original_record = row["record"]
        audit.update(status="passed_exact_replay", finished_at=runner.utc_now(), steps=len(actions),
            success=original_record["success"], all_robot_states_exact=True, actions_exact=True,
            initial_image_equal=True, initial_wrist_image_equal=True, final_image_equal=True,
            terminal_flags_equal=True, success_equal=True, source_evidence_unchanged=True,
            max_state_absolute_difference=0.0,
            original_timings={key: original_record[key] for key in ("wall_seconds", "policy_inference_seconds", "simulator_seconds")},
            replay_timings={"helper_elapsed_seconds": elapsed, "simulator_seconds": float(sum(recorder.step_seconds)),
                            "simulator_step_max_seconds": float(max(recorder.step_seconds)),
                            "scope": "step timings wrap the original simulator.step; helper elapsed also includes reset/rendering/GIF encoding."},
            files_sha256={name: runner.file_sha(output / name) for name in EXPORTS})
        validate_replay(output, root=root, _audit=audit)
        runner._exclusive_json(output / "flow_replay_audit.json", audit)
        return audit
    except Exception as error:
        if not (output / "flow_replay_audit.json").exists():
            # A candidate may fail its final independent validation after the
            # comparisons ran. Preserve evidence without publishing pass claims.
            for key in ("all_robot_states_exact", "actions_exact", "initial_image_equal", "initial_wrist_image_equal",
                        "final_image_equal", "terminal_flags_equal", "success_equal", "source_evidence_unchanged",
                        "max_state_absolute_difference", "success"):
                audit.pop(key, None)
            audit.update(status="failed", finished_at=runner.utc_now(), error_type=type(error).__name__,
                         error=_portable_error(str(error), root), traceback=_portable_error(traceback.format_exc(), root),
                         files_sha256={path.relative_to(output).as_posix(): runner.file_sha(path)
                                       for path in output.rglob("*") if path.is_file()})
            runner._exclusive_json(output / "flow_replay_audit.json", audit)
        raise


def validate_replay(output, root=ROOT, *, _audit=None):
    """Strict read-only consumer for the packager; no Torch/model/environment imports.

    Missing/failed diagnostics raise rather than becoming an exact-replay claim.
    The original16-case study remains independently counted in either outcome.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    _require(output.is_relative_to(root / "outputs/replays"), "Replay must remain in project diagnostics")
    audit = runner.read_json(output / "flow_replay_audit.json") if _audit is None else _audit
    _require(audit["schema"] == "feedbackbench.flow_replay_audit.v1" and audit["status"] == "passed_exact_replay",
             "Replay is incomplete or failed; no exact-replay claim is allowed")
    original = _safe(root, audit["original"]["episode"] + "/result.json").parent
    study, row, binding = _source_inputs(original, root)
    _require(audit["original"] == binding, "Replay original evidence binding changed")
    _require(audit["runtime_versions"] == study["protocol"]["runtime_versions"], "Replay runtime identity differs")
    expected_sources = {name: runner.file_sha(root / name) for name in SOURCE_FILES}
    _require(audit["sources_sha256"] == expected_sources, "Replay implementation changed")
    _require(set(audit["files_sha256"]) == set(EXPORTS), "Replay export inventory differs")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected_inventory = set(EXPORTS) | ({"flow_replay_audit.json"} if _audit is None else set())
    _require(actual == expected_inventory, "Unbound replay file exists")
    for name, expected in audit["files_sha256"].items():
        _require(runner.file_sha(_safe(output, name)) == expected, f"Replay artifact hash differs: {name}")
    for snapshot_name, source_name in (("wrapper_source.py", SOURCE_FILES[0]), ("base_helper_source.py", SOURCE_FILES[1])):
        _require(runner.file_sha(output / snapshot_name) == expected_sources[source_name], "Replay source snapshot differs")
    attempt = runner.read_json(output / "attempt.json")
    _require(attempt["original"] == binding and attempt["sources_sha256"] == expected_sources
             and attempt["runtime_versions"] == audit["runtime_versions"], "Replay attempt binding differs")
    result = row["record"]
    with np.load(original / "trajectory.npz", allow_pickle=False) as saved:
        expected_states = np.concatenate((saved["states"], saved["final_state"][None]), axis=0)
        expected_actions = saved["actions"].copy()
    with np.load(output / "replayed_trajectory.npz", allow_pickle=False) as replayed:
        _require(set(replayed.files) == {"states", "actions", "flags", "simulator_seconds"}, "Replay trajectory inventory differs")
        _require(np.array_equal(replayed["states"], expected_states)
                 and np.array_equal(replayed["actions"], expected_actions), "Recorded replay states/actions are not identical")
        flags, timings = replayed["flags"], replayed["simulator_seconds"]
        _require(flags.dtype == np.bool_ and flags.shape == (result["steps"], 3), "Invalid replay terminal flags")
        _require(not flags[:-1].any() and flags[-1].tolist() ==
                 [result["terminated"], result["truncated"], result["success"]], "Recorded replay terminal/success flags differ")
        _require(timings.shape == (result["steps"],) and np.isfinite(timings).all() and np.all(timings >= 0),
                 "Invalid replay timing evidence")
        _require(float(timings.sum()) == audit["replay_timings"]["simulator_seconds"]
                 or np.isclose(timings.sum(), audit["replay_timings"]["simulator_seconds"], rtol=1e-12, atol=1e-12),
                 "Replay timing summary differs")
        _require(float(timings.max()) == audit["replay_timings"]["simulator_step_max_seconds"], "Replay max-step timing differs")
    for name in ("initial.png", "initial_wrist.png", "final.png"):
        with Image.open(original / name) as source, Image.open(output / name) as replayed_image:
            _require(np.array_equal(np.asarray(source.convert("RGB")), np.asarray(replayed_image.convert("RGB"))),
                     f"Replay image differs: {name}")
    with Image.open(output / "base/replay.gif") as image:
        _require(image.format == "GIF" and image.size == (result["environment"]["image_size"],) * 2, "Invalid replay GIF")
        image.verify()
    base_audit = runner.read_json(output / "base/replay_audit.json")
    _require(base_audit["replay_helper_sha256"] == expected_sources[SOURCE_FILES[1]]
             and base_audit["original_protocol_sha256"] == binding["protocol_sha256"]
             and base_audit["original_result_sha256"] == binding["result_sha256"]
             and base_audit["original_protocol_fingerprint"] == binding["protocol_fingerprint"]
             and base_audit["source_episode"] == binding["episode_id"]
             and base_audit["source_files_sha256"] == {name: binding["files_sha256"][name] for name in base.SOURCE_FILES}
             and base_audit["steps"] == result["steps"] and base_audit["success"] is result["success"]
             and base_audit["max_state_absolute_difference"] == 0.0, "Base-helper audit binding differs")
    for key in ("initial_image_equal", "initial_wrist_image_equal", "final_image_equal", "terminal_flags_equal", "success_equal",
                "source_evidence_unchanged", "trajectory_states_close", "final_robot_state_close"):
        _require(base_audit[key] is True, f"Base-helper verification failed: {key}")
    for key in ("all_robot_states_exact", "actions_exact", "initial_image_equal", "initial_wrist_image_equal",
                "final_image_equal", "terminal_flags_equal", "success_equal", "source_evidence_unchanged",
                "excluded_from_benchmark_counts"):
        _require(audit[key] is True, f"Replay verification flag differs: {key}")
    _require(audit["new_policy_inference_calls"] == 0 and audit["steps"] == result["steps"]
             and audit["success"] is result["success"] and audit["max_state_absolute_difference"] == 0.0,
             "Replay scope/outcome differs")
    _require(audit["original_timings"] == {key: result[key] for key in
                                         ("wall_seconds", "policy_inference_seconds", "simulator_seconds")},
             "Original timing summary differs")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    audit = validate_replay(args.output) if args.validate_only else replay(args.episode, args.output)
    print(audit["status"])
    print(args.output / "flow_replay_audit.json")


if __name__ == "__main__":
    main()
