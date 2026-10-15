"""Replay saved actions for diagnosis; never count replay as another evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from embodied_research.runtime import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("result.json", "trajectory.npz", "initial.png", "initial_wrist.png", "final.png")


def validate_output(original: Path, output: Path) -> None:
    protected = [original]
    protected.extend(parent for parent in original.parents if
                     (parent / "protocol.json").is_file() or (parent / "campaign_freeze.json").is_file())
    if output.exists() or any(output.is_relative_to(parent) for parent in protected):
        raise ValueError("Choose a new diagnostic directory outside the original episode and experiment/campaign")


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_hashes(original: Path) -> dict[str, str]:
    return {name: hashlib.sha256((original / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def validate_provenance(root: Path, original: Path, result: dict) -> tuple[Path, bytes]:
    protocol_path = original.parent.parent / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    fingerprint = _digest_json({key: protocol[key] for key in ("plan", "sources", "runtime_versions", "artifacts")})
    if fingerprint != protocol["fingerprint"] or result["protocol_fingerprint"] != fingerprint:
        raise ValueError("Original protocol fingerprint does not match its source material")
    cases = [case for case in protocol["cases"] if case["id"] == result["id"]]
    if len(cases) != 1 or any(result.get(key) != value for key, value in cases[0].items()):
        raise ValueError("Original episode identity/configuration differs from its protocol")
    for name, checksum in protocol["sources"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe protocol source name")
        if hashlib.sha256((root / "src/embodied_research" / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Replay source differs from the original protocol: {name}")
    for name in ("mujoco", "robosuite", "hf-libero", "numpy", "torch"):
        if importlib.metadata.version(name) != protocol["runtime_versions"][name]:
            raise ValueError(f"Replay dependency differs from the original protocol: {name}")
    return protocol_path, protocol_bytes


def replay_episode(original: Path, output: Path, *, simulator_factory=None, root: Path = ROOT) -> dict:
    original, output = original.resolve(), output.resolve()
    validate_output(original, output)
    result_bytes = (original / "result.json").read_bytes()
    result = json.loads(result_bytes)
    if result.get("status") != "complete":
        raise ValueError("Replay requires a completed original episode")
    protocol_path, protocol_bytes = validate_provenance(root, original, result)
    original_hashes = _source_hashes(original)
    import numpy as np
    from PIL import Image

    with np.load(original / "trajectory.npz", allow_pickle=False) as data:
        actions = data["actions"].copy()
        states = data["states"].copy()
        final_state = data["final_state"].copy()
    steps = result["steps"]
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
            or actions.shape != (steps, 7) or states.shape != (steps, 8) or final_state.shape != (8,)
            or not all(np.isfinite(value).all() for value in (actions, states, final_state))):
        raise ValueError("Saved trajectory does not match the original episode")
    np.testing.assert_array_equal(states[0], np.asarray(result["initial_robot_state"], dtype=states.dtype))
    np.testing.assert_array_equal(final_state, np.asarray(result["final_robot_state"], dtype=final_state.dtype))
    if simulator_factory is None:
        configure_runtime()
        from embodied_research.simulator import LiberoSimulator

        simulator_factory = LiberoSimulator
    frames = []
    max_state_difference = 0.0
    with simulator_factory(result["suite"], result["task_id"],
                           image_size=result["environment"]["image_size"]) as simulator:
        if simulator.metadata() != result["environment"]:
            raise ValueError("Replay environment configuration differs from the original")
        observation = simulator.reset(seed=result["seed"], init_state_id=result["init_state_id"])
        first = simulator.frame()
        if hashlib.sha256(first.tobytes()).hexdigest() != result["initial_image_sha256"]:
            raise ValueError("Initial camera differs from original; refusing an unverified replay")
        np.testing.assert_allclose(observation["observation.state"], result["initial_robot_state"], rtol=1e-6, atol=1e-7)
        for name, frame in (("initial.png", first), ("initial_wrist.png", simulator.frame("image2"))):
            with Image.open(original / name) as expected:
                np.testing.assert_array_equal(frame, np.asarray(expected.convert("RGB")))
        frames.append(Image.fromarray(first))
        for index, action in enumerate(actions):
            actual = observation["observation.state"]
            np.testing.assert_allclose(actual, states[index], rtol=1e-6, atol=1e-7)
            max_state_difference = max(max_state_difference, float(np.max(np.abs(actual - states[index]))))
            observation, _, terminated, truncated, info = simulator.step(action)
            if (index + 1) % 4 == 0 or index + 1 == len(actions):
                frames.append(Image.fromarray(simulator.frame()))
            if terminated or truncated:
                if index + 1 != len(actions):
                    raise ValueError("Replay terminated at a different control step")
        if bool(terminated) != result["terminated"] or bool(truncated) != result["truncated"]:
            raise ValueError("Replay termination/truncation flags differ from the original")
        actual = observation["observation.state"]
        np.testing.assert_allclose(actual, final_state, rtol=1e-6, atol=1e-7)
        max_state_difference = max(max_state_difference, float(np.max(np.abs(actual - final_state))))
        with Image.open(original / "final.png") as final_image:
            np.testing.assert_array_equal(simulator.frame(), np.asarray(final_image.convert("RGB")))
        if bool(info["is_success"]) != result["success"]:
            raise ValueError("Replay success predicate differs from original")
    if _source_hashes(original) != original_hashes or protocol_path.read_bytes() != protocol_bytes:
        raise ValueError("Original evidence changed during replay; refusing to publish diagnostics")
    output.mkdir(parents=True)
    frames[0].save(output / "replay.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    audit = {"kind": "diagnostic replay, excluded from benchmark counts", "source_episode": result["id"],
             "original_protocol_fingerprint": result["protocol_fingerprint"],
             "original_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
             "source_files_sha256": original_hashes, "source_evidence_unchanged": True,
             "original_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
             "replay_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "initial_image_equal": True, "initial_wrist_image_equal": True, "final_robot_state_close": True,
             "trajectory_states_close": True, "state_rtol": 1e-6, "state_atol": 1e-7,
             "max_state_absolute_difference": max_state_difference, "terminal_flags_equal": True,
             "final_image_equal": True,
             "success_equal": True, "steps": len(actions), "success": result["success"]}
    (output / "replay_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Original episode directory containing result.json and trajectory.npz")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic directory outside formal evidence")
    args = parser.parse_args()
    replay_episode(args.episode, args.output)
    output = args.output.resolve()
    print(output / "replay.gif")


if __name__ == "__main__":
    main()
