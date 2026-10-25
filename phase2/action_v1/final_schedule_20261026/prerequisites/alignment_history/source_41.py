"""Validate released LIBERO observation/action alignment without a VLA model.

Revision 2 checks every demonstration's robot qpos against adjacent stored states
and reconstructs recorded observations directly in one simulator. An additional
isolated action replay is retained as a diagnostic: flattened MuJoCo states do
not preserve the gripper/controller's hidden state, so dynamics are NOT verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def infer_offset(states, joint_states, gripper_states):
    import numpy as np

    states = np.asarray(states)
    observed = np.concatenate((joint_states, gripper_states), axis=1)
    if states.ndim != 2 or states.shape[0] != len(observed) or states.shape[1] < 10 or len(states) < 3:
        raise ValueError("Expected time+Panda qpos in aligned MuJoCo state rows")
    if observed.shape[1] != 9 or not np.isfinite(states).all() or not np.isfinite(observed).all():
        raise ValueError("Invalid robot qpos/state data")
    errors = {offset: float(np.max(np.abs(observed[:-1] - states[offset:len(states)-1+offset, 1:10])))
              for offset in (0, 1)}
    matches = [offset for offset, error in errors.items() if error <= 1e-9]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous or inconsistent observation/state alignment: {errors}")
    return {"action_offset": matches[0], "same_state_max_abs": errors[0], "next_state_max_abs": errors[1],
            "checked_rows": len(states) - 1, "state_time_step_min": float(np.min(np.diff(states[:, 0]))),
            "state_time_step_max": float(np.max(np.diff(states[:, 0])))}


def rewrite_asset_paths(xml, assets_root, robosuite_root):
    """Relocate file references only; refuse missing files or escapes."""
    assets_root = Path(assets_root).resolve()
    robosuite_root = Path(robosuite_root).resolve()
    tree = ET.fromstring(xml)
    replacements = []
    for element in tree.findall("asset/*"):
        old = element.get("file")
        if old is None:
            continue
        parts = old.replace("\\", "/").split("/")
        if "robosuite" in parts:
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
            base, suffix = robosuite_root, parts[index + 1:]
        elif "assets" in parts and any(p in ("chiliocosm", "libero", "libero-dev") for p in parts):
            index = max(i for i, part in enumerate(parts) if part == "assets")
            base, suffix = assets_root, parts[index + 1:]
        else:
            raise ValueError(f"Unrecognized source asset prefix: {old}")
        resolved = base.joinpath(*suffix).resolve()
        if not resolved.is_relative_to(base) or not resolved.is_file():
            raise ValueError(f"Missing or escaped source asset: {old}")
        element.set("file", resolved.as_posix())
        replacements.append({"original": old, "resolved": str(resolved)})
    return ET.tostring(tree, encoding="unicode"), replacements


def image_orientation_errors(actual, reference):
    import numpy as np

    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[2] != 3:
        raise ValueError("Expected matching RGB HWC images")
    variants = {"identity": actual, "vertical": actual[::-1], "horizontal": actual[:, ::-1],
                "rot180": actual[::-1, ::-1]}
    return {name: float(np.mean(np.abs(value.astype(np.float32) - reference.astype(np.float32))) / 255)
            for name, value in variants.items()}


def common_offset(records):
    offsets = {r["action_offset"] for r in records}
    if len(offsets) != 1 or not offsets.issubset({0, 1}):
        raise ValueError("Sources/demonstrations do not share one verified action offset")
    return next(iter(offsets))


def inspect_source(source):
    import h5py
    from embodied_research.action_demo_data import sha256_file

    path = ROOT / source["path"]
    if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
        raise ValueError("Source SHA256/size does not match the pinned source manifest")
    with h5py.File(path, "r") as dataset:
        group = dataset["data"]
        expected = {f"demo_{i}" for i in range(50)}
        if set(group.keys()) != expected:
            raise ValueError("Fixed source does not contain exactly demo_0..49; do not replace task")
        problem = json.loads(group.attrs["problem_info"])
        if problem["language_instruction"].lower() != source["language"].lower():
            raise ValueError("Published task language disagrees with fixed manifest")
        env_args = json.loads(group.attrs["env_args"])
        config = env_args["env_kwargs"]["controller_configs"]
        if config.get("type") != "OSC_POSE" or config.get("control_delta") is not True:
            raise ValueError("Unsupported action controller semantics")
        demos = []
        for index in range(50):
            demo = group[f"demo_{index}"]
            result = infer_offset(demo["states"][:], demo["obs/joint_states"][:], demo["obs/gripper_states"][:])
            if abs(result["state_time_step_min"] - .05) > 1e-9 or abs(result["state_time_step_max"] - .05) > 1e-9:
                raise ValueError("Unexpected source time grid; action chunk adjacency not established")
            demos.append({"episode_key": f"demo_{index}", "frame_count": len(demo["actions"]), **result})
        return {"suite": source["suite"], "task_id": 0, "task_uid": source["task_uid"],
                "path": source["path"], "sha256": source["sha256"], "bytes": source["bytes"],
                "action_offset": common_offset(demos), "demonstrations": demos,
                "env_args": env_args, "image_convention": str(group.attrs["macros_image_convention"]),
                "dataset_tag": str(group.attrs.get("tag", "unknown")), "status": "state_rows_verified"}


def replay_source(source, record, count=5):
    import h5py
    import numpy as np
    import psutil
    from embodied_research.runtime import configure_runtime, configure_libero_paths, MUJOCO_VERSION

    configure_runtime(ROOT)
    configure_libero_paths(ROOT)
    import importlib.metadata
    import robosuite
    import robosuite.utils.transform_utils as transforms
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING

    if importlib.metadata.version("mujoco") != MUJOCO_VERSION:
        raise RuntimeError("Unexpected simulator version")
    if psutil.virtual_memory().available < 2 * 1024 ** 3:
        raise RuntimeError("At least 2 GiB free RAM required for isolated simulation diagnostic")
    kwargs = dict(record["env_args"]["env_kwargs"])
    kwargs["bddl_file_name"] = str(Path(get_libero_path("bddl_files")) / source["suite"] / (source["task_name"] + ".bddl"))
    kwargs.update({"has_renderer": False, "has_offscreen_renderer": True, "ignore_done": True,
                   "use_camera_obs": True, "initialization_noise": None})
    env = None
    rows = []
    with h5py.File(ROOT / source["path"], "r") as dataset:
        demo = dataset["data/demo_0"]
        states, actions = demo["states"][:], demo["actions"][:]
        xml = demo.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        relocated, substitutions = rewrite_asset_paths(xml, ROOT / "assets/libero", Path(robosuite.__file__).parent)
        indices = sorted(set(np.linspace(0, len(states) - 2, count, dtype=int).tolist()))
        try:
            env = TASK_MAPPING[record["env_args"]["problem_name"]](**kwargs)
            for index in indices:
                if psutil.virtual_memory().available < 768 * 1024 ** 2:
                    raise RuntimeError("Memory floor reached; stop only this diagnostic")
                env.reset_from_xml_string(relocated)
                env.sim.reset()
                env.sim.set_state_from_flattened(states[index])
                env.sim.forward()
                robot = env.robots[0]
                qpos_indices = list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
                if qpos_indices != list(range(9)):
                    raise RuntimeError("Panda qpos layout differs from inspected state rows")
                robot.controller.update(force=True)
                robot.controller.reset_goal()
                before = env._get_observations(force_update=True)
                after, _, _, _ = env.step(actions[index])
                flat_after = env.sim.get_state().flatten()
                target = before if record["action_offset"] == 0 else after
                target_eef = np.concatenate((target["robot0_eef_pos"],
                                             transforms.quat2axisangle(target["robot0_eef_quat"])))
                eef_error = float(np.max(np.abs(target_eef - demo["obs/ee_states"][index])))
                gripper_error = float(np.max(np.abs(target["robot0_gripper_qpos"] - demo["obs/gripper_states"][index])))
                state_error = float(np.max(np.abs(flat_after - states[index + 1])))
                cameras = {}
                for camera, stored in (("agentview_image", "agentview_rgb"),
                                       ("robot0_eye_in_hand_image", "eye_in_hand_rgb")):
                    errors = image_orientation_errors(target[camera], demo["obs/" + stored][index])
                    cameras[camera] = {"mae_by_transform": errors, "best": min(errors, key=errors.get)}
                # Reconstruct the actual recorded physical state, independently
                # of unrecorded controller/gripper histories. This is the v2
                # observation/action-index gate; dynamics remain diagnostic.
                recorded_state_index = index + record["action_offset"]
                env.sim.set_state_from_flattened(states[recorded_state_index])
                env.sim.forward()
                reconstructed = env._get_observations(force_update=True)
                reconstructed_eef = np.concatenate((reconstructed["robot0_eef_pos"],
                    transforms.quat2axisangle(reconstructed["robot0_eef_quat"])))
                reconstruction = {
                    "state_index": recorded_state_index,
                    "eef_max_abs": float(np.max(np.abs(reconstructed_eef - demo["obs/ee_states"][index]))),
                    "gripper_max_abs": float(np.max(np.abs(reconstructed["robot0_gripper_qpos"]
                                                           - demo["obs/gripper_states"][index]))),
                    "cameras": {},
                }
                for camera, stored in (("agentview_image", "agentview_rgb"),
                                       ("robot0_eye_in_hand_image", "eye_in_hand_rgb")):
                    errors = image_orientation_errors(reconstructed[camera], demo["obs/" + stored][index])
                    reconstruction["cameras"][camera] = {"mae_by_transform": errors,
                                                          "best": min(errors, key=errors.get)}
                rows.append({"source_state_index": index, "source_action_index": index,
                             "stored_observation_index": index, "next_state_index": index + 1,
                             "eef_max_abs": eef_error, "gripper_max_abs": gripper_error,
                             "next_state_max_abs": state_error, "cameras": cameras,
                             "recorded_state_reconstruction": reconstruction,
                             "rss_bytes": psutil.Process().memory_info().rss,
                             "available_ram_bytes": psutil.virtual_memory().available})
        finally:
            if env is not None:
                env.close()
    # Thresholds declared before actual replay. Camera absolute error tolerates
    # renderer-version lighting changes; unique orientation remains mandatory.
    reconstructed_rows = [r["recorded_state_reconstruction"] for r in rows]
    passed = all(r["eef_max_abs"] <= 1e-3 and r["gripper_max_abs"] <= 1e-3
                 and all(c["best"] == "identity" and c["mae_by_transform"]["identity"] <= .12
                         and min(v for k, v in c["mae_by_transform"].items() if k != "identity")
                         - c["mae_by_transform"]["identity"] >= .01 for c in r["cameras"].values())
                 for r in reconstructed_rows)
    return {"status": "passed" if passed else "needs_attention", "sample_selection": "5 evenly spaced demo_0 rows",
            "samples": rows, "mujoco_version": MUJOCO_VERSION, "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "relocated_xml_sha256": hashlib.sha256(relocated.encode()).hexdigest(),
            "asset_relocations": substitutions,
            "thresholds": {"eef_max_abs": 1e-3, "gripper_max_abs": 1e-3,
                           "next_state_max_abs": 1e-2, "camera_mae": .12, "orientation_margin": .01},
            "scope": "observation_action_alignment", "dynamics_replay_verified": False,
            "pass_requirements": "All 50 demos have exact robot qpos/state mapping and 20Hz row adjacency; "
                                 "recorded-state reconstruction passes unchanged EEF/gripper/camera thresholds",
            "dynamics_limitation": "Original isolated action replay metrics are retained but not a pass claim. "
                                   "PandaGripper.current_action is integrated hidden controller state absent from "
                                   "flattened MuJoCo states; resetting each sample does not restore it.",
            "policy_image_operation": "source raw RGB rotated 180 degrees once, as current policy_observation",
            "normalization": "Original checkpoint preprocessor; no new gripper or controller scaling"}


def main():
    from embodied_research.action_demo_data import sha256_file, write_json
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/action_adaptation/study_v1/source_manifest.json")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-receipt", type=Path, default=ROOT / ".runtime/action_alignment_spatial_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an alignment report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(args.source_manifest.read_text())
    sources = [s for s in source_manifest["sources"] if args.suite is None or s["suite"] == args.suite]
    if not sources:
        raise ValueError("Requested suite absent from fixed source manifest")
    process = psutil.Process()
    if not args.v1_receipt.is_file():
        raise FileNotFoundError("Preserved v1 dynamics diagnostic receipt required")
    report = {"status": "running", "action_offset": None, "records": [], "started_unix": time.time(),
              "diagnostic_revision": 2, "scope": "observation_action_alignment",
              "dynamics_replay_verified": False,
              "v1_receipt": {"path": str(args.v1_receipt), "sha256": sha256_file(args.v1_receipt),
                             "status": json.loads(args.v1_receipt.read_text())["status"]},
              "owner_pid": process.pid, "owner_create_time": process.create_time(),
              "source_manifest_sha256": sha256_file(args.source_manifest),
              "script_sha256": sha256_file(Path(__file__)),
              "coverage": "single_source" if args.suite else "four_source_study"}
    status_path = args.output.with_name(args.output.stem + ".status.json")
    write_json(status_path, report)
    try:
        for source in sources:
            record = inspect_source(source)
            report["records"].append(record)
            write_json(status_path, report)
            record["replay"] = replay_source(source, record)
            record["status"] = record["replay"]["status"]
            write_json(status_path, report)
        report["action_offset"] = common_offset(report["records"])
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["records"]) else "needs_attention"
        report["finished_unix"] = time.time()
        write_json(status_path, report)
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "action_offset": report["action_offset"], "output": str(args.output)}))
        return 0 if report["status"] == "passed" else 1
    except BaseException as error:
        report.update({"status": "needs_attention", "error": f"{type(error).__name__}: {error}"})
        write_json(status_path, report)
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
