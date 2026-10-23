"""Pinned human-demonstration data for an explicitly development-only probe.

Metadata/download operations use only the standard library. Numpy and h5py are
imported only while preparing arrays; neither Torch nor simulator code is loaded.
The source HDF5 is retained byte-for-byte. This is not a train/test data split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


DATASET_REPO = "yifengzhu-hf/LIBERO-datasets"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
DATASET_FILE = f"libero_spatial/{TASK_NAME}_demo.hdf5"
SOURCE_BYTES = 508_779_600
SOURCE_SHA256 = "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
SOURCE_CODE_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SOURCE_CODE_URL = (
    "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/"
    f"{SOURCE_CODE_REVISION}/scripts/create_dataset.py"
)
METADATA_URL = (
    f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}"
    "/libero_spatial?limit=100"
)
DOWNLOAD_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{DATASET_FILE}"
TASK_LANGUAGE = TASK_NAME.replace("_", " ")
CHUNK_SIZE = 50
EPISODE_KEY = "demo_0"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomic metadata replacement, never a partially written success receipt."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_source(*, opener=urllib.request.urlopen) -> dict:
    """Read a small pinned HF tree response, refusing unexpected bytes/hash/path."""
    with opener(METADATA_URL, timeout=60) as response:
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Dataset metadata response unexpectedly exceeds 1 MiB")
    entries = json.loads(payload)
    matches = [item for item in entries if item.get("path") == DATASET_FILE]
    if len(matches) != 1:
        raise ValueError("Pinned official task file is missing or ambiguous")
    entry = matches[0]
    if entry.get("type") != "file" or entry.get("size") != SOURCE_BYTES:
        raise ValueError("Pinned task file size/type differs from reviewed metadata")
    if entry.get("lfs", {}).get("oid") != SOURCE_SHA256:
        raise ValueError("Pinned task file SHA256 differs from reviewed metadata")
    if entry.get("lfs", {}).get("size") != SOURCE_BYTES or SOURCE_BYTES > MAX_DOWNLOAD_BYTES:
        raise ValueError("Pinned task file exceeds the download cap or has inconsistent LFS size")
    return {
        "repo_id": DATASET_REPO, "revision": DATASET_REVISION,
        "path": DATASET_FILE, "bytes": SOURCE_BYTES, "sha256": SOURCE_SHA256,
        "download_url": DOWNLOAD_URL, "metadata_url": METADATA_URL,
        "metadata_response_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata_response_bytes": len(payload), "hf_file_metadata": entry,
        "maximum_download_bytes": MAX_DOWNLOAD_BYTES,
    }


def download_file(
    destination: Path, source: dict, *, opener=urllib.request.urlopen,
    byte_limit: int = MAX_DOWNLOAD_BYTES,
) -> dict:
    """Stream exactly one file with a hard cap; failed .part files are preserved.

    No automatic retry/resume: a partial previous attempt requires inspection,
    so repeated invocations cannot silently consume another full download.
    """
    size = source["bytes"]
    if not isinstance(size, int) or size < 1 or size > byte_limit:
        raise ValueError("Refusing download above byte limit")
    if destination.exists():
        if destination.stat().st_size != size or sha256_file(destination) != source["sha256"]:
            raise ValueError("Existing source file does not match the pinned file; refusing overwrite")
        return {"downloaded_bytes": 0, "reused_verified_file": True}
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Preserved partial download requires inspection before retry: {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    digest = hashlib.sha256()
    try:
        with opener(source["download_url"], timeout=60) as response, partial.open("xb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != size:
                raise ValueError("HTTP Content-Length differs from pinned file size")
            while total < size:
                block = response.read(min(1024 * 1024, size - total))
                if not block:
                    raise ValueError(f"Truncated download: {total} of {size} bytes")
                total += len(block)
                if total > size or total > byte_limit:
                    raise ValueError("Download crossed the declared byte limit")
                digest.update(block)
                output.write(block)
        # A trusted immutable LFS SHA binds the complete file. Reading an extra
        # byte is unnecessary, and avoiding it preserves the exact network cap.
        if total != size or digest.hexdigest() != source["sha256"]:
            raise ValueError("Downloaded source hash/size does not match the pinned file")
        partial.replace(destination)
    except BaseException as error:
        evidence = {
            "status": "needs_attention", "error": f"{type(error).__name__}: {error}",
            "bytes_read_this_attempt": total, "sha256_read_this_attempt": digest.hexdigest(),
            "partial_file": partial.name,
            "partial_bytes_on_disk": partial.stat().st_size if partial.exists() else 0,
        }
        write_json(destination.with_name(destination.name + ".failure.json"), evidence)
        raise
    return {"downloaded_bytes": total, "reused_verified_file": False}


def action_chunk(actions, observation_index: int, *, chunk_size: int = CHUNK_SIZE) -> dict:
    """Probe alignment: obs[t] -> original expert actions[t+1:t+1+K].

    Official create_dataset.py records the observation returned by env.step(a).
    This +1 is an explicit probe hypothesis, not an asserted property of every
    published LIBERO conversion. Simulator replay must validate full training.
    Padded rows repeat the final real action and must be masked out of the loss.
    """
    import numpy as np

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Expected finite expert actions with shape (T, 7)")
    if (not isinstance(observation_index, int) or isinstance(observation_index, bool)
            or not 0 <= observation_index < len(actions) - 1):
        raise ValueError("Observation must have at least one following expert action")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("Chunk size must be a positive integer")
    start = observation_index + 1
    count = min(chunk_size, len(actions) - start)
    chunk = np.repeat(np.asarray(actions[-1:], dtype=np.float32), chunk_size, axis=0)
    chunk[:count] = actions[start:start + count]
    is_pad = np.arange(chunk_size) >= count
    indices = np.full(chunk_size, -1, dtype=np.int64)
    indices[:count] = np.arange(start, start + count)
    return {"actions": chunk, "action_is_pad": is_pad, "action_indices": indices}


def convert_observation(image, image2, ee_state, gripper_state) -> dict:
    """Raw HDF5 RGB -> project's policy image orientation, exactly one rotation.

    HDF5 ee_states already contains axis-angle, not a quaternion. Preserve those
    three values and finger qpos; checkpoint processors own state/action stats.
    """
    import numpy as np

    result = {}
    for key, value in (("image", image), ("image2", image2)):
        value = np.asarray(value)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{key} must be uint8 RGB HWC")
        result[key] = np.ascontiguousarray(value[::-1, ::-1])
    ee_state = np.asarray(ee_state)
    gripper_state = np.asarray(gripper_state)
    if ee_state.shape != (6,) or gripper_state.shape != (2,):
        raise ValueError("Expected existing EEF position+axis-angle (6,) and finger qpos (2,)")
    state = np.concatenate((ee_state, gripper_state)).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite proprioception")
    result["state"] = state
    return result


def prepare_example(source_path: Path, output_dir: Path, *, source: dict) -> dict:
    """Read only demo_0, retaining original HDF5 and producing one causal chunk."""
    import h5py
    import numpy as np

    if source_path.stat().st_size != source["bytes"] or sha256_file(source_path) != source["sha256"]:
        raise ValueError("Source file changed before preparation")
    with h5py.File(source_path, "r") as dataset:
        demo = dataset[f"data/{EPISODE_KEY}"]
        actions = demo["actions"][:]
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < CHUNK_SIZE + 1:
            raise ValueError("Fixed demo_0 cannot provide the requested complete future action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite source actions")
        obs = demo["obs"]
        expected_shapes = {"agentview_rgb": (len(actions), None, None, 3),
                           "eye_in_hand_rgb": (len(actions), None, None, 3),
                           "ee_states": (len(actions), 6), "gripper_states": (len(actions), 2)}
        shapes = {}
        for key, expected in expected_shapes.items():
            actual = obs[key].shape
            if len(actual) != len(expected) or any(b is not None and a != b for a, b in zip(actual, expected)):
                raise ValueError(f"Unexpected source shape for {key}: {actual}")
            shapes[key] = {"shape": list(actual), "dtype": str(obs[key].dtype)}
        if int(demo.attrs.get("num_samples", -1)) != len(actions):
            raise ValueError("Source num_samples disagrees with action count")
        # Inspect small published metadata, never expose model XML/object state
        # as model inputs. This cannot by itself prove the regeneration history.
        published_attributes = {}
        for group_name, group in (("root", dataset), ("data", dataset["data"]), (EPISODE_KEY, demo)):
            attributes = {}
            for name in ("env_args", "problem_info", "bddl_file_name", "bddl_file", "num_samples", "total"):
                if name in group.attrs:
                    value = group.attrs[name]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    if hasattr(value, "item"):
                        value = value.item()
                    if len(str(value)) <= 32000:
                        attributes[name] = value
            published_attributes[group_name] = attributes
        example = convert_observation(obs["agentview_rgb"][0], obs["eye_in_hand_rgb"][0],
                                      obs["ee_states"][0], obs["gripper_states"][0])
        example.update(action_chunk(actions, 0))
        example.update({"episode_index": np.asarray(0, dtype=np.int64),
                        "observation_index": np.asarray(0, dtype=np.int64),
                        "task": np.asarray(TASK_LANGUAGE)})
        source_action_hash = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
        array_manifest = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in example.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    example_path = output_dir / "example.npz"
    if example_path.exists() or (output_dir / "metadata.json").exists():
        raise FileExistsError("Probe output already exists; refusing to overwrite evidence")
    with example_path.open("xb") as stream:
        np.savez_compressed(stream, **example)
    metadata = {
        "schema": "feedbackbench.action_adaptation_probe.v1", "status": "prepared",
        "purpose": "development_resource_probe_only", "final_evaluation_data": False,
        "selection": {"suite": "libero_spatial", "task_order_index": 0, "task_id": 0,
                      "task_name": TASK_NAME, "task_language": TASK_LANGUAGE,
                      "episode_key": EPISODE_KEY, "episode_index": 0,
                      "observation_index": 0, "chunk_size": CHUNK_SIZE,
                      "rule": "Fixed task 0 and demo_0 before inspecting success; no replacement/fallback"},
        "source": source, "source_file": source_path.name,
        "source_action_array_sha256": source_action_hash,
        "source_actions": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
        "source_observations": shapes, "published_hdf5_attributes": published_attributes,
        "example_file": example_path.name, "example_npz_sha256": sha256_file(example_path),
        "arrays": array_manifest, "expert_label": "Published human teleoperation actions, not policy predictions",
        "camera_orientation": "policy_ready_rot180_once",
        "camera_semantics": {"image": "agentview_rgb", "image2": "eye_in_hand_rgb",
                             "source": "Raw HDF5 RGB as published; source file is unchanged",
                             "operation": "image[::-1, ::-1], both spatial axes, exactly once",
                             "probe_next_operation": "HWC uint8 -> CHW float32 /255; DO NOT flip again",
                             "reference": "src/embodied_research/simulator.py:policy_observation"},
        "state_semantics": ["eef_x", "eef_y", "eef_z", "eef_axis_angle_x", "eef_axis_angle_y",
                            "eef_axis_angle_z", "gripper_qpos_0", "gripper_qpos_1"],
        "action_semantics": ["OSC_POSE_dx", "OSC_POSE_dy", "OSC_POSE_dz", "OSC_POSE_drx",
                             "OSC_POSE_dry", "OSC_POSE_drz", "gripper_command"],
        "action_units": "Original 7D controller input commands, not absolute position/orientation or SI pose "
                        "increments; preserve the published controller scaling in env_args",
        "normalization": "No fitted normalization, action clipping, gripper remapping or extra quaternion conversion; "
                         "pass state and actions through the pinned checkpoint's original preprocessor",
        "action_pad_mask": "action_is_pad=True excludes padded action rows from training loss; "
                           "repeat-final-action padding, action_indices=-1 for padding",
        "alignment": {
            "name": "explicit_probe_next_action_hypothesis", "observation_source_index": 0,
            "first_action_source_index": 1, "action_indices": list(range(1, CHUNK_SIZE + 1)),
            "index_units": "rows in the released HDF5 after its upstream filtering, not raw teleop timestamps",
            "source_code_revision": SOURCE_CODE_REVISION, "source_code_url": SOURCE_CODE_URL,
            "source_code_evidence": "create_dataset.py lines 175-177: obs=env.step(action); "
                                    "lines 203-221: save returned state/images; "
                                    "lines 226-227: states/actions filtered by the same valid_index",
            "published_dataset_generation_revision_known": False,
            "simulator_replay_verified": False,
            "limitation": "Current pinned official source suggests post-action observations. The dataset does not "
                          "identify its exact generation commit; the +1 probe offset is explicit and provisional. "
                          "Verify HDF5 states and simulator replay before freezing any training/evaluation protocol.",
        },
        "split_policy": "Entire demo_0 is development-only; no random frame split and no held-out claim",
        "pretraining_overlap": "Public LIBERO checkpoint overlap is unknown; no unseen-task generalization claim",
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
