import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from embodied_research import action_demo_data as data


class Response(io.BytesIO):
    def __init__(self, payload, content_length=None):
        super().__init__(payload)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


def test_pinned_metadata_checks_size_sha_and_path():
    entry = {"path": data.DATASET_FILE, "type": "file", "size": data.SOURCE_BYTES,
             "lfs": {"oid": data.SOURCE_SHA256, "size": data.SOURCE_BYTES}}
    def opener(*args, **kwargs):
        return Response(json.dumps([entry]).encode())
    assert data.inspect_source(opener=opener)["revision"] == data.DATASET_REVISION
    entry["lfs"]["oid"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        data.inspect_source(opener=opener)


def test_download_stream_hash_reuse_and_cap(tmp_path):
    payload = b"real demonstration fixture"
    source = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "download_url": "fixture"}
    dest = tmp_path / "demo.hdf5"
    with pytest.raises(ValueError, match="byte limit"):
        data.download_file(dest, source, byte_limit=1)
    result = data.download_file(dest, source, opener=lambda *a, **k: Response(payload, len(payload)))
    assert result["downloaded_bytes"] == len(payload)
    assert dest.read_bytes() == payload
    assert data.download_file(dest, source)["reused_verified_file"]
    dest.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        data.download_file(dest, source)


def test_failure_preserves_partial_bytes_and_hash(tmp_path):
    dest = tmp_path / "demo.hdf5"
    source = {"bytes": 20, "sha256": "0" * 64, "download_url": "fixture"}
    with pytest.raises(ValueError, match="Truncated"):
        data.download_file(dest, source, opener=lambda *a, **k: Response(b"partial"))
    assert not dest.exists()
    assert (tmp_path / "demo.hdf5.part").read_bytes() == b"partial"
    evidence = json.loads((tmp_path / "demo.hdf5.failure.json").read_text())
    assert evidence["sha256_read_this_attempt"] == hashlib.sha256(b"partial").hexdigest()
    with pytest.raises(FileExistsError, match="inspection"):
        data.download_file(dest, source)


def test_chunk_is_future_supervision_and_padding_is_masked():
    actions = np.arange(60 * 7, dtype=np.float32).reshape(60, 7)
    chunk = data.action_chunk(actions, 0)
    np.testing.assert_array_equal(chunk["actions"], actions[1:51])
    assert not chunk["action_is_pad"].any()
    np.testing.assert_array_equal(chunk["action_indices"], np.arange(1, 51))
    tail = data.action_chunk(actions, 57)
    assert tail["action_is_pad"].sum() == 48
    np.testing.assert_array_equal(tail["actions"][:2], actions[58:])
    np.testing.assert_array_equal(tail["actions"][2:], np.repeat(actions[-1:], 48, axis=0))
    assert (tail["action_indices"][2:] == -1).all()
    with pytest.raises(ValueError, match="following"):
        data.action_chunk(actions, 59)


def test_rgb_rotation_once_existing_axisangle_and_gripper_preserved():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    eef = np.asarray([1., 2., 3., .4, .5, .6])
    grip = np.asarray([.02, -.02])
    result = data.convert_observation(image, image + 1, eef, grip)
    np.testing.assert_array_equal(result["image"], image[::-1, ::-1])
    np.testing.assert_array_equal(result["state"], np.concatenate((eef, grip)).astype(np.float32))
    assert result["image"].flags.c_contiguous
    with pytest.raises(ValueError, match="EEF"):
        data.convert_observation(image, image, np.zeros(7), grip)


def make_demo(path, frames=60):
    with h5py.File(path, "w") as dataset:
        demo = dataset.create_group("data/demo_0")
        demo.attrs["num_samples"] = frames
        demo.create_dataset("actions", data=np.arange(frames * 7).reshape(frames, 7) / 1000.)
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.arange(frames * 4 * 5 * 3, dtype=np.uint8).reshape(frames, 4, 5, 3))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((frames, 4, 5, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((frames, 6)))
        obs.create_dataset("gripper_states", data=np.zeros((frames, 2)))
    return {"bytes": path.stat().st_size, "sha256": data.sha256_file(path)}


def test_prepare_binds_actual_bytes_selection_and_provisional_alignment(tmp_path):
    source = tmp_path / "source.hdf5"
    provenance = make_demo(source)
    output = tmp_path / "prepared"
    result = data.prepare_example(source, output, source=provenance)
    assert data.sha256_file(source) == provenance["sha256"]
    assert result["selection"]["episode_key"] == "demo_0"
    assert result["camera_orientation"] == "policy_ready_rot180_once"
    assert result["alignment"]["simulator_replay_verified"] is False
    assert result["final_evaluation_data"] is False
    with np.load(output / "example.npz", allow_pickle=False) as example:
        assert example["task"].item() == data.TASK_LANGUAGE
        assert example["actions"].shape == (50, 7)
        np.testing.assert_array_equal(example["action_indices"], np.arange(1, 51))
    assert result["example_npz_sha256"] == data.sha256_file(output / "example.npz")
    with pytest.raises(FileExistsError, match="overwrite"):
        data.prepare_example(source, output, source=provenance)


def test_refuses_short_demo_without_selecting_another(tmp_path):
    source = tmp_path / "source.hdf5"
    provenance = make_demo(source, frames=50)
    with pytest.raises(ValueError, match="complete future"):
        data.prepare_example(source, tmp_path / "out", source=provenance)


def test_import_metadata_module_never_imports_torch_or_h5py():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; sys.path.insert(0, 'src'); import embodied_research.action_demo_data; " \
           "assert 'torch' not in sys.modules; assert 'h5py' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root, check=True, timeout=20)
