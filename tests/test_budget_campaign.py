"""Evidence, resource ownership and deployment-label regression tests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from budget_campaign_common import (canonical_hash, case_directory, deadline_reached,
    digest, owned_path, read_json, seal_episode, verify_episode, write_json)
from budget_episode_worker import deviation_targets, noise_seed
from run_budget_campaign import campaign_lock, freeze_trained, is_alive, process_identity


def test_deadline_is_utc_inclusive():
    protocol = {"deadline_utc": "2026-10-30T06:08:00Z"}
    assert not deadline_reached(protocol, datetime(2026, 10, 30, 6, 7, 59, tzinfo=timezone.utc))
    assert deadline_reached(protocol, datetime(2026, 10, 30, 6, 8, 0, tzinfo=timezone.utc))


@pytest.mark.parametrize("name", ["..", "../outside", "."])
def test_owned_path_rejects_parent_and_root(tmp_path, name):
    with pytest.raises(ValueError):
        owned_path(tmp_path, name)


def test_noise_pairing_and_windows_names():
    case = {"case_id": "eval:libero_spatial:7:init41:fixed_2", "task_uid": "libero_spatial:7", "initial_state_id": 41}
    other = dict(case, case_id="eval:libero_spatial:7:init41:full_seed_0", method_id="full_seed_0")
    assert noise_seed(case, 2) == noise_seed(other, 2)
    assert noise_seed(case, 2) != noise_seed(case, 3)
    assert case_directory(case) != case_directory(other)
    assert len(case_directory(case)) == 16 and ":" not in case_directory(case)


def test_gripper_label_uses_postprocessed_actions():
    normalized = {k: np.zeros((1, 50, 7), dtype=np.float32) for k in (2, 4, 10)}
    normalized[2][..., 6] = .02
    normalized[10][..., 6] = .1
    environment = {k: values.copy() for k, values in normalized.items()}
    for value in environment.values():
        value[..., 6] -= .0496
    labels = deviation_targets(normalized, environment)
    assert labels.shape == (2, 3) and labels.dtype == np.float32
    assert labels[0, 2] == 1  # Both normalized values positive, real commands disagree.
    assert np.array_equal(labels[:, :2], np.zeros((2, 2)))


def episode(tmp_path):
    case = {"case_id": "collect:libero_spatial:0:init0", "task_uid": "libero_spatial:0",
            "suite": "libero_spatial", "split": "train"}
    protocol = {"campaign_fingerprint": "fixture"}
    write_json(tmp_path / "metadata.json", {"case_id": case["case_id"], "task_key": case["task_uid"],
               "case": case, "split": "train", "phase": "collection", "method": None,
               "campaign_fingerprint": "fixture"})
    write_json(tmp_path / "result.json", {"complete": True, "success": False, "terminated": False, "truncated": True,
               "steps": 280, "horizon": 280, "decisions": 1, "actual_nfe": 14,
               "budget_frequencies": {"2": 1, "4": 0, "10": 0}})
    write_json(tmp_path / "decisions/000.json", {"decision": 0, "budget": 2, "actual_nfe": 14, "executed_steps": 280})
    np.savez_compressed(tmp_path / "trajectory.npz", actions=np.zeros((280, 7)), states=np.zeros((280, 8)))
    np.savez_compressed(tmp_path / "samples.npz", features=np.zeros((1, 3)), targets=np.zeros((1, 2, 3)))
    seal_episode(tmp_path)
    return case, protocol


def test_completed_failure_is_valid_evidence(tmp_path):
    case, protocol = episode(tmp_path)
    _, result = verify_episode(tmp_path, case, protocol, strict=True)
    assert result["success"] is False


@pytest.mark.parametrize("mutation", ["tamper", "extra", "early_truncation", "nfe"])
def test_receipt_and_terminal_contract_reject_bad_evidence(tmp_path, mutation):
    case, protocol = episode(tmp_path)
    if mutation == "tamper":
        (tmp_path / "samples.npz").write_bytes(b"changed")
    elif mutation == "extra":
        (tmp_path / "unexpected.txt").write_text("late write")
    else:
        result = read_json(tmp_path / "result.json")
        result["steps" if mutation == "early_truncation" else "actual_nfe"] = 279
        write_json(tmp_path / "result.json", result)
        (tmp_path / "receipt.json").unlink()
        seal_episode(tmp_path)
    with pytest.raises(ValueError):
        verify_episode(tmp_path, case, protocol, strict=True)


def test_only_one_supervisor_and_pid_reuse_check(tmp_path):
    identity = process_identity()
    assert is_alive(identity)
    assert not is_alive({**identity, "created": identity["created"] - 1})
    with campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active supervisor"):
            with campaign_lock(tmp_path):
                pass
    assert not (tmp_path / "supervisor.lock").exists()


def test_restore_complete_training_requires_all_checkpoint_hashes(tmp_path):
    trained = tmp_path / "trained"
    config = {"training": "fixture"}
    data = {"episodes": []}
    data["fingerprint"] = canonical_hash(data)
    binding = {"campaign_fingerprint": "fixture", "config_sha256": canonical_hash(config),
               "data_fingerprint": data["fingerprint"], "sources": {}}
    models = []
    for mode in ("full", "no_probe", "proprio_probe"):
        for seed in (0, 1, 2):
            relative = f"{mode}_seed{seed}/selector.pt"
            path = trained / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(bytes([seed]))
            models.append({"variant": mode, "seed": seed, "checkpoint": relative, "checkpoint_sha256": digest(path)})
    write_json(trained / "summary.json", {"status": "complete", "models": models, "binding": binding})
    write_json(trained / "binding.json", binding)
    write_json(trained / "config.json", config)
    write_json(trained / "data_manifest.json", data)
    protocol = {"campaign_fingerprint": "fixture", "training": {"feature_modes": ["full", "no_probe", "proprio_probe"],
                "seeds": [0, 1, 2]}, "collection": {"cases": []}}
    checkpoint = trained / models[0]["checkpoint"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        freeze_trained(tmp_path, protocol)
    checkpoint.write_bytes(bytes([0]))
    freeze_trained(tmp_path, protocol)
    assert read_json(tmp_path / "trained_manifest.json")["controller_count"] == 9
