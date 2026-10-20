"""Data/math tests; opt-in test trains only tiny interface fixtures, never a VLA."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from embodied_research import budget_learning as budget


SLICES = {"image": [0, 2], "state": [2, 3], "noise": [3, 4], "v0": [4, 6]}
LAYOUT = {"modes": {
    "full": {"blocks": ["image", "state", "noise", "v0"], "dimension": 6},
    "no_probe": {"blocks": ["image", "state", "noise"], "dimension": 4},
    "proprio_probe": {"blocks": ["state", "noise", "v0"], "dimension": 4},
}}


def write_receipt(directory):
    files = {}
    for name in ("metadata.json", "samples.npz", "trajectory.npz", "result.json"):
        path = directory / name
        files[name] = {"sha256": budget.sha256_file(path), "bytes": path.stat().st_size}
    (directory / "receipt.json").write_text(json.dumps({
        "schema": budget.RECEIPT_SCHEMA, "status": "complete", "files": files,
    }), encoding="utf-8")


def write_episode(collection, case_id, task_key, split, n=3, offset=0):
    directory = collection / case_id
    directory.mkdir(parents=True)
    features = np.arange(n * 6, dtype=np.float32).reshape(n, 6) / 20 + offset
    targets = np.full((n, 2, 3), 0.02, dtype=np.float32)
    targets[:, 1] /= 2
    np.savez_compressed(directory / "samples.npz", features=features, targets=targets,
                        candidate_actions=np.array([{"never": "read this object array"}], dtype=object))
    np.savez_compressed(directory / "trajectory.npz", actions=np.zeros((1, 7), dtype=np.float32))
    (directory / "result.json").write_text('{"status":"complete"}', encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps({
        "case_id": case_id, "task_key": task_key, "split": split,
        "feature_slices": SLICES, "feature_layout": LAYOUT,
        "checkpoint_id": "test-fixture-not-a-vla", "campaign_fingerprint": "frozen-synthetic-campaign",
    }), encoding="utf-8")
    write_receipt(directory)
    return directory


def edit_metadata(directory, **changes):
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    write_receipt(directory)


@pytest.fixture
def collection(tmp_path):
    result = tmp_path / "collection"
    for split, offset in zip(budget.SPLITS, (0, 10, 20)):
        write_episode(result, f"case_{split}", f"task_{split}", split, offset=offset)
    return result


def test_loader_reads_only_declared_features_and_targets(collection):
    dataset = budget.load_dataset(collection)
    assert len(dataset.episodes) == 3
    assert dataset.split("train")[0].features.shape == (3, 6)
    assert dataset.split("calibration")[0].targets.shape == (3, 2, 3)
    assert dataset.input_manifest["fingerprint"] == budget.load_dataset(collection).input_manifest["fingerprint"]
    with pytest.raises(ValueError, match="Not a training partition"):
        dataset.split("test")


@pytest.mark.parametrize("split", ["test", "validation"])
def test_loader_rejects_unplanned_partition(collection, split):
    edit_metadata(collection / "case_calibration", split=split)
    with pytest.raises(ValueError, match="Unexpected partition"):
        budget.load_dataset(collection)


def test_loader_rejects_test_directory_before_traversing_it(collection):
    (collection / "test").mkdir()
    (collection / "test" / "metadata.json").write_text("not even valid json")
    with pytest.raises(ValueError, match="Test directory"):
        budget.load_dataset(collection)


def test_loader_rejects_task_overlap_and_repeated_case_id(collection):
    directory = collection / "case_tuning"
    edit_metadata(directory, task_key="task_train")
    with pytest.raises(ValueError, match="across partitions"):
        budget.load_dataset(collection)
    edit_metadata(directory, task_key="task_tuning", case_id="case_train")
    with pytest.raises(ValueError, match="Duplicate case_id"):
        budget.load_dataset(collection)


@pytest.mark.parametrize("mutation", ["no_receipt", "incomplete", "changed_metadata", "missing_bound_file"])
def test_loader_rejects_incomplete_or_tampered_episode(collection, mutation):
    directory = collection / "case_train"
    receipt_path = directory / "receipt.json"
    if mutation == "no_receipt":
        receipt_path.unlink()
    elif mutation == "incomplete":
        receipt = json.loads(receipt_path.read_text())
        receipt["status"] = "running"
        receipt_path.write_text(json.dumps(receipt))
    elif mutation == "changed_metadata":
        with (directory / "metadata.json").open("a") as stream:
            stream.write(" ")
    else:
        (directory / "trajectory.npz").unlink()
    with pytest.raises(ValueError):
        budget.load_dataset(collection)


def test_loader_rejects_path_escape_even_with_correct_file_hash(collection):
    directory = collection / "case_train"
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["../case_tuning/metadata.json"] = receipt["files"]["metadata.json"]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Unsafe receipt path"):
        budget.load_dataset(collection)


@pytest.mark.parametrize("mutation", ["float64", "negative", "gripper", "nan", "shape"])
def test_loader_rejects_invalid_arrays(collection, mutation):
    directory = collection / "case_train"
    x, y = np.zeros((3, 6), np.float32), np.zeros((3, 2, 3), np.float32)
    if mutation == "float64":
        x = x.astype(np.float64)
    elif mutation == "negative":
        y[0, 0, 0] = -1
    elif mutation == "gripper":
        y[0, 1, 2] = 1.01
    elif mutation == "nan":
        x[0, 0] = np.nan
    else:
        y = y[:, 0]
    np.savez_compressed(directory / "samples.npz", features=x, targets=y)
    write_receipt(directory)
    with pytest.raises(ValueError):
        budget.load_dataset(collection)


@pytest.mark.parametrize("field,value", [("checkpoint_id", "other"), ("campaign_fingerprint", "changed")])
def test_loader_binds_one_checkpoint_and_campaign(collection, field, value):
    edit_metadata(collection / "case_tuning", **{field: value})
    with pytest.raises(ValueError, match="differs across episodes"):
        budget.load_dataset(collection)


def test_feature_indices_follow_declared_modes_and_reject_bad_layout():
    np.testing.assert_array_equal(budget.feature_indices(SLICES, LAYOUT, "proprio_probe", 6), [2, 3, 4, 5])
    bad = {**SLICES, "v0": [3, 6]}
    with pytest.raises(ValueError, match="exactly once"):
        budget.feature_indices(bad, LAYOUT, "full", 6)
    layout = json.loads(json.dumps(LAYOUT))
    layout["modes"]["no_probe"]["blocks"] = ["image", "image"]
    with pytest.raises(ValueError, match="feature mode"):
        budget.feature_indices(SLICES, layout, "no_probe", 6)


def episode(task, n):
    return budget.EpisodeData("case", task, "train", np.zeros((n, 2), np.float32),
                              np.zeros((n, 2, 3), np.float32), {}, {})


def test_hierarchical_weights_equal_tasks_then_episodes_not_frames():
    weights = budget.balanced_weights((episode("a", 2), episode("a", 4), episode("b", 1)))
    np.testing.assert_allclose(weights, [0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0.5])
    assert weights.sum() == 1
    # Duplicating samples within one episode leaves its contribution unchanged.
    doubled = budget.balanced_weights((episode("a", 4), episode("a", 4), episode("b", 1)))
    assert doubled[:4].sum() == weights[:2].sum()


def test_standardizer_uses_weighted_training_values_and_handles_constant_features():
    features = np.array([[1, 5], [3, 5]], dtype=np.float32)
    mean, scale = budget.fit_standardizer(features, np.array([0.25, 0.75]))
    np.testing.assert_allclose(mean, [2.5, 5])
    np.testing.assert_allclose(scale, [np.sqrt(0.75), 1])
    assert np.isfinite((np.array([[10000, 5]]) - mean) / scale).all()


def test_log_prediction_inverse_nonnegative_and_nonfinite_rejected():
    truth = np.arange(12).reshape(2, 2, 3) / 100
    np.testing.assert_allclose(budget.decode_predictions(np.log1p(truth).reshape(2, 6)), truth)
    assert not budget.decode_predictions(np.full((2, 6), -1)).any()
    with pytest.raises((ValueError, FloatingPointError)):
        budget.decode_predictions(np.full((2, 6), 1000))


def test_calibration_residual_is_dimensionless_and_scaled_back_per_target():
    predicted = np.zeros((1, 2, 3))
    targets = np.array([[[0.05, 0.1, 0.3], [0, 0, 0]]])
    scales = (0.05, 0.05, 0.1)
    q = budget.calibrate_residual(predicted, targets, [1], scales, 0.9)
    assert q == pytest.approx(3)
    assert budget.budget_decisions(predicted, residual_quantile=q, scales=scales, threshold=3)[0] == 2
    assert budget.budget_decisions(predicted, residual_quantile=q, scales=scales, threshold=2.99)[0] == 10


def test_weighted_quantile_preserves_task_weight_instead_of_frame_count():
    assert budget.weighted_quantile([0, 0, 0, 9], [1, 1, 1, 9], 0.5) == 9
    assert budget.weighted_quantile([3, 1, 2], [1, 1, 1], 1) == 3
    with pytest.raises(ValueError):
        budget.weighted_quantile([1, 2], [0, 1], 0.9)


def test_policy_selection_respects_budget_and_actual_nonmonotonic_disagreement():
    predictions = np.array([[[0.01] * 3, [0.3] * 3], [[2] * 3, [0.01] * 3]])
    targets = np.array([[[0.1] * 3, [0.5] * 3], [[2] * 3, [0.3] * 3]])
    result = budget.select_calibration_policy(predictions, targets, [1, 1], scales=[1, 1, 1],
                                               thresholds=[0.1], residual_quantile=0)
    assert result["selected"]["policy"] == {"threshold": 0.1}
    assert result["selected"]["mean_nfe"] == 3
    assert result["selected"]["reference_disagreement_risk"] == pytest.approx(0.2)
    assert result["selected"]["budget_probabilities"] == [0.5, 0.5, 0.0]
    # A worse four-step candidate is legal; no monotonicity is imposed.
    targets[:, 0] = 0.01
    targets[:, 1] = 0.8
    result = budget.select_calibration_policy(predictions, targets, [1, 1], scales=[1, 1, 1],
                                               thresholds=[0.1], residual_quantile=0)
    assert result["selected"]["policy"] == {"fixed_budget": 2}


def test_explicit_fixed_baselines_allow_feasible_selection_and_record_reference_zero():
    targets = np.ones((2, 2, 3))
    targets[:, 1] = 0.1
    result = budget.select_calibration_policy(np.ones_like(targets), targets, [1, 1], scales=[1] * 3,
                                               thresholds=[0], residual_quantile=10)
    assert result["selected"]["policy"] == {"fixed_budget": 4}
    reference = next(item for item in result["candidates"] if item["policy"] == {"fixed_budget": 10})
    assert reference["reference_disagreement_risk"] == 0
    assert not reference["feasible"]


@pytest.mark.parametrize("change", [{"epochs": 101}, {"threads": 4}, {"hidden_dim": 32},
                                     {"thresholds": (1, 1)}, {"risk_scales": (0, 1, 1)}])
def test_training_configuration_keeps_prespecified_limits(change):
    with pytest.raises(ValueError):
        budget.TrainingConfig(**change).validate()


@pytest.mark.skipif(os.environ.get("FEEDBACKBENCH_BUDGET_TRAIN_TEST") != "1",
                    reason="Opt-in tiny CPU MLP training; no VLA/checkpoint download")
def test_tiny_training_roundtrip_all_seeds_variants_and_bound_evidence(collection, tmp_path):
    output = tmp_path / "trained"
    summary = budget.train_campaign(collection, output, config=budget.TrainingConfig(epochs=2, batch_size=2))
    assert summary["status"] == "complete"
    assert {(model["variant"], model["seed"]) for model in summary["models"]} == {
        (variant, seed) for variant in budget.VARIANTS for seed in budget.SEEDS}
    train = budget.load_dataset(collection).split("train")[0]
    for model in summary["models"]:
        selector = budget.load_selector(output / model["checkpoint"], expected_sha256=model["checkpoint_sha256"])
        expected_mean = train.features[:, selector.indices].mean(axis=0)
        np.testing.assert_allclose(selector.mean, expected_mean, atol=1e-7)
        prediction = selector.select_budget(train.features[0])
        assert prediction["budget"] in budget.BUDGETS
        assert prediction["predictions"].shape == (2, 3)
        assert np.all(prediction["predictions"] >= 0)
        assert prediction["budget_probabilities"].sum() == 1
        assert prediction["calibration_budget_probabilities"].sum() == pytest.approx(1)
        batch = selector.select_budget(train.features)
        assert batch["budget"].shape == (len(train.features),)
        np.testing.assert_allclose(batch["predictions"][0], prediction["predictions"], atol=1e-7)
        assert selector.metadata["binding"]["data_fingerprint"] == summary["binding"]["data_fingerprint"]
    with pytest.raises(ValueError, match="new or empty"):
        budget.train_campaign(collection, output, config=budget.TrainingConfig(epochs=1))
    first = output / summary["models"][0]["checkpoint"]
    with pytest.raises(ValueError, match="hash differs"):
        budget.load_selector(first, expected_sha256="0" * 64)
    assert Path(output / "source_snapshot/src/embodied_research/budget_learning.py").is_file()
