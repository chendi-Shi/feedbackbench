"""Small CPU regressors for disagreement with a frozen 10-step action reference.

This learns action disagreement, not failure probability or a safety guarantee.
Torch is imported only when training or loading a trained selector. All fitting,
epoch selection and threshold selection use disjoint task partitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any

import numpy as np


SPLITS = ("train", "tuning", "calibration")
VARIANTS = ("full", "no_probe", "proprio_probe")
SEEDS = (0, 1, 2)
BUDGETS = (2, 4, 10)
RECEIPT_SCHEMA = "feedbackbench.budget_episode_receipt.v1"
SELECTOR_SCHEMA = "feedbackbench.budget_selector.v1"
SOURCE_FILES = (
    "src/embodied_research/budget_learning.py", "scripts/train_budget_selector.py",
    "tests/test_budget_learning.py",
)
TARGET_NAMES = ("translation_rmse", "rotation_rmse", "gripper_sign_disagreement")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path}")
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class EpisodeData:
    case_id: str
    task_key: str
    split: str
    features: np.ndarray
    targets: np.ndarray
    metadata: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class BudgetDataset:
    episodes: tuple[EpisodeData, ...]
    feature_slices: dict[str, Any]
    feature_layout: dict[str, Any]
    checkpoint_id: str
    campaign_fingerprint: str
    input_manifest: dict[str, Any]

    def split(self, name: str) -> tuple[EpisodeData, ...]:
        if name not in SPLITS:
            raise ValueError(f"Not a training partition: {name}")
        return tuple(episode for episode in self.episodes if episode.split == name)


def feature_indices(feature_slices: dict, feature_layout: dict, variant: str,
                    dimension: int) -> np.ndarray:
    """Resolve explicitly declared blocks; never infer a block from its name."""
    if variant not in VARIANTS or not isinstance(feature_slices, dict) or not feature_slices:
        raise ValueError("Invalid variant or feature_slices")
    blocks = {}
    coverage = np.zeros(dimension, dtype=np.int8)
    for name, bounds in feature_slices.items():
        if (not isinstance(name, str) or not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(index) is not int for index in bounds)):
            raise ValueError("Each feature block must be a [start, stop] integer pair")
        start, stop = bounds
        if not 0 <= start < stop <= dimension:
            raise ValueError(f"Invalid feature block: {name}")
        coverage[start:stop] += 1
        blocks[name] = np.arange(start, stop)
    if not np.all(coverage == 1):
        raise ValueError("Feature blocks must cover the feature vector exactly once")
    modes = feature_layout.get("modes", {})
    resolved = {}
    for mode in VARIANTS:
        specification = modes.get(mode, {})
        names = specification.get("blocks") if isinstance(specification, dict) else None
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(name not in blocks for name in names)):
            raise ValueError(f"Missing or invalid declared feature mode: {mode}")
        indices = np.concatenate([blocks[name] for name in names]).astype(np.int64)
        for dim_key in ("dim", "dimension"):
            if dim_key in specification and specification[dim_key] != len(indices):
                raise ValueError(f"Declared feature dimension differs: {mode}")
        resolved[mode] = indices
    if set(resolved["full"]) != set(range(dimension)):
        raise ValueError("The full variant must contain every feature")
    for mode in ("no_probe", "proprio_probe"):
        if len(resolved[mode]) >= dimension:
            raise ValueError(f"Ablation must exclude at least one feature: {mode}")
    return resolved[variant]


def _verify_receipt(directory: Path) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Incomplete episode: missing receipt in {directory}")
    receipt = _read_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "complete":
        raise ValueError(f"Episode receipt is not complete: {directory}")
    files = receipt.get("files", {})
    if not isinstance(files, dict) or not {
        "metadata.json", "samples.npz", "trajectory.npz", "result.json",
    }.issubset(files):
        raise ValueError(f"Receipt does not bind all episode artifacts: {directory}")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe receipt path")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping receipt artifact: {name}")
        if (not isinstance(expected, dict) or path.stat().st_size != expected.get("bytes")
                or sha256_file(path) != expected.get("sha256")):
            raise ValueError(f"Episode artifact differs from receipt: {path}")
    return {"receipt_sha256": sha256_file(receipt_path), "files": files}


def load_dataset(collection: str | Path) -> BudgetDataset:
    """Load complete training episodes, rejecting test data and partial episodes.

    Only features/targets are read from each NPZ. Candidate actions and reference
    observations cannot accidentally become input features. Receipt hashes bind
    the remaining artifacts without interpreting them as training data.
    """
    collection = Path(collection).resolve()
    if not collection.is_dir() or any(part.lower() == "test" for part in collection.parts):
        raise ValueError("Expected a training collection outside any test directory")
    episode_directories = []
    for parent, directories, names in os.walk(collection, followlinks=False):
        if any(name.lower() == "test" for name in directories):
            raise ValueError("Test directory present in training collection; refusing to read it")
        if any((Path(parent) / name).is_symlink() for name in directories):
            raise ValueError("Symlinked training directories are not allowed")
        if {"metadata.json", "samples.npz", "receipt.json"}.intersection(names):
            episode_directories.append(Path(parent))
    if not episode_directories:
        raise ValueError("No completed training episodes")
    episodes = []
    task_splits = {}
    case_ids = set()
    binding = None
    for directory in sorted(episode_directories):
        evidence = _verify_receipt(directory)
        metadata = _read_json(directory / "metadata.json")
        for name in ("case_id", "task_key", "split", "checkpoint_id", "campaign_fingerprint"):
            if not isinstance(metadata.get(name), str) or not metadata[name]:
                raise ValueError(f"Missing metadata {name}: {directory}")
        split, task_key, case_id = (metadata[name] for name in ("split", "task_key", "case_id"))
        if split not in SPLITS:
            raise ValueError(f"Unexpected partition {split!r}; test/validation are not training partitions")
        if case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if task_key in task_splits and task_splits[task_key] != split:
            raise ValueError(f"Task occurs across partitions: {task_key}")
        task_splits[task_key] = split
        with np.load(directory / "samples.npz", allow_pickle=False) as archive:
            features, targets = archive["features"], archive["targets"]
        if features.dtype != np.float32 or targets.dtype != np.float32:
            raise ValueError("features and targets must be float32")
        if (features.ndim != 2 or min(features.shape) < 1
                or targets.shape != (len(features), 2, 3)):
            raise ValueError("Expected features (N,D) and targets (N,2,3), with N,D > 0")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("Non-finite feature or target")
        if np.any(targets < 0) or np.any(targets[:, :, 2] > 1):
            raise ValueError("Targets must be nonnegative; gripper disagreement must be <= 1")
        feature_slices, feature_layout = metadata.get("feature_slices"), metadata.get("feature_layout")
        if not isinstance(feature_layout, dict):
            raise ValueError("Missing feature_layout")
        for variant in VARIANTS:
            feature_indices(feature_slices, feature_layout, variant, features.shape[1])
        current_binding = {
            "feature_slices": feature_slices, "feature_layout": feature_layout,
            "checkpoint_id": metadata["checkpoint_id"], "campaign_fingerprint": metadata["campaign_fingerprint"],
            "dimension": features.shape[1],
        }
        if binding is not None and binding != current_binding:
            raise ValueError("Feature layout, checkpoint or campaign differs across episodes")
        binding = current_binding
        evidence = {"relative_directory": directory.relative_to(collection).as_posix(), **evidence,
                    "case_id": case_id, "task_key": task_key, "split": split, "samples": len(features)}
        episodes.append(EpisodeData(case_id, task_key, split, features, targets, metadata, evidence))
    if {episode.split for episode in episodes} != set(SPLITS):
        raise ValueError("Nonempty train, tuning and calibration partitions are all required")
    manifest = {"schema": "feedbackbench.budget_training_inputs.v1", "binding": binding,
                "episodes": [episode.evidence for episode in episodes]}
    manifest["fingerprint"] = _digest(manifest)
    return BudgetDataset(tuple(episodes), binding["feature_slices"], binding["feature_layout"],
                         binding["checkpoint_id"], binding["campaign_fingerprint"], manifest)


def balanced_weights(episodes: tuple[EpisodeData, ...]) -> np.ndarray:
    """Equal tasks, then equal episodes within a task, then equal samples."""
    if not episodes:
        raise ValueError("Cannot weight an empty partition")
    counts = {task: sum(episode.task_key == task for episode in episodes)
              for task in {episode.task_key for episode in episodes}}
    return np.concatenate([
        np.full(len(episode.features), 1.0 / (len(counts) * counts[episode.task_key] * len(episode.features)))
        for episode in episodes
    ])


def fit_standardizer(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = _weights(weights, len(features))
    values = np.asarray(features, dtype=np.float64)
    mean = np.einsum("n,nd->d", weights, values)
    variance = np.einsum("n,nd->d", weights, (values - mean) ** 2)
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _weights(weights, count: int) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Expected finite positive sample weights")
    return result / result.sum()


def _scales(scales) -> np.ndarray:
    result = np.asarray(scales, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Three finite positive proxy risk scales are required")
    return result


def _errors(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 3 or result.shape[1:] != (2, 3) or len(result) == 0
            or not np.all(np.isfinite(result)) or np.any(result < 0)):
        raise ValueError("Expected nonnegative finite errors (N,2,3)")
    return result


def decode_predictions(log_predictions: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        decoded = np.maximum(0, np.expm1(np.asarray(log_predictions, dtype=np.float64)))
    return _errors(decoded.reshape(-1, 2, 3))


def weighted_quantile(values, weights, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or not 0 < quantile <= 1:
        raise ValueError("Invalid weighted quantile inputs")
    weights = _weights(weights, len(values))
    order = np.argsort(values, kind="stable")
    index = min(int(np.searchsorted(np.cumsum(weights[order]), quantile, side="left")), len(values) - 1)
    return float(values[order[index]])


def calibrate_residual(predictions, targets, weights, scales, quantile: float) -> float:
    predictions, targets = _errors(predictions), _errors(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Calibration prediction/target shapes differ")
    residuals = np.max((targets - predictions) / _scales(scales), axis=(1, 2))
    return max(0.0, weighted_quantile(residuals, weights, quantile))


def budget_decisions(predictions, *, residual_quantile: float, scales,
                     threshold: float | None = None, fixed_budget: int | None = None) -> np.ndarray:
    predictions = _errors(predictions)
    scales = _scales(scales)
    if not np.isfinite(residual_quantile) or residual_quantile < 0:
        raise ValueError("Invalid residual quantile")
    if (threshold is None) == (fixed_budget is None):
        raise ValueError("Specify either a threshold or a fixed budget")
    if fixed_budget is not None:
        if fixed_budget not in BUDGETS:
            raise ValueError("Invalid fixed budget")
        return np.full(len(predictions), fixed_budget, dtype=np.int64)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    # q is dimensionless: convert back to each raw target unit before comparison.
    upper = predictions + residual_quantile * scales
    accepted = np.all(upper <= threshold * scales, axis=2)
    return np.where(accepted[:, 0], 2, np.where(accepted[:, 1], 4, 10)).astype(np.int64)


def policy_metrics(decisions, targets, weights, scales) -> dict[str, Any]:
    targets = _errors(targets)
    decisions = np.asarray(decisions)
    if decisions.shape != (len(targets),) or not np.all(np.isin(decisions, BUDGETS)):
        raise ValueError("Invalid budget decisions")
    weights = _weights(weights, len(targets))
    risks = np.zeros(len(targets), dtype=np.float64)
    for index, budget in enumerate(BUDGETS[:2]):
        mask = decisions == budget
        risks[mask] = np.max(targets[mask, index] / _scales(scales), axis=1)
    return {
        "mean_nfe": float(weights @ decisions), "reference_disagreement_risk": float(weights @ risks),
        "budget_probabilities": [float(weights[decisions == budget].sum()) for budget in BUDGETS],
    }


def select_calibration_policy(predictions, targets, weights, *, scales, thresholds,
                              residual_quantile: float, max_mean_nfe: float = 4.0) -> dict[str, Any]:
    """Select on calibration only; returned risk is selection-set, not test risk."""
    if not 2 <= max_mean_nfe <= 10:
        raise ValueError("Mean NFE constraint must lie between 2 and 10")
    policies = [{"fixed_budget": budget} for budget in BUDGETS]
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Threshold grid must be nonempty and unique")
    policies.extend({"threshold": threshold} for threshold in thresholds)
    candidates = []
    for policy in policies:
        decisions = budget_decisions(predictions, residual_quantile=residual_quantile, scales=scales, **policy)
        metrics = policy_metrics(decisions, targets, weights, scales)
        candidates.append({"policy": policy, **metrics, "feasible": metrics["mean_nfe"] <= max_mean_nfe + 1e-10})
    feasible = [index for index, candidate in enumerate(candidates) if candidate["feasible"]]
    selected_index = min(feasible, key=lambda index: (
        candidates[index]["reference_disagreement_risk"], candidates[index]["mean_nfe"], index))
    return {"selected": candidates[selected_index], "selected_index": selected_index,
            "candidates": candidates, "max_mean_nfe": max_mean_nfe,
            "metric_scope": "calibration selection-set estimate; not an independent evaluation",
            "reference10_risk_zero_by_definition": True}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    threads: int = 2
    risk_scales: tuple[float, float, float] = (0.05, 0.05, 0.1)
    residual_quantile: float = 0.9
    thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0)
    max_mean_nfe: float = 4.0

    def validate(self) -> None:
        if not 1 <= self.epochs <= 100 or self.batch_size < 1 or self.hidden_dim != 64 or self.threads != 2:
            raise ValueError("Require epochs 1..100, positive batch size, hidden_dim=64 and CPU threads=2")
        if (not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not np.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("Invalid optimizer configuration")
        _scales(self.risk_scales)
        if not 0 < self.residual_quantile <= 1 or not 2 <= self.max_mean_nfe <= 10:
            raise ValueError("Invalid calibration configuration")
        if (not self.thresholds or len(set(self.thresholds)) != len(self.thresholds)
                or any(not np.isfinite(value) or value < 0 for value in self.thresholds)):
            raise ValueError("Invalid fixed threshold grid")


def _make_model(torch, dimension: int, hidden_dim: int):
    return torch.nn.Sequential(torch.nn.Linear(dimension, hidden_dim), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_dim, 6))


def _predict(torch, model, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            chunks.append(model(torch.from_numpy(values[start:start + batch_size])).numpy())
    predictions = np.concatenate(chunks)
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("Non-finite model predictions")
    return predictions


def _regression_metrics(log_predictions, targets, weights) -> dict[str, Any]:
    weights = _weights(weights, len(targets))
    raw = decode_predictions(log_predictions)
    errors = np.asarray(log_predictions, dtype=np.float64) - np.log1p(targets.reshape(-1, 6))
    return {"weighted_log1p_mse": float(weights @ np.mean(errors ** 2, axis=1)),
            "weighted_raw_mae_by_budget_target": np.einsum("n,nij->ij", weights, np.abs(raw - targets)).tolist()}


def _source_binding(root: Path, output: Path) -> dict[str, Any]:
    result = {}
    for relative in SOURCE_FILES:
        source = root / relative
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result[relative] = {"sha256": sha256_file(source), "snapshot": destination.relative_to(output).as_posix()}
        if sha256_file(destination) != result[relative]["sha256"]:
            raise ValueError("Source changed while taking its snapshot")
    return result


def train_campaign(data: str | Path, output: str | Path, *, config: TrainingConfig | None = None,
                   project_root: str | Path | None = None) -> dict[str, Any]:
    """Fit all nine predeclared regressors serially; never choose a seed/variant."""
    config = config or TrainingConfig()
    config.validate()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be new or empty; preserve prior runs")
    dataset = load_dataset(data)
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)
    sources = _source_binding(root, output)
    config_record = {"training": asdict(config), "variants": VARIANTS, "seeds": SEEDS,
                     "target_names": TARGET_NAMES, "candidate_budgets": BUDGETS[:2], "reference_budget": 10,
                     "objective": "MSE of log1p for all six nonnegative disagreement targets, including gripper fraction",
                     "risk_scale_units": "normalized action discrepancy; not physical robot tolerances",
                     "probe_cost_accounting": "shared first velocity reused; deploy must verify actual NFE parity",
                     "selection": "epoch on tuning; residual and fixed policy on calibration; no seed/variant selection",
                     "claims": "no success, failure-probability, safety or cross-task coverage guarantee"}
    binding = {"data_fingerprint": dataset.input_manifest["fingerprint"],
               "campaign_fingerprint": dataset.campaign_fingerprint,
               "config_sha256": _digest(config_record), "sources": sources}
    _write_json(output / "data_manifest.json", dataset.input_manifest)
    _write_json(output / "config.json", config_record)
    _write_json(output / "binding.json", binding)
    summary = {"schema": "feedbackbench.budget_training_run.v1", "status": "running", "models": [],
               "started_at": datetime.now(timezone.utc).isoformat(), "binding": binding,
               "split_counts": {split: {"tasks": len({episode.task_key for episode in dataset.split(split)}),
                                         "episodes": len(dataset.split(split)),
                                         "samples": sum(len(episode.features) for episode in dataset.split(split))}
                                for split in SPLITS}}
    _write_json(output / "summary.json", summary)
    try:
        import torch
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        summary["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                              "threads": torch.get_num_threads(), "deterministic_algorithms": True}
        dimension = dataset.episodes[0].features.shape[1]
        for variant in VARIANTS:
            indices = feature_indices(dataset.feature_slices, dataset.feature_layout, variant, dimension)
            arrays = {}
            for split in SPLITS:
                episodes = dataset.split(split)
                arrays[split] = {
                    "x": np.concatenate([episode.features[:, indices] for episode in episodes]),
                    "y": np.concatenate([episode.targets for episode in episodes]), "w": balanced_weights(episodes),
                }
            mean, scale = fit_standardizer(arrays["train"]["x"], arrays["train"]["w"])
            for split in SPLITS:
                arrays[split]["x"] = ((arrays[split]["x"] - mean) / scale).astype(np.float32)
            for seed in SEEDS:
                model_dir = output / f"{variant}_seed{seed}"
                model_dir.mkdir()
                started = time.monotonic()
                torch.manual_seed(seed)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                model = _make_model(torch, len(indices), config.hidden_dim)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                             weight_decay=config.weight_decay)
                train = arrays["train"]
                x = torch.from_numpy(train["x"])
                y = torch.from_numpy(np.log1p(train["y"].reshape(-1, 6)))
                # Mean batch gradient is an unbiased estimate of the weighted full objective.
                w = torch.from_numpy((train["w"] * len(x)).astype(np.float32))
                curve, best_loss, best_state, best_epoch = [], float("inf"), None, None
                for epoch in range(1, config.epochs + 1):
                    model.train()
                    permutation = torch.randperm(len(x), generator=generator)
                    for start in range(0, len(x), config.batch_size):
                        batch = permutation[start:start + config.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        error = (model(x[batch]) - y[batch]).square().mean(dim=1)
                        loss = (error * w[batch]).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        optimizer.step()
                    metrics = {}
                    for split in ("train", "tuning"):
                        values = arrays[split]
                        metrics[split] = _regression_metrics(
                            _predict(torch, model, values["x"], config.batch_size), values["y"], values["w"])
                    tuning_loss = metrics["tuning"]["weighted_log1p_mse"]
                    if tuning_loss < best_loss:
                        best_loss, best_epoch = tuning_loss, epoch
                        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                    curve.append({"epoch": epoch, **metrics})
                    _write_json(model_dir / "curve.json", curve)
                model.load_state_dict(best_state)
                calibration = arrays["calibration"]
                calibration_log = _predict(torch, model, calibration["x"], config.batch_size)
                predictions = decode_predictions(calibration_log)
                q = calibrate_residual(predictions, calibration["y"], calibration["w"], config.risk_scales,
                                       config.residual_quantile)
                selection = select_calibration_policy(
                    predictions, calibration["y"], calibration["w"], scales=config.risk_scales,
                    thresholds=config.thresholds, residual_quantile=q, max_mean_nfe=config.max_mean_nfe)
                validation = {}
                for split in ("tuning", "calibration"):
                    values = arrays[split]
                    logs = _predict(torch, model, values["x"], config.batch_size)
                    validation[split] = _regression_metrics(logs, values["y"], values["w"])
                    np.savez_compressed(model_dir / f"{split}_predictions.npz", predictions=decode_predictions(logs),
                                        targets=values["y"], weights=values["w"])
                checkpoint = {
                    "schema": SELECTOR_SCHEMA, "variant": variant, "seed": seed, "best_epoch": best_epoch,
                    "input_dimension": dimension, "feature_indices": indices.tolist(),
                    "feature_layout": dataset.feature_layout, "feature_slices": dataset.feature_slices,
                    "checkpoint_id": dataset.checkpoint_id, "hidden_dim": config.hidden_dim,
                    "standardizer_mean": mean.tolist(), "standardizer_scale": scale.tolist(),
                    "state_dict": best_state, "risk_scales": list(config.risk_scales),
                    "residual_quantile_value": q, "residual_quantile_level": config.residual_quantile,
                    "policy": selection["selected"]["policy"],
                    "calibration_budget_probabilities": selection["selected"]["budget_probabilities"],
                    "binding": binding, "model_kind": "log1p_reference_disagreement_regressor",
                    "probe_cost_accounting": config_record["probe_cost_accounting"],
                }
                temporary = model_dir / "selector.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(model_dir / "selector.pt")
                _write_json(model_dir / "selection.json", selection)
                _write_json(model_dir / "validation_metrics.json", validation)
                model_record = {
                    "variant": variant, "seed": seed, "best_epoch": best_epoch, "tuning_loss": best_loss,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "input_features": len(indices), "seconds": time.monotonic() - started,
                    "residual_quantile_value": q, "selection": selection["selected"],
                    "checkpoint": f"{model_dir.name}/selector.pt",
                    "checkpoint_sha256": sha256_file(model_dir / "selector.pt"),
                }
                _write_json(model_dir / "model.json", model_record)
                summary["models"].append(model_record)
                _write_json(output / "summary.json", summary)
                del model, optimizer, x, y, w, best_state, checkpoint
            del arrays
        # Detect any input/source edits during training, rather than silently binding stale bytes.
        if load_dataset(data).input_manifest["fingerprint"] != binding["data_fingerprint"]:
            raise ValueError("Training inputs changed during fitting")
        if any(sha256_file(root / name) != record["sha256"] for name, record in sources.items()):
            raise ValueError("Training source changed during fitting")
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        _write_json(output / "summary.json", summary)
        raise
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "summary.json", summary)
    return summary


class BudgetSelector:
    """Immutable fitted selector; observation-independent random control is separate."""

    def __init__(self, torch, checkpoint: dict[str, Any]):
        if checkpoint.get("schema") != SELECTOR_SCHEMA:
            raise ValueError("Unknown budget selector checkpoint schema")
        self._torch, self.metadata = torch, checkpoint
        self.indices = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
        expected = feature_indices(checkpoint["feature_slices"], checkpoint["feature_layout"],
                                   checkpoint["variant"], checkpoint["input_dimension"])
        if not np.array_equal(expected, self.indices):
            raise ValueError("Checkpoint feature indices disagree with layout")
        self.mean = np.asarray(checkpoint["standardizer_mean"], dtype=np.float64)
        self.scale = np.asarray(checkpoint["standardizer_scale"], dtype=np.float64)
        if (self.mean.shape != self.indices.shape or self.scale.shape != self.indices.shape
                or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale))
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid checkpoint standardizer")
        probabilities = np.asarray(checkpoint["calibration_budget_probabilities"], dtype=np.float64)
        if (probabilities.shape != (3,) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("Invalid calibration budget probabilities")
        self.calibration_budget_probabilities = probabilities.copy()
        self.model = _make_model(torch, len(self.indices), checkpoint["hidden_dim"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def select_budget(self, features: np.ndarray) -> dict[str, Any]:
        values = np.asarray(features, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if (values.ndim != 2 or len(values) == 0 or values.shape[1] != self.metadata["input_dimension"]
                or not np.all(np.isfinite(values))):
            raise ValueError("Invalid selector features")
        x = ((values[:, self.indices] - self.mean) / self.scale).astype(np.float32)
        predictions = decode_predictions(_predict(self._torch, self.model, x, 256))
        q = self.metadata["residual_quantile_value"]
        scales = np.asarray(self.metadata["risk_scales"])
        decisions = budget_decisions(predictions, residual_quantile=q, scales=scales, **self.metadata["policy"])
        upper = predictions + q * scales
        probabilities = (decisions[:, None] == np.asarray(BUDGETS)).astype(np.float64)
        return {
            "budget": int(decisions[0]) if single else decisions,
            "predictions": predictions[0] if single else predictions,
            "upper_errors": upper[0] if single else upper,
            "budget_probabilities": probabilities[0] if single else probabilities,
            "calibration_budget_probabilities": self.calibration_budget_probabilities.copy(),
            "variant": self.metadata["variant"], "seed": self.metadata["seed"],
        }


def load_selector(checkpoint: str | Path, *, expected_sha256: str | None = None) -> BudgetSelector:
    path = Path(checkpoint)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Selector checkpoint hash differs")
    import torch
    torch.set_num_threads(2)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return BudgetSelector(torch, payload)
