"""Auditable statistics for a frozen, paired pretrained-policy experiment.

Task macro averages give each (suite, task_id) equal weight. Confidence
intervals resample task clusters, never pretend all episodes are independent,
and do not measure model-training uncertainty or sim-to-real generalization.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IDENTITY_FIELDS = ("model", "suite", "task_id", "init_state_id", "seed", "condition", "method")
TIMING_FIELDS = (
    "policy_inference_seconds", "preprocess_seconds", "scheduler_seconds", "simulator_seconds", "wall_seconds",
)


class ExperimentIntegrityError(ValueError):
    """The corpus does not match the declared experiment or contains invalid data."""


@dataclass(frozen=True)
class ExperimentData:
    output_dir: Path
    protocol: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]
    missing_ids: tuple[str, ...]
    partial: bool
    input_sha256: dict[str, str]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ExperimentIntegrityError(f"non-finite JSON constant {value}: {path.name}")

    try:
        result = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentIntegrityError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ExperimentIntegrityError(f"expected a JSON object: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def _identity(record: dict[str, Any]) -> tuple[Any, ...]:
    missing = [key for key in ("id", *IDENTITY_FIELDS) if key not in record]
    if missing:
        raise ExperimentIntegrityError(f"missing identity fields {missing} in {record.get('id', '<unknown>')}")
    for key in ("id", "model", "suite", "condition", "method"):
        if not isinstance(record[key], str) or not record[key]:
            raise ExperimentIntegrityError(f"{key} must be a nonempty string")
    for key in ("task_id", "init_state_id", "seed"):
        if isinstance(record[key], bool) or not isinstance(record[key], int) or record[key] < 0:
            raise ExperimentIntegrityError(f"{key} must be a nonnegative integer")
    return tuple(record[key] for key in IDENTITY_FIELDS)


def _validate_complete(record: dict[str, Any], path: Path) -> None:
    if not isinstance(record.get("success"), bool):
        raise ExperimentIntegrityError(f"success must be boolean: {record['id']}")
    if not isinstance(record.get("task"), str) or not record["task"]:
        raise ExperimentIntegrityError(f"task must be a nonempty string: {record['id']}")
    for key in ("steps", "horizon", "inference_calls"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExperimentIntegrityError(f"{key} must be a nonnegative integer: {record['id']}")
    if record["horizon"] < 1 or record["steps"] > record["horizon"]:
        raise ExperimentIntegrityError(f"invalid episode horizon/steps: {record['id']}")
    if record["inference_calls"] > record["steps"]:
        raise ExperimentIntegrityError(f"more model calls than control ticks: {record['id']}")
    for key in TIMING_FIELDS:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ExperimentIntegrityError(f"{key} must be finite and nonnegative: {record['id']}")
    if not isinstance(record.get("model_info"), dict):
        raise ExperimentIntegrityError(f"model_info must be a JSON object: {record['id']}")
    explicit_upstream = "scheduler" in record and record["scheduler"] is None
    if not isinstance(record.get("scheduler_config"), dict) and not (
        explicit_upstream and "scheduler_config" in record and record["scheduler_config"] is None
    ):
        raise ExperimentIntegrityError(f"scheduler_config must be a JSON object or explicit upstream null: {record['id']}")
    if "video" in record and record["video"] is not None:
        if not isinstance(record["video"], str):
            raise ExperimentIntegrityError(f"video must be a relative path: {record['id']}")
        video = (path.parent / record["video"]).resolve()
        if not video.is_relative_to(path.parent.resolve()) or not video.is_file():
            raise ExperimentIntegrityError(f"video is missing or outside its episode folder: {record['id']}")


def validate_initial_parity(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Verify methods in each paired block began from the same observed state.

    Old minimal records may omit both fields. Once one record in a block has a
    field, every method in that block must have it; incomplete evidence cannot
    silently disable that check. Current runners always write both fields.
    Image bytes must match exactly, while proprioception allows float32-scale
    rounding (rtol=1e-6, atol=1e-7). Different models/conditions are separate blocks.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = tuple(row[field] for field in ("model", "condition", "suite", "task_id", "init_state_id", "seed"))
        groups[key].append(row)
    audit: dict[str, Any] = {
        "blocks": len(groups), "image_checked_blocks": 0, "state_checked_blocks": 0,
        "image_method_comparisons": 0, "state_method_comparisons": 0,
        "proprio_rtol": 1e-6, "proprio_atol": 1e-7,
    }
    for key, rows in groups.items():
        for field, prefix in (("initial_image_sha256", "image"), ("initial_robot_state", "state")):
            presence = [field in row for row in rows]
            if not any(presence):
                continue
            if not all(presence):
                raise ExperimentIntegrityError(f"incomplete initial parity field {field} for block {key}")
            values = []
            for row in rows:
                value = row[field]
                if prefix == "image":
                    if (not isinstance(value, str) or len(value) != 64
                            or any(character not in "0123456789abcdefABCDEF" for character in value)):
                        raise ExperimentIntegrityError(f"invalid initial_image_sha256: {row['id']}")
                    values.append(value.lower())
                else:
                    if (not isinstance(value, list) or len(value) != 8 or any(
                        isinstance(component, bool) or not isinstance(component, (int, float))
                        or not np.isfinite(component) for component in value
                    )):
                        raise ExperimentIntegrityError(f"invalid initial_robot_state: {row['id']}")
                    values.append(np.asarray(value, dtype=np.float64))
            for index, value in enumerate(values[1:], start=1):
                matches = value == values[0] if prefix == "image" else np.allclose(
                    value, values[0], rtol=audit["proprio_rtol"], atol=audit["proprio_atol"], equal_nan=False,
                )
                if not matches:
                    raise ExperimentIntegrityError(
                        f"initial {prefix} parity mismatch between {rows[0]['id']} and {rows[index]['id']}"
                    )
            audit[f"{prefix}_checked_blocks"] += 1
            audit[f"{prefix}_method_comparisons"] += len(values) - 1
    audit["blocks_without_image_evidence"] = len(groups) - audit["image_checked_blocks"]
    audit["blocks_without_state_evidence"] = len(groups) - audit["state_checked_blocks"]
    return audit


def load_experiment(output_dir: str | Path, *, partial: bool = False) -> ExperimentData:
    """Read every declared result exactly once; never select the best retry.

    Missing/error episodes are accepted only in explicit partial mode. Duplicate
    identities, unexpected results, malformed data, and identity mismatches are
    always rejected. Source bytes remain untouched and their hashes are kept.
    """
    root = Path(output_dir).resolve()
    protocol, protocol_hash = _read_json(root / "protocol.json")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExperimentIntegrityError("protocol.cases must be a nonempty list")
    expected: dict[str, dict[str, Any]] = {}
    identities: set[tuple[Any, ...]] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ExperimentIntegrityError("every protocol case must be an object")
        identity = _identity(case)
        if case["id"] in expected or identity in identities:
            raise ExperimentIntegrityError(f"duplicate protocol case: {case['id']}")
        expected[case["id"]] = case
        identities.add(identity)
    complete: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    hashes = {"protocol.json": protocol_hash}
    for path in sorted((root / "episodes").glob("*/result.json")):
        record, checksum = _read_json(path)
        identity = _identity(record)
        identifier = record["id"]
        if identifier in seen:
            raise ExperimentIntegrityError(f"duplicate episode result: {identifier}")
        if identifier not in expected:
            raise ExperimentIntegrityError(f"unexpected episode result: {identifier}")
        if identity != _identity(expected[identifier]):
            raise ExperimentIntegrityError(f"result identity differs from protocol: {identifier}")
        if "fingerprint" in protocol and record.get("protocol_fingerprint") != protocol["fingerprint"]:
            raise ExperimentIntegrityError(f"result provenance differs from protocol: {identifier}")
        for key in ("num_steps", "scheduler", "record_video"):
            if key in expected[identifier] and record.get(key) != expected[identifier][key]:
                raise ExperimentIntegrityError(f"result {key} differs from protocol: {identifier}")
        if path.parent.name != identifier:
            raise ExperimentIntegrityError(f"episode folder does not match result id: {identifier}")
        seen.add(identifier)
        hashes[path.relative_to(root).as_posix()] = checksum
        if record.get("status") == "complete":
            _validate_complete(record, path)
            complete.append(record)
        elif record.get("status") == "error":
            errors.append(record)
        else:
            raise ExperimentIntegrityError(f"unknown episode status: {identifier}")
    missing = tuple(sorted(set(expected) - seen))
    validate_initial_parity(complete)
    if not partial and (missing or errors):
        raise ExperimentIntegrityError(
            f"full report requires all planned episodes: {len(missing)} missing, {len(errors)} error; "
            "use partial=True only for an explicitly incomplete report"
        )
    return ExperimentData(root, protocol, tuple(complete), tuple(errors), missing, partial, hashes)


def _task(record: dict[str, Any]) -> tuple[str, int]:
    return record["suite"], record["task_id"]


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(np.mean(data)) if data else 0.0


def aggregate_groups(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group complete episodes, macro-average tasks, and keep raw outcome counts."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "complete":
            raise ExperimentIntegrityError("aggregate_groups accepts complete records only")
        grouped[record["model"], record["condition"], record["method"]].append(record)
    result = []
    for (model, condition, method), episodes in sorted(grouped.items()):
        tasks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for episode in episodes:
            tasks[_task(episode)].append(episode)
        per_task = [{
            "suite": suite, "task_id": task_id, "task": rows[0]["task"],
            "episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": _mean(float(row["success"]) for row in rows),
        } for (suite, task_id), rows in sorted(tasks.items())]
        successes = sum(row["success"] for row in episodes)
        total_steps = sum(row["steps"] for row in episodes)
        result.append({
            "model": model, "condition": condition, "method": method,
            "tasks": len(tasks), "episodes": len(episodes), "successes": successes,
            "failures": len(episodes) - successes,
            "macro_task_success": _mean(row["success_rate"] for row in per_task),
            "episode_success": successes / len(episodes),
            "mean_inference_calls": _mean(row["inference_calls"] for row in episodes),
            "mean_steps": _mean(row["steps"] for row in episodes),
            "inference_calls_per_100_steps": (
                100 * sum(row["inference_calls"] for row in episodes) / total_steps if total_steps else None
            ),
            "mean_model_seconds": _mean(row["policy_inference_seconds"] for row in episodes),
            "total_wall_seconds": sum(row["wall_seconds"] for row in episodes),
            "mean_timings": {field: _mean(row[field] for row in episodes) for field in TIMING_FIELDS},
            "per_task": per_task,
        })
    return result


def task_cluster_interval(
    task_differences: Iterable[float], *, resamples: int = 2000, seed: int = 20261014,
) -> tuple[float, float] | None:
    """Percentile bootstrap of equal-weight task effects, conditional on episodes.

    One task cannot estimate between-task uncertainty, so its interval is None.
    A zero-width interval across identical observed task effects is descriptive;
    it does not prove the true uncertainty is zero.
    """
    values = np.asarray(list(task_differences), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
        raise ValueError("task differences must be finite rates in [-1,1]")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _is_fixed8(method: str) -> bool:
    return method.lower().replace("_", "").replace("-", "") == "fixed8"


def paired_comparisons(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare every available method against fixed8 on matching trial identities."""
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("status") != "complete":
            raise ExperimentIntegrityError("paired_comparisons accepts complete records only")
        groups[record["model"], record["condition"]][record["method"]].append(record)

    def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["suite"], row["task_id"], row["init_state_id"], row["seed"]

    comparisons = []
    for (model, condition), methods in sorted(groups.items()):
        baselines = [name for name in methods if _is_fixed8(name)]
        if not baselines:
            continue
        if len(baselines) > 1:
            raise ExperimentIntegrityError(f"ambiguous fixed8 baseline for {model}/{condition}")
        baseline_name = baselines[0]
        baseline = {pair_key(row): row for row in methods[baseline_name]}
        if len(baseline) != len(methods[baseline_name]):
            raise ExperimentIntegrityError("duplicate baseline pairing identity")
        for method, rows in sorted(methods.items()):
            if method == baseline_name:
                continue
            candidate = {pair_key(row): row for row in rows}
            if len(candidate) != len(rows):
                raise ExperimentIntegrityError("duplicate candidate pairing identity")
            matched = sorted(baseline.keys() & candidate.keys())
            task_pairs: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for key in matched:
                task_pairs[key[:2]].append((candidate[key], baseline[key]))
            per_task = [{
                "suite": suite, "task_id": task_id, "pairs": len(pairs),
                "candidate_success": _mean(float(c["success"]) for c, _ in pairs),
                "baseline_success": _mean(float(b["success"]) for _, b in pairs),
                "difference": _mean(float(c["success"]) - float(b["success"]) for c, b in pairs),
            } for (suite, task_id), pairs in sorted(task_pairs.items())]
            differences = [row["difference"] for row in per_task]
            interval = task_cluster_interval(differences)
            base_calls = _mean(baseline[key]["inference_calls"] for key in matched)
            candidate_calls = _mean(candidate[key]["inference_calls"] for key in matched)
            wins = sum(candidate[key]["success"] and not baseline[key]["success"] for key in matched)
            losses = sum(baseline[key]["success"] and not candidate[key]["success"] for key in matched)
            comparisons.append({
                "model": model, "condition": condition, "method": method, "baseline": baseline_name,
                "paired_tasks": len(per_task), "paired_episodes": len(matched),
                "unmatched_candidate": len(candidate) - len(matched),
                "unmatched_baseline": len(baseline) - len(matched),
                "candidate_macro_success": _mean(row["candidate_success"] for row in per_task) if matched else None,
                "baseline_macro_success": _mean(row["baseline_success"] for row in per_task) if matched else None,
                "macro_success_difference": _mean(differences) if matched else None,
                "task_cluster_95ci": list(interval) if interval is not None else None,
                "bootstrap_resamples": 2000, "bootstrap_seed": 20261014,
                "candidate_wins": wins, "baseline_wins": losses, "ties": len(matched) - wins - losses,
                "candidate_mean_calls": candidate_calls if matched else None,
                "baseline_mean_calls": base_calls if matched else None,
                "inference_call_reduction": 1 - candidate_calls / base_calls if base_calls else None,
                "mean_model_seconds_difference": _mean(
                    candidate[key]["policy_inference_seconds"] - baseline[key]["policy_inference_seconds"]
                    for key in matched
                ) if matched else None,
                "small_task_count": len(per_task) < 10,
                "per_task": per_task,
            })
    return comparisons


def _validate_controlled_pair(base: dict[str, Any], changed: dict[str, Any]) -> None:
    """Require observed reset parity and unchanged model/environment evidence."""
    for field in ("model", "suite", "task_id", "init_state_id", "seed", "task", "horizon", "num_steps", "environment"):
        if base.get(field) != changed.get(field):
            raise ExperimentIntegrityError(f"controlled comparison changed {field}")
    for row in (base, changed):
        if "initial_image_sha256" not in row or "initial_robot_state" not in row:
            raise ExperimentIntegrityError("controlled comparison requires initial image and state parity evidence")
        if not row["model_info"].get("model_sha256") or not row["model_info"].get("revision"):
            raise ExperimentIntegrityError("controlled comparison requires verified model weights and revision")
    ignore = {"load_seconds"}
    if ({key: value for key, value in base["model_info"].items() if key not in ignore}
            != {key: value for key, value in changed["model_info"].items() if key not in ignore}):
        raise ExperimentIntegrityError("controlled comparison changed weights or model configuration")
    # For clean/delay comparisons the condition differs by design. Validate the
    # physical reset evidence in a shared block without modifying raw records.
    validate_initial_parity([base, {**changed, "condition": base["condition"]}])


def _controlled_summary(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *,
    model: str, condition: str, method: str, baseline: str, partial: bool,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Summarize a predeclared contrast; full corpora cannot discard unmatched trials."""
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in ("suite", "task_id", "init_state_id", "seed"))

    left = {key(row): row for row in baseline_rows}
    right = {key(row): row for row in candidate_rows}
    if len(left) != len(baseline_rows) or len(right) != len(candidate_rows):
        raise ExperimentIntegrityError("duplicate controlled pairing identity")
    weights = {(row["model_info"].get("model_sha256"), row["model_info"].get("revision"))
               for row in [*baseline_rows, *candidate_rows]}
    if len(weights) > 1:
        raise ExperimentIntegrityError("controlled comparison changed weights across paired tasks")
    if not partial and (not left or left.keys() != right.keys()):
        raise ExperimentIntegrityError("controlled comparison requires fully matched pairs")
    pairs = [(left[identity], right[identity]) for identity in sorted(left.keys() & right.keys())]
    tasks: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for base, changed in pairs:
        _validate_controlled_pair(base, changed)
        tasks[_task(base)].append((base, changed))
    per_task = [{
        "suite": suite, "task_id": task_id, "pairs": len(values),
        "baseline_success": _mean(float(base["success"]) for base, _ in values),
        "candidate_success": _mean(float(changed["success"]) for _, changed in values),
        "difference": _mean(float(changed["success"]) - float(base["success"]) for base, changed in values),
        "mean_calls_difference": _mean(changed["inference_calls"] - base["inference_calls"] for base, changed in values),
    } for (suite, task_id), values in sorted(tasks.items())]
    interval = task_cluster_interval(row["difference"] for row in per_task)
    result = {
        "model": model, "condition": condition, "method": method, "baseline": baseline,
        "paired_tasks": len(tasks), "paired_episodes": len(pairs),
        "unmatched_baseline": len(left) - len(pairs), "unmatched_candidate": len(right) - len(pairs),
        "macro_success_difference": _mean(row["difference"] for row in per_task) if pairs else None,
        "task_cluster_95ci": list(interval) if interval is not None else None,
        "candidate_mean_calls": _mean(changed["inference_calls"] for _, changed in pairs) if pairs else None,
        "baseline_mean_calls": _mean(base["inference_calls"] for base, _ in pairs) if pairs else None,
        "mean_calls_difference": _mean(changed["inference_calls"] - base["inference_calls"]
                                       for base, changed in pairs) if pairs else None,
        "macro_calls_difference": _mean(row["mean_calls_difference"] for row in per_task) if pairs else None,
        "mean_model_seconds_difference": _mean(changed["policy_inference_seconds"] - base["policy_inference_seconds"]
                                               for base, changed in pairs) if pairs else None,
        "candidate_wins": sum(changed["success"] and not base["success"] for base, changed in pairs),
        "baseline_wins": sum(base["success"] and not changed["success"] for base, changed in pairs),
        "ties": sum(base["success"] == changed["success"] for base, changed in pairs),
        "bootstrap_resamples": 2000, "bootstrap_seed": 20261014,
        "small_task_count": len(tasks) < 10, "per_task": per_task,
        "per_pair": [{
            "suite": base["suite"], "task_id": base["task_id"], "task": base["task"],
            "init_state_id": base["init_state_id"], "seed": base["seed"],
            "baseline_id": base["id"], "candidate_id": changed["id"],
            "baseline_success": base["success"], "candidate_success": changed["success"],
            "success_difference": int(changed["success"]) - int(base["success"]),
            "baseline_calls": base["inference_calls"], "candidate_calls": changed["inference_calls"],
            "calls_difference": changed["inference_calls"] - base["inference_calls"],
        } for base, changed in pairs],
    }
    return result, pairs


def ablation_comparisons(records: Iterable[dict[str, Any]], *, partial: bool = False) -> list[dict[str, Any]]:
    """Direct no_vision/no_gripper/no_ensemble minus full adaptive contrasts.

    Each variant must change exactly its named enabled component to False.
    Model, reset, task, environment, and all remaining scheduler fields agree.
    """
    variants = {"no_vision": "use_vision", "no_gripper": "use_gripper", "no_ensemble": "temporal_ensemble"}
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("status") != "complete":
            raise ExperimentIntegrityError("ablation_comparisons accepts complete records only")
        if row["method"] in {"adaptive", *variants}:
            groups[row["model"], row["condition"]][row["method"]].append(row)
    results = []
    for (model, condition), methods in sorted(groups.items()):
        for method, component in variants.items():
            if method not in methods:
                continue
            result, pairs = _controlled_summary(methods.get("adaptive", []), methods[method], model=model,
                                                condition=condition, method=method, baseline="adaptive", partial=partial)
            for base, changed in pairs:
                config = base["scheduler_config"]
                altered = changed["scheduler_config"]
                if (not isinstance(config, dict) or config.get("mode") != "adaptive"
                        or config.get(component) is not True or not isinstance(altered, dict)
                        or altered.get(component) is not False
                        or {**config, component: False} != altered):
                    raise ExperimentIntegrityError(f"ablation {method} must change only enabled {component} to False")
            result["changed_component"] = component
            result["difference_direction"] = "variant minus full adaptive"
            results.append(result)
    return results


def compare_observation_delay(baseline: ExperimentData, delayed: ExperimentData) -> dict[str, Any]:
    """Match delay4 to clean for fixed8/adaptive, without conflating method effects."""
    if any(data.partial or data.errors or data.missing_ids for data in (baseline, delayed)):
        raise ExperimentIntegrityError("delay comparison requires two complete stage corpora")
    for field in ("sources", "runtime_versions", "artifacts"):
        if baseline.protocol.get(field) != delayed.protocol.get(field):
            raise ExperimentIntegrityError(f"delay stages differ in frozen {field}")
    expected_methods = {"fixed8", "adaptive"}
    changed = list(delayed.records)
    if not changed or any(row["condition"] != "delay4" or row["method"] not in expected_methods for row in changed):
        raise ExperimentIntegrityError("delay stage must contain only delay4 fixed8/adaptive")
    results = []
    for model in sorted({row["model"] for row in changed}):
        model_keys = []
        for method in sorted(expected_methods):
            base_rows = [row for row in baseline.records
                         if row["model"] == model and row["method"] == method and row["condition"] == "clean"]
            delay_rows = [row for row in changed if row["model"] == model and row["method"] == method]
            result, pairs = _controlled_summary(base_rows, delay_rows, model=model, condition="delay4 minus clean",
                                                method=method, baseline=method, partial=False)
            for base, altered in pairs:
                if base["scheduler_config"] != altered["scheduler_config"]:
                    raise ExperimentIntegrityError("delay comparison changed scheduler configuration")
            model_keys.append({(row["suite"], row["task_id"], row["init_state_id"], row["seed"]) for row in delay_rows})
            result["difference_direction"] = "delay4 minus clean"
            results.append(result)
        if model_keys[0] != model_keys[1]:
            raise ExperimentIntegrityError("delay methods must evaluate the same declared trials")
    return {
        "comparison": "Controlled observation delay: delay4 minus clean, within each method",
        "comparisons": results, "baseline_input_sha256": baseline.input_sha256,
        "delayed_input_sha256": delayed.input_sha256,
        "warning": "受控观测延迟的配对描述；小任务样本及多重比较限制结论，不代表真实实时延迟或稳健性保证。",
        "uncertainty": "2000 task-cluster percentile bootstrap draws; fixed observed trials and model checkpoints",
    }


def summarize_experiment(data: ExperimentData) -> dict[str, Any]:
    records = list(data.records)
    warnings = []
    task_count = len({_task(row) for row in records})
    if data.partial:
        warnings.append("部分报告仅统计已完成试验；缺失与错误试验没有补值，方法间可能不可直接比较。")
    if task_count < 10:
        warnings.append("任务簇少于 10 个，跨任务效应和置信区间不稳定；不能据此声称普遍优于基线。")
    comparisons = paired_comparisons(records)
    ablations = ablation_comparisons(records, partial=data.partial)
    if ablations and task_count >= 10 and any(row["small_task_count"] for row in ablations):
        warnings.append("部分组件消融少于 10 个配对任务，区间不稳定；不能借报告中的其他任务扩大消融样本量。")
    if any(row["unmatched_candidate"] or row["unmatched_baseline"] for row in comparisons):
        warnings.append("部分方法没有完整配对；对照表仅使用共同试验，并明确列出未配对数量。")
    if any(row["task_cluster_95ci"] is None for row in comparisons):
        warnings.append("少于两个配对任务时不计算任务簇置信区间；单任务结果不能估计跨任务不确定性。")
    return {
        "schema_version": 1, "partial": data.partial,
        "planned_episodes": len(data.protocol["cases"]), "completed_episodes": len(records),
        "error_episodes": len(data.errors), "missing_episodes": len(data.missing_ids),
        "missing_ids": list(data.missing_ids), "error_ids": [row["id"] for row in data.errors],
        "models": sorted({row["model"] for row in records}), "tasks": task_count,
        "successes": sum(row["success"] for row in records),
        "total_wall_seconds": sum(row["wall_seconds"] for row in records),
        "groups": aggregate_groups(records), "paired_comparisons": comparisons, "ablation_comparisons": ablations,
        "warnings": warnings, "input_sha256": data.input_sha256.copy(),
        "initial_state_parity": validate_initial_parity(records),
        "estimand": "equal-weight (suite,task_id) means over observed init-state/seed trials",
        "uncertainty": "2000 task-cluster percentile bootstrap draws; fixed observed trials and model checkpoints",
    }


def compare_flow_steps(baseline: ExperimentData, candidate: ExperimentData, *, expected_pairs: int = 4) -> dict[str, Any]:
    """Strict cross-stage SmolVLA fixed15 comparison: 5 versus 10 flow steps.

    Baseline's other methods remain in its own report, but are not included in
    this predeclared comparison. No favorable trial matching or dropping occurs.
    """
    if baseline.partial or candidate.partial or baseline.errors or candidate.errors:
        raise ExperimentIntegrityError("flow comparison requires two complete stage corpora")
    if baseline.missing_ids or candidate.missing_ids:
        raise ExperimentIntegrityError("flow comparison requires every planned stage episode")
    base_rows = [row for row in baseline.records if row["model"] == "smolvla" and row["method"] == "fixed15"]
    candidate_rows = list(candidate.records)
    if any(row["model"] != "smolvla" or row["method"] != "fixed15_steps5" for row in candidate_rows):
        raise ExperimentIntegrityError("flow candidate stage must contain only SmolVLA fixed15_steps5")
    weight_identities = {(row["model_info"].get("model_sha256"), row["model_info"].get("revision"))
                         for row in [*base_rows, *candidate_rows]}
    if len(weight_identities) != 1:
        raise ExperimentIntegrityError("flow comparison changed base weights across paired tasks")
    for field in ("sources", "runtime_versions", "artifacts"):
        if baseline.protocol.get(field) != candidate.protocol.get(field):
            raise ExperimentIntegrityError(f"flow stages differ in frozen {field}")

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in ("suite", "task_id", "init_state_id", "seed", "condition"))

    base_map = {key(row): row for row in base_rows}
    candidate_map = {key(row): row for row in candidate_rows}
    if len(base_map) != len(base_rows) or len(candidate_map) != len(candidate_rows):
        raise ExperimentIntegrityError("duplicate flow pairing identity")
    if base_map.keys() != candidate_map.keys() or len(base_map) != expected_pairs:
        raise ExperimentIntegrityError(f"flow comparison requires exactly {expected_pairs} fully matched pairs")
    per_pair = []
    per_task: dict[tuple[str, int], list[float]] = defaultdict(list)
    for identity in sorted(base_map):
        base, changed = base_map[identity], candidate_map[identity]
        if base.get("num_steps") != 10 or changed.get("num_steps") != 5:
            raise ExperimentIntegrityError("flow comparison requires declared num_steps=10 versus 5")
        if base.get("scheduler_config") != changed.get("scheduler_config"):
            raise ExperimentIntegrityError("flow comparison changed scheduler configuration")
        config = base.get("scheduler_config") or {}
        if config.get("mode") != "fixed" or config.get("fixed_interval") != 15 or config.get("temporal_ensemble") is not False:
            raise ExperimentIntegrityError("flow comparison requires fixed15 without temporal ensembling")
        if base["horizon"] != changed["horizon"] or base["task"] != changed["task"]:
            raise ExperimentIntegrityError("flow comparison changed task or horizon")
        if base.get("environment") != changed.get("environment"):
            raise ExperimentIntegrityError("flow comparison changed environment metadata")
        for row, steps in ((base, 10), (changed, 5)):
            info = row["model_info"]
            if (not info.get("model_sha256") or not info.get("revision") or info.get("num_steps") != steps):
                raise ExperimentIntegrityError("flow model metadata lacks verified weights/revision/step count")
            if "initial_image_sha256" not in row or "initial_robot_state" not in row:
                raise ExperimentIntegrityError("flow comparison requires initial image and state parity evidence")
        ignore = {"num_steps", "load_seconds"}
        base_info = {name: value for name, value in base["model_info"].items() if name not in ignore}
        changed_info = {name: value for name, value in changed["model_info"].items() if name not in ignore}
        if base_info != changed_info:
            raise ExperimentIntegrityError("flow comparison changed base weights or model configuration")
        validate_initial_parity([base, changed])
        difference = float(changed["success"]) - float(base["success"])
        per_task[_task(base)].append(difference)
        per_pair.append({
            "suite": base["suite"], "task_id": base["task_id"], "task": base["task"],
            "init_state_id": base["init_state_id"], "seed": base["seed"], "condition": base["condition"],
            "baseline_id": base["id"], "candidate_id": changed["id"],
            "success_10": base["success"], "success_5": changed["success"], "success_difference": difference,
            "calls_10": base["inference_calls"], "calls_5": changed["inference_calls"],
            "steps_10": base["steps"], "steps_5": changed["steps"],
            "model_seconds_10": base["policy_inference_seconds"],
            "model_seconds_5": changed["policy_inference_seconds"],
            "model_seconds_difference": changed["policy_inference_seconds"] - base["policy_inference_seconds"],
        })
    differences = [_mean(values) for _, values in sorted(per_task.items())]
    interval = task_cluster_interval(differences)
    return {
        "comparison": "SmolVLA fixed15 without ensemble: 5 minus 10 flow integration steps",
        "paired_episodes": len(per_pair), "paired_tasks": len(per_task), "per_pair": per_pair,
        "macro_success_difference": _mean(differences),
        "task_cluster_95ci": list(interval) if interval is not None else None,
        "bootstrap_resamples": 2000, "bootstrap_seed": 20261014,
        "mean_model_seconds_difference": _mean(row["model_seconds_difference"] for row in per_pair),
        "small_task_count": len(per_task) < 10,
        "warning": "仅少量任务的配对试验；区间不稳定，不支持普遍优劣、非劣效或 SOTA 结论。",
        "initial_state_parity": validate_initial_parity([*base_rows, *candidate_rows]),
        "baseline_input_sha256": baseline.input_sha256,
        "candidate_input_sha256": candidate.input_sha256,
    }
