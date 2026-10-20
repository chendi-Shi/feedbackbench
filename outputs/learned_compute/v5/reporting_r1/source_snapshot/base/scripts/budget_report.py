"""Read sealed campaign evidence and report task-group-weighted results.

No model dependencies or unpickling. Partial campaigns get descriptive values,
never full-study confidence intervals. Receipt verification precedes statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import math
from pathlib import Path
import random
import statistics

from budget_campaign_common import (case_directory, digest, fingerprint, owned_path,
                                    read_json, utc_now, verify_episode, write_json)


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261016
METRICS = ("success", "nfe_per_decision", "latency_seconds_per_decision",
           "controller_seconds_per_decision", "episode_wall_seconds")


def _number(value, name, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _validate_result(result):
    if result.get("complete") is not True or any(type(result.get(name)) is not bool for name in ("terminated", "truncated")):
        raise ValueError("completion and terminal flags must be explicit Booleans")
    if type(result.get("success")) is not bool:
        raise ValueError("success must be a Boolean; completed failures remain in the denominator")
    for name in ("steps", "decisions", "actual_nfe"):
        _number(result.get(name), name, integer=True, positive=True)
    for name in ("decision_wall_seconds", "episode_wall_seconds", "controller_overhead_seconds"):
        _number(result.get(name), name)
    frequencies = result.get("budget_frequencies")
    if not isinstance(frequencies, dict) or set(frequencies) != {"2", "4", "10"}:
        raise ValueError("budget_frequencies must contain 2/4/10")
    for name, value in frequencies.items():
        _number(value, f"budget_frequency_{name}", integer=True)
    if sum(frequencies.values()) != result["decisions"]:
        raise ValueError("budget frequency counts differ from decision count")


def _metrics(result):
    decisions = result["decisions"]
    return {"success": float(result["success"]), "nfe_per_decision": result["actual_nfe"] / decisions,
            "latency_seconds_per_decision": result["decision_wall_seconds"] / decisions,
            "controller_seconds_per_decision": result["controller_overhead_seconds"] / decisions,
            "episode_wall_seconds": result["episode_wall_seconds"]}


def _validate_trained_binding(campaign, method, binding):
    manifest_path = campaign / "trained_manifest.json"
    if (not isinstance(binding, dict)
            or not {"manifest_sha256", "selector", "calibration_budget_probabilities"} <= binding.keys()
            or binding.get("manifest_sha256") != digest(manifest_path)):
        raise ValueError("evaluation trained manifest SHA-256 binding mismatch")
    manifest = read_json(manifest_path)
    selector = binding.get("selector")
    probabilities = binding.get("calibration_budget_probabilities")
    if method["family"] == "fixed":
        if selector is not None or probabilities is not None:
            raise ValueError("fixed baseline must bind the trained manifest without a selector or random probabilities")
        return
    if method["family"] not in {"learned", "random_calibrated"}:
        raise ValueError("unknown planned evaluation method family")
    variant = method["feature_mode"] if method["family"] == "learned" else "full"
    relative = f"trained/{variant}_seed{method['seed']}/selector.pt"
    expected_hash = manifest["files"][relative]["sha256"]
    if selector != {"path": relative, "sha256": expected_hash} or digest(owned_path(campaign, relative)) != expected_hash:
        raise ValueError("evaluation selector path/hash differs from the planned method and frozen trained manifest")
    if method["family"] == "random_calibrated":
        if not isinstance(probabilities, list) or len(probabilities) != 3:
            raise ValueError("random calibrated baseline requires three calibration probabilities")
        for value in probabilities:
            _number(value, "calibration_budget_probability")
        if any(value > 1 for value in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-9):
            raise ValueError("calibration probabilities must sum to one")
    elif probabilities is not None:
        raise ValueError("learned controller must not bind random-baseline probabilities")


def _group_values(rows):
    # Equal initial states within each official task, then equal tasks within
    # each exact-match group, then equal groups. No control-step pseudoreplication.
    tasks = defaultdict(list)
    task_groups = {}
    for row in rows:
        task = row["case"]["task_uid"]
        tasks[task].append(row["metrics"])
        task_groups[task] = row["case"]["duplicate_group"]
    groups = defaultdict(list)
    for task, values in tasks.items():
        groups[task_groups[task]].append({name: statistics.fmean(value[name] for value in values) for name in METRICS})
    return {group: {name: statistics.fmean(value[name] for value in values) for name in METRICS}
            for group, values in sorted(groups.items())}


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] * (high - position) + sorted_values[high] * (position - low) if low != high else sorted_values[low]


def _statistics(rows, allow_ci):
    groups = _group_values(rows)
    point = {name: statistics.fmean(value[name] for value in groups.values()) if groups else None for name in METRICS}
    intervals = None
    if allow_ci and len(groups) >= 2:
        values = list(groups.values())
        rng = random.Random(BOOTSTRAP_SEED)
        bootstrap = {name: [] for name in METRICS}
        for _ in range(BOOTSTRAP_SAMPLES):
            selected = [values[rng.randrange(len(values))] for _ in values]
            for name in METRICS:
                bootstrap[name].append(statistics.fmean(value[name] for value in selected))
        intervals = {name: [_quantile(sorted(samples), .025), _quantile(sorted(samples), .975)]
                     for name, samples in bootstrap.items()}
    return {"point_estimates": point, "full_study_ci95": intervals, "observed_task_groups": len(groups),
            "observed_tasks": len({row["case"]["task_uid"] for row in rows}), "group_values": groups,
            "scope": "complete_planned_test_set" if allow_ci else "observed_completed_subset_descriptive_only",
            "ci_unavailable_reason": None if intervals else "incomplete_study_or_fewer_than_two_task_groups"}


def _training_progress(campaign, protocol):
    expected = {(mode, seed) for mode in protocol["training"]["feature_modes"] for seed in protocol["training"]["seeds"]}
    result = {"planned_models": len(expected), "reported_models": 0, "verified_models": 0,
              "summary_status": "missing", "frozen_manifest_valid": False, "complete": False, "issues": []}
    summary_path = campaign / "trained/summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            result["summary_status"] = summary.get("status", "unknown")
            models = summary.get("models", [])
            result["reported_models"] = len(models)
            identities = set()
            for model in models:
                identity = (model["variant"], model["seed"])
                if identity not in expected or identity in identities:
                    raise ValueError("unknown or repeated trained model identity")
                identities.add(identity)
                path = owned_path(campaign, "trained", model["checkpoint"])
                if digest(path) != model["checkpoint_sha256"]:
                    raise ValueError(f"trained checkpoint digest mismatch: {model['checkpoint']}")
                result["verified_models"] += 1
            if summary.get("binding", {}).get("campaign_fingerprint") != protocol["campaign_fingerprint"]:
                raise ValueError("training summary campaign binding mismatch")
        except (KeyError, ValueError, OSError, TypeError) as error:
            result["issues"].append(str(error))
    manifest_path = campaign / "trained_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            if (manifest.get("schema") != "feedbackbench.budget_trained_manifest.v1"
                    or manifest.get("campaign_fingerprint") != protocol["campaign_fingerprint"]
                    or manifest.get("controller_count") != len(expected)
                    or manifest.get("test_opened_after_freeze") is not True):
                raise ValueError("trained manifest schema/campaign/count/freeze boundary mismatch")
            if not manifest.get("files"):
                raise ValueError("empty trained manifest")
            inventory = {path.relative_to(campaign).as_posix() for path in (campaign / "trained").rglob("*") if path.is_file()}
            if inventory != manifest["files"].keys():
                raise ValueError("trained manifest inventory mismatch")
            for relative, record in manifest["files"].items():
                path = owned_path(campaign, relative)
                if digest(path) != record["sha256"]:
                    raise ValueError(f"trained manifest digest mismatch: {relative}")
                if "bytes" in record and path.stat().st_size != record["bytes"]:
                    raise ValueError(f"trained manifest size mismatch: {relative}")
            result["frozen_manifest_valid"] = True
        except (KeyError, ValueError, OSError, TypeError) as error:
            result["issues"].append(str(error))
    result["complete"] = (result["summary_status"] == "complete" and result["verified_models"] == len(expected)
                          and result["frozen_manifest_valid"] and not result["issues"])
    return result


def _attempts(campaign, protocol):
    records = []
    for phase in ("collection", "evaluation"):
        for case in protocol[phase]["cases"]:
            path = campaign / "attempts" / phase / case_directory(case)
            if not path.is_dir():
                continue
            for directory in sorted(path.iterdir()):
                if not directory.is_dir():
                    continue
                record = {"phase": phase, "case_id": case["case_id"],
                          "path": directory.relative_to(campaign).as_posix(),
                          "sealed_unpublished": (directory / "receipt.json").exists(), "failures": []}
                audit = campaign / "runs" / f"{phase}-{case_directory(case)}-{directory.name}.json"
                for filename, file in (("failure.json", directory / "failure.json"),
                                       ("supervisor.json", directory / "supervisor.json"), ("supervisor_run", audit)):
                    if file.exists():
                        try:
                            evidence = read_json(file)
                            record[filename] = {key: evidence[key] for key in
                                                ("error_type", "error", "reason", "stop_reason", "exit_code", "status", "stage", "steps", "decisions", "at")
                                                if key in evidence}
                            if (filename == "failure.json" or evidence.get("stop_reason") or evidence.get("reason")
                                    or evidence.get("exit_code", 0) != 0 or evidence.get("status") in {"failed", "error"}):
                                record["failures"].append(filename)
                        except (ValueError, OSError, TypeError) as error:
                            record[filename] = {"read_error": str(error)}
                text = str([record.get(name, {}) for name in ("failure.json", "supervisor.json", "supervisor_run")]).casefold()
                record["resource_or_deadline_related"] = any(term in text for term in
                                                               ("resourcewait", "memory", "rss", "deadline", "resource stop", "resource_guard"))
                record["status"] = "sealed_unpublished" if record["sealed_unpublished"] else (
                    "failed_unsealed" if record["failures"] else "unsealed_status_unknown")
                records.append(record)
    return {"attempt_directories": len(records), "failed_unsealed_attempts": sum(row["status"] == "failed_unsealed" for row in records),
            "resource_or_deadline_attempts": sum(row["resource_or_deadline_related"] for row in records),
            "sealed_unpublished_attempts": sum(row["sealed_unpublished"] for row in records),
            "records": records, "included_in_success_denominator": False}


def generate_report(campaign: str | Path) -> dict:
    """Verify completed episodes and return a report without writing anything."""
    campaign = Path(campaign).resolve()
    protocol = read_json(campaign / "protocol.json")
    if protocol.get("campaign_fingerprint") != fingerprint(protocol):
        raise ValueError("campaign protocol fingerprint changed")
    methods = protocol["evaluation"]["methods"]
    method_map = {method["method_id"]: method for method in methods}
    if len(method_map) != len(methods):
        raise ValueError("duplicate evaluation method IDs")
    verified = {"collection": [], "evaluation": []}
    progress = {}
    for phase in verified:
        planned = protocol[phase]["cases"]
        if len({case["case_id"] for case in planned}) != len(planned):
            raise ValueError(f"duplicate planned {phase} case IDs")
        if len({case_directory(case) for case in planned}) != len(planned):
            raise ValueError(f"case directory hash collision in {phase}")
        missing, invalid = [], []
        for case in planned:
            directory = campaign / phase / case_directory(case)
            if not directory.exists():
                missing.append(case["case_id"])
                continue
            try:
                metadata, result = verify_episode(directory, case, protocol)
                if metadata.get("case") != case or metadata.get("phase") != phase or metadata.get("split") != case["split"]:
                    raise ValueError("episode case/phase/split metadata differs from protocol")
                if metadata.get("method") != (method_map[case["method_id"]] if phase == "evaluation" else None):
                    raise ValueError("episode method metadata differs from protocol")
                if phase == "evaluation":
                    _validate_trained_binding(campaign, method_map[case["method_id"]], metadata.get("trained_binding"))
                _validate_result(result)
                verified[phase].append({"case": case, "result": result, "metrics": _metrics(result),
                                        "receipt_sha256": digest(directory / "receipt.json")})
            except (KeyError, ValueError, OSError, TypeError) as error:
                invalid.append({"case_id": case["case_id"], "error": str(error)})
        expected = {case_directory(case) for case in planned}
        directory = campaign / phase
        unexpected = [path.name for path in sorted(directory.iterdir()) if path.is_dir() and path.name not in expected] if directory.is_dir() else []
        progress[phase] = {"planned": len(planned), "verified_complete": len(verified[phase]),
                           "completed_successes": sum(row["result"]["success"] for row in verified[phase]),
                           "completed_task_failures": sum(not row["result"]["success"] for row in verified[phase]),
                           "missing": len(missing), "missing_case_ids": missing, "invalid": invalid,
                           "unexpected_directories": unexpected,
                           "complete": not missing and not invalid and not unexpected}
    training = _training_progress(campaign, protocol)
    complete = all(row["complete"] for row in progress.values()) and training["complete"]
    method_results = []
    for method in methods:
        rows = [row for row in verified["evaluation"] if row["case"]["method_id"] == method["method_id"]]
        planned = [case for case in protocol["evaluation"]["cases"] if case["method_id"] == method["method_id"]]
        frequencies = Counter()
        for row in rows:
            frequencies.update(row["result"]["budget_frequencies"])
        method_results.append({"method": method, "planned_episodes": len(planned), "verified_episodes": len(rows),
                               "completed_successes": sum(row["result"]["success"] for row in rows),
                               "completed_task_failures": sum(not row["result"]["success"] for row in rows),
                               "complete": len(rows) == len(planned),
                               "total_actual_nfe": sum(row["result"]["actual_nfe"] for row in rows),
                               "total_decisions": sum(row["result"]["decisions"] for row in rows),
                               "budget_frequency_counts": {str(k): frequencies[str(k)] for k in (2, 4, 10)},
                               **_statistics(rows, complete)})
    paired = []
    indexed = {method["method_id"]: {(row["case"]["task_uid"], row["case"]["initial_state_id"]): row
                                     for row in verified["evaluation"] if row["case"]["method_id"] == method["method_id"]}
               for method in methods}
    for left, right in combinations(method_map, 2):
        common = sorted(indexed[left].keys() & indexed[right].keys())
        differences = [{"case": indexed[left][key]["case"],
                        "metrics": {name: indexed[left][key]["metrics"][name] - indexed[right][key]["metrics"][name]
                                    for name in METRICS}} for key in common]
        paired.append({"left": left, "right": right, "difference": "left_minus_right", "paired_episodes": len(common),
                       "pairs": [{"task_uid": task, "initial_state_id": initial} for task, initial in common],
                       "both_failed_pairs_included": sum(not indexed[left][key]["result"]["success"] and
                                                         not indexed[right][key]["result"]["success"] for key in common),
                       **_statistics(differences, complete)})
    return {"schema": "feedbackbench.learned_compute.report.v1", "status": "complete" if complete else "partial",
            "generated_at": utc_now(), "campaign_fingerprint": protocol["campaign_fingerprint"],
            "report_source_sha256": digest(Path(__file__)), "progress": progress, "training": training,
            "attempts": _attempts(campaign, protocol), "methods": method_results, "paired_differences": paired,
            "statistical_contract": {"weighting": "equal_group_then_equal_task_then_equal_initial_state",
                                     "bootstrap_unit": "held_out_exact_match_task_group",
                                     "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                                     "interval": "percentile_95_percent", "multiple_comparison_adjustment": None,
                                     "paired_resampling": "same_group_same_task_same_initial_state_across_methods",
                                     "training_seed_selection": "none_all_prespecified_methods_reported",
                                     "partial_study_full_study_confidence_intervals": False,
                                     "causal_same_observation_claim_for_diverged_rollouts": False},
            "claims": {"full_planned_study_complete": complete, "performance_improvement_claim": None,
                       "resource_attempts_are_task_failures": False,
                       "reported_intervals_are_simultaneous_familywise_intervals": False},
            "evidence": {phase: [{"case_id": row["case"]["case_id"], "receipt_sha256": row["receipt_sha256"]}
                                  for row in rows] for phase, rows in verified.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = generate_report(args.campaign)
    write_json(args.output or args.campaign / "report.json", report)
    print(f"{report['status']}: {report['progress']['evaluation']['verified_complete']} / "
          f"{report['progress']['evaluation']['planned']} evaluation episodes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
