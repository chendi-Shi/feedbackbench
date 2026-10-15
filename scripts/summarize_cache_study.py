"""Strict, read-only evidence analysis for the offline ActionCache CPU pilot.

Only report.json/report.md/CSV, figures and their exports.json seal are written.
No model or simulator is imported. Incomplete studies require --partial and retain planned denominators.
The report's own source hash is recorded separately from frozen run sources.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np

from embodied_research.experiment import digest_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]
RESAMPLES, BOOTSTRAP_SEED = 2000, 20261014
EXPORT_SCHEMA = "feedbackbench.cache_study_exports.v1"
EXPORT_FILES = (
    "report.json", "report.md", "groups.csv", "episodes.csv", "pairs.csv", "calls.csv",
    "figures/cache_primary_success_nfe.png", "figures/cache_primary_success_nfe.svg",
    "figures/cache_ablation_success_nfe.png", "figures/cache_ablation_success_nfe.svg",
    "figures/cache_latency_components.png", "figures/cache_latency_components.svg",
)
TIMINGS = ("vlm_seconds", "vision_prefix_seconds", "vlm_transformer_seconds", "action_head_seconds",
           "projection_lookup_seconds", "postprocess_seconds", "preprocess_seconds", "end_to_end_seconds",
           "amortized_end_to_end_seconds", "prediction_seconds", "inference_seconds")
MODE_DESCRIPTIONS = {
    "full10": "当前观测条件下从纯噪声完整采样10步",
    "noise2": "当前观测条件下从纯噪声采样2步",
    "cache2": "命中开发集 t=.2 的32维中间潜变量后沿原网格细化2步；未命中完整10步",
    "cache0": "命中开发集 t=0 的7维最终归一化动作后直接反归一化；未命中完整10步",
}


class IntegrityError(ValueError):
    """Invalid evidence must never be downgraded into a missing-data warning."""


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise IntegrityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise IntegrityError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=reject)


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _distribution(values):
    values = list(values)
    return {"n": len(values), "mean": _mean(values),
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def _cluster_ci(task_effects):
    values = np.asarray(list(task_effects), dtype=np.float64)
    if len(values) < 2:
        return None
    _require(np.isfinite(values).all(), "nonfinite task effect")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values), (RESAMPLES, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def _identity(case):
    return tuple(case[key] for key in ("model", "suite", "task_id", "init_state_id", "seed", "condition"))


def _task(case):
    return case["suite"], case["task_id"]


def _verify_sources(protocol, root):
    fields = ("spec", "sources", "runtime_versions", "artifacts")
    _require(all(key in protocol for key in (*fields, "fingerprint")), "incomplete protocol provenance")
    _require(digest_json({key: protocol[key] for key in fields}) == protocol["fingerprint"],
             "protocol fingerprint mismatch")
    _require(bool(protocol["sources"]), "protocol has no frozen sources")
    root = Path(root).resolve()
    for name, expected in protocol["sources"].items():
        path = (root / name).resolve()
        _require(not Path(name).is_absolute() and path.is_relative_to(root), "source path escapes project")
        _require(path.is_file() and sha(path) == expected, f"frozen source mismatch: {name}")


def _validate_complete(output, case, record, protocol):
    directory = output / "episodes" / case["id"]
    receipt = read_json(directory / "receipt.json")
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "receipt.json"}
    _require(actual == set(receipt["files"]), f"receipt inventory mismatch: {case['id']}")
    _require({"result.json", "calls.json", "trajectory.npz", "initial.png"} <= actual,
             "required completed evidence missing")
    for name, expected in receipt["files"].items():
        _require(not any(char in name for char in "/\\:") and name not in {".", ".."}, "unsafe receipt path")
        _require(sha(directory / name) == expected, f"receipt hash mismatch: {case['id']}/{name}")
    model = record["model_info"]
    checkpoint = protocol["artifacts"]["models"][case["model"]]
    _require((model["revision"], model["model_sha256"]) ==
             (checkpoint["revision"], checkpoint["files"]["model.safetensors"]), "record model differs from protocol")
    _require(type(record["success"]) is bool, "success must be boolean")
    initial_state = np.asarray(record["initial_robot_state"])
    _require(initial_state.shape == (8,) and np.isfinite(initial_state).all(), "invalid initial proprioception")
    steps, count = record["steps"], record["inference_calls"]
    _require(type(steps) is int and 0 < steps <= record["horizon"], "invalid episode steps")
    calls = read_json(directory / "calls.json")
    _require(isinstance(calls, list) and type(count) is int and count == len(calls) > 0,
             "call log/inference count mismatch")
    _require(calls == model["cache_policy"]["call_log"], "duplicate call evidence disagrees")
    _require(model["cache_policy"]["mode"] == case["method"], "mode disagrees with case")
    interval = protocol["spec"]["execution"]["replan_interval"]
    expected_steps = list(range(0, steps, interval))
    _require(count == len(expected_steps), "call count differs from fixed execution interval")
    for index, call in enumerate(calls):
        _require(call["call_index"] == index and call["observation_count"] == expected_steps[index] + 1,
                 "call indices/observation indices disagree with executed steps")
        _require(call["mode"] == case["method"], "call mode differs from episode")
        _require(isinstance(call["noise_sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", call["noise_sha256"]),
                 "invalid sampling-noise fingerprint")
        hit = call["hit"]
        if case["method"] in ("cache2", "cache0"):
            _require(type(hit) is bool, "cache hit must be boolean")
            expected_nfe = (2 if case["method"] == "cache2" else 0) if hit else 10
            _require(not hit or isinstance(call["retrieved_case_id"], str), "hit lacks retrieved development identity")
        else:
            _require(hit is None, "noise baseline cannot claim a cache hit")
            expected_nfe = 10 if case["method"] == "full10" else 2
        _require(type(call["nfe"]) is int and call["nfe"] == expected_nfe, "NFE disagrees with mode/hit")
        _require(all(_finite(call.get(key)) for key in TIMINGS), "invalid/missing call timings")
        _require(call["similarity"] is None or (-1 <= call["similarity"] <= 1), "invalid cosine similarity")
    _require(model["cache_policy"]["total_nfe"] == sum(c["nfe"] for c in calls), "total NFE mismatch")
    _require(_finite(record["wall_seconds"]), "invalid raw episode wall time")
    _require(math.isclose(record["policy_inference_seconds"], sum(c["inference_seconds"] for c in calls),
                         abs_tol=1e-7, rel_tol=1e-9), "episode inference time differs from call log")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        actions = trajectory["actions"].copy()
        _require(actions.shape == (steps, 7) and np.isfinite(actions).all(), "invalid action trajectory")
        _require(np.array_equal(trajectory["replan_steps"], expected_steps), "trajectory replans differ from logs")
        _require(trajectory["states"].shape == (steps, 8), "invalid state trajectory")
        _require(np.allclose(trajectory["states"][0], record["initial_robot_state"], rtol=0, atol=1e-6),
                 "initial proprioception differs from trajectory")
    from PIL import Image
    with Image.open(directory / "initial.png") as image:
        pixels = np.asarray(image.convert("RGB"))
    _require(hashlib.sha256(pixels.tobytes()).hexdigest() == record["initial_image_sha256"],
             "initial image does not match recorded pixel fingerprint")
    return {"case": case, "record": record, "calls": calls, "actions": actions,
            "result_sha256": sha(directory / "result.json"), "receipt_sha256": sha(directory / "receipt.json"),
            "artifact_sha256": receipt["files"]}


def _verify_binding(output, protocol, records):
    heldout = [row for row in records.values() if row["case"]["split"] == "heldout"]
    path = output / "frozen_cache/binding.json"
    if not path.exists():
        _require(not heldout, "heldout results exist without a frozen cache binding")
        return None
    from embodied_research.action_cache import FrozenActionCache
    binding = read_json(path)
    _require(binding["protocol_fingerprint"] == protocol["fingerprint"], "cache binding/protocol mismatch")
    development = [c for c in protocol["spec"]["cases"] if c["split"] == "development"]
    _require(all(c["id"] in records for c in development), "binding exists before development completion")
    receipts = {c["id"]: records[c["id"]]["receipt_sha256"] for c in development}
    admitted = {c["id"] for c in development if records[c["id"]]["record"]["success"]}
    _require(binding["development_receipts"] == receipts, "cache binding/development receipts mismatch")
    _require(set(binding["admitted_successful_case_ids"]) == admitted and binding["test_admission"] is False,
             "cache admits other than successful development cases")
    checkpoint = protocol["artifacts"]["models"]["smolvla"]
    configuration = protocol["spec"]["cache"]
    for mode in ("cache2", "cache0"):
        cache = FrozenActionCache.load(output / "frozen_cache" / mode,
                                      expected_provenance_sha256=binding[mode]["provenance_sha256"])
        _require(cache.cache_sha256 == binding[mode]["cache_sha256"] and len(cache) == binding[mode]["entries"],
                 "frozen cache content differs from binding")
        _require(set(cache.entry_case_ids) <= admitted, "cache contains non-successful development entries")
        provenance = cache.provenance.to_dict()
        _require((provenance["model_revision"], provenance["model_sha256"]) ==
                 (checkpoint["revision"], checkpoint["files"]["model.safetensors"]),
                 "cache model provenance differs from protocol")
        expected_shape, native_time = ((50, 32), .2) if mode == "cache2" else ((50, 7), 0.)
        _require(cache.latent_shape == expected_shape and provenance["flow_time"] == native_time,
                 "cache latent shape/time differs from mode")
        _require(set(provenance["development_case_ids"]) == {c["id"] for c in development},
                 "cache declared development split differs from protocol")
        _require(cache.threshold == configuration["cosine_threshold"] and cache.capacity == configuration["capacity"],
                 "cache threshold/capacity differs from protocol")
        projected = cache.projection.config
        _require((projected.output_dim, projected.density, projected.seed) ==
                 (configuration["projection_dim"], configuration["density"], configuration["projection_seed"]),
                 "cache projection differs from protocol")
        _require(projected.input_dim == provenance["feature_contract"]["input_dim"], "cache feature dimension mismatch")
        for case_id in admitted:
            _require(cache.entry_case_ids.count(case_id) == records[case_id]["record"]["inference_calls"],
                     "cache did not admit exactly every call from successful development episodes")
    for row in heldout:
        _require(row["record"]["cache_binding_sha256"] == digest_json(binding), "heldout cache binding changed")
        mode = row["case"]["method"]
        if mode in ("cache2", "cache0"):
            _require(row["record"]["model_info"]["cache_policy"]["cache"]["cache_sha256"] ==
                     binding[mode]["cache_sha256"], "policy-reported cache differs from binding")
        for call in row["calls"]:
            _require(not call["hit"] or call["retrieved_case_id"] in admitted,
                     "retrieved case is not an admitted successful development episode")
    return binding


def load_study(output_dir, *, partial=False, root=ROOT):
    output = Path(output_dir)
    protocol = read_json(output / "protocol.json")
    _verify_sources(protocol, root)
    cases = protocol["spec"]["cases"]
    _require(cases and len({c["id"] for c in cases}) == len(cases), "empty or duplicate case denominator")
    case_ids = {c["id"] for c in cases}
    for directory in (output / "episodes").glob("*"):
        if directory.is_dir():
            _require(directory.name in case_ids, "unplanned episode directory exists")
    records, statuses = {}, {}
    for case in cases:
        _require(re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) is not None, "unsafe case identifier")
        _require(case["method"] in MODE_DESCRIPTIONS and case["split"] in ("development", "heldout"),
                 "unsupported declared method or split")
        directory = output / "episodes" / case["id"]
        result_path = directory / "result.json"
        if not result_path.exists():
            statuses[case["id"]] = "unsealed" if directory.exists() else "missing"
            continue
        record = read_json(result_path)
        _require(all(record.get(key) == value for key, value in case.items()), "case identity/configuration changed")
        _require(record.get("protocol_fingerprint") == protocol["fingerprint"], "record protocol fingerprint mismatch")
        if record.get("status") == "error":
            _require(not (directory / "receipt.json").exists(), "error record has a completed receipt")
            statuses[case["id"]] = "error"
            continue
        _require(record.get("status") == "complete", "unexpected result status")
        if not (directory / "receipt.json").exists():
            statuses[case["id"]] = "unsealed"
            continue
        records[case["id"]] = _validate_complete(output, case, record, protocol)
        statuses[case["id"]] = "complete"
    complete = len(records) == len(cases)
    if not complete and not partial:
        raise IntegrityError(f"Study incomplete: {len(records)}/{len(cases)} sealed complete; use --partial explicitly")
    binding = _verify_binding(output, protocol, records)
    # Initial-state and noise audits cover all methods in every observed block,
    # even when full10 has not completed yet in an explicitly partial report.
    blocks = defaultdict(list)
    for row in records.values():
        blocks[(row["case"]["split"], *_identity(row["case"]))].append(row)
    paired_blocks = aligned_calls = 0
    for rows in blocks.values():
        if len(rows) < 2:
            continue
        paired_blocks += 1
        first = rows[0]
        for other in rows[1:]:
            a, b = first["record"], other["record"]
            _require(a["initial_image_sha256"] == b["initial_image_sha256"], "paired initial images differ")
            _require(np.allclose(a["initial_robot_state"], b["initial_robot_state"], rtol=0, atol=1e-6),
                     "paired initial robot states differ")
            for x, y in zip(first["calls"], other["calls"]):
                _require(x["noise_sha256"] == y["noise_sha256"], "paired decision noise differs")
                aligned_calls += 1
    evidence = {"protocol.json": sha(output / "protocol.json")}
    for case_id, row in records.items():
        evidence[f"episodes/{case_id}/receipt.json"] = row["receipt_sha256"]
        evidence.update({f"episodes/{case_id}/{name}": digest for name, digest in row["artifact_sha256"].items()})
    if binding is not None:
        for name in ("binding.json", "cache2.json", "cache2.npz", "cache0.json", "cache0.npz"):
            evidence[f"frozen_cache/{name}"] = sha(output / "frozen_cache" / name)
    return {"protocol": protocol, "records": records, "statuses": statuses, "complete": complete,
            "evidence_sha256": evidence,
            "binding": binding, "pairing_audit": {"paired_initial_blocks": paired_blocks,
            "same_ordinal_noise_pairs_checked": aligned_calls, "proprio_atol": 1e-6,
            "scope": "identical initial RGB/proprioception and first/all shared decision noise; later observations may diverge"}}


def _group(scope, method, planned, study):
    cases = [c for c in planned if c["method"] == method]
    rows = [study["records"][c["id"]] for c in cases if c["id"] in study["records"]]
    calls = [call for row in rows for call in row["calls"]]
    task_success = defaultdict(list)
    for row in rows:
        task_success[_task(row["case"])].append(int(row["record"]["success"]))
    cache_calls = [c for c in calls if c["hit"] is not None]
    return {"scope": scope, "method": method, "planned_episodes": len(cases), "completed_episodes": len(rows),
            "planned_tasks": len({_task(c) for c in cases}), "observed_tasks": len(task_success),
            "statuses": {status: sum(study["statuses"][c["id"]] == status for c in cases)
                         for status in ("complete", "missing", "unsealed", "error")},
            "complete": len(rows) == len(cases), "successes": sum(r["record"]["success"] for r in rows),
            "observed_macro_task_success": _mean(_mean(outcomes) for outcomes in task_success.values()),
            "replans": len(calls), "total_nfe": sum(c["nfe"] for c in calls),
            "mean_nfe_per_replan": _mean(c["nfe"] for c in calls),
            "cache_queries": len(cache_calls), "cache_hits": sum(c["hit"] for c in cache_calls),
            "hit_rate": _mean(int(c["hit"]) for c in cache_calls),
            "full_sampling_calls": sum(c["nfe"] == 10 for c in calls),
            "refinement_calls": sum(c["mode"] == "cache2" and c["hit"] is True for c in calls),
            "fallback_calls": sum(c["hit"] is False for c in cache_calls),
            "direct_cached_final_calls": sum(c["mode"] == "cache0" and c["hit"] is True for c in calls),
            "noise2_calls": sum(c["mode"] == "noise2" for c in calls),
            "timings_seconds": {key: _distribution(c[key] for c in calls) for key in TIMINGS}}


def _compare(scope, candidate, baseline, planned, study):
    indexed = {(_identity(c), c["method"]): c for c in planned}
    _require(len(indexed) == len(planned), "duplicate trial identity within comparison")
    identities = sorted(key for key, method in indexed if method == candidate and (key, baseline) in indexed)
    pairs = []
    for identity in identities:
        ca, cb = indexed[identity, candidate], indexed[identity, baseline]
        if ca["id"] not in study["records"] or cb["id"] not in study["records"]:
            continue
        a, b = study["records"][ca["id"]], study["records"][cb["id"]]
        ar, br = a["record"], b["record"]
        available = min(len(a["actions"]), len(b["actions"]))
        first_10_complete = available >= 10
        difference = (a["actions"][:10].astype(np.float64) - b["actions"][:10]
                      if first_10_complete else None)
        pairs.append({"suite": ca["suite"], "task_id": ca["task_id"], "init_state_id": ca["init_state_id"],
                      "candidate_id": ca["id"], "baseline_id": cb["id"],
                      "candidate_success": ar["success"], "baseline_success": br["success"],
                      "success_difference": int(ar["success"]) - int(br["success"]),
                      "candidate_mean_call_seconds": _mean(c["end_to_end_seconds"] for c in a["calls"]),
                      "baseline_mean_call_seconds": _mean(c["end_to_end_seconds"] for c in b["calls"]),
                      "mean_call_seconds_difference": (_mean(c["end_to_end_seconds"] for c in a["calls"])
                                                       - _mean(c["end_to_end_seconds"] for c in b["calls"])),
                      "candidate_wall_seconds": ar["wall_seconds"], "baseline_wall_seconds": br["wall_seconds"],
                      "both_success": ar["success"] and br["success"],
                      "wall_difference_both_success_only": ar["wall_seconds"] - br["wall_seconds"]
                      if ar["success"] and br["success"] else None,
                      "actual_available_steps": available,
                      "first_action_steps_compared": 10 if first_10_complete else 0,
                      "first_10_complete": first_10_complete,
                      "first_10_missing_reason": None if first_10_complete else
                      f"Requires 10 actions from both episodes; candidate={len(a['actions'])}, baseline={len(b['actions'])}",
                      "first_actions_rmse_per_dim": np.sqrt(np.mean(difference ** 2, axis=0)).tolist()
                      if first_10_complete else None,
                      "first_actions_mae_per_dim": np.mean(np.abs(difference), axis=0).tolist()
                      if first_10_complete else None,
                      "first_actions_rmse_all_dims": float(np.sqrt(np.mean(difference ** 2)))
                      if first_10_complete else None,
                      "first_actions_max_abs_difference": float(np.max(np.abs(difference)))
                      if first_10_complete else None,
                      "first_decision_noise_sha256": a["calls"][0]["noise_sha256"]})
    tasks = defaultdict(list)
    for pair in pairs:
        tasks[_task(pair)].append(pair)
    task_rows = [{"suite": key[0], "task_id": key[1], "pairs": len(rows),
                  "success_difference": _mean(row["success_difference"] for row in rows),
                  "mean_call_seconds_difference": _mean(row["mean_call_seconds_difference"] for row in rows)}
                 for key, rows in sorted(tasks.items())]
    return {"scope": scope, "candidate": candidate, "baseline": baseline,
            "planned_pairs": len(identities), "completed_pairs": len(pairs), "observed_task_clusters": len(tasks),
            "complete": len(pairs) == len(identities),
            "macro_success_difference": _mean(r["success_difference"] for r in task_rows),
            "success_task_cluster_95ci": _cluster_ci(r["success_difference"] for r in task_rows),
            "macro_mean_call_seconds_difference": _mean(r["mean_call_seconds_difference"] for r in task_rows),
            "latency_task_cluster_95ci_seconds": _cluster_ci(r["mean_call_seconds_difference"] for r in task_rows),
            "both_success_pairs": sum(p["both_success"] for p in pairs),
            "mean_wall_difference_both_success_only": _mean(p["wall_difference_both_success_only"]
                                                           for p in pairs if p["both_success"]),
            "first_10_complete_pairs": sum(p["first_10_complete"] for p in pairs),
            "first_10_missing_pairs": sum(not p["first_10_complete"] for p in pairs),
            "mean_first_actions_rmse_all_dims": _mean(p["first_actions_rmse_all_dims"]
                                                     for p in pairs if p["first_10_complete"]),
            "per_task": task_rows, "per_pair": pairs}


def summarize(study):
    protocol = study["protocol"]
    cases, spec = protocol["spec"]["cases"], protocol["spec"]
    development = [c for c in cases if c["split"] == "development"]
    heldout = [c for c in cases if c["split"] == "heldout"]
    primary = [c for c in heldout if c["method"] in spec["primary"]["methods"]
               and c["init_state_id"] in spec["primary"]["initial_states"]]
    ablations = [c for c in heldout if c["method"] in spec["ablations"]["methods"]
                 and c["init_state_id"] in spec["ablations"]["initial_states"]]
    _require(len(development) == spec["development"]["episodes"], "development denominator differs from spec")
    _require(len({_task(c) for c in primary}) == spec["primary"]["task_clusters"],
             "primary task-cluster denominator differs from spec")
    groups = [_group("development", "full10", development, study)]
    groups += [_group("primary", method, primary, study) for method in spec["primary"]["methods"]]
    groups += [_group("ablation_init11", method, ablations, study) for method in spec["ablations"]["methods"]]
    comparisons = [_compare("primary", "cache2", "full10", primary, study)]
    comparisons += [_compare("ablation_init11", candidate, baseline, ablations, study)
                    for candidate, baseline in (("cache2", "full10"), ("noise2", "full10"),
                                                ("cache0", "full10"), ("cache2", "noise2"), ("cache2", "cache0"))]
    _require(comparisons[0]["planned_pairs"] == spec["primary"]["pairs"], "primary denominator differs from spec")
    _require(all(c["planned_pairs"] == spec["ablations"]["pairs"] for c in comparisons[1:]),
             "ablation denominator differs from spec")
    episodes, models = [], {}
    for case in cases:
        row = study["records"].get(case["id"])
        episode = {**{k: case[k] for k in ("id", "model", "suite", "task_id", "init_state_id", "method", "split")},
                   "status": study["statuses"][case["id"]], "result_path": f"episodes/{case['id']}/result.json"}
        if row is not None:
            record = row["record"]
            episode.update({key: record[key] for key in ("success", "steps", "horizon", "inference_calls",
                                                         "wall_seconds", "policy_inference_seconds", "preprocess_seconds")})
            episode.update({"total_nfe": sum(c["nfe"] for c in row["calls"]),
                            "result_sha256": row["result_sha256"], "receipt_sha256": row["receipt_sha256"]})
            info = {k: v for k, v in record["model_info"].items() if k != "cache_policy"}
            models[digest_json(info)] = info
        episodes.append(episode)
    return {"schema": "feedbackbench.cache_study_report.v1", "generated_at": utc_now(),
            "status": "complete" if study["complete"] else "PARTIAL",
            "name": spec["name"], "question": spec["question"],
            "counts": {"planned": len(cases), **{state: list(study["statuses"].values()).count(state)
                                               for state in ("complete", "missing", "unsealed", "error")}},
            "protocol_fingerprint": protocol["fingerprint"], "frozen_sources": protocol["sources"],
            "evidence_sha256": study["evidence_sha256"],
            "report_source_sha256": sha(Path(__file__)), "runtime_versions": protocol["runtime_versions"],
            "model_details_from_records": list(models.values()), "model_artifacts": protocol["artifacts"]["models"],
            "cache_configuration": spec["cache"], "cache_binding": study["binding"],
            "method_descriptions": MODE_DESCRIPTIONS, "pairing_audit": study["pairing_audit"],
            "groups": groups, "comparisons": comparisons, "episodes": episodes,
            "uncertainty": {"unit": "task cluster; equal-weight mean of within-task paired episode effects",
                            "resamples": RESAMPLES, "seed": BOOTSTRAP_SEED,
                            "small_n": "Four planned task clusters; percentile intervals are unstable and exploratory.",
                            "zero_width": "Zero width means no observed task-effect variation; not equivalence, noninferiority or zero true uncertainty."},
            "interpretation_limits": [
                "Partial reports describe only observed complete pairs; missing/error/unsealed cases remain in planned denominators.",
                "Development, primary and init11 ablation panels overlap; do not add their totals or treat all cases as independent evaluation.",
                "Call p50/p95 pool calls descriptively; paired latency compares episode mean call cost, not identical later observations.",
                "Wall difference is shown only for both-success pairs and remains outcome-conditioned descriptive evidence, not causal speedup.",
                "Every episode raw wall is retained; shorter failed rollouts are not speedups.",
                "First-ten-action distance requires 10 actions from both episodes; shorter episodes retain their outcomes but have missing diagnostics.",
                "Action distance uses denormalized mixed-unit coordinates; behavior distance is not task quality.",
                "Fixed offline successful-development cache differs from original ActionCache online successful-rollout LRU.",
                "Threshold is fixed before pilot and untuned. No test-time cache admission, training, SOTA or real-time deployment claim.",
                *spec.get("limitations", []),
            ]}


def _csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _format(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown(report):
    counts = report["counts"]
    title = "完整试验记录" if report["status"] == "complete" else "PARTIAL：试验未完成，不给出完整结论"
    lines = [f"# {report['name']} — {title}", "", report["question"], "",
             f"已核验 {counts['complete']}/{counts['planned']} 个计划回合；缺失 {counts['missing']}、"
             f"未封存 {counts['unsealed']}、错误 {counts['error']}。开发集与测试集分开；主比较与消融的重叠回合不重复相加。", "",
             "所有完整回合均核对协议与源码指纹、receipt 文件哈希、模型权重来源、调用日志及轨迹。"
             "配对核对初始图像/机器人状态和首决策噪声，并检查共有决策序号的噪声一致性。", "",
             "## 方法与范围", ""]
    lines += [f"- `{method}`：{description}。" for method, description in MODE_DESCRIPTIONS.items()]
    lines += ["", f"缓存阈值 **{report['cache_configuration']['cosine_threshold']}**，预先固定、未经调优。"
              "只接纳成功开发回合，测试期冻结；这与 [ActionCache 论文](https://arxiv.org/abs/2607.06370v2) "
              "的在线成功回合 LRU 缓存不同。", "",
              "## 成功与采样预算", "",
              "成功数以已完成回合计，旁列计划分母；部分数据不能代表整个计划成功率。", "",
              "| 范围 / 方法 | 完成 / 计划 | 任务数 | 成功 / 已完成 | NFE / 规划 | hit / query | 完整 / 细化 / 回退 / 直接缓存 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["groups"]:
        lines.append(f"| {row['scope']} / {row['method']} | {row['completed_episodes']}/{row['planned_episodes']} | "
                     f"{row['observed_tasks']}/{row['planned_tasks']} | {row['successes']}/{row['completed_episodes']} | "
                     f"{_format(row['mean_nfe_per_replan'])} | {row['cache_hits']}/{row['cache_queries']} | "
                     f"{row['full_sampling_calls']} / {row['refinement_calls']} / {row['fallback_calls']} / {row['direct_cached_final_calls']} |")
    lines += ["", "回退次数是完整10步调用的子集，不能与完整次数相加。noise2 计数另存 report.json。", "",
              "![Primary success and NFE](figures/cache_primary_success_nfe.png)", "",
              "![Common-init ablations](figures/cache_ablation_success_nfe.png)", "",
              "## 配对比较", "",
              "以下均为 candidate − baseline。成功差先在每个任务内平均，再等权平均任务；"
              f"按任务簇重抽样 {RESAMPLES} 次，seed={BOOTSTRAP_SEED}。仅4个计划任务簇，区间不稳定；"
              "零宽区间仅说明观察到的任务效应无变异，不能证明等价、非劣或真实不确定性为零。", "",
              "| 范围 / 对照 | 配对完成 / 计划 | 任务簇 | 成功差 pp [95% CI] | 每调用均值差秒 [95% CI] | 双成功对数 / wall差秒 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["comparisons"]:
        ci, latency_ci = row["success_task_cluster_95ci"], row["latency_task_cluster_95ci_seconds"]
        effect = None if row["macro_success_difference"] is None else 100 * row["macro_success_difference"]
        ci_text = "不可估计" if ci is None else f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]"
        latency_text = "不可估计" if latency_ci is None else f"[{latency_ci[0]:+.3f}, {latency_ci[1]:+.3f}]"
        lines.append(f"| {row['scope']} / {row['candidate']} − {row['baseline']} | "
                     f"{row['completed_pairs']}/{row['planned_pairs']} | {row['observed_task_clusters']} | "
                     f"{_format(effect, 2)} {ci_text} | {_format(row['macro_mean_call_seconds_difference'])} {latency_text} | "
                     f"{row['both_success_pairs']} / {_format(row['mean_wall_difference_both_success_only'])} |")
    lines += ["", "配对延迟比较的是各回合的平均调用耗时；首决策之后轨迹/观测可能不同，不能解释为相同输入的纯算法计时。"
              "wall差只列共同成功回合，仍是受结果选择影响的参考量。所有回合原始wall在 episodes.csv；失败更早结束不是加速。", "",
              "## 分项时延", "", "每格为 p50 / p95 秒；调用池分位数为描述性统计，较长回合贡献更多调用。"
              "VLM 总项包含 vision-prefix 和 transformer，不能将三列相加。end-to-end 包含最近一次观测预处理；"
              "amortized 另计上次规划以来所有非规划 tick 的预处理，详见 JSON/CSV。", "",
              "| 范围 / 方法 | calls | VLM总项 | vision-prefix | transformer | action head | lookup | end-to-end |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["groups"]:
        keys = ("vlm_seconds", "vision_prefix_seconds", "vlm_transformer_seconds", "action_head_seconds",
                "projection_lookup_seconds", "end_to_end_seconds")
        cells = [f"{_format(row['timings_seconds'][key]['p50'])} / {_format(row['timings_seconds'][key]['p95'])}" for key in keys]
        lines.append(f"| {row['scope']} / {row['method']} | {row['replans']} | " + " | ".join(cells) + " |")
    lines += ["", "![Timing components](figures/cache_latency_components.png)", "", "## 行为差与限制", "",
              "pairs.csv 列出双方都有完整前10步时的7维逐通道 RMSE/MAE，以及混合坐标的距离。"
              "任何一方少于10步时，距离记为缺失，保留可用步数和原因，不纳入均值；该回合的成功结果仍计入主分析。"
              "这是已执行动作的行为距离，不是质量或成功的替代指标；后续观测可能已分叉。", "",
              "| 范围 / 对照 | first10有效 / 已完成配对 | 前10步混合坐标 RMSE 的回合均值 |", "|---|---:|---:|"]
    for comparison in report["comparisons"]:
        lines.append(f"| {comparison['scope']} / {comparison['candidate']} − {comparison['baseline']} | "
                     f"{comparison['first_10_complete_pairs']}/{comparison['completed_pairs']} | "
                     f"{_format(comparison['mean_first_actions_rmse_all_dims'], 6)} |")
    lines += ["",
              "这是少量已知预训练任务上的初始状态泛化试验。CPU 同步仿真在推理时暂停；不支持真实实时部署、SOTA 或新任务泛化声明。", "",
              "## 可复核证据", "",
              "[协议](protocol.json) · [完整机器可读报告](report.json) · [方法汇总](groups.csv) · "
              "[所有回合](episodes.csv) · [逐次调用](calls.csv) · [配对结果与动作差](pairs.csv)", "",
              f"Protocol SHA fingerprint: `{report['protocol_fingerprint']}`。模型 revision、权重 SHA、参数量与运行时版本取自原始记录和协议，见 report.json。", ""]
    for model in report["model_details_from_records"]:
        lines.append(f"- `{model.get('name', 'model')}`：{model.get('parameters', '未记录')} 参数，"
                     f"{model.get('device', '未记录')} / {model.get('dtype', '未记录')}；"
                     f"revision `{model['revision']}`，权重 SHA256 `{model['model_sha256']}`。")
    lines.append("")
    return "\n".join(lines)


def _plots(output, report):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    for scope, filename, title in (("primary", "cache_primary_success_nfe", "Primary paired comparison"),
                                   ("ablation_init11", "cache_ablation_success_nfe", "Common-init exploratory ablations")):
        groups = [g for g in report["groups"] if g["scope"] == scope]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        labels = [f"{g['method']}\n{g['observed_tasks']}/{g['planned_tasks']} tasks\n"
                  f"{g['completed_episodes']}/{g['planned_episodes']} episodes" for g in groups]
        for ax, key, ylabel in ((axes[0], "observed_macro_task_success", "Observed task-macro success"),
                                (axes[1], "mean_nfe_per_replan", "Mean flow NFE per replan")):
            values = [g[key] if g[key] is not None else np.nan for g in groups]
            colors = {"full10": "#3976aa", "noise2": "#cb843a", "cache2": "#658d68", "cache0": "#9374ab"}
            bars = ax.bar(range(len(groups)), values, color=[colors[group["method"]] for group in groups])
            ax.set_xticks(range(len(groups)), labels, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.18 if key == "observed_macro_task_success" else 12)
            ax.grid(axis="y", alpha=.2)
            for bar, value, group in zip(bars, values, groups):
                text = (f"{group['successes']}/{group['completed_episodes']} successful" if key == "observed_macro_task_success"
                        else f"{value:.2f}; {group['replans']} replans") if np.isfinite(value) else "No completed data"
                ax.text(bar.get_x() + bar.get_width()/2, (value if np.isfinite(value) else 0) +
                        (.035 if key == "observed_macro_task_success" else .25), text, ha="center", fontsize=9)
        fig.suptitle(f"SmolVLA offline-cache pilot — {title}\n{report['status']}; counts show observed/planned", fontsize=12)
        for suffix in ("png", "svg"):
            fig.savefig(destination / f"{filename}.{suffix}", dpi=160)
        plt.close(fig)
    groups = [g for g in report["groups"] if g["scope"] != "development"]
    components = ("vision_prefix_seconds", "vlm_transformer_seconds", "action_head_seconds",
                  "projection_lookup_seconds", "postprocess_seconds", "end_to_end_seconds")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    labels = [f"{'P' if g['scope'] == 'primary' else 'A'}:{g['method']}\n"
              f"{g['observed_tasks']} tasks; {g['completed_episodes']}/{g['planned_episodes']} ep\n"
              f"n={g['replans']} calls" for g in groups]
    for ax, component in zip(axes.flat, components):
        for x, group in enumerate(groups):
            dist = group["timings_seconds"][component]
            if dist["n"]:
                ax.plot([x, x], [dist["p50"], dist["p95"]], color="#38759f", linewidth=2)
                ax.scatter(x, dist["p50"], color="#38759f", label="p50" if x == 0 else None)
                ax.scatter(x, dist["p95"], color="#ce853d", marker="^", label="p95" if x == 0 else None)
        ax.set_xticks(range(len(groups)), labels, fontsize=8)
        ax.set_title(component.removesuffix("_seconds").replace("_", " "))
        ax.set_ylabel("Seconds / call")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    if any(g["replans"] for g in groups):
        axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"SmolVLA offline-cache pilot — descriptive call timings ({report['status']})\n"
                 "P: primary; A: common-init ablation; panels overlap, not independent samples", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"cache_latency_components.{suffix}", dpi=160)
    plt.close(fig)


def _export_path(output, name):
    root = Path(output).resolve()
    path = (root / name).resolve()
    _require(path.is_relative_to(root) and path.is_file(), f"Missing or out-of-stage export: {name}")
    return path


def _validate_markdown_resources(output):
    """Check generated inline Markdown links/images without rendering or network I/O."""
    root = Path(output).resolve()
    markdown = (root / "report.md").read_text(encoding="utf-8")
    for match in re.finditer(r"!?\[[^\]\r\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))[^)]*\)", markdown):
        reference = urlsplit(match.group(1) or match.group(2))
        if reference.scheme in {"https", "http", "mailto"} or reference.netloc:
            continue
        _require(not reference.scheme, "Unsupported local Markdown URL scheme")
        if not reference.path:
            continue
        relative = unquote(reference.path).replace("\\", "/")
        _require(not relative.startswith("/") and re.match(r"^[A-Za-z]:", relative) is None,
                 "Markdown local resource must use a relative stage path")
        target = (root / relative).resolve()
        _require(target.is_relative_to(root) and target.is_file(),
                 f"Missing or out-of-stage Markdown resource: {relative}")


def _validate_exports(output):
    manifest_path = _export_path(output, "exports.json")
    manifest = read_json(manifest_path)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "files"}
             and manifest["schema"] == EXPORT_SCHEMA, "invalid export manifest schema")
    _require(isinstance(manifest["files"], dict) and set(manifest["files"]) == set(EXPORT_FILES),
             "export manifest differs from the fixed export whitelist")
    for name in EXPORT_FILES:
        expected = manifest["files"][name]
        _require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected) is not None,
                 f"invalid export SHA256: {name}")
        _require(sha(_export_path(output, name)) == expected, f"export hash mismatch: {name}")
    _validate_markdown_resources(output)


def build_report(output_dir, *, partial=False, root=ROOT):
    output = Path(output_dir)
    study = load_study(output, partial=partial, root=root)
    report = summarize(study)
    _plots(output, report)
    _csv(output / "groups.csv", report["groups"])
    _csv(output / "episodes.csv", report["episodes"])
    _csv(output / "pairs.csv", [{"scope": comparison["scope"], "candidate": comparison["candidate"],
                                "baseline": comparison["baseline"], **pair}
                               for comparison in report["comparisons"] for pair in comparison["per_pair"]])
    _csv(output / "calls.csv", [{"id": row["case"]["id"], "split": row["case"]["split"],
                                "suite": row["case"]["suite"], "task_id": row["case"]["task_id"],
                                "init_state_id": row["case"]["init_state_id"], **call}
                               for row in study["records"].values() for call in row["calls"]])
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    write_json(output / "report.json", report)
    _validate_markdown_resources(output)
    # This is the final write: an interrupted rebuild leaves the previous seal
    # invalid rather than silently mixing figures/CSVs from different rounds.
    write_json(output / "exports.json", {"schema": EXPORT_SCHEMA,
               "files": {name: sha(_export_path(output, name)) for name in EXPORT_FILES}})
    return output / "report.json"


def validate_report(output_dir, *, partial=False, root=ROOT):
    """Read-only packager check: strict raw validation and deterministic recompute.

    Does not write figures or import plotting/model libraries. Checks the fixed
    export inventory, all export hashes and relative Markdown resources before
    recomputing statistics. By default both evidence and report must be complete.
    A changed report implementation also
    requires regeneration because its source hash is included in the summary.
    """
    _validate_exports(output_dir)
    current = read_json(Path(output_dir) / "report.json")
    expected = summarize(load_study(output_dir, partial=partial, root=root))
    left = {key: value for key, value in current.items() if key != "generated_at"}
    right = {key: value for key, value in expected.items() if key != "generated_at"}
    _require(left == right, "report is stale or differs from verified evidence/report implementation")
    _require((Path(output_dir) / "report.md").read_text(encoding="utf-8") == _markdown(current),
             "Markdown content differs from the verified report")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/frontier/cache_study")
    parser.add_argument("--partial", action="store_true", help="Explicitly summarize incomplete observed evidence")
    args = parser.parse_args()
    print(build_report(args.output, partial=args.partial))


if __name__ == "__main__":
    main()
