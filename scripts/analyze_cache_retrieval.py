"""Post-hoc descriptive diagnostics for every declared heldout cache2/cache0 episode.

Uses summarize_cache_study.load_study for strict sealed-evidence validation; a
complete study is required unless --partial is explicit. No model or simulator
is imported, and neither the protocol nor official report exports are changed.

Writes retrieval_analysis.json (episode summaries plus every decision) and
retrieval_analysis.csv (one row per planned cache episode). Existing outputs are
never overwritten; use a new --destination for a later snapshot. A partial
snapshot preserves all planned episodes and never produces formal conclusions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
_LOADER_PATH = Path(__file__).with_name("summarize_cache_study.py")
_SPEC = importlib.util.spec_from_file_location("cache_retrieval_evidence_loader", _LOADER_PATH)
_REPORTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORTER)
load_study = _REPORTER.load_study
IntegrityError = _REPORTER.IntegrityError
sha = _REPORTER.sha
MODES = ("cache2", "cache0")
INTERPRETATION = [
    "Post-hoc descriptive diagnostics; no protocol, threshold, cache, or method selection is changed.",
    "All declared heldout cache2/cache0 episodes are included in protocol order, including failures; "
    "unsealed, missing, and error episodes have unknown statistics, never zero-filled outcomes.",
    "Repeated retrieval and episode failure may co-occur; these observations do not establish causality.",
    "Cosine similarity measures the projected feature key, not action quality or task success.",
    "A hit run is consecutive decisions retrieving the same index within one episode and cache; "
    "any miss or different index ends it. Decision counts are not independent trials or time durations.",
    "Same task means identical suite and task_id in the frozen protocol; development and heldout "
    "initial states differ. A cross-task hit does not by itself establish transfer or generalization.",
    "NFE counts action-head evaluations and is neither end-to-end latency nor quality-adjusted speedup. "
    "These diagnostics do not support effectiveness claims, including when the study is complete.",
]


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def analyze_episode(case, calls, source_cases, entry_case_ids):
    """Pure statistics over one verified episode; source indices are checked again.

    Empty calls are supported for synthetic boundaries. The evidence loader
    itself rejects empty completed real episodes. Miss similarities remain in
    the decision trace but never enter hit similarity summaries or hit runs.
    """
    _require(case["split"] == "heldout" and case["method"] in MODES,
             "retrieval diagnostics require a heldout cache episode")
    decisions, similarities, indices = [], [], set()
    same_task_hits = cross_task_hits = total_nfe = current_run = longest_run = 0
    previous_index = None
    for ordinal, call in enumerate(calls):
        _require(type(call["call_index"]) is int and call["call_index"] == ordinal,
                 "call_index must be sequential from zero")
        hit, similarity = call["hit"], call["similarity"]
        _require(type(hit) is bool, "hit must be boolean")
        _require(similarity is None or (type(similarity) in (float, int)
                 and math.isfinite(similarity) and -1 <= similarity <= 1), "invalid similarity")
        expected_nfe = (2 if case["method"] == "cache2" else 0) if hit else 10
        _require(type(call["nfe"]) is int and call["nfe"] == expected_nfe,
                 "NFE differs from declared mode/hit")
        index, source = call.get("retrieved_index"), call.get("retrieved_case_id")
        relation = None
        if hit:
            _require(type(index) is int and 0 <= index < len(entry_case_ids),
                     "retrieved index is absent or outside the frozen cache")
            _require(isinstance(source, str) and source == entry_case_ids[index],
                     "retrieved source differs from the frozen cache index")
            _require(source in source_cases and source_cases[source]["split"] == "development",
                     "retrieved source must be a declared development case")
            _require(similarity is not None, "hit similarity is missing")
            source_case = source_cases[source]
            same_task = (source_case["suite"], source_case["task_id"]) == (case["suite"], case["task_id"])
            relation = "same_task" if same_task else "cross_task"
            same_task_hits += int(same_task)
            cross_task_hits += int(not same_task)
            similarities.append(float(similarity))
            indices.add(index)
            current_run = current_run + 1 if previous_index == index else 1
            previous_index = index
            longest_run = max(longest_run, current_run)
        else:
            _require(index is None and source is None, "miss cannot contain a retrieved index or source")
            previous_index, current_run = None, 0
        total_nfe += call["nfe"]
        decisions.append({"call_index": ordinal, "source_case_id": source, "retrieved_index": index,
                          "similarity": similarity, "hit": hit, "nfe": call["nfe"],
                          "source_task_relation": relation})
    queries, hits = len(calls), len(similarities)
    return {"queries": queries, "hits": hits, "misses": queries - hits,
            "hit_rate": hits / queries if queries else None, "total_nfe": total_nfe,
            "mean_nfe_per_query": total_nfe / queries if queries else None,
            "same_task_hits": same_task_hits, "cross_task_hits": cross_task_hits,
            "unique_cache_indices": sorted(indices), "unique_cache_index_count": len(indices),
            "longest_same_index_hit_run": longest_run,
            "hit_similarity": {"min": min(similarities) if similarities else None,
                               "p50": median(similarities) if similarities else None,
                               "max": max(similarities) if similarities else None},
            "decisions": decisions}


def _entry_sources(directory, study):
    """Read the index/source map already authenticated by the common loader.

    Recheck its file hash so the mapping cannot change between load_study and
    this extra read. The loader has already validated cache integrity, entry
    count, success-only development admission, and protocol provenance.
    """
    if study["binding"] is None:
        return {}
    result = {}
    for mode in MODES:
        relative = f"frozen_cache/{mode}.json"
        payload = (Path(directory) / relative).read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == study["evidence_sha256"].get(relative),
                 f"frozen cache manifest changed after evidence validation: {mode}")
        metadata = json.loads(payload)
        entries = metadata["entry_case_ids"]
        _require(isinstance(entries, list) and all(isinstance(item, str) for item in entries)
                 and len(entries) == study["binding"][mode]["entries"], "cache index/source count mismatch")
        result[mode] = entries
    return result


def summarize(study, cache_entry_case_ids):
    """Include every planned cache episode; no success filtering or causal tests."""
    cases = study["protocol"]["spec"]["cases"]
    source_cases = {case["id"]: case for case in cases}
    selected = [case for case in cases if case["split"] == "heldout" and case["method"] in MODES]
    episodes = []
    for case in selected:
        case_id = case["id"]
        row = {key: case[key] for key in ("id", "suite", "task_id", "init_state_id", "method", "split")}
        row.update(status=study["statuses"][case_id], success=None, steps=None, receipt_sha256=None)
        if row["status"] == "complete":
            _require(case_id in study["records"], "complete episode lacks verified record")
            _require(case["method"] in cache_entry_case_ids, "complete cache episode lacks frozen index mapping")
            record = study["records"][case_id]
            row.update(success=record["record"]["success"], steps=record["record"]["steps"],
                       receipt_sha256=record["receipt_sha256"])
            row.update(analyze_episode(case, record["calls"], source_cases, cache_entry_case_ids[case["method"]]))
        else:
            # No completed data does not mean a zero-query or zero-hit episode.
            empty = analyze_episode(case, [], source_cases, [])
            row.update({key: None for key in empty})
            row["hit_similarity"] = {"min": None, "p50": None, "max": None}
            row["decisions"] = []
        episodes.append(row)
    return {"schema": "feedbackbench.cache_retrieval_analysis.v1", "generated_at": _REPORTER.utc_now(),
            "status": "COMPLETE_DESCRIPTIVE_ONLY" if study["complete"] else "PARTIAL_DESCRIPTIVE_ONLY",
            "study_complete": study["complete"], "post_hoc": True, "formal_conclusions": False,
            "protocol_fingerprint": study["protocol"]["fingerprint"],
            "analysis_source_sha256": sha(Path(__file__)), "evidence_loader_source_sha256": sha(_LOADER_PATH),
            "evidence_sha256": study["evidence_sha256"], "cache_binding": study["binding"],
            "planned_study_episodes": len(cases), "completed_study_episodes": len(study["records"]),
            "study_episode_statuses": study["statuses"], "planned_cache_episodes": len(selected),
            "completed_cache_episodes": sum(row["status"] == "complete" for row in episodes),
            "interpretation": INTERPRETATION,
            "csv_layout": "One row per planned heldout cache episode; null means unobserved or undefined. "
                          "unique_cache_indices is a JSON array. Every decision is retained in JSON episodes[].decisions.",
            "episodes": episodes}


def _csv_bytes(episodes):
    fields = ("id", "suite", "task_id", "init_state_id", "method", "split", "status", "success", "steps",
              "queries", "hits", "misses", "hit_rate", "total_nfe", "mean_nfe_per_query", "same_task_hits",
              "cross_task_hits", "unique_cache_index_count", "unique_cache_indices", "longest_same_index_hit_run",
              "hit_similarity_min", "hit_similarity_p50", "hit_similarity_max", "receipt_sha256")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for episode in episodes:
        row = {key: episode[key] for key in fields if key in episode}
        if row["unique_cache_indices"] is not None:
            row["unique_cache_indices"] = json.dumps(row["unique_cache_indices"], separators=(",", ":"))
        row.update({f"hit_similarity_{key}": value for key, value in episode["hit_similarity"].items()})
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def build_analysis(study_dir, *, partial=False, root=ROOT, destination=None):
    directory = Path(study_dir)
    output = directory if destination is None else Path(destination)
    paths = (output / "retrieval_analysis.csv", output / "retrieval_analysis.json")
    if any(path.exists() for path in paths):
        raise FileExistsError("retrieval analysis already exists; choose a new --destination")
    study = load_study(directory, partial=partial, root=root)
    result = summarize(study, _entry_sources(directory, study))
    result["partial_requested"] = bool(partial)
    csv_payload = _csv_bytes(result["episodes"])
    result["csv_sha256"] = hashlib.sha256(csv_payload).hexdigest()
    json_payload = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, payload in zip(paths, (csv_payload, json_payload)):
            with path.open("xb") as handle:
                created.append(path)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return paths[1]


def validate_analysis(study_dir, analysis_dir=None, partial=False, root=ROOT):
    """Read-only delivery check against freshly validated, sealed source evidence.

    Default validation requires the full study and an export made without
    --partial. Explicit partial validation must match the export's recorded
    choice, even when all episodes happened to be complete. Generation time is
    the only ignored JSON field; CSV bytes must exactly equal reconstruction.
    Returns the verified JSON object and never creates or repairs any file.
    """
    directory = Path(study_dir)
    output = directory if analysis_dir is None else Path(analysis_dir)
    study = load_study(directory, partial=partial, root=root)
    expected = summarize(study, _entry_sources(directory, study))
    expected["partial_requested"] = bool(partial)
    expected_csv = _csv_bytes(expected["episodes"])
    expected["csv_sha256"] = hashlib.sha256(expected_csv).hexdigest()
    try:
        current = _REPORTER.read_json(output / "retrieval_analysis.json")
        current_csv = (output / "retrieval_analysis.csv").read_bytes()
    except FileNotFoundError as error:
        raise IntegrityError("retrieval analysis JSON/CSV export missing") from error
    _require(isinstance(current, dict), "retrieval analysis JSON must be an object")
    _require(current.get("partial_requested") is bool(partial),
             "partial_requested differs from the selected validation mode")
    _require(hashlib.sha256(current_csv).hexdigest() == current.get("csv_sha256"),
             "retrieval analysis CSV checksum differs from its JSON binding")
    _require(current_csv == expected_csv, "retrieval analysis CSV is stale or modified")
    left = {key: value for key, value in current.items() if key != "generated_at"}
    right = {key: value for key, value in expected.items() if key != "generated_at"}
    # Canonical serialization preserves JSON types: Python equality alone would
    # accept a modified true as 1, including in success and hit evidence.
    _require(json.dumps(left, sort_keys=True, allow_nan=False)
             == json.dumps(right, sort_keys=True, allow_nan=False),
             "retrieval analysis JSON is stale or differs from verified evidence/implementation")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/frontier/cache_study")
    parser.add_argument("--destination", type=Path, help="Separate export directory; default is the study directory")
    parser.add_argument("--partial", action="store_true",
                        help="Explicit incomplete snapshot; retains planned cases and makes no formal conclusions")
    args = parser.parse_args()
    print(build_analysis(args.study, destination=args.destination, partial=args.partial))


if __name__ == "__main__":
    main()
