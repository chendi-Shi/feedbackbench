"""Synthetic retrieval logs only: no model, simulation, or actual study outputs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cache_retrieval_under_test", ROOT / "scripts/analyze_cache_retrieval.py"
)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def case(identifier, suite="libero_spatial", *, split="heldout", method="cache2"):
    return {"id": identifier, "suite": suite, "task_id": 1, "init_state_id": 11,
            "split": split, "method": method}


def call(index, retrieved=None, *, mode="cache2", similarity=None):
    hit = retrieved is not None
    entry_index, source = retrieved if hit else (None, None)
    return {"call_index": index, "hit": hit, "retrieved_index": entry_index,
            "retrieved_case_id": source, "similarity": similarity,
            "nfe": (2 if mode == "cache2" else 0) if hit else 10}


@pytest.fixture
def sources():
    # IDs deliberately encode no task identity; task comparison must use the protocol.
    return {"dev_a": case("dev_a", split="development", method="full10"),
            "dev_b": case("dev_b", "libero_goal", split="development", method="full10")}


def test_empty_and_misses_are_not_hits(sources):
    empty = analysis.analyze_episode(case("test"), [], sources, [])
    assert empty["queries"] == empty["hits"] == empty["total_nfe"] == 0
    assert empty["hit_rate"] is None
    assert empty["mean_nfe_per_query"] is None
    assert empty["unique_cache_indices"] == []
    assert empty["longest_same_index_hit_run"] == 0
    assert empty["hit_similarity"] == {"min": None, "p50": None, "max": None}
    miss = call(0, similarity=.94)
    miss["nearest_case_id"] = "dev_a"
    result = analysis.analyze_episode(case("test"), [miss, call(1)], sources, [])
    assert (result["queries"], result["hits"], result["misses"], result["total_nfe"]) == (2, 0, 2, 20)
    assert result["same_task_hits"] == result["cross_task_hits"] == 0
    assert result["hit_similarity"] == empty["hit_similarity"]
    assert result["decisions"][0]["source_case_id"] is None
    assert result["decisions"][0]["similarity"] == .94


@pytest.mark.parametrize("mode,expected_nfe", [("cache2", 22), ("cache0", 10)])
def test_miss_resets_run_and_cross_task_uses_suite(mode, expected_nfe, sources):
    entries = ["dev_a", "dev_b", "dev_a"]
    sequence = [(0, "dev_a"), (0, "dev_a"), None,
                (0, "dev_a"), (0, "dev_a"), (1, "dev_b"), (2, "dev_a")]
    calls = [call(i, value, mode=mode, similarity=.95 + i * .005 if value else .2)
             for i, value in enumerate(sequence)]
    result = analysis.analyze_episode(case("test", method=mode), calls, sources, entries)
    assert (result["queries"], result["hits"], result["total_nfe"]) == (7, 6, expected_nfe)
    assert (result["same_task_hits"], result["cross_task_hits"]) == (5, 1)
    assert result["unique_cache_indices"] == [0, 1, 2]
    assert result["unique_cache_index_count"] == 3
    assert result["longest_same_index_hit_run"] == 2
    assert result["hit_similarity"] == pytest.approx({"min": .95, "p50": .9675, "max": .98})
    assert result["decisions"][5] == {
        "call_index": 5, "source_case_id": "dev_b", "retrieved_index": 1,
        "similarity": .975, "hit": True, "nfe": 2 if mode == "cache2" else 0,
        "source_task_relation": "cross_task"}


@pytest.mark.parametrize("change,match", [
    ({"retrieved_index": 1}, "index"),
    ({"retrieved_index": True}, "index"),
    ({"retrieved_case_id": "dev_b"}, "source"),
    ({"similarity": float("nan")}, "similarity"),
    ({"call_index": 2}, "call_index"),
])
def test_rejects_malformed_hit_evidence(change, match, sources):
    row = {**call(0, (0, "dev_a"), similarity=.99), **change}
    with pytest.raises(analysis.IntegrityError, match=match):
        analysis.analyze_episode(case("test"), [row], sources, ["dev_a"])


def synthetic_study(sources, *, complete):
    heldout = [case("passed"), case("failed", "libero_goal", method="cache0"),
               case("not_yet", method="cache2")]
    records = {}
    for current in heldout[:3 if complete else 2]:
        records[current["id"]] = {
            "case": current,
            "record": {"success": current["id"] == "passed", "steps": 10},
            "calls": [call(0, (0, "dev_a"), mode=current["method"], similarity=.96)],
            "receipt_sha256": "b" * 64}
    cases = [*sources.values(), *heldout, case("baseline", method="full10")]
    return {"protocol": {"fingerprint": "a" * 64, "spec": {"cases": cases}},
            "complete": complete, "records": records,
            "statuses": {c["id"]: "complete" if c["id"] in records else "missing" for c in cases},
            "evidence_sha256": {"protocol.json": "c" * 64,
                                "episodes/failed/receipt.json": "b" * 64},
            "binding": {"cache2": {"entries": 1}, "cache0": {"entries": 1}},
            "pairing_audit": {"synthetic": True}}


def test_summary_keeps_failed_and_unfinished_planned_cache_cases(sources):
    study = synthetic_study(sources, complete=False)
    result = analysis.summarize(study, {"cache2": ["dev_a"], "cache0": ["dev_a"]})
    assert result["status"] == "PARTIAL_DESCRIPTIVE_ONLY"
    assert result["planned_cache_episodes"] == 3
    assert result["completed_cache_episodes"] == 2
    assert [row["id"] for row in result["episodes"]] == ["passed", "failed", "not_yet"]
    assert result["episodes"][1]["success"] is False
    assert result["episodes"][1]["cross_task_hits"] == 1
    assert result["episodes"][2]["status"] == "missing"
    assert result["episodes"][2]["queries"] is None  # unknown is not zero queries
    assert result["episodes"][2]["decisions"] == []
    assert result["evidence_sha256"] == study["evidence_sha256"]
    assert result["protocol_fingerprint"] == "a" * 64
    assert result["analysis_source_sha256"] == analysis.sha(Path(analysis.__file__))


def test_build_forwards_complete_default_and_writes_only_separate_exports(tmp_path, monkeypatch, sources):
    study = synthetic_study(sources, complete=True)
    forwarded = []

    def fake_load(directory, *, partial, root):
        forwarded.append((directory, partial, root))
        return study

    monkeypatch.setattr(analysis, "load_study", fake_load)
    monkeypatch.setattr(analysis, "_entry_sources", lambda directory, data: {
        "cache2": ["dev_a"], "cache0": ["dev_a"]})
    sentinel = tmp_path / "report.json"
    sentinel.write_text("unchanged", encoding="utf-8")
    path = analysis.build_analysis(tmp_path)
    assert forwarded == [(tmp_path, False, analysis.ROOT)]
    result = json.loads(path.read_text(encoding="utf-8"))
    csv_path = tmp_path / "retrieval_analysis.csv"
    assert result["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert result["status"] == "COMPLETE_DESCRIPTIVE_ONLY"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and rows[1]["success"] == "False"
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert {p.name for p in tmp_path.iterdir()} == {
        "report.json", "retrieval_analysis.json", "retrieval_analysis.csv"}
    with pytest.raises(FileExistsError):
        analysis.build_analysis(tmp_path)
    other = tmp_path / "partial"
    analysis.build_analysis(tmp_path, destination=other, partial=True)
    assert forwarded[-1][1] is True


def test_cache_index_mapping_is_bound_to_evidence_hash(tmp_path, sources):
    study = synthetic_study(sources, complete=True)
    cache_dir = tmp_path / "frozen_cache"
    cache_dir.mkdir()
    for mode in ("cache2", "cache0"):
        path = cache_dir / f"{mode}.json"
        path.write_text(json.dumps({"entry_case_ids": ["dev_a"]}), encoding="utf-8")
        study["evidence_sha256"][f"frozen_cache/{mode}.json"] = analysis.sha(path)
    assert analysis._entry_sources(tmp_path, study) == {"cache2": ["dev_a"], "cache0": ["dev_a"]}
    (cache_dir / "cache2.json").write_text('{"entry_case_ids": ["dev_b"]}', encoding="utf-8")
    with pytest.raises(analysis.IntegrityError, match="changed"):
        analysis._entry_sources(tmp_path, study)


def test_import_does_not_load_models_or_simulators():
    script = (
        "import importlib.util, sys; "
        "spec=importlib.util.spec_from_file_location('retrieval_import_check', sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert not any(n.split('.')[0] in "
        "{'torch','lerobot','transformers','libero','robosuite','mujoco','matplotlib'} for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script, str(Path(analysis.__file__))],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def analysis_export(tmp_path, monkeypatch, sources):
    study = synthetic_study(sources, complete=True)
    forwarded = []

    def fake_load(directory, *, partial, root):
        forwarded.append((directory, partial, root))
        if not study["complete"] and not partial:
            raise analysis.IntegrityError("Study incomplete")
        return study

    monkeypatch.setattr(analysis, "load_study", fake_load)
    monkeypatch.setattr(analysis, "_entry_sources", lambda directory, data: {
        "cache2": ["dev_a"], "cache0": ["dev_a"]})
    analysis.build_analysis(tmp_path)
    return tmp_path, study, forwarded


def test_validation_is_read_only_and_ignores_only_generation_time(analysis_export):
    directory, _, forwarded = analysis_export
    path = directory / "retrieval_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["generated_at"] = "another generation time"
    path.write_text(json.dumps(data), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert analysis.validate_analysis(directory) == data
    assert forwarded[-1] == (directory, False, analysis.ROOT)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    elsewhere = directory / "separate"
    analysis.build_analysis(directory, destination=elsewhere)
    assert analysis.validate_analysis(directory, analysis_dir=elsewhere)["study_complete"] is True


@pytest.mark.parametrize("damage", ["missing_json", "missing_csv", "changed_decision", "changed_hit_type", "changed_csv"])
def test_validation_rejects_missing_or_modified_exports(analysis_export, damage):
    directory, _, _ = analysis_export
    json_path, csv_path = directory / "retrieval_analysis.json", directory / "retrieval_analysis.csv"
    if damage.startswith("missing"):
        (json_path if damage.endswith("json") else csv_path).unlink()
    else:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if damage == "changed_decision":
            data["episodes"][0]["decisions"][0]["similarity"] = .97
        elif damage == "changed_hit_type":
            data["episodes"][0]["decisions"][0]["hit"] = 1
        else:
            csv_path.write_bytes(csv_path.read_bytes() + b"modified\n")
            # Even a matching self-reported checksum cannot replace recomputation.
            data["csv_sha256"] = analysis.sha(csv_path)
        json_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(analysis.IntegrityError):
        analysis.validate_analysis(directory)


def test_validation_rejects_stale_evidence_and_implementation(analysis_export):
    directory, study, _ = analysis_export
    study["evidence_sha256"]["episodes/failed/receipt.json"] = "d" * 64
    with pytest.raises(analysis.IntegrityError, match="stale"):
        analysis.validate_analysis(directory)
    study["evidence_sha256"]["episodes/failed/receipt.json"] = "b" * 64
    path = directory / "retrieval_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["analysis_source_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(analysis.IntegrityError, match="stale"):
        analysis.validate_analysis(directory)


def test_validation_requires_complete_by_default_and_matching_partial_choice(analysis_export):
    directory, study, forwarded = analysis_export
    study["complete"] = False
    partial_dir = directory / "partial_snapshot"
    analysis.build_analysis(directory, partial=True, destination=partial_dir)
    with pytest.raises(analysis.IntegrityError, match="Study incomplete"):
        analysis.validate_analysis(directory, analysis_dir=partial_dir)
    assert forwarded[-1][1] is False
    assert analysis.validate_analysis(directory, analysis_dir=partial_dir, partial=True)["partial_requested"] is True
    study["complete"] = True
    complete_partial_dir = directory / "complete_but_partial_requested"
    analysis.build_analysis(directory, partial=True, destination=complete_partial_dir)
    with pytest.raises(analysis.IntegrityError, match="partial_requested"):
        analysis.validate_analysis(directory, analysis_dir=complete_partial_dir)
    assert analysis.validate_analysis(directory, analysis_dir=complete_partial_dir, partial=True)["study_complete"] is True
