"""Synthetic sealed evidence only; no actual model or environment execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from embodied_research.action_cache import (
    CacheProvenance, FrozenActionCache, ProjectionConfig, SparseTernaryProjection,
)
from embodied_research.experiment import digest_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reporter = _module("cache_study_report_under_test", "summarize_cache_study.py")
runner = _module("cache_study_design_for_report_fixture", "run_cache_study.py")


def _seal(directory):
    write_json(directory / "receipt.json", {"files": {
        p.name: reporter.sha(p) for p in directory.iterdir() if p.is_file() and p.name != "receipt.json"
    }})


@pytest.fixture
def evidence(tmp_path):
    project, output = tmp_path / "project", tmp_path / "pilot"
    project.mkdir()
    output.mkdir()
    source = project / "fixture_source.py"
    source.write_text("# synthetic fixture source; never imported\n", encoding="utf-8")
    protocol = {"spec": runner.study_spec(), "sources": {"fixture_source.py": reporter.sha(source)},
                "runtime_versions": {"fixture": "1"},
                "artifacts": {"models": {"smolvla": {"revision": "synthetic-revision", "files": {
                    "model.safetensors": "a" * 64}}}}}
    protocol["fingerprint"] = digest_json(protocol)
    write_json(output / "protocol.json", protocol)
    cases = protocol["spec"]["cases"]
    dev_ids = tuple(c["id"] for c in cases if c["split"] == "development")
    admitted = []
    for case in cases:
        directory = output / "episodes" / case["id"]
        directory.mkdir(parents=True)
        method, suite, initial = case["method"], case["suite"], case["init_state_id"]
        if case["split"] == "development":
            success = suite != runner.SUITES[-1]
        elif method == "cache2":
            success = not (suite == runner.SUITES[0] and initial == 29)
        elif method == "noise2":
            success = suite != runner.SUITES[0]
        elif method == "cache0":
            success = suite == runner.SUITES[-1]
        else:
            success = True
        if case["split"] == "development" and success:
            admitted.append(case["id"])
        steps = 21 if success else 11
        calls = []
        for index, step in enumerate(range(0, steps, 10)):
            hit = index != 1 if method.startswith("cache") else None
            nfe = ((2 if method == "cache2" else 0) if hit else 10) if hit is not None else (2 if method == "noise2" else 10)
            inference = 6.2 + nfe * .4
            call = {"call_index": index, "observation_count": step + 1, "mode": method,
                    "nfe": nfe, "hit": hit, "similarity": 1. if hit else -.2 if hit is False else None,
                    "retrieved_case_id": dev_ids[0] if hit else None,
                    "noise_sha256": hashlib.sha256(f"noise-{index}".encode()).hexdigest(),
                    "vlm_seconds": 6., "vision_prefix_seconds": 2., "vlm_transformer_seconds": 4.,
                    "action_head_seconds": nfe * .4, "projection_lookup_seconds": .1 if hit is not None else 0.,
                    "postprocess_seconds": .3, "preprocess_seconds": .2,
                    "inference_seconds": inference, "prediction_seconds": inference + .5,
                    "end_to_end_seconds": inference + .7, "amortized_end_to_end_seconds": inference + 2.5}
            calls.append(call)
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / "initial.png")
        offset = {"full10": 0., "cache2": .1, "noise2": .2, "cache0": .3}[method]
        actions = np.full((steps, 7), offset, dtype=np.float32)
        np.savez_compressed(directory / "trajectory.npz", actions=actions,
                            states=np.zeros((steps, 8), dtype=np.float32),
                            replan_steps=np.arange(0, steps, 10))
        wall = 100. if method == "full10" else 80. if success else 3.
        record = {**case, "status": "complete", "protocol_fingerprint": protocol["fingerprint"],
                  "success": success, "steps": steps, "horizon": 21, "inference_calls": len(calls),
                  "wall_seconds": wall, "policy_inference_seconds": sum(c["inference_seconds"] for c in calls),
                  "preprocess_seconds": steps * .2, "initial_robot_state": [0.] * 8,
                  "initial_image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                  "model_info": {"name": "smolvla", "revision": "synthetic-revision", "model_sha256": "a" * 64,
                                 "parameters": 123, "device": "cpu", "dtype": "float32",
                                 "cache_policy": {"mode": method, "call_log": calls,
                                                  "total_nfe": sum(c["nfe"] for c in calls)}}}
        write_json(directory / "calls.json", calls)
        write_json(directory / "result.json", record)
        if case["split"] == "development":
            _seal(directory)
    binding = {"protocol_fingerprint": protocol["fingerprint"], "admitted_successful_case_ids": admitted,
               "test_admission": False, "development_receipts": {
                   case_id: reporter.sha(output / "episodes" / case_id / "receipt.json") for case_id in dev_ids}}
    for mode, shape, native_time in (("cache2", (50, 32), .2), ("cache0", (50, 7), 0.)):
        projection = SparseTernaryProjection(ProjectionConfig(4, output_dim=500, density=.01, seed=20261014))
        provenance = CacheProvenance("synthetic-revision", "a" * 64, {"input_dim": 4}, native_time, dev_ids)
        cache = FrozenActionCache(projection, shape, provenance, threshold=.95, capacity=3000)
        for case_id in admitted:
            count = reporter.read_json(output / "episodes" / case_id / "result.json")["inference_calls"]
            for _ in range(count):
                cache.add(np.arange(4, dtype=np.float32), np.zeros(shape, dtype=np.float32), case_id=case_id)
        cache.freeze().save(output / "frozen_cache" / mode)
        binding[mode] = {"cache_sha256": cache.cache_sha256, "provenance_sha256": cache.provenance_sha256,
                         "entries": len(cache)}
    write_json(output / "frozen_cache/binding.json", binding)
    for case in cases:
        if case["split"] != "heldout":
            continue
        directory = output / "episodes" / case["id"]
        record = reporter.read_json(directory / "result.json")
        record["cache_binding_sha256"] = digest_json(binding)
        if case["method"] in ("cache2", "cache0"):
            record["model_info"]["cache_policy"]["cache"] = {
                "cache_sha256": binding[case["method"]]["cache_sha256"]}
        write_json(directory / "result.json", record)
        _seal(directory)
    return project, output, cases


def _load(evidence, partial=False):
    project, output, _ = evidence
    return reporter.load_study(output, partial=partial, root=project)


def _case(evidence, method="cache2", initial=29, suite=None):
    return next(c for c in evidence[2] if c["method"] == method and c["init_state_id"] == initial
                and (suite is None or c["suite"] == suite))


def _mutate_record(evidence, case, change, *, reseal=True):
    directory = evidence[1] / "episodes" / case["id"]
    record = reporter.read_json(directory / "result.json")
    change(record)
    write_json(directory / "result.json", record)
    if reseal:
        _seal(directory)
    return directory


def test_full_report_preserves_denominators_budget_categories_and_cluster_estimate(evidence):
    study = _load(evidence)
    summary = reporter.summarize(study)
    assert summary["status"] == "complete"
    assert summary["counts"] == {"planned": 28, "complete": 28, "missing": 0, "unsealed": 0, "error": 0}
    primary = summary["comparisons"][0]
    assert primary["completed_pairs"] == primary["planned_pairs"] == 8
    assert primary["observed_task_clusters"] == 4
    assert primary["macro_success_difference"] == -.125
    assert primary["success_task_cluster_95ci"] == [-.375, 0.]
    assert primary["both_success_pairs"] == 7
    assert primary["mean_wall_difference_both_success_only"] == -20.
    assert any(p["candidate_wall_seconds"] == 3 and p["wall_difference_both_success_only"] is None
               for p in primary["per_pair"])
    groups = {(g["scope"], g["method"]): g for g in summary["groups"]}
    cached = groups["primary", "cache2"]
    assert cached["successes"] == 7 and cached["planned_episodes"] == 8
    assert cached["replans"] == 23 and cached["cache_hits"] == 15 and cached["fallback_calls"] == 8
    assert cached["full_sampling_calls"] == 8 and cached["refinement_calls"] == 15
    assert cached["mean_nfe_per_replan"] == (8*10 + 15*2)/23
    assert all(g["planned_episodes"] == 4 for g in summary["groups"] if g["scope"] == "ablation_init11")
    assert all(c["completed_pairs"] == 4 for c in summary["comparisons"][1:])
    assert summary["pairing_audit"]["paired_initial_blocks"] == 8
    assert summary["model_details_from_records"][0]["parameters"] == 123  # not a hardcoded real-model count
    assert len(summary["episodes"]) == 28  # repeated comparison panels do not duplicate episode evidence
    for pair in primary["per_pair"]:
        assert pair["first_action_steps_compared"] == 10
        assert pair["first_actions_rmse_all_dims"] == pytest.approx(.1)


def test_zero_width_interval_is_descriptive_and_single_task_uncertainty_is_unavailable():
    assert reporter._cluster_ci([0., 0., 0., 0.]) == [0., 0.]
    assert reporter._cluster_ci([0.]) is None


def test_short_success_keeps_outcome_but_has_missing_first_ten_diagnostic(evidence, tmp_path):
    case = _case(evidence, initial=11)
    directory = evidence[1] / "episodes" / case["id"]
    record = reporter.read_json(directory / "result.json")
    assert record["success"] is True
    calls = record["model_info"]["cache_policy"]["call_log"][:1]
    record.update(steps=5, inference_calls=1, preprocess_seconds=1.,
                  policy_inference_seconds=calls[0]["inference_seconds"])
    record["model_info"]["cache_policy"].update(call_log=calls, total_nfe=calls[0]["nfe"])
    # The short trajectory has a much larger distance; including it would
    # visibly contaminate the diagnostic mean while its true outcome is success.
    np.savez_compressed(directory / "trajectory.npz", actions=np.full((5, 7), 9., dtype=np.float32),
                        states=np.zeros((5, 8), dtype=np.float32), replan_steps=np.array([0]))
    write_json(directory / "calls.json", calls)
    write_json(directory / "result.json", record)
    _seal(directory)
    summary = reporter.summarize(_load(evidence))
    primary = summary["comparisons"][0]
    pair = next(row for row in primary["per_pair"] if row["candidate_id"] == case["id"])
    assert pair["candidate_success"] is True and pair["success_difference"] == 0
    assert pair["actual_available_steps"] == 5 and pair["first_action_steps_compared"] == 0
    assert pair["first_10_complete"] is False and "candidate=5" in pair["first_10_missing_reason"]
    for field in ("first_actions_rmse_per_dim", "first_actions_mae_per_dim", "first_actions_rmse_all_dims",
                  "first_actions_max_abs_difference"):
        assert pair[field] is None
    assert primary["completed_pairs"] == 8 and primary["macro_success_difference"] == -.125
    assert primary["first_10_complete_pairs"] == 7 and primary["first_10_missing_pairs"] == 1
    assert primary["mean_first_actions_rmse_all_dims"] == pytest.approx(.1)
    cached = next(g for g in summary["groups"] if (g["scope"], g["method"]) == ("primary", "cache2"))
    assert cached["successes"] == 7 and cached["completed_episodes"] == 8
    markdown = reporter._markdown(summary)
    assert "少于10步" in markdown and "7/8 | 0.100000" in markdown
    path = tmp_path / "short_pair.csv"
    reporter._csv(path, [pair])
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["first_actions_rmse_all_dims"] == "" and row["actual_available_steps"] == "5"


def test_empty_partial_study_has_no_invented_success_or_cost(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("fixture", encoding="utf-8")
    protocol = {"spec": runner.study_spec(), "sources": {"source.py": reporter.sha(source)},
                "runtime_versions": {}, "artifacts": {"models": {}}}
    protocol["fingerprint"] = digest_json(protocol)
    write_json(tmp_path / "protocol.json", protocol)
    summary = reporter.summarize(reporter.load_study(tmp_path, root=tmp_path, partial=True))
    assert summary["counts"]["planned"] == summary["counts"]["missing"] == 28
    assert summary["status"] == "PARTIAL"
    assert all(group["observed_macro_task_success"] is None and group["mean_nfe_per_replan"] is None
               for group in summary["groups"])
    assert summary["comparisons"][0]["macro_success_difference"] is None


@pytest.mark.parametrize("state", ["missing", "error", "unsealed"])
def test_incomplete_studies_refuse_default_and_keep_planned_denominator_when_partial(evidence, state):
    case = _case(evidence)
    directory = evidence[1] / "episodes" / case["id"]
    if state == "missing":
        # Only fixture-owned files; no project/output evidence is touched.
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    elif state == "error":
        write_json(directory / "result.json", {**case, "status": "error",
                   "protocol_fingerprint": reporter.read_json(evidence[1] / "protocol.json")["fingerprint"]})
        (directory / "receipt.json").unlink()
    else:
        (directory / "receipt.json").unlink()
    with pytest.raises(reporter.IntegrityError, match="incomplete"):
        _load(evidence)
    summary = reporter.summarize(_load(evidence, partial=True))
    assert summary["status"] == "PARTIAL" and summary["counts"][state] == 1
    assert summary["comparisons"][0]["planned_pairs"] == 8
    assert summary["comparisons"][0]["completed_pairs"] == 7
    assert not summary["comparisons"][0]["complete"]
    assert "PARTIAL" in reporter._markdown(summary)


@pytest.mark.parametrize("kind", ["source", "protocol", "receipt", "binding", "model", "noise", "proprio", "image", "nfe"])
def test_tampering_or_pairing_mismatch_fails_even_in_partial_mode(evidence, kind):
    case = _case(evidence)
    directory = evidence[1] / "episodes" / case["id"]
    if kind == "source":
        (evidence[0] / "fixture_source.py").write_text("changed", encoding="utf-8")
    elif kind == "protocol":
        path = evidence[1] / "protocol.json"
        protocol = reporter.read_json(path)
        protocol["spec"]["cache"]["cosine_threshold"] = .1
        write_json(path, protocol)
    elif kind == "receipt":
        _mutate_record(evidence, case, lambda r: r.update(wall_seconds=1.), reseal=False)
    elif kind == "binding":
        _mutate_record(evidence, case, lambda r: r.update(cache_binding_sha256="b" * 64))
    elif kind == "model":
        _mutate_record(evidence, case, lambda r: r["model_info"].update(parameters=123, model_sha256="b" * 64))
    elif kind in ("noise", "nfe"):
        record = reporter.read_json(directory / "result.json")
        calls = record["model_info"]["cache_policy"]["call_log"]
        calls[0]["noise_sha256" if kind == "noise" else "nfe"] = "b" * 64 if kind == "noise" else 9
        write_json(directory / "calls.json", calls)
        write_json(directory / "result.json", record)
        _seal(directory)
    elif kind == "proprio":
        _mutate_record(evidence, case, lambda r: r.update(initial_robot_state=[1.] * 8))
    elif kind == "image":
        _mutate_record(evidence, case, lambda r: r.update(initial_image_sha256="b" * 64))
    with pytest.raises(reporter.IntegrityError):
        _load(evidence, partial=True)


def test_artifacts_are_written_without_mutating_sealed_raw_files(evidence):
    project, output, _ = evidence
    before = {path.relative_to(output).as_posix(): reporter.sha(path) for path in output.rglob("*") if path.is_file()}
    path = reporter.build_report(output, root=project)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert reporter.validate_report(output, root=project)["status"] == "complete"
    exports = reporter.read_json(output / "exports.json")
    assert set(exports["files"]) == set(reporter.EXPORT_FILES)
    assert "exports.json" not in exports["files"]
    assert exports["files"]["report.json"] == reporter.sha(path)
    assert report["evidence_sha256"] == before
    assert all(reporter.sha(output / name) == digest for name, digest in before.items())
    for name in ("report.md", "groups.csv", "episodes.csv", "pairs.csv", "calls.csv"):
        assert (output / name).is_file()
    for stem in ("cache_primary_success_nfe", "cache_ablation_success_nfe", "cache_latency_components"):
        assert (output / "figures" / f"{stem}.png").stat().st_size > 1000
        assert (output / "figures" / f"{stem}.svg").is_file()
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "失败更早结束不是加速" in markdown and "不能证明等价" in markdown
    assert "不是质量" in markdown and "未经调优" in markdown
    assert str(project) not in path.read_text(encoding="utf-8")
    report["comparisons"][0]["macro_success_difference"] = 1.
    write_json(path, report)
    # Even a re-sealed manipulated summary must fail deterministic recompute.
    exports["files"]["report.json"] = reporter.sha(path)
    write_json(output / "exports.json", exports)
    with pytest.raises(reporter.IntegrityError, match="stale"):
        reporter.validate_report(output, root=project)


def test_export_binding_rejects_deleted_modified_and_stale_files_without_plot_import(evidence):
    project, output, _ = evidence
    reporter.build_report(output, root=project)
    # A fresh validation process must not load the plotting or model stacks.
    script = (
        "import importlib.util, pathlib, sys; "
        "spec=importlib.util.spec_from_file_location('verify_exports', sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.validate_report(pathlib.Path(sys.argv[2]), root=pathlib.Path(sys.argv[3])); "
        "assert not any(n.split('.')[0] in {'matplotlib','torch','lerobot','transformers'} for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script, str(Path(reporter.__file__)), str(output), str(project)],
                   check=True, capture_output=True, text=True)
    image = output / "figures/cache_primary_success_nfe.png"
    saved_image = image.read_bytes()
    image.unlink()
    with pytest.raises(reporter.IntegrityError, match="Missing.*export"):
        reporter.validate_report(output, root=project)
    image.write_bytes(saved_image + b"modified")
    with pytest.raises(reporter.IntegrityError, match="export hash mismatch"):
        reporter.validate_report(output, root=project)
    image.write_bytes(saved_image)

    report_path, exports_path = output / "report.json", output / "exports.json"
    saved_report, saved_exports = report_path.read_bytes(), exports_path.read_bytes()
    report = reporter.read_json(report_path)
    # generated_at is intentionally ignored by statistical recompute, but the
    # export seal must still bind this exact generation's JSON bytes.
    report["generated_at"] = "stale report from another generation"
    write_json(report_path, report)
    with pytest.raises(reporter.IntegrityError, match="export hash mismatch: report.json"):
        reporter.validate_report(output, root=project)
    report_path.write_bytes(saved_report)
    exports = reporter.read_json(exports_path)
    exports["files"]["report.json"] = "0" * 64
    write_json(exports_path, exports)
    with pytest.raises(reporter.IntegrityError, match="export hash mismatch: report.json"):
        reporter.validate_report(output, root=project)
    exports_path.write_bytes(saved_exports)
    exports = reporter.read_json(exports_path)
    exports["files"].pop("figures/cache_primary_success_nfe.png")
    write_json(exports_path, exports)
    with pytest.raises(reporter.IntegrityError, match="whitelist"):
        reporter.validate_report(output, root=project)
    exports_path.write_bytes(saved_exports)

    markdown_path = output / "report.md"
    saved_markdown = markdown_path.read_bytes()
    markdown_path.write_text(saved_markdown.decode("utf-8") + "\n![missing](figures/missing.png)\n", encoding="utf-8")
    exports = reporter.read_json(exports_path)
    exports["files"]["report.md"] = reporter.sha(markdown_path)
    write_json(exports_path, exports)
    with pytest.raises(reporter.IntegrityError, match="Markdown resource"):
        reporter.validate_report(output, root=project)
    markdown_path.write_bytes(saved_markdown)
    exports_path.write_bytes(saved_exports)
    assert reporter.validate_report(output, root=project)["status"] == "complete"
