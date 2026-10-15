import copy
import hashlib
import json

import numpy as np
import pytest

from embodied_research import reporting
from embodied_research.statistics import (
    ExperimentIntegrityError,
    ablation_comparisons,
    aggregate_groups,
    compare_flow_steps,
    compare_observation_delay,
    load_experiment,
    paired_comparisons,
    summarize_experiment,
    task_cluster_interval,
    validate_initial_parity,
)


def episode(identifier, *, task_id=0, init_state_id=0, seed=1, method="fixed8", success=True, suite="spatial"):
    return {
        "id": identifier, "model": "minerva", "suite": suite, "task_id": task_id,
        "task": f"task {task_id}", "init_state_id": init_state_id, "seed": seed,
        "condition": "nominal", "method": method, "status": "complete", "success": success,
        "steps": 20, "horizon": 520, "inference_calls": 3, "policy_inference_seconds": 1.5,
        "preprocess_seconds": 0.1, "scheduler_seconds": 0.02, "simulator_seconds": 0.5,
        "wall_seconds": 2.2, "model_info": {"checkpoint_sha256": "frozen"},
        "scheduler_config": {"fixed_interval": 8},
    }


def write_experiment(root, records, *, cases=None, extra_protocol=None):
    cases = copy.deepcopy(records if cases is None else cases)
    protocol = {"cases": cases, "question": "Budget/quality?", "split": "held-out init states"}
    protocol.update(extra_protocol or {})
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for row in records:
        folder = root / "episodes" / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")
    return protocol


def test_task_macro_gives_equal_weights_despite_unequal_episode_counts():
    records = [episode(f"a{i}", init_state_id=i, success=True) for i in range(9)]
    records.append(episode("b", task_id=1, success=False))
    row = aggregate_groups(records)[0]
    assert row["macro_task_success"] == 0.5
    assert row["episode_success"] == 0.9
    assert row["episodes"] == 10
    assert row["successes"] == 9
    assert row["failures"] == 1
    assert row["tasks"] == 2
    assert row["mean_inference_calls"] == 3
    assert row["mean_model_seconds"] == 1.5
    assert row["inference_calls_per_100_steps"] == 15
    assert row["total_wall_seconds"] == pytest.approx(22)


def test_suite_and_task_id_jointly_define_task_cluster():
    rows = [episode("a", suite="spatial"), episode("b", suite="object", success=False)]
    result = aggregate_groups(rows)[0]
    assert result["tasks"] == 2
    assert result["macro_task_success"] == 0.5


def test_paired_difference_uses_matched_init_and_seed_then_macro_tasks():
    rows = []
    for task in range(2):
        for state in range(3 if task == 0 else 1):
            rows.extend([
                episode(f"b{task}-{state}", task_id=task, init_state_id=state, success=task == 1),
                episode(f"c{task}-{state}", task_id=task, init_state_id=state,
                        method="adaptive", success=task == 0),
            ])
    rows.append(episode("unmatched", task_id=0, init_state_id=0, seed=999, method="adaptive"))
    result = paired_comparisons(rows)[0]
    assert result["paired_tasks"] == 2
    assert result["paired_episodes"] == 4
    assert result["macro_success_difference"] == 0
    assert result["candidate_wins"] == 3
    assert result["baseline_wins"] == 1
    assert result["ties"] == 0
    assert result["unmatched_candidate"] == 1
    assert result["unmatched_baseline"] == 0
    assert result["task_cluster_95ci"] == [-1, 1]


def test_comparisons_include_all_candidates_and_never_select_best():
    rows = [episode("base", success=True)]
    for method, success in (("fixed1", False), ("fixed4", True), ("fixed15", False), ("adaptive", False)):
        row = episode(method, method=method, success=success)
        row["inference_calls"] = 1
        rows.append(row)
    results = paired_comparisons(rows)
    assert len(results) == 4
    assert {row["method"] for row in results} == {"fixed1", "fixed4", "fixed15", "adaptive"}
    for row in results:
        assert row["task_cluster_95ci"] is None
        assert row["small_task_count"]
        assert row["inference_call_reduction"] == pytest.approx(2 / 3)


def test_unmatched_groups_do_not_invent_zero_difference():
    rows = [episode("base"), episode("candidate", method="adaptive", task_id=2)]
    result = paired_comparisons(rows)[0]
    assert result["paired_episodes"] == 0
    assert result["macro_success_difference"] is None
    assert result["candidate_macro_success"] is None
    assert result["inference_call_reduction"] is None


def test_bootstrap_is_task_level_deterministic_and_honest_with_one_task():
    first = task_cluster_interval([-1, -0.5, 0, 0.5, 1])
    assert first == task_cluster_interval([-1, -0.5, 0, 0.5, 1])
    assert first[0] < 0 < first[1]
    assert task_cluster_interval([0.5]) is None
    assert task_cluster_interval([]) is None
    assert task_cluster_interval([0, 0, 0]) == (0, 0)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [1.1], [[0, 1]]])
def test_bootstrap_rejects_non_rates(values):
    with pytest.raises(ValueError):
        task_cluster_interval(values)


def test_complete_corpus_is_read_without_modifying_raw_files(tmp_path):
    records = [episode("base"), episode("candidate", method="adaptive", success=False)]
    write_experiment(tmp_path, records)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    data = load_experiment(tmp_path)
    result = summarize_experiment(data)
    assert result["completed_episodes"] == result["planned_episodes"] == 2
    assert result["successes"] == 1
    assert result["partial"] is False
    assert len(result["input_sha256"]) == 3
    for path, contents in before.items():
        assert path.read_bytes() == contents
        assert result["input_sha256"][path.relative_to(tmp_path).as_posix()] == hashlib.sha256(contents).hexdigest()
    json.dumps(result, allow_nan=False)


def test_missing_and_error_rejected_for_full_report_but_visible_in_partial(tmp_path):
    good = episode("good")
    error = episode("error", method="adaptive")
    error["status"] = "error"
    missing = episode("missing", method="fixed4")
    write_experiment(tmp_path, [good, error], cases=[good, error, missing])
    with pytest.raises(ExperimentIntegrityError, match="1 missing, 1 error"):
        load_experiment(tmp_path)
    result = summarize_experiment(load_experiment(tmp_path, partial=True))
    assert result["completed_episodes"] == 1
    assert result["missing_ids"] == ["missing"]
    assert result["error_ids"] == ["error"]
    assert result["successes"] == 1  # Does not count error.success=True.
    assert result["partial"]
    assert any("部分" in warning for warning in result["warnings"])


@pytest.mark.parametrize("alias_id", [False, True])
def test_protocol_duplicates_rejected_even_in_partial_mode(tmp_path, alias_id):
    original = episode("a")
    duplicate = copy.deepcopy(original)
    if alias_id:
        duplicate["id"] = "b"
    write_experiment(tmp_path, [], cases=[original, duplicate])
    with pytest.raises(ExperimentIntegrityError, match="duplicate"):
        load_experiment(tmp_path, partial=True)


def test_unexpected_result_rejected(tmp_path):
    write_experiment(tmp_path, [episode("extra", method="adaptive")], cases=[episode("planned")])
    with pytest.raises(ExperimentIntegrityError, match="unexpected"):
        load_experiment(tmp_path, partial=True)


def test_identity_mismatch_rejected(tmp_path):
    expected = episode("a")
    actual = copy.deepcopy(expected)
    actual["seed"] = 999
    write_experiment(tmp_path, [actual], cases=[expected])
    with pytest.raises(ExperimentIntegrityError, match="identity differs"):
        load_experiment(tmp_path)


def test_frozen_protocol_fingerprint_mismatch_rejected(tmp_path):
    row = episode("a")
    row["protocol_fingerprint"] = "older"
    write_experiment(tmp_path, [row], extra_protocol={"fingerprint": "current"})
    with pytest.raises(ExperimentIntegrityError, match="provenance"):
        load_experiment(tmp_path)


def test_explicit_upstream_scheduler_null_is_valid(tmp_path):
    row = episode("a", method="upstream")
    row["scheduler"] = None
    row["scheduler_config"] = None
    write_experiment(tmp_path, [row])
    assert len(load_experiment(tmp_path).records) == 1


def test_changed_case_parameters_rejected_even_when_identity_matches(tmp_path):
    expected = episode("a")
    expected["num_steps"] = 10
    actual = copy.deepcopy(expected)
    actual["num_steps"] = 1
    write_experiment(tmp_path, [actual], cases=[expected])
    with pytest.raises(ExperimentIntegrityError, match="num_steps"):
        load_experiment(tmp_path)


def with_initial_evidence(row, *, image="a" * 64, state=None):
    row["initial_image_sha256"] = image
    row["initial_robot_state"] = [0.1] * 8 if state is None else state
    return row


def test_same_initial_image_and_allclose_proprio_are_audited_across_methods(tmp_path):
    base = with_initial_evidence(episode("base"))
    candidate = with_initial_evidence(episode("candidate", method="adaptive"), state=[0.10000005] * 8)
    write_experiment(tmp_path, [base, candidate])
    audit = summarize_experiment(load_experiment(tmp_path))["initial_state_parity"]
    assert audit["image_method_comparisons"] == 1
    assert audit["state_method_comparisons"] == 1
    assert audit["blocks_without_image_evidence"] == 0
    assert audit["blocks_without_state_evidence"] == 0


@pytest.mark.parametrize("field,value,expected", [
    ("initial_image_sha256", "b" * 64, "image"),
    ("initial_robot_state", [0.1001] * 8, "state"),
])
def test_real_initial_parity_mismatch_rejects_full_and_partial_report(tmp_path, field, value, expected):
    base = with_initial_evidence(episode("base"))
    candidate = with_initial_evidence(episode("candidate", method="adaptive"))
    candidate[field] = value
    write_experiment(tmp_path, [base, candidate])
    for partial in (False, True):
        with pytest.raises(ExperimentIntegrityError, match=f"initial {expected} parity mismatch"):
            load_experiment(tmp_path, partial=partial)


@pytest.mark.parametrize("field", ["initial_image_sha256", "initial_robot_state"])
def test_partial_missing_initial_evidence_cannot_silently_disable_check(tmp_path, field):
    base = with_initial_evidence(episode("base"))
    candidate = with_initial_evidence(episode("candidate", method="adaptive"))
    del candidate[field]
    write_experiment(tmp_path, [base, candidate])
    with pytest.raises(ExperimentIntegrityError, match="incomplete initial parity"):
        load_experiment(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("initial_image_sha256", "invalid"), ("initial_image_sha256", "x" * 64),
    ("initial_robot_state", [0.1] * 7), ("initial_robot_state", [True] * 8),
    ("initial_robot_state", [float("inf")] * 8),
])
def test_initial_evidence_validated_even_for_single_method(field, value):
    row = with_initial_evidence(episode("base"))
    row[field] = value
    with pytest.raises(ExperimentIntegrityError, match="invalid initial"):
        validate_initial_parity([row])


@pytest.mark.parametrize("field,value", [
    ("model", "smolvla"), ("condition", "delay4"), ("suite", "goal"), ("task_id", 1),
    ("init_state_id", 2), ("seed", 3),
])
def test_different_pairing_blocks_are_not_required_to_have_same_images_or_states(field, value):
    base = with_initial_evidence(episode("base"))
    candidate = with_initial_evidence(episode("candidate", method="adaptive"), image="b" * 64, state=[1] * 8)
    candidate[field] = value
    audit = validate_initial_parity([base, candidate])
    assert audit["blocks"] == 2
    assert audit["image_method_comparisons"] == audit["state_method_comparisons"] == 0


def test_legacy_minimal_records_report_missing_parity_evidence_instead_of_claiming_check():
    audit = validate_initial_parity([episode("base"), episode("candidate", method="adaptive")])
    assert audit["blocks"] == 1
    assert audit["image_checked_blocks"] == audit["state_checked_blocks"] == 0
    assert audit["blocks_without_image_evidence"] == audit["blocks_without_state_evidence"] == 1


def test_second_result_with_same_id_is_never_chosen_as_better_retry(tmp_path):
    row = episode("a", success=False)
    write_experiment(tmp_path, [row])
    duplicate = tmp_path / "episodes" / "z"
    duplicate.mkdir()
    row["success"] = True
    (duplicate / "result.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError, match="duplicate"):
        load_experiment(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("success", 1), ("steps", 521), ("steps", -1), ("horizon", 0), ("inference_calls", 21),
    ("policy_inference_seconds", -1), ("wall_seconds", "2.2"), ("model_info", []),
    ("scheduler_config", None), ("task", ""), ("status", "running"),
])
def test_invalid_completed_records_rejected(tmp_path, key, value):
    row = episode("a")
    row[key] = value
    write_experiment(tmp_path, [row])
    with pytest.raises(ExperimentIntegrityError):
        load_experiment(tmp_path)


def test_nonfinite_json_rejected_before_aggregation(tmp_path):
    row = episode("a")
    row["wall_seconds"] = np.nan
    write_experiment(tmp_path, [row])
    with pytest.raises(ExperimentIntegrityError, match="non-finite"):
        load_experiment(tmp_path)


def test_duplicate_json_keys_rejected(tmp_path):
    (tmp_path / "protocol.json").write_text('{"cases":[],"cases":[]}', encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError, match="duplicate JSON"):
        load_experiment(tmp_path)


@pytest.mark.parametrize("video", ["missing.gif", "../../outside.gif"])
def test_video_must_exist_inside_its_episode_folder(tmp_path, video):
    row = episode("a")
    row["video"] = video
    write_experiment(tmp_path, [row])
    (tmp_path / "outside.gif").write_bytes(b"GIF89a")
    with pytest.raises(ExperimentIntegrityError, match="video"):
        load_experiment(tmp_path)


def test_valid_video_reference_and_safe_html_with_raw_evidence(tmp_path, monkeypatch):
    row = episode("a")
    row["video"] = "rollout.gif"
    write_experiment(tmp_path, [row], extra_protocol={"question": "<script>alert('bad')</script>"})
    (tmp_path / "episodes" / "a" / "rollout.gif").write_bytes(b"GIF89a")
    monkeypatch.setattr(reporting, "_plot_figures", lambda root, summary: [])
    before = (tmp_path / "protocol.json").read_bytes()
    path = reporting.build_report(tmp_path)
    contents = path.read_text(encoding="utf-8")
    assert "<script>" not in contents
    assert "&lt;script&gt;" in contents
    assert 'href="episodes/a/result.json"' in contents
    assert 'src="episodes/a/rollout.gif"' in contents
    assert 'href="protocol.json"' in contents
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "protocol.json").read_bytes() == before
    assert "部分实验报告" not in contents


def test_partial_report_heading_cannot_masquerade_as_full_report(tmp_path, monkeypatch):
    write_experiment(tmp_path, [], cases=[episode("planned")])
    monkeypatch.setattr(reporting, "_plot_figures", lambda root, summary: [])
    with pytest.raises(ExperimentIntegrityError):
        reporting.build_report(tmp_path)
    path = reporting.build_report(tmp_path, partial=True)
    contents = path.read_text(encoding="utf-8")
    assert "部分实验报告 · 尚未完成" in contents
    assert "0 / 1" in contents
    assert "missing" in contents


def test_real_matplotlib_figures_are_exported_with_report_links(tmp_path):
    pytest.importorskip("matplotlib")
    rows = []
    for task in range(2):
        rows.extend([
            episode(f"base{task}", task_id=task, success=task == 0),
            episode(f"adaptive{task}", task_id=task, method="adaptive", success=task == 1),
        ])
    write_experiment(tmp_path, rows, extra_protocol={"plan": {"name": "paired-fixture-stage"}})
    report = reporting.build_report(tmp_path)
    contents = report.read_text(encoding="utf-8")
    for name in ("task_macro_success", "quality_budget", "paired_differences"):
        png = tmp_path / "figures" / f"{name}.png"
        svg = tmp_path / "figures" / f"{name}.svg"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        svg_text = svg.read_text(encoding="utf-8")
        assert "<svg " in svg_text
        assert "paired-fixture-stage" in svg_text
        assert "tasks=2" in svg_text
        assert "episodes=2" in svg_text
        assert png.stat().st_size > 1000
        assert f'figures/{name}.svg' in contents


def test_one_episode_smoke_figures_identify_stage_and_sample_size(tmp_path):
    write_experiment(tmp_path, [episode("smoke")], extra_protocol={"plan": {"name": "integration-smoke"}})
    reporting.build_report(tmp_path)
    for name in ("task_macro_success", "quality_budget"):
        text = (tmp_path / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        assert "integration-smoke" in text
        assert "tasks=1, episodes=1" in text
        assert "Observed benchmark performance" not in text
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "integration-smoke"


def test_budget_labels_avoid_nearby_points_deterministically_without_moving_data():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    coordinates = [(value, 100.0) for value in (8, 13.8, 14.7, 16.2, 21.2, 22, 28, 111)]
    labels = {coordinate: [str(index)] for index, coordinate in enumerate(coordinates, start=1)}
    placements = []
    for items in (list(labels.items()), list(reversed(list(labels.items())))):
        fig, ax = plt.subplots(figsize=(10, 6))
        points = ax.scatter(*zip(*coordinates))
        before = points.get_offsets().copy()
        ax.set_ylim(-4, 109)
        ax.margins(x=.2)
        annotations = reporting._place_budget_labels(ax, dict(items))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [annotation.get_bbox_patch().get_window_extent(renderer) for annotation in annotations]
        assert all(not left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1:])
        for box in boxes:
            assert all(not box.contains(*ax.transData.transform(point)) for point in coordinates)
        np.testing.assert_array_equal(points.get_offsets(), before)
        assert all(annotation.arrow_patch is not None for annotation in annotations)
        placements.append([annotation.get_position() for annotation in annotations])
        plt.close(fig)
    assert placements[0] == placements[1]
    assert len(set(placements[0])) > 1


def test_zero_width_pair_interval_is_not_presented_as_equivalence(tmp_path, monkeypatch):
    rows = [episode(f"{method}-{task}", task_id=task, method=method, success=True)
            for task in range(2) for method in ("fixed8", "adaptive")]
    write_experiment(tmp_path, rows)
    monkeypatch.setattr(reporting, "_plot_figures", lambda root, summary: [])
    report = reporting.build_report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired_comparisons"][0]["task_cluster_95ci"] == [0, 0]
    text = report.read_text(encoding="utf-8")
    assert "已观察任务的配对平均效应没有变异" in text
    assert "不证明两种方法等效或非劣效" in text
    assert "不表示真实不确定性为零" in text
    assert "4 / 4" in text


@pytest.mark.parametrize("all_missing", [False, True])
def test_task_heatmap_shows_all_planned_cells_with_missing_mask_and_initial_counts(tmp_path, monkeypatch, all_missing):
    from matplotlib.axes import Axes

    observed = [
        episode("base0seed1", task_id=0, init_state_id=0, seed=1, success=True),
        episode("base0seed2", task_id=0, init_state_id=0, seed=2, success=False),
        episode("base0init1", task_id=0, init_state_id=1, seed=1, success=True),
        episode("adaptive0", task_id=0, method="adaptive", success=False),
        episode("base1", task_id=1, success=False),
    ]
    missing = [episode(f"missing-{task}-{method}", task_id=task, method=method)
               for task, method in ((0, "fixed15"), (1, "fixed15"), (1, "adaptive"),
                                    (2, "fixed8"), (2, "fixed15"), (2, "adaptive"))]
    write_experiment(tmp_path, [] if all_missing else observed, cases=[*observed, *missing],
                     extra_protocol={"plan": {"name": "partial-heatmap-fixture"}})
    images = []
    original = Axes.imshow

    def capture(self, values, *args, **kwargs):
        artist = original(self, values, *args, **kwargs)
        images.append(artist)
        return artist

    monkeypatch.setattr(Axes, "imshow", capture)
    report = reporting.build_report(tmp_path, partial=True)
    assert len(images) == 1
    assert images[0].get_clim() == (0, 100)
    values = images[0].get_array()
    assert values.shape == (3, 3)  # Rows: all tasks 0/1/2; columns: adaptive/fixed15/fixed8.
    if all_missing:
        assert np.ma.getmaskarray(values).all()
    else:
        np.testing.assert_array_equal(np.ma.getmaskarray(values), [[False, True, False], [True, True, False], [True, True, True]])
        assert values[0, 2] == pytest.approx(200 / 3)
        assert values[0, 0] == values[1, 2] == 0  # Real failures remain observed, not missing.
    svg = (tmp_path / "figures/task_method_success.svg").read_text(encoding="utf-8")
    assert "PARTIAL | partial-heatmap-fixture" in svg
    assert "spatial:2" in svg
    assert "fixed15" in svg
    assert "no data" in svg
    assert "i=2, e=3" in svg if not all_missing else "i=2, e=3" not in svg
    assert (tmp_path / "figures/task_method_success.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert 'href="figures/task_method_success.svg"' in report.read_text(encoding="utf-8")


def flow_records():
    baseline, changed = [], []
    for index, suite in enumerate(("spatial", "object", "goal", "long")):
        for target, prefix, steps, method, success in (
            (baseline, "b", 10, "fixed15", index % 2 == 0),
            (changed, "c", 5, "fixed15_steps5", index % 2 == 1),
        ):
            row = with_initial_evidence(episode(f"{prefix}{index}", suite=suite, task_id=1,
                                               method=method, success=success))
            row.update(model="smolvla", num_steps=steps, environment={"controller": "fixed"})
            row["scheduler_config"] = {"mode": "fixed", "fixed_interval": 15, "temporal_ensemble": False}
            row["model_info"] = {"name": "smolvla", "model_sha256": "a" * 64, "revision": "frozen",
                                 "num_steps": steps, "load_seconds": float(steps)}
            target.append(row)
    return baseline, changed


def write_flow_stages(tmp_path, baseline, changed):
    left, right = tmp_path / "50_smolvla", tmp_path / "60_flow_steps"
    left.mkdir()
    right.mkdir()
    write_experiment(left, baseline)
    write_experiment(right, changed)
    return load_experiment(left), load_experiment(right)


def test_flow_steps_are_strictly_paired_across_stages_and_reported(tmp_path):
    rows = flow_records()
    baseline, candidate = write_flow_stages(tmp_path, *rows)
    result = compare_flow_steps(baseline, candidate)
    assert result["paired_tasks"] == result["paired_episodes"] == 4
    assert result["macro_success_difference"] == 0
    assert result["task_cluster_95ci"] == [-1, 1]
    assert result["small_task_count"]
    assert result["initial_state_parity"]["image_method_comparisons"] == 4
    path = reporting.build_flow_report(baseline.output_dir, candidate.output_dir, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "少量任务" in text
    assert "不证明两种方法等效或非劣效" in text
    assert "50_smolvla/episodes/b0/result.json" in text
    assert "60_flow_steps/episodes/c0/result.json" in text
    assert len(json.loads((tmp_path / "flow_comparison.json").read_text(encoding="utf-8"))["per_pair"]) == 4


@pytest.mark.parametrize("mutation,expected", [
    ("weight", "weights"), ("scheduler", "scheduler"), ("steps", "num_steps"),
    ("horizon", "horizon"), ("environment", "environment"), ("image", "image parity"),
    ("state", "state parity"), ("missing_evidence", "parity evidence"), ("missing_pair", "4 fully matched"),
])
def test_flow_comparison_rejects_confounders_or_missing_pairs(tmp_path, mutation, expected):
    baseline_rows, changed_rows = flow_records()
    target = changed_rows[0]
    if mutation == "weight":
        target["model_info"]["model_sha256"] = "b" * 64
    elif mutation == "scheduler":
        target["scheduler_config"]["fixed_interval"] = 8
    elif mutation == "steps":
        target["num_steps"] = 6
    elif mutation == "horizon":
        target["horizon"] = 600
    elif mutation == "environment":
        target["environment"]["controller"] = "different"
    elif mutation == "image":
        target["initial_image_sha256"] = "b" * 64
    elif mutation == "state":
        target["initial_robot_state"][0] += 1
    elif mutation == "missing_evidence":
        del target["initial_image_sha256"]
    elif mutation == "missing_pair":
        changed_rows.pop()
    baseline, changed = write_flow_stages(tmp_path, baseline_rows, changed_rows)
    with pytest.raises(ExperimentIntegrityError, match=expected):
        compare_flow_steps(baseline, changed)


def controlled_episode(identifier, *, method="adaptive", task_id=0, init_state_id=0, success=True, condition="clean"):
    row = with_initial_evidence(episode(identifier, task_id=task_id, init_state_id=init_state_id,
                                       method=method, success=success))
    row.update(condition=condition, num_steps=None, environment={"controller": "fixed"})
    row["model_info"] = {"model_sha256": "a" * 64, "revision": "frozen", "load_seconds": 1.2}
    row["scheduler_config"] = {"mode": "adaptive", "max_interval": 15, "min_interval": 2,
                               "use_vision": True, "use_gripper": True, "temporal_ensemble": True,
                               "image_threshold": 0.04, "gripper_threshold": 0.003}
    component = {"no_vision": "use_vision", "no_gripper": "use_gripper", "no_ensemble": "temporal_ensemble"}.get(method)
    if component:
        row["scheduler_config"][component] = False
    if method == "fixed8":
        row["scheduler_config"].update(mode="fixed", fixed_interval=8)
    return row


def ablation_records():
    rows = []
    # Unequal initial-state counts deliberately distinguish task and episode means.
    for task_id, initial_states in ((0, range(3)), (1, range(1))):
        for initial in initial_states:
            for method in ("adaptive", "no_vision", "no_gripper", "no_ensemble"):
                row = controlled_episode(f"{method}-{task_id}-{initial}", task_id=task_id,
                                         init_state_id=initial, method=method,
                                         success=(task_id == 0) if method == "adaptive" else (task_id == 1))
                row["inference_calls"] = 4 if method == "adaptive" else 2
                rows.append(row)
    return rows


def test_direct_ablations_use_adaptive_and_task_weights_not_fixed8():
    rows = ablation_records()
    original = copy.deepcopy(rows)
    result = ablation_comparisons(rows)
    assert len(result) == 3
    assert {row["method"] for row in result} == {"no_vision", "no_gripper", "no_ensemble"}
    for row in result:
        assert row["baseline"] == "adaptive"
        assert row["paired_tasks"] == 2
        assert row["paired_episodes"] == 4
        assert row["macro_success_difference"] == 0  # Episode-weighted difference would be -0.5.
        assert row["task_cluster_95ci"] == [-1, 1]
        assert row["mean_calls_difference"] == -2
        assert row["candidate_wins"] == 1
        assert row["baseline_wins"] == 3
        assert row["difference_direction"] == "variant minus full adaptive"
    assert rows == original


@pytest.mark.parametrize("mutation,expected", [
    ("threshold", "change only"), ("wrong_switch", "change only"), ("already_disabled", "change only"),
    ("weights", "weights"), ("horizon", "horizon"), ("image", "image parity"),
    ("state", "state parity"), ("missing_evidence", "parity evidence"), ("missing_pair", "fully matched"),
    ("duplicate", "duplicate"),
])
def test_direct_ablation_rejects_confounded_or_incomplete_pairs(mutation, expected):
    base = controlled_episode("base")
    variant = controlled_episode("variant", method="no_vision")
    rows = [base, variant]
    if mutation == "threshold":
        variant["scheduler_config"]["image_threshold"] = 0.5
    elif mutation == "wrong_switch":
        variant["scheduler_config"]["use_gripper"] = False
    elif mutation == "already_disabled":
        base["scheduler_config"]["use_vision"] = False
    elif mutation == "weights":
        variant["model_info"]["model_sha256"] = "b" * 64
    elif mutation == "horizon":
        variant["horizon"] += 1
    elif mutation == "image":
        variant["initial_image_sha256"] = "b" * 64
    elif mutation == "state":
        variant["initial_robot_state"][0] += 1
    elif mutation == "missing_evidence":
        del variant["initial_robot_state"]
    elif mutation == "missing_pair":
        variant["seed"] += 1
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(variant))
    with pytest.raises(ExperimentIntegrityError, match=expected):
        ablation_comparisons(rows)


def test_partial_ablation_exposes_unmatched_instead_of_inventing_effect():
    rows = [controlled_episode("base"), controlled_episode("variant", method="no_vision", init_state_id=1)]
    result = ablation_comparisons(rows, partial=True)[0]
    assert result["paired_episodes"] == 0
    assert result["unmatched_baseline"] == result["unmatched_candidate"] == 1
    assert result["macro_success_difference"] is None
    assert result["mean_calls_difference"] is None
    assert result["task_cluster_95ci"] is None


def test_ablation_report_exports_actual_contrast_plot_and_labels(tmp_path):
    write_experiment(tmp_path, ablation_records(), extra_protocol={"plan": {"name": "ablation-fixture-stage"}})
    report = reporting.build_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "关闭组件减完整 adaptive" in text
    assert "每回合均值调用差" in text
    assert "未做多重比较校正" in text
    assert "不表示真实不确定性为零" in text
    assert (tmp_path / "figures/ablation_differences.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg_text = (tmp_path / "figures/ablation_differences.svg").read_text(encoding="utf-8")
    assert "<svg " in svg_text
    assert "ablation-fixture-stage" in svg_text
    assert "tasks=2, paired episodes=4" in svg_text
    # Coincident budget points share a readable label; their plotted values are not jittered.
    budget_text = (tmp_path / "figures/quality_budget.svg").read_text(encoding="utf-8")
    assert "2, 3, 4" in budget_text
    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(result["ablation_comparisons"]) == 3


def delay_records():
    clean, delay = [], []
    for task_id in (1, 6):
        for method in ("fixed8", "adaptive"):
            for target, condition in ((clean, "clean"), (delay, "delay4")):
                row = controlled_episode(f"{condition}-{method}-{task_id}", task_id=task_id, method=method,
                                         condition=condition, success=(condition == "clean"))
                row["inference_calls"] = 3 if condition == "clean" else 5
                target.append(row)
    return clean, delay


def write_delay_stages(tmp_path, clean, delayed):
    base_dir, delay_dir = tmp_path / "30_methods", tmp_path / "40_delay"
    base_dir.mkdir()
    delay_dir.mkdir()
    write_experiment(base_dir, clean)
    write_experiment(delay_dir, delayed)
    return load_experiment(base_dir), load_experiment(delay_dir)


def test_delay_comparison_preserves_direction_raw_evidence_and_report(tmp_path):
    baseline, delayed = write_delay_stages(tmp_path, *delay_records())
    result = compare_observation_delay(baseline, delayed)
    assert len(result["comparisons"]) == 2
    for comparison in result["comparisons"]:
        assert comparison["macro_success_difference"] == -1
        assert comparison["mean_calls_difference"] == 2
        assert comparison["task_cluster_95ci"] == [-1, -1]
        assert comparison["small_task_count"]
    path = reporting.build_delay_report(baseline.output_dir, delayed.output_dir, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "本表未检验方法与条件的交互" in text
    assert "不证明两种方法等效或非劣效" in text
    assert "30_methods/episodes/clean-fixed8-1/result.json" in text
    assert "40_delay/episodes/delay4-fixed8-1/result.json" in text
    svg_text = (tmp_path / "figures/delay_differences.svg").read_text(encoding="utf-8")
    assert "40_delay vs matched clean from 30_methods" in svg_text
    assert "tasks=2, paired episodes=2" in svg_text
    assert (tmp_path / "delay_comparison.json").is_file()


@pytest.mark.parametrize("mutation,expected", [
    ("scheduler", "scheduler"), ("image", "image parity"), ("state", "state parity"),
    ("seed", "fully matched"), ("missing_pair", "fully matched"), ("wrong_condition", "only delay4"),
    ("model", "weights"), ("source", "frozen sources"),
])
def test_delay_comparison_rejects_confounders_or_unmatched_trials(tmp_path, mutation, expected):
    base_rows, delayed_rows = delay_records()
    if mutation == "scheduler":
        delayed_rows[0]["scheduler_config"]["max_interval"] = 14
    elif mutation == "image":
        # Both methods agree within delay condition; cross-condition check must still fail.
        for row in delayed_rows:
            row["initial_image_sha256"] = "b" * 64
    elif mutation == "state":
        for row in delayed_rows:
            row["initial_robot_state"][0] += 1
    elif mutation == "seed":
        for row in delayed_rows:
            row["seed"] += 1
    elif mutation == "missing_pair":
        delayed_rows.pop()
    elif mutation == "wrong_condition":
        for row in delayed_rows:
            row["condition"] = "dim"
    elif mutation == "model":
        for row in delayed_rows:
            row["model_info"]["model_sha256"] = "b" * 64
    base, delayed = write_delay_stages(tmp_path, base_rows, delayed_rows)
    if mutation == "source":
        delayed.protocol["sources"] = {"runtime.py": "modified"}
    with pytest.raises(ExperimentIntegrityError, match=expected):
        compare_observation_delay(base, delayed)
