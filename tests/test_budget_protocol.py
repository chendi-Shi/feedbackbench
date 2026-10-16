"""Pure-stdlib task manifest tests; no installed LIBERO or model is required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/budget_protocol.py"
SPEC = importlib.util.spec_from_file_location("budget_protocol_test", PATH)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def _unicode(value):
    encoded = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def _initial_archive(path, rows=50):
    # A protocol-2 NumPy reconstruction archive, generated as bytes so tests do
    # not import NumPy or execute the reconstruction instructions.
    payload = (b"\x80\x02cnumpy.core.multiarray\n_reconstruct\ncnumpy\nndarray\nK\x00\x85"
               b"c_codecs\nencode\n" + _unicode("b") + _unicode("latin1") + b"\x86R\x87R(K\x01K"
               + bytes([rows]) + b"K\x02\x86cnumpy\ndtype\n" + _unicode("f8") + b"\x89\x88\x87R(K\x03"
               + _unicode("<") + b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89c_codecs\nencode\n"
               + _unicode("\x00" * rows * 2 * 8) + _unicode("latin1") + b"\x86Rtb.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", "3\n")


@pytest.fixture
def installation(tmp_path):
    package = protocol._root_path(tmp_path) / protocol.PACKAGE_PATH
    benchmark = package / "benchmark"
    benchmark.mkdir(parents=True)
    mapping = {suite: [f"{suite}_action_{index}" for index in range(10)] for suite in protocol.SUITES}
    (benchmark / "libero_suite_task_map.py").write_text(
        "raise RuntimeError('must never execute installed code')\nlibero_task_map = " + repr(mapping), encoding="utf-8")
    (benchmark / "__init__.py").write_text(
        "raise RuntimeError('must never import libero')\ntask_orders = " + repr([list(range(10))]), encoding="utf-8")
    for suite, names in mapping.items():
        for task_key in names:
            bddl = package / "bddl_files" / suite / f"{task_key}.bddl"
            bddl.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(f"(define (:language {task_key.replace('_', ' ')}))\n", encoding="utf-8")
            _initial_archive(package / "init_files" / suite / f"{task_key}.pruned_init")
    return tmp_path


def test_full_manifest_has_planned_counts_and_no_side_effects(installation):
    before = {str(path): path.read_bytes() for path in installation.rglob("*") if path.is_file()}
    result = protocol.make_protocol(installation)
    assert result["status"] == "planned"
    assert result["counts"] == {"tasks": 40, "exact_match_groups": 40, "collection_episodes": 56,
                                "evaluation_methods": 15, "evaluation_episodes": 360, "total_planned_episodes": 416}
    assert result["split_policy"]["actual_task_counts"] == {"train": 20, "tuning": 4, "calibration": 4, "test": 12}
    assert result["split_policy"]["independent_semantic_task_count"] is None
    assert result["source_binding"]["status"] == "pending_runner_freeze"
    assert result["source_binding"]["sources"] == []
    assert json.loads(json.dumps(result)) == result
    assert protocol.make_protocol(installation) == result
    after = {str(path): path.read_bytes() for path in installation.rglob("*") if path.is_file()}
    assert after == before


def test_catalog_paths_hashes_and_legacy_tasks_are_train(installation):
    tasks = protocol.task_catalog(installation)
    assert len({task["task_uid"] for task in tasks}) == 40
    for task in tasks:
        assert task["task"] == task["task_key"]
        assert not Path(task["path"]).is_absolute() and not Path(task["initial_path"]).is_absolute()
        assert task["bddl_sha"] == hashlib.sha256((installation / task["path"]).read_bytes()).hexdigest()
        assert task["hash"] == hashlib.sha256((installation / task["initial_path"]).read_bytes()).hexdigest()
        assert task["initial_state_count"] == 50
        assert task["initial_state_dimension"] == 2
        if task["task_id"] == 1:
            assert task["split"] == "train"


def test_collection_behavior_pairing_and_test_isolation(installation):
    result = protocol.make_protocol(installation)
    collection = result["collection"]
    assert collection["different_initial_states_are_paired_causal_rollouts"] is False
    assert collection["pairing_unit"] == "same_decision_same_observation_same_context_same_initial_noise"
    for case in collection["cases"]:
        assert case["split"] != "test"
        assert case["paired_prediction_budgets"] == [2, 4, 10]
        assert case["behavior_budget"] == {0: 2, 1: 10}[case["initial_state_id"]]
    assert {case["initial_state_id"] for case in result["evaluation"]["cases"]} == {41, 43}
    assert {case["task_uid"] for case in collection["cases"]}.isdisjoint(
        {case["task_uid"] for case in result["evaluation"]["cases"]})
    assert all(case["split"] == "test" for case in result["evaluation"]["cases"])
    assert result["training"]["test_is_locked_until_all_models_are_frozen"] is True
    assert result["training"]["preprocessing_fit_split"] == "train"


def test_all_fifteen_methods_cover_identical_test_cases(installation):
    result = protocol.make_protocol(installation)
    methods = result["evaluation"]["methods"]
    assert len({method["method_id"] for method in methods}) == 15
    for method in methods:
        cases = [case for case in result["evaluation"]["cases"] if case["method_id"] == method["method_id"]]
        assert len(cases) == 24
        assert len({(case["task_uid"], case["initial_state_id"]) for case in cases}) == 24
    fixed = [method for method in methods if method["family"] == "fixed"]
    assert [method["budget"] for method in fixed] == [2, 4, 10]
    assert all(method["seed"] is None for method in fixed)
    for mode in protocol.FEATURE_MODES:
        assert [method["seed"] for method in methods if method["feature_mode"] == mode] == [0, 1, 2]
    for method in methods:
        if method["family"] == "random_calibrated":
            assert method["matched_controller"] == f"full_seed_{method['seed']}"
    assert result["evaluation"]["random_probabilities_may_use_test_outcomes"] is False
    assert result["evaluation"]["select_best_training_seed_on_test"] is False


def _record(uid, split, bddl, language):
    return {"task_uid": uid, "original_split": split, "bddl_sha": bddl,
            "normalized_language": protocol.normalize_language(language)}


def test_transitive_duplicate_group_cannot_leak_across_splits():
    tasks = [_record("a", "train", "same", "first"), _record("b", "test", "same", "second"),
             _record("c", "calibration", "different", " SECOND  "), _record("d", "test", "other", "unique")]
    protocol._assign_duplicate_groups(tasks)
    assert {task["split"] for task in tasks[:3]} == {"train"}
    assert len({task["duplicate_group"] for task in tasks[:3]}) == 1
    assert all(task["duplicate_group_size"] == 3 for task in tasks[:3])
    assert tasks[3]["split"] == "test"
    assert tasks[3]["duplicate_group"] != tasks[0]["duplicate_group"]


def test_any_cross_split_group_moves_to_train_even_without_original_train_member():
    tasks = [_record("a", "tuning", "a", "ＳＡＭＥ task"), _record("b", "test", "b", "same TASK")]
    protocol._assign_duplicate_groups(tasks)
    assert [task["split"] for task in tasks] == ["train", "train"]
    assert all(task["split_reassigned"] for task in tasks)


def test_within_split_duplicate_remains_in_its_split():
    tasks = [_record("a", "test", "same", "first"), _record("b", "test", "same", "second")]
    protocol._assign_duplicate_groups(tasks)
    assert [task["split"] for task in tasks] == ["test", "test"]
    assert not any(task["split_reassigned"] for task in tasks)


def test_real_duplicate_changes_actual_plan_denominators(installation):
    tasks = protocol.task_catalog(installation)
    source = installation / tasks[0]["path"]
    destination = installation / tasks[7]["path"]
    destination.write_bytes(source.read_bytes())
    result = protocol.make_protocol(installation)
    assert result["split_policy"]["actual_task_counts"] == {"train": 21, "tuning": 4, "calibration": 4, "test": 11}
    assert result["counts"]["exact_match_groups"] == 39
    assert result["counts"]["collection_episodes"] == 58
    assert result["counts"]["evaluation_episodes"] == 330
    assert result["counts"]["total_planned_episodes"] == 388
    assert result["split_policy"]["reassigned_task_uids"] == ["libero_spatial:7"]
    assert len(result["split_policy"]["duplicate_groups"]) == 1


@pytest.mark.parametrize("name,value", [("libero_task_map", "dict()"), ("task_orders", "list()")])
def test_nonliteral_installed_assignment_is_rejected(installation, name, value):
    filename = "libero_suite_task_map.py" if name == "libero_task_map" else "__init__.py"
    (installation / protocol.PACKAGE_PATH / "benchmark" / filename).write_text(f"{name} = {value}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a literal"):
        protocol.make_protocol(installation)


def test_nondefault_task_order_is_rejected(installation):
    path = installation / protocol.PACKAGE_PATH / "benchmark/__init__.py"
    path.write_text("task_orders = " + repr([list(reversed(range(10)))]), encoding="utf-8")
    with pytest.raises(ValueError, match="default task order"):
        protocol.task_catalog(installation)


def test_missing_evaluation_initial_state_is_rejected(installation):
    task = protocol.task_catalog(installation)[0]
    _initial_archive(installation / task["initial_path"], rows=43)
    with pytest.raises(ValueError, match="initial state 43 is unavailable"):
        protocol.task_catalog(installation)


def test_unrecognized_initial_pickle_globals_are_never_executed(installation):
    task = protocol.task_catalog(installation)[0]
    with zipfile.ZipFile(installation / task["initial_path"], "w") as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02cos\nsystem\n.")
        archive.writestr("archive/version", "3\n")
    with pytest.raises(ValueError, match="unsupported initial-state pickle structure"):
        protocol.task_catalog(installation)


def test_resource_and_deadline_contract(installation):
    result = protocol.make_protocol(installation)
    assert result["deadline_utc"] == "2026-10-30T06:08:00Z"
    assert result["execution"]["max_concurrent_workers"] == 1
    assert result["execution"]["local_only"] is True
    assert result["execution"]["cloud_allowed"] is False
    assert result["execution"]["gpu_allowed"] is False
    assert result["resource_limits"]["max_process_rss_bytes"] == 5905580032
    assert result["resource_limits"]["min_system_available_bytes"] == 805306368
    assert result["resource_limits"]["initial_min_system_available_bytes"] == 4294967296


def test_import_and_catalog_do_not_import_heavy_dependencies(installation):
    code = ("import runpy,sys; m=runpy.run_path(sys.argv[1]); m['make_protocol'](sys.argv[2]);"
            "assert not ({'torch','numpy','mujoco','libero','robosuite'} & sys.modules.keys())")
    result = subprocess.run([sys.executable, "-c", code, str(PATH), str(installation)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
