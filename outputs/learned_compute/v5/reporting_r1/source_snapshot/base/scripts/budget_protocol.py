"""Read the local LIBERO task catalog and describe the planned compute study.

This module uses only the standard library. It parses installed Python sources
with AST and inspects initial-state pickle opcodes without executing them. It
does not import LIBERO, Torch, a simulator, or a model, and never writes files.
The final experiment source freeze belongs to the runner, not this catalog.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import pickletools
import re
import unicodedata
import zipfile


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPLITS = ("train", "tuning", "calibration", "test")
BUDGETS = (2, 4, 10)
TRAINING_SEEDS = (0, 1, 2)
FEATURE_MODES = ("full", "no_probe", "proprio_probe")
DEADLINE_UTC = "2026-10-30T06:08:00Z"
PACKAGE_PATH = Path(".venv/Lib/site-packages/libero/libero")
GIB = 1024 ** 3


def _root_path(root: str | os.PathLike) -> Path:
    absolute = os.path.abspath(root)
    # Some official task filenames exceed the legacy Windows MAX_PATH limit
    # under this workspace. Keep manifest paths relative and extend only I/O.
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
    return Path(absolute)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            values.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            values.append(node.value)
    if len(values) != 1:
        raise ValueError(f"expected one literal assignment to {name} in {path}")
    try:
        return ast.literal_eval(values[0])
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"{name} must be a literal; installed code is never executed") from error


def normalize_language(language: str) -> str:
    """Conservative exact-match normalization, not a semantic similarity claim."""
    return " ".join(unicodedata.normalize("NFKC", language).casefold().split())


def _language(task_key: str) -> str:
    # Equivalent to the installed benchmark's grab_language_from_filename for
    # these suites; the source file and task order are hashed in the protocol.
    if task_key[0].isupper():
        match = re.search(r"SCENE\d+_(.+)$", task_key)
        if match is None:
            raise ValueError(f"scene task has no recognized language suffix: {task_key}")
        task_key = match.group(1)
    return " ".join(task_key.split("_"))


def _initial_shape(path: Path) -> tuple[int, int]:
    """Read the known local NumPy archive shape, without pickle.load/torch.load."""
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"archive/data.pkl", "archive/version"}:
            raise ValueError(f"unsupported initial-state archive layout: {path}")
        payload = archive.read("archive/data.pkl")
    operations = [(op.name, value) for op, value, _ in pickletools.genops(payload)
                  if op.name not in {"BINPUT", "LONG_BINPUT"}]
    allowed_globals = {"numpy.core.multiarray _reconstruct", "numpy._core.multiarray _reconstruct",
                       "numpy ndarray", "_codecs encode", "numpy dtype"}
    globals_seen = {value for name, value in operations if name == "GLOBAL"}
    if (not operations or operations[0] != ("PROTO", 2) or operations[-1][0] != "STOP"
            or not globals_seen <= allowed_globals or "numpy ndarray" not in globals_seen):
        raise ValueError(f"unsupported initial-state pickle structure: {path}")
    first_mark = next((index for index, (name, _) in enumerate(operations) if name == "MARK"), -1)
    shape = operations[first_mark + 1:first_mark + 5] if first_mark >= 0 else []
    integers = {"BININT1", "BININT2", "BININT", "INT"}
    if (len(shape) != 4 or shape[0][0] not in integers or shape[0][1] != 1
            or any(name not in integers for name, _ in shape[1:3]) or shape[3][0] != "TUPLE2"):
        raise ValueError(f"initial-state array has no recognized two-dimensional shape: {path}")
    rows, columns = shape[1][1], shape[2][1]
    strings = [value for name, value in operations if name == "BINUNICODE"]
    if (not all(isinstance(value, int) and 0 < value < 100000 for value in (rows, columns))
            or not {"f8", "<", "latin1"} <= set(strings)
            or sum(len(value) == rows * columns * 8 for value in strings) != 1):
        raise ValueError(f"initial-state shape/dtype/payload contract mismatch: {path}")
    return rows, columns


def _original_split(task_id: int) -> str:
    return "train" if task_id <= 4 else "tuning" if task_id == 5 else "calibration" if task_id == 6 else "test"


def _assign_duplicate_groups(tasks: list[dict]) -> None:
    """Union exact BDDL or normalized-language matches, including transitivity."""
    parents = list(range(len(tasks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, task in enumerate(tasks):
        for key in (("bddl", task["bddl_sha"]), ("language", task["normalized_language"])):
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index, task in enumerate(tasks):
        groups.setdefault(find(index), []).append(task)
    for members in groups.values():
        identity = "\n".join(sorted(task["task_uid"] for task in members))
        group_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        original_splits = {task["original_split"] for task in members}
        # Every cross-split exact-match group goes to train, including a group
        # that originally touched only tuning/calibration/test.
        split = "train" if len(original_splits) > 1 else next(iter(original_splits))
        for task in members:
            task.update(split=split, duplicate_group=group_id, duplicate_group_size=len(members),
                        split_reassigned=split != task["original_split"])


def task_catalog(root: str | os.PathLike) -> list[dict]:
    """Return 40 ordered task records; all recorded paths are root-relative.

    ``task`` and ``task_key`` are the official task name; ``language`` is the
    benchmark instruction. ``path``/``bddl_sha`` identify BDDL bytes, while
    ``initial_path``/``hash`` identify the official initial-state archive.
    """
    root = _root_path(root)
    package = root / PACKAGE_PATH
    mapping = _literal_assignment(package / "benchmark/libero_suite_task_map.py", "libero_task_map")
    orders = _literal_assignment(package / "benchmark/__init__.py", "task_orders")
    if not isinstance(mapping, dict) or not isinstance(orders, list) or not orders or orders[0] != list(range(10)):
        raise ValueError("this protocol requires official default task order 0 == [0, ..., 9]")
    tasks = []
    for suite in SUITES:
        names = mapping.get(suite)
        if not isinstance(names, list) or len(names) != 10 or len(set(names)) != 10:
            raise ValueError(f"expected exactly 10 distinct task keys in {suite}")
        for task_id, task_key in enumerate(names):
            if not isinstance(task_key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", task_key):
                raise ValueError(f"invalid task filename in {suite}: {task_key!r}")
            bddl = package / "bddl_files" / suite / f"{task_key}.bddl"
            initial = package / "init_files" / suite / f"{task_key}.pruned_init"
            language = _language(task_key)
            bddl_text = bddl.read_text(encoding="utf-8-sig")
            matches = re.findall(r"\(:language\s+([^()]+)\)", bddl_text)
            if len(matches) != 1:
                raise ValueError(f"expected one BDDL language clause: {bddl}")
            rows, columns = _initial_shape(initial)
            if rows <= 43:
                raise ValueError(f"initial state 43 is unavailable in {initial}")
            tasks.append({"suite": suite, "task_id": task_id, "task_uid": f"{suite}:{task_id}",
                          "task_key": task_key, "task": task_key, "language": language,
                          "normalized_language": normalize_language(language),
                          "bddl_language": " ".join(matches[0].split()),
                          "bddl_sha": _sha256(bddl), "path": bddl.relative_to(root).as_posix(),
                          "initial_path": initial.relative_to(root).as_posix(), "hash": _sha256(initial),
                          "initial_state_count": rows, "initial_state_dimension": columns,
                          "original_split": _original_split(task_id)})
    _assign_duplicate_groups(tasks)
    return tasks


def _method_catalog() -> list[dict]:
    methods = [{"method_id": f"fixed_{budget}", "family": "fixed", "feature_mode": None,
                "seed": None, "budget": budget} for budget in BUDGETS]
    for mode in FEATURE_MODES:
        for seed in TRAINING_SEEDS:
            methods.append({"method_id": f"{mode}_seed_{seed}", "family": "learned", "feature_mode": mode,
                            "seed": seed, "budget": None})
    for seed in TRAINING_SEEDS:
        methods.append({"method_id": f"random_seed_{seed}", "family": "random_calibrated", "feature_mode": None,
                        "seed": seed, "budget": None, "probability_source": "calibration_split_full_controller",
                        "matched_controller": f"full_seed_{seed}"})
    return methods


def make_protocol(root: str | os.PathLike) -> dict:
    """Describe the planned experiment as JSON-compatible data; no side effects."""
    root = _root_path(root)
    tasks = task_catalog(root)
    groups = {}
    for task in tasks:
        groups.setdefault(task["duplicate_group"], []).append(task)
    methods = _method_catalog()
    collection_cases = []
    evaluation_cases = []
    for task in tasks:
        base = {key: task[key] for key in ("suite", "task_id", "task_uid", "task_key", "split", "duplicate_group")}
        if task["split"] != "test":
            for initial, behavior in ((0, 2), (1, 10)):
                collection_cases.append({**base, "case_id": f"collect:{task['task_uid']}:init{initial}",
                                         "initial_state_id": initial, "behavior_budget": behavior,
                                         "paired_prediction_budgets": list(BUDGETS)})
        else:
            for initial in (41, 43):
                for method in methods:
                    evaluation_cases.append({**base,
                                             "case_id": f"eval:{task['task_uid']}:init{initial}:{method['method_id']}",
                                             "initial_state_id": initial, "method_id": method["method_id"]})
    actual_counts = {split: sum(task["split"] == split for task in tasks) for split in SPLITS}
    group_counts = {split: sum(members[0]["split"] == split for members in groups.values()) for split in SPLITS}
    sources = []
    for relative in (PACKAGE_PATH / "benchmark/__init__.py", PACKAGE_PATH / "benchmark/libero_suite_task_map.py"):
        sources.append({"path": relative.as_posix(), "sha256": _sha256(root / relative)})
    duplicate_groups = [{"group": group, "task_uids": [task["task_uid"] for task in members],
                         "original_splits": sorted({task["original_split"] for task in members}),
                         "resolved_split": members[0]["split"],
                         "distinct_bddl_hashes": len({task["bddl_sha"] for task in members}),
                         "distinct_normalized_languages": len({task["normalized_language"] for task in members})}
                        for group, members in groups.items() if len(members) > 1]
    return {
        "schema": "feedbackbench.learned_compute.protocol.v1", "status": "planned", "deadline_utc": DEADLINE_UTC,
        "execution": {"device": "cpu", "local_only": True, "cloud_allowed": False, "gpu_allowed": False,
                      "max_concurrent_workers": 1, "deadline_action": "checkpoint_and_stop"},
        "resource_limits": {"max_process_rss_bytes": int(5.5 * GIB), "min_system_available_bytes": int(.75 * GIB),
                            "initial_min_system_available_bytes": 4 * GIB, "status": "provisional_until_source_freeze"},
        "official_sources": sources, "suites": list(SUITES), "task_order_index": 0, "tasks": tasks,
        "split_policy": {"original_task_ids": {"train": [0, 1, 2, 3, 4], "tuning": [5], "calibration": [6],
                                                "test": [7, 8, 9]},
                         "duplicate_keys": ["exact_bddl_sha256", "normalized_benchmark_language"],
                         "language_normalization": "Unicode NFKC + casefold + whitespace collapse",
                         "duplicate_combination": "transitive_union_of_either_key",
                         "cross_split_duplicate_destination": "train",
                         "old_four_task_id_1_cases": "train_only_not_test",
                         "requested_task_counts": {"train": 20, "tuning": 4, "calibration": 4, "test": 12},
                         "actual_task_counts": actual_counts, "exact_match_group_counts": group_counts,
                         "exact_match_groups": len(groups), "independent_semantic_task_count": None,
                         "duplicate_groups": duplicate_groups,
                         "reassigned_task_uids": [task["task_uid"] for task in tasks if task["split_reassigned"]]},
        "collection": {"splits": ["train", "tuning", "calibration"], "initial_state_ids": [0, 1],
                       "behavior_budget_by_initial_state": {"0": 2, "1": 10},
                       "paired_prediction_budgets": list(BUDGETS),
                       "pairing_unit": "same_decision_same_observation_same_context_same_initial_noise",
                       "different_initial_states_are_paired_causal_rollouts": False,
                       "test_collection_allowed": False, "cases": collection_cases},
        "training": {"seeds": list(TRAINING_SEEDS), "feature_modes": list(FEATURE_MODES),
                     "fit_split": "train", "hyperparameter_selection_split": "tuning",
                     "calibration_split": "calibration", "test_is_locked_until_all_models_are_frozen": True,
                     "preprocessing_fit_split": "train", "split_unit": "whole_task_exact_match_group",
                     "old_trajectories_may_be_relabelled_as_new_independent_tasks": False},
        "evaluation": {"split": "test", "initial_state_ids": [41, 43], "methods": methods,
                       "method_count": len(methods), "episodes_per_method": actual_counts["test"] * 2,
                       "cases": evaluation_cases,
                       "random_probabilities_fit_on": "calibration_only_for_matching_full_controller_seed",
                       "random_probabilities_may_use_test_outcomes": False,
                       "comparison_unit": "same_test_task_and_initial_state",
                       "uncertainty_unit": "held_out_task_group_not_individual_control_step",
                       "required_metrics": ["task_success", "actual_nfe", "decision_wall_seconds", "episode_wall_seconds",
                                            "controller_overhead_seconds", "budget_frequencies", "resource_interruptions"],
                       "interrupted_episode_is_task_failure": False,
                       "select_best_training_seed_on_test": False},
        "counts": {"tasks": len(tasks), "exact_match_groups": len(groups),
                   "collection_episodes": len(collection_cases), "evaluation_methods": len(methods),
                   "evaluation_episodes": len(evaluation_cases),
                   "total_planned_episodes": len(collection_cases) + len(evaluation_cases)},
        "source_binding": {"status": "pending_runner_freeze", "sources": [],
                           "required_roles": ["protocol", "collector_runner", "learning", "sampler", "evaluation_runner"],
                           "note": "The final runner must hash existing finalized sources before any collection; this is no freeze receipt."},
    }
