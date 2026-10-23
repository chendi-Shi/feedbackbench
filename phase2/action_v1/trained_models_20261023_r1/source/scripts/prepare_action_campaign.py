"""Freeze the complete prospective action-adaptation study before any fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from action_campaign_common import ROOT, DEADLINE, deadline_reached, digest, now, read, resolve, verify_protocol, verify_seal, write_json

TASKS = {
    "libero_spatial": ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                       "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    "libero_object": ("pick_up_the_alphabet_soup_and_place_it_in_the_basket", "pick up the alphabet soup and place it in the basket"),
    "libero_goal": ("open_the_middle_drawer_of_the_cabinet", "open the middle drawer of the cabinet"),
    "libero_10": ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                  "put both the alphabet soup and the tomato sauce in the basket"),
}
SEEDS = (17, 23, 47)
STEPS = 2000


def make_schedule(demos, seed, steps=STEPS, offset=1):
    """Balanced tasks, whole-demo split, fixed samples shared by both families."""
    rng = random.Random(seed)
    train = [d for d in demos if d["split"] == "train"]
    tasks = sorted({d["task"] for d in train})
    by_task = {task: sorted((d for d in train if d["task"] == task), key=lambda d: d["id"]) for task in tasks}
    items = []
    while len(items) < steps:
        order = list(tasks)
        rng.shuffle(order)
        for task in order:
            demo = rng.choice(by_task[task])
            items.append({"demo_id": demo["id"], "observation_index": rng.randrange(demo["frame_count"] - offset)})
            if len(items) == steps:
                break
    validation = []
    # Three fixed positions per validation episode. Neither targets nor outcomes determine selection.
    for demo in sorted((d for d in demos if d["split"] == "validation"), key=lambda d: d["id"]):
        last = demo["frame_count"] - offset - 1
        for index, fraction in enumerate((0.0, 0.5, 1.0)):
            validation.append({"demo_id": demo["id"], "observation_index": int(last * fraction),
                               "noise_seed": 100000 + len(validation), "time": (.25, .5, .75)[index]})
    return {"schema": "feedbackbench.action_sample_schedule.v1", "seed": seed, "train": items, "validation": validation,
            "task_sample_counts": dict(Counter(next(d["task"] for d in train if d["id"] == item["demo_id"]) for item in items))}


def make_evaluation():
    methods = [{"method_id": "frozen", "family": "frozen", "training_run_id": None}]
    for seed in SEEDS:
        for family in ("lora", "output_head"):
            name = f"{family}_s{seed}"
            methods.append({"method_id": name, "family": family, "training_run_id": name})
    cases = []
    # Interleave methods per condition/state/task so incomplete runs are not all one method.
    for initial in (5, 15, 25, 35, 45):
        for suite, (_, language) in TASKS.items():
            for condition in ("clean", "brightness075"):
                for method in methods:
                    cases.append({"case_id": f"{suite}_t0_i{initial}_{condition}_{method['method_id']}",
                                  "method_id": method["method_id"], "suite": suite, "task_id": 0,
                                  "task_uid": suite + ":0", "language": language,
                                  "initial_state_id": initial, "condition": condition})
    return {"num_steps": 10, "replan_steps": 10, "image_size": 360, "arena_bytes": 256 * 1024**2,
            "environment_seed": 1000, "methods": methods, "cases": cases,
            "noise_seed_rule": "sha256('action-eval-v1:{task_uid}:{initial_state_id}:{decision}') first four bytes little-endian; excludes method, training seed and condition",
            "horizons": {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
            "conditions": {"clean": "unchanged camera observations", "brightness075": "multiply both policy RGB camera tensors by 0.75 on every observation; physical state unchanged"},
            "complete_failure_policy": "retain every completed failure; never rerun it for success"}


def source_inventory():
    names = ["scripts/action_campaign_common.py", "scripts/prepare_action_campaign.py", "scripts/run_action_campaign.py",
             "scripts/train_action_adaptation.py", "scripts/evaluate_action_adaptation.py", "scripts/report_action_campaign.py",
             "scripts/run_action_adaptation_probe.py", "scripts/probe_action_adaptation.py",
             "scripts/budget_campaign_common.py", "scripts/budget_campaign_common_v5.py", "scripts/budget_startup_memory.py",
             "scripts/libero_arena_budget.py", "scripts/run.py", "docs/ACTION_ADAPTATION_RUNBOOK.md",
             "src/embodied_research/action_adaptation.py", "src/embodied_research/action_head_baseline.py",
             "src/embodied_research/action_demo_data.py", "scripts/prepare_action_adaptation_data.py",
             "src/embodied_research/policies.py", "src/embodied_research/runtime.py", "src/embodied_research/simulator.py",
             "vendor/MINERVA/src/lerobot/envs/libero.py", "vendor/MINERVA/src/lerobot/policies/factory.py",
             "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
             "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py"]
    names += ["tests/" + name for name in (
        "test_action_adaptation.py", "test_action_adaptation_data.py", "test_action_adaptation_probe.py",
        "test_action_adaptation_supervisor.py", "test_action_adaptation_train.py", "test_action_cache.py",
        "test_action_campaign.py", "test_action_demo_alignment.py", "test_action_head_baseline.py",
        "test_action_study_sources.py")]
    names += ["tests/test_evaluate_action_adaptation.py", "scripts/validate_action_demo_alignment.py",
              "scripts/validate_action_demo_alignment_v2.py", "scripts/combine_action_demo_alignment.py",
              "tests/test_combine_action_demo_alignment.py", "scripts/prepare_action_study_sources.py"]
    return {name: digest(ROOT / name) for name in sorted(set(names))}


def verify_pipeline_check(report_path):
    """Check both actual training families against their preserved historical sources."""
    from run_action_campaign import completed_training

    campaign = report_path.parent
    protocol_path = campaign / "protocol.json"
    fingerprint = (campaign / "protocol.sha256").read_text().strip()
    if digest(protocol_path) != fingerprint:
        raise ValueError("Development training protocol changed")
    protocol = read(protocol_path)
    report = read(report_path)
    if (protocol.get("development_only") is not True or report.get("status") != "complete"
            or report.get("protocol_sha256") != fingerprint
            or len(protocol["runs"]) != 2
            or {run["mode"] for run in protocol["runs"].values()} != {"lora", "output_head"}
            or any(run["steps"] < 2 for run in protocol["runs"].values())):
        raise ValueError("Both real development training pipelines must have completed")
    snapshot = campaign / "source_snapshot"
    index = read(snapshot / "index.json")
    if set(index) != set(protocol["source_files"]):
        raise ValueError("Historical pipeline source coverage changed")
    for relative, expected in protocol["source_files"].items():
        if digest(resolve(index[relative], snapshot)) != expected:
            raise ValueError("Historical pipeline source changed")
    manifest_path = campaign / "trained_manifest.json"
    manifest = read(manifest_path)
    completed = completed_training(campaign, protocol, fingerprint)
    if (len(completed) != 2 or manifest.get("protocol_sha256") != fingerprint
            or manifest.get("runs") != completed
            or report.get("trained_manifest_sha256") != digest(manifest_path)):
        raise ValueError("Development training checkpoints or report binding changed")
    return {"report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": digest(report_path)},
            "protocol": {"path": protocol_path.relative_to(ROOT).as_posix(), "sha256": fingerprint},
            "trained_manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": digest(manifest_path)},
            "excluded_from_formal_results": True}


def verify_development_episode(report_path, pipeline_evidence):
    """Validate execution and sealing, regardless of the development task outcome."""
    report = read(report_path)
    campaign = report_path.parents[3]
    protocol_path = campaign / "protocol.json"
    fingerprint = (campaign / "protocol.sha256").read_text().strip()
    protocol = verify_protocol(protocol_path, fingerprint)
    historical = protocol["historical_training"]
    expected = {"evaluation_protocol_sha256": fingerprint,
                "training_protocol_sha256": pipeline_evidence["protocol"]["sha256"],
                "trained_manifest_sha256": pipeline_evidence["trained_manifest"]["sha256"],
                "case_id": report["case"]["case_id"], "excluded_from_formal_study": True}
    if (protocol.get("development_only") is not True or report.get("status") != "complete"
            or report.get("development_only") is not True or report.get("study_result") is not False
            or report["case"]["initial_state_id"] != 0
            or historical["protocol"]["sha256"] != expected["training_protocol_sha256"]
            or historical["trained_manifest"]["sha256"] != expected["trained_manifest_sha256"]
            or any(report.get(key) != value for key, value in expected.items() if key != "case_id")):
        raise ValueError("Development episode is not bound to the verified training pipelines")
    verify_seal(report_path.parent, expected)
    return {"report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": digest(report_path)},
            "protocol": {"path": protocol_path.relative_to(ROOT).as_posix(), "sha256": fingerprint},
            "receipt": {"path": (report_path.parent / "receipt.json").relative_to(ROOT).as_posix(),
                        "sha256": digest(report_path.parent / "receipt.json")},
            "excluded_from_formal_results": True, "success_was_not_a_gate": True}


def register(sources_path, alignment_path, probe_path, pipeline_path, development_episode_path, output):
    import h5py
    from combine_action_demo_alignment import combine

    if deadline_reached():
        raise ValueError("Hard deadline passed; no new campaign")
    output = output.resolve()
    if not output.is_relative_to((ROOT / "outputs/action_adaptation").resolve()) or output.exists():
        raise ValueError("Campaign requires a new directory below outputs/action_adaptation")
    sources = read(sources_path)["sources"]
    if {r["suite"] for r in sources} != set(TASKS) or len(sources) != 4:
        raise ValueError("All four prospectively fixed task-0 sources are required")
    alignment = read(alignment_path)
    if alignment.get("status") != "passed" or alignment.get("action_offset") not in (0, 1):
        raise ValueError("Passed demonstration alignment is required")
    records = alignment.get("records", [])
    expected_sources = {r["suite"]: r["sha256"] for r in sources}
    if (len(records) != 4 or {r.get("suite"): r.get("sha256") for r in records} != expected_sources
            or any(r.get("status") != "passed" for r in records)):
        raise ValueError("Alignment must verify each of the four exact source files")
    reconstructed = combine(sources_path, [r["path"] for r in alignment["single_source_receipts"]])
    reconstructed["combiner_sha256"] = digest(ROOT / "scripts/combine_action_demo_alignment.py")
    if (alignment != reconstructed
            or alignment["validator_script_sha256"] != digest(ROOT / "scripts/validate_action_demo_alignment_v2.py")):
        raise ValueError("Alignment receipt chain or validator changed")
    pipeline_evidence = verify_pipeline_check(pipeline_path)
    development_evaluation_evidence = verify_development_episode(development_episode_path, pipeline_evidence)
    probe = read(probe_path)
    if probe.get("status") != "passed" or not probe.get("base_exactly_unchanged") or not probe.get("reload_actions_exact"):
        raise ValueError("Passed real-checkpoint training/reload probe is required")
    demos = []
    excluded = []
    for source in sources:
        suite = source["suite"]
        if source["task_id"] != 0 or source["task_name"] != TASKS[suite][0]:
            raise ValueError("Task selection differs from prospective task 0")
        path = resolve(source["path"])
        if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
            raise ValueError("Pinned expert source is incomplete or changed")
        with h5py.File(path, "r") as data:
            if set(data["data"]) != {f"demo_{i}" for i in range(50)}:
                raise ValueError("Expected all 50 published demonstrations; no task substitution")
            excluded.append({"suite": suite, "episode_key": "demo_0", "reason": "development"})
            for index in range(1, 50):
                key = f"demo_{index}"
                episode = data[f"data/{key}"]
                count = len(episode["actions"])
                if count <= alignment["action_offset"] or episode["actions"].shape != (count, 7):
                    raise ValueError("Invalid action trajectory")
                demos.append({"id": f"{suite}_{key}", "split": "train" if index <= 40 else "validation",
                              "task": suite + ":0", "language": TASKS[suite][1], "path": source["path"],
                              "sha256": source["sha256"], "episode_key": key, "frame_count": count})
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "data_manifest.json"
    write_json(data_path, {"schema": "feedbackbench.action_demo_manifest.v1", "demos": demos, "development_exclusions": excluded,
                           "source_manifest": {"path": sources_path.relative_to(ROOT).as_posix(), "sha256": digest(sources_path)}})
    runs = {}
    for seed in SEEDS:
        schedule_path = output / f"schedule_s{seed}.json"
        write_json(schedule_path, make_schedule(demos, seed, offset=alignment["action_offset"]))
        for mode in ("lora", "output_head"):
            runs[f"{mode}_s{seed}"] = {"mode": mode, "initialization_seed": seed, "flow_seed": seed,
                "steps": STEPS, "schedule": {"path": schedule_path.relative_to(ROOT).as_posix(), "sha256": digest(schedule_path)}}
    model_files = {}
    for name in ("smolvla", "smolvlm_assets"):
        manifest = ROOT / "models" / name / "download_manifest.json"
        for path in [manifest, *(manifest.parent / r["file"] for r in read(manifest)["files"])]:
            model_files[path.relative_to(ROOT).as_posix()] = digest(path)
    protocol = {"schema": "feedbackbench.action_adaptation_train.v1", "frozen": True, "frozen_at": now(),
        "study_id": "action-adaptation-v1", "deadline_utc": DEADLINE,
        "research_question": "Does expert-supervised action-expert LoRA improve same-domain closed-loop success over frozen SmolVLA and matched output-head training?",
        "claims": {"base_training_overlap": "unknown", "unseen_task_claim": False, "hardware_robot_claim": False,
                   "evaluation_initial_state_independence": "not established against demonstration generation; prospective same-domain evaluation only",
                   "alignment_scope": "observation indexing and reconstruction; complete dynamics trajectory replay not verified",
                   "task_selection": "official task_order_index=0, task_id=0 from each suite, fixed before new results"},
        "model": {"models_dir": "models", "files": model_files, "num_steps": 10},
        "data": {"manifest": {"path": data_path.relative_to(ROOT).as_posix(), "sha256": digest(data_path)},
                 "alignment_evidence": {"path": alignment_path.relative_to(ROOT).as_posix(), "sha256": digest(alignment_path)},
                 "action_offset": alignment["action_offset"], "image_transform": "rot180_once"},
        "probe_evidence": {"path": probe_path.relative_to(ROOT).as_posix(), "sha256": digest(probe_path)},
        "pipeline_evidence": pipeline_evidence,
        "development_evaluation_evidence": development_evaluation_evidence,
        "optimizer": {"type": "AdamW", "lr": 1e-4, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
        "flow_time_distribution": "beta_1.5_1_scaled_0.999_plus_0.001", "checkpoint_every": 100, "validation_every": 500,
        "checkpoint_selection": "last scheduled step for every seed; validation never selects or changes training",
        "resources": {"torch_threads": 2, "interop_threads": 1, "max_seconds": 8 * 3600,
                      "evaluation_max_seconds": 7200, "start_available_gib": 4.25, "runtime_available_gib": .75,
                      "free_disk_gib": 4, "model_workers": 1, "cloud_cost": 0},
        "attempt_policy": {"training_max_attempts": 3, "evaluation_max_partial_attempts": 3,
                           "completed_or_terminal_evaluation_retry": False, "post_resource_stop_backoff_seconds": 300},
        "runs": runs, "evaluation": make_evaluation(),
        "reporting": {"primary": "mean paired success difference, LoRA versus frozen and output_head, clean condition",
                      "secondary": "brightness075 paired success, per-task outcomes, all three seeds, latency and flow losses",
                      "bootstrap": "10000 resamples of four task groups, seed20261022; descriptive with only four tasks",
                      "failure_policy": "all planned cases shown; incomplete is partial; no test-driven tuning or seed selection"},
        "source_files": source_inventory()}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, (relative, expected) in enumerate(protocol["source_files"].items()):
        copied = snapshot / f"{index:03d}_{Path(relative).name}"
        shutil.copyfile(ROOT / relative, copied)
        if digest(copied) != expected:
            raise ValueError("Source changed while freezing")
    write_json(snapshot / "index.json", {relative: f"{index:03d}_{Path(relative).name}"
                                        for index, relative in enumerate(protocol["source_files"])})
    write_json(output / "protocol.json", protocol)
    with (output / "protocol.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(output / "protocol.json") + "\n")
    return {"campaign": str(output), "training_runs": len(runs), "training_steps": 6 * STEPS,
            "training_demos": 160, "validation_demos": 36, "evaluation_cases": 280, "protocol_sha256": digest(output / "protocol.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--probe-report", required=True, type=Path)
    parser.add_argument("--pipeline-report", required=True, type=Path)
    parser.add_argument("--development-evaluation-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(register(args.sources.resolve(), args.alignment.resolve(), args.probe_report.resolve(),
                              args.pipeline_report.resolve(), args.development_evaluation_report.resolve(), args.output), indent=2))


if __name__ == "__main__":
    main()
