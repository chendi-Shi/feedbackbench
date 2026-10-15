"""Materialize the predeclared screening protocol; never run or score an episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CANDIDATES = {
    "adaptive_low": {"vision_threshold": .015, "gripper_threshold": .001},
    "adaptive_medium": {"vision_threshold": .04, "gripper_threshold": .003},
    "adaptive_high": {"vision_threshold": .08, "gripper_threshold": .01},
}


def fixed(interval, ensemble=True):
    return {"name": f"fixed{interval}", "scheduler": {
        "mode": "fixed", "fixed_interval": interval, "temporal_ensemble": ensemble, "ensemble_coeff": .01}}


def adaptive(name, thresholds, **overrides):
    return {"name": name, "scheduler": {"mode": "adaptive", "min_interval": 2, "max_interval": 15,
            "temporal_ensemble": True, "ensemble_coeff": .01, **thresholds, **overrides}}


def plan(name, model, methods, tasks, initials, split, conditions=None):
    return {"name": name, "models": [{"name": model, "methods": methods}], "tasks": tasks,
            "initial_states": initials, "conditions": conditions or ["clean"], "seed": 1000,
            "threads": 2, "image_size": 360, "warmup_calls": 2, "rotate_method_order": True,
            "split": split,
            "question": "How does feedback frequency affect task success and inference cost on a fixed CPU policy?",
            "hypothesis": "Observation-triggered replanning may preserve success with fewer inference calls; a negative result is allowed.",
            "limitations": ["Frozen published checkpoint; no retraining-seed claim.",
                            "Known tasks and published held-out initial states; not unseen-language generalization.",
                            "Simulation pauses during inference; delay4 is prescribed sensor delay, not asynchronous real-time control.",
                            "Calls are the primary compute proxy. Wall-clock measurements on a shared laptop are descriptive.",
                            "Few initial states per task; screening evidence, not a full published benchmark reproduction."]}


def save(name, data):
    destination = ROOT / "plans" / name
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", choices=list(CANDIDATES), help="Winner of the declared development-only selection rule")
    args = parser.parse_args()
    baseline_tasks = [{"suite": suite, "task_id": task, "record_video": task == 0}
                      for suite in SUITES for task in range(10)]
    save("10_coverage.json", plan("40-task-coverage", "minerva", [{"name": "upstream", "scheduler": None}],
                                 baseline_tasks, [0], "coverage diagnostic; not used for selecting thresholds"))
    dev_tasks = [{"suite": suite, "task_id": 0} for suite in SUITES]
    dev = plan("development-calibration", "minerva", [fixed(8)] + [adaptive(n, t) for n, t in CANDIDATES.items()],
               dev_tasks, [1, 2], "development only; these tasks/initials do not enter the method comparison")
    dev["selection_rule"] = ("Among candidates with development macro success >= fixed8, choose lowest mean inference calls; "
                             "ties use higher success then lexical method name. If none qualifies, choose highest success, "
                             "then lowest calls, then lexical name. Freeze the selected configuration before test.")
    save("20_development.json", dev)
    if not args.selected:
        return
    thresholds = CANDIDATES[args.selected]
    test_tasks = [{"suite": suite, "task_id": task,
                   "record_video": suite == "libero_spatial" and task == 1}
                  for suite in SUITES for task in (1, 6)]
    methods = [fixed(k) for k in (1, 4, 8, 15)] + [
        adaptive("adaptive", thresholds), adaptive("no_vision", thresholds, use_vision=False),
        adaptive("no_gripper", thresholds, use_gripper=False),
        adaptive("no_ensemble", thresholds, temporal_ensemble=False)]
    main_test = plan("held-out-method-comparison", "minerva", methods, test_tasks, [11, 29], "held-out test")
    main_test["selected_on_development"] = args.selected
    save("30_methods.json", main_test)
    delay = plan("paired-sensor-delay", "minerva", [fixed(8), adaptive("adaptive", thresholds)], test_tasks,
                 [11, 29], "held-out sensor stress test", ["delay4"])
    delay["selected_on_development"] = args.selected
    save("40_delay.json", delay)
    # A transfer pilot is deliberately separate from the broad MINERVA study.
    smol_tasks = [{"suite": suite, "task_id": 1, "record_video": suite == "libero_spatial"} for suite in SUITES]
    smol_methods = [fixed(8, False), fixed(15, False), adaptive("adaptive", thresholds, temporal_ensemble=False)]
    smol = plan("language-policy-transfer-pilot", "smolvla", smol_methods, smol_tasks, [11], "limited CPU transfer pilot")
    smol["selected_on_development"] = args.selected + " on MINERVA development only; no SmolVLA tuning"
    save("50_smolvla.json", smol)
    steps = fixed(15, False)
    steps["name"] = "fixed15_steps5"
    steps["num_steps"] = 5
    save("60_flow_steps.json", plan("flow-step-pilot", "smolvla", [steps], smol_tasks, [11],
                                    "compare descriptively with the same fixed15 num_steps=10 cases in stage50"))


if __name__ == "__main__":
    main()
