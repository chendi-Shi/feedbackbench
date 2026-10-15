"""Development-only selection: freeze one candidate before reading test outcomes."""
from __future__ import annotations

from .experiment import digest_json, write_json
from .statistics import load_experiment, summarize_experiment


def choose_candidate(groups):
    scores = {group["method"]: group for group in groups}
    expected = {"fixed8", "adaptive_low", "adaptive_medium", "adaptive_high"}
    if set(scores) != expected or len(groups) != len(expected):
        raise ValueError("Selection requires exactly the three declared development candidates and fixed8")
    if any((group["model"], group["condition"]) != ("minerva", "clean") for group in groups):
        raise ValueError("This selection rule is defined only on clean MINERVA development data")
    candidates = [scores[name] for name in expected - {"fixed8"}]
    baseline = scores["fixed8"]["macro_task_success"]
    eligible = [group for group in candidates if group["macro_task_success"] >= baseline]
    if eligible:
        chosen = min(eligible, key=lambda x: (x["mean_inference_calls"], -x["macro_task_success"], x["method"]))
    else:
        chosen = min(candidates, key=lambda x: (-x["macro_task_success"], x["mean_inference_calls"], x["method"]))
    return {"selected": chosen["method"], "baseline_success": baseline,
            "eligible": sorted(group["method"] for group in eligible), "scores": groups,
            "fallback_used": not eligible}


def select_development(output_dir):
    data = load_experiment(output_dir)
    if data.protocol["plan"].get("name") != "development-calibration":
        raise ValueError("Only the declared development-calibration protocol can select thresholds")
    summary = summarize_experiment(data)
    selection = {**choose_candidate(summary["groups"]),
                 "development_protocol_fingerprint": data.protocol["fingerprint"],
                 "development_inputs_sha256": digest_json(data.input_sha256),
                 "rule": data.protocol["plan"]["selection_rule"],
                 "test_results_read": False}
    write_json(data.output_dir / "selection.json", selection)
    return selection
