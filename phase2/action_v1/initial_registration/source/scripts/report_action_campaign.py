"""Report every planned seed/case of the frozen expert-action study."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_campaign_common import digest, now, read, verify_protocol, verify_seal, write_json
from run_action_campaign import completed_training


def verify_episode_report(result, case, protocol, fingerprint, manifest_sha):
    horizon = protocol["evaluation"]["horizons"][case["suite"]]
    if (result.get("status") != "complete" or result.get("case") != case
            or type(result.get("official_success")) is not bool
            or type(result.get("terminated")) is not bool or type(result.get("truncated")) is not bool
            or type(result.get("steps")) is not int or not 0 < result["steps"] <= horizon
            or result.get("horizon") != horizon
            or not (result["terminated"] or (result["truncated"] and result["steps"] == horizon))
            or result.get("protocol_sha256") != fingerprint or result.get("trained_manifest_sha256") != manifest_sha):
        raise ValueError("Incomplete or differently bound episode cannot enter the success denominator")


def paired_summary(rows, left_family, right_family, condition):
    """Average training seeds within each task before four-task group bootstrap."""
    import numpy as np

    paired, per_task = [], {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        differences = []
        for initial in (5, 15, 25, 35, 45):
            for seed in (17, 23, 47):
                left = f"{left_family}_s{seed}"
                right = "frozen" if right_family == "frozen" else f"{right_family}_s{seed}"
                a = rows.get((suite, initial, condition, left))
                b = rows.get((suite, initial, condition, right))
                if a is None or b is None:
                    return {"status": "pending_complete_pairs", "estimate": None}
                delta = int(a) - int(b)
                differences.append(delta)
                paired.append({"suite": suite, "initial_state_id": initial, "seed": seed, "difference": delta})
        per_task[suite] = float(np.mean(differences))
    values = np.asarray(list(per_task.values()))
    generator = np.random.default_rng(20261022)
    bootstrap = values[generator.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {"status": "complete_pairs", "estimate": float(values.mean()),
            "descriptive_task_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "task_groups": 4, "initial_states_per_task": 5, "training_seeds": 3,
            "per_task": per_task, "paired_differences": paired,
            "limitation": "Only four task groups. Training seeds share evaluation cases; this is not 60 independent tasks."}


def build_report(campaign):
    campaign = Path(campaign).resolve()
    fingerprint = (campaign / "protocol.sha256").read_text().strip()
    protocol = verify_protocol(campaign / "protocol.json", fingerprint)
    trained = completed_training(campaign, protocol, fingerprint)
    manifest = campaign / "trained_manifest.json"
    manifest_sha = digest(manifest) if manifest.exists() else None
    if manifest.exists():
        frozen = read(manifest)
        if (frozen.get("protocol_sha256") != fingerprint or frozen.get("runs") != trained
                or len(trained) != len(protocol["runs"])):
            raise ValueError("Frozen trained manifest differs from all completed runs")
    methods = {m["method_id"]: m for m in protocol["evaluation"]["methods"]}
    results, missing, receipts, rows = [], [], [], {}
    for case in protocol["evaluation"]["cases"]:
        paths = list((campaign / "evaluation" / case["case_id"]).glob("attempt_*/receipt.json"))
        if len(paths) > 1:
            raise ValueError("Duplicate case results cannot select a favorable attempt")
        if not paths:
            missing.append(case["case_id"])
            continue
        if manifest_sha is None:
            raise ValueError("Evaluation was run before all trained checkpoints froze")
        path = paths[0].parent
        verify_seal(path, {"protocol_sha256": fingerprint, "trained_manifest_sha256": manifest_sha, "case_id": case["case_id"]})
        result = read(path / "report.json")
        verify_episode_report(result, case, protocol, fingerprint, manifest_sha)
        results.append(result)
        receipts.append({"path": str(paths[0].relative_to(campaign)).replace("\\", "/"), "sha256": digest(paths[0])})
        rows[(case["suite"], case["initial_state_id"], case["condition"], case["method_id"])] = result["official_success"]
    rates = []
    for method in methods:
        for condition in ("clean", "brightness075"):
            selected = [r for r in results if r["case"]["method_id"] == method and r["case"]["condition"] == condition]
            planned = [c for c in protocol["evaluation"]["cases"] if c["method_id"] == method and c["condition"] == condition]
            successful = sum(r["official_success"] for r in selected)
            rates.append({"method_id": method, "condition": condition, "completed": len(selected), "planned": len(planned),
                          "successes": successful, "failures": len(selected) - successful,
                          "observed_success_fraction": successful / len(selected) if selected else None,
                          "coverage_complete": len(selected) == len(planned),
                          "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected),
                          "inference_seconds": sum(r["inference_seconds"] for r in selected)})
    complete = not missing and len(trained) == len(protocol["runs"])
    return {"schema": "feedbackbench.action_report.v1", "generated_at": now(), "status": "complete" if complete else "partial",
            "protocol_sha256": fingerprint, "trained_manifest_sha256": manifest_sha,
            "training_completed": len(trained), "training_planned": len(protocol["runs"]),
            "evaluation_completed": len(results), "evaluation_planned": len(protocol["evaluation"]["cases"]),
            "rates": rates, "missing_cases": missing, "sealed_receipts": receipts,
            "contrasts": {f"lora_minus_{right}_{condition}": paired_summary(rows, "lora", right, condition)
                          for right in ("frozen", "output_head") for condition in ("clean", "brightness075")},
            "claims": {"same_domain_adaptation": True, "unseen_pretraining_tasks": False, "real_robot": False,
                       "all_seed_results_retained": True, "training_loss_is_task_success": False},
            "limitations": ["Four fixed tasks and five paired initial states per task limit generalization and precision.",
                            "Foundation-model pretraining overlap with LIBERO is unknown.",
                            "Evaluation initial-state independence from demonstration generation is unverified; this is prospective same-domain adaptation.",
                            "Demonstration indexing and observation reconstruction were checked; complete dynamics trajectory replay was not verified.",
                            "Brightness stress is one controlled visual condition, not general robustness.",
                            "All training uses this CPU computer; the visual-language backbone stays frozen.",
                            "No efficacy claim is established by a partial run or a decreasing training loss."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.campaign)
    destination = args.output or args.campaign / "report.json"
    write_json(destination, report)
    print({key: report[key] for key in ("status", "training_completed", "training_planned", "evaluation_completed", "evaluation_planned")})


if __name__ == "__main__":
    main()
