"""One independent LIBERO-10 VLA-JEPA resource-budget diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vlajepa_compact as compact

DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_long_horizon"
MIN_FREE_GIB, MAX_RSS_GIB, MIN_RUNTIME_FREE_GIB = 5.75, 6.0, 0.5
SOURCES = ("scripts/run_vlajepa_long_horizon.py", "docs/VLAJEPA_LONG_HORIZON_PROTOCOL.md")


def separate_output(output):
    output = compact.separate_output(output)
    for previous in (compact.DEFAULT_OUTPUT.resolve(),
                     (ROOT / "outputs/frontier/vlajepa_isolated_closed_loop").resolve()):
        if output.is_relative_to(previous) or previous.is_relative_to(output):
            raise ValueError("The previous experiment must remain separate")
    return output


def configured_runner():
    runner = compact.configured_runner()
    prior_cases, prior_spec, prior_verify = runner.cases_for_study, runner.study_spec, runner.verify_study
    runner.SOURCES = (*runner.SOURCES, *SOURCES)
    runner.MIN_FREE_GIB, runner.MAX_RSS_GIB = MIN_FREE_GIB, MAX_RSS_GIB
    runner.MIN_RUNTIME_FREE_GIB = MIN_RUNTIME_FREE_GIB

    def cases():
        selected = [row for row in prior_cases() if row["suite"] == "libero_10"]
        runner.require(len(selected) == 1, "Expected exactly one inherited LIBERO-10 case")
        row = selected[0]
        runner.require((row["task_id"], row["init_state_id"], row["seed"]) == (1, 41, 1000),
                       "Inherited long-task identity differs")
        row["model"] = "vlajepa_long_horizon"
        row["split"] = "long_horizon_resource_feasibility"
        row["id"] = runner.episode_id(row)
        return selected

    def spec():
        result = prior_spec()
        result["name"] = "VLA-JEPA single LIBERO-10 task with a separate 6 GiB RSS budget"
        result["cases"] = cases()
        result["memory_guard"] = {"minimum_available_gib": MIN_FREE_GIB, "maximum_rss_gib": MAX_RSS_GIB,
                                  "minimum_runtime_available_gib": MIN_RUNTIME_FREE_GIB, "sampling_seconds": 1}
        result["resource_diagnostic"] = {
            "planned_episodes": 1, "previous_results_imported": False,
            "selection": "Previously resource-interrupted LIBERO-10 task 1, init41, seed1000",
            "numeric_physics_and_control_settings": "unchanged from compact protocol",
            "budget_change": "5.75 GiB before imports and loading; 6 GiB RSS; unchanged 0.5 GiB runtime floor"}
        result["limitations"] = [item for item in result["limitations"]
                                 if not item.startswith("four known tasks,") and not item.startswith("3.5 GiB gate")]
        result["limitations"].extend([
            "One selected previously used task and initial state; exploratory resource feasibility, not generalization",
            "Independent one-case protocol and budget; cannot combine previous successes into a four-of-four experiment",
            "Every earlier interrupted attempt remains retained separately; no automatic retry of unsealed evidence",
            "Frozen engine stdout may display [1/4]; this protocol and all saved reports/progress plan exactly one case",
            "Frozen LIBERO reset may swallow repeated native-warning exceptions; a hung reset requires external timeout"])
        return result

    def verify(output, *, partial=False):
        result = prior_verify(output, partial=True)
        case_id = cases()[0]["id"]
        runner.require(set(result["statuses"]) == {case_id}, "Single-task episode inventory differs")
        rows = result["results"]
        runner.require(len(rows) <= 1 and result["completed"] == len(rows)
                       and all(row["id"] == case_id for row in rows), "Single-task result inventory differs")
        complete = len(rows) == 1
        runner.require((result["statuses"][case_id] == "complete") == complete,
                       "Single-task completion status differs")
        result.update(planned=1, complete=complete)
        runner.require(partial or complete, "Single-task diagnostic incomplete; partial must be explicit")
        return result

    runner.cases_for_study, runner.study_spec, runner.verify_study = cases, spec, verify
    runner.write_report = lambda output: write_report(runner, output)
    return runner


def write_report(runner, output):
    output = Path(output)
    study = runner.verify_study(output, partial=True)
    study["run_audits"] = []
    for path in sorted((output / "runs").glob("run_*.json")):
        record = runner.read_json(path)
        study["run_audits"].append({"path": path.relative_to(output).as_posix(), "sha256": runner.file_sha(path),
            "status": record["status"], "weights_loaded": record.get("weights_loaded", False),
            "memory": {key: value for key, value in record.get("memory", {}).items() if key != "samples"},
            "load_quantize_seconds": record.get("load_quantize_seconds"), "imports_seconds": record.get("imports_seconds")})
    compact.reliable_write_json(runner.write_json, output / "report.json", study)
    rows = study["results"]
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ("id", "suite", "task", "success", "steps", "horizon", "inference_calls", "policy_inference_seconds", "wall_seconds")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# VLA-JEPA LIBERO-10 single-task resource diagnostic", "",
             f"Planned **1** episode; completed **{study['completed']}**; official successes **{study['successes']}**; completed task failures **{study['task_failures']}**.",
             "Completion and task success are separate. Missing, unsealed and error statuses remain in report.json.", "",
             "This independent one-case protocol uses a 5.75 GiB preflight gate, 6 GiB RSS cap and 0.5 GiB runtime available-memory floor.",
             "Earlier successes and interrupted attempts are separate evidence, not a combined four-of-four result.", "",
             "| Suite | Official success | Steps / horizon | Policy calls | Action-head NFE | Inference seconds | Episode wall seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['suite']} | {row['success']} | {row['steps']} / {row['horizon']} | {row['inference_calls']} | {row['model_info']['action_head_nfe']} | {row['policy_inference_seconds']:.3f} | {row['wall_seconds']:.3f} |")
    lines += ["", "Simulation pauses during inference. The 20 Hz simulated control frequency is not model throughput. Episode wall time excludes model loading and GIF encoding.",
              "Shared-machine timings and one-second RSS samples do not establish stable throughput or instantaneous peak memory.", "", "## Run and memory evidence", ""]
    for audit in study["run_audits"]:
        memory = audit["memory"]
        lines.append(f"- [Run record]({audit['path']}): `{audit['status']}`; sampled peak RSS `{memory.get('peak_rss_bytes')}` bytes; minimum available system memory `{memory.get('minimum_available_bytes')}` bytes.")
    for row in rows:
        path = f"episodes/{row['id']}"
        lines += ["", f"## {row['suite']}", "", row["task"], "",
                  f"[Result]({path}/result.json) · [Calls]({path}/calls.json) · [Trajectory]({path}/trajectory.npz)", "",
                  f"![Recorded closed loop]({path}/rollout.gif)"]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in study["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {key: value for key, value in study.items() if key != "results"}


def run(runner, output):
    output = separate_output(output)
    original_write = runner.write_json
    original_seal_write = runner.seal_episode.__globals__["write_json"]

    def progress_writer(path, value):
        if Path(path).resolve() == (output / "progress.json").resolve() and "planned" in value:
            value = {**value, "planned": 1}
        return original_write(path, value)

    seal_writer = lambda path, value: compact.reliable_write_json(original_seal_write, path, value)
    with patch.object(runner, "write_json", progress_writer), patch.dict(runner.seal_episode.__globals__, {"write_json": seal_writer}):
        outcome = compact.run(runner, output, limit=1)
    return runner.write_report(output) if "completed" in outcome else outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    output = separate_output(args.output)
    runner = configured_runner()
    if args.command == "freeze":
        result = runner.run(output, freeze_only=True)
    elif args.command == "run":
        result = run(runner, output)
    elif args.command == "report":
        result = runner.write_report(output)
    else:
        result = runner.verify_study(output, partial=args.partial)
        result.pop("results")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
