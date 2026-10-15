"""Build a final evidence summary only after every frozen campaign stage completes.

Run with: python scripts/run.py scripts/summarize_campaign.py --output outputs/campaign
Optionally add --markdown docs/RESULTS.md. No policies, simulators, or plots run.
The development choice is verified against development evidence, never reselected
using test outcomes. Raw results, protocols, and frozen plans remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from embodied_research.experiment import digest_json, expand_plan, utc_now, write_json
from embodied_research.selection import choose_candidate
from embodied_research.statistics import (
    ExperimentIntegrityError, compare_flow_steps, compare_observation_delay,
    load_experiment, summarize_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ("coverage", "10_coverage"), ("development", "20_development"),
    ("methods", "30_methods"), ("delay", "40_delay"),
    ("smolvla", "50_smolvla"), ("flow_steps", "60_flow_steps"),
)
RUNTIME_SOURCES = ("experiment.py", "policies.py", "runtime.py", "simulator.py", "scheduling.py")


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentIntegrityError(f"duplicate JSON key {key}: {path.name}")
            result[key] = value
        return result

    def reject(value):
        raise ExperimentIntegrityError(f"non-finite JSON value {value}: {path.name}")

    if not path.is_file():
        raise ExperimentIntegrityError(f"required completed-campaign evidence is missing: {path}")
    raw = path.read_bytes()
    result = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ExperimentIntegrityError(f"expected JSON object: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def _verify_frozen_sources(freeze: dict[str, Any]) -> None:
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or not {
        "src/embodied_research/statistics.py", "src/embodied_research/selection.py",
    }.issubset(sources):
        raise ExperimentIntegrityError("campaign freeze must bind statistics and development selection sources")
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ExperimentIntegrityError(f"invalid frozen source path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ExperimentIntegrityError(f"frozen source changed since campaign start: {name}")


def _exact_comparison(comparisons: list[dict[str, Any]], model: str, method: str, condition: str) -> dict[str, Any]:
    rows = [row for row in comparisons if (row["model"], row["method"], row["condition"]) == (model, method, condition)]
    if len(rows) != 1 or rows[0]["unmatched_candidate"] or rows[0]["unmatched_baseline"]:
        raise ExperimentIntegrityError(f"expected one fully paired {model}/{condition}/{method} comparison")
    return rows[0]


def _link(path: Path, markdown: Path) -> str:
    return quote(Path(os.path.relpath(path, markdown.parent)).as_posix(), safe="/")


def _ci(row: dict[str, Any]) -> str:
    interval = row["task_cluster_95ci"]
    return "不可估计" if interval is None else f"[{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}]"


def _method_table(groups: list[dict[str, Any]]) -> str:
    lines = ["| 方法 / 条件 | 任务 / 回合 | 成功 / 回合 | 任务宏成功率 | 均值调用 | 调用 / 百步 | 均值模型秒数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        rate = row["inference_calls_per_100_steps"]
        lines.append(f"| {row['method']} / {row['condition']} | {row['tasks']} / {row['episodes']} | "
                     f"{row['successes']} / {row['episodes']} | {row['macro_task_success'] * 100:.1f}% | "
                     f"{row['mean_inference_calls']:.2f} | {'—' if rate is None else f'{rate:.2f}'} | "
                     f"{row['mean_model_seconds']:.2f} |")
    return "\n".join(lines)


def _contrast_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| 方法 | 任务 / 配对回合 | 成功率差 / 百分点 | 任务簇 95% 区间 | 均值调用差 |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        calls = row.get("mean_calls_difference", row["candidate_mean_calls"] - row["baseline_mean_calls"])
        lines.append(f"| {row['method']} | {row['paired_tasks']} / {row['paired_episodes']} | "
                     f"{row['macro_success_difference'] * 100:+.1f} | {_ci(row)} | {calls:+.2f} |")
    return "\n".join(lines)


def _markdown(summary: dict[str, Any], output: Path, destination: Path) -> str:
    counts, stages, comparisons = summary["counts"], summary["stages"], summary["comparisons"]
    selection = summary["development_selection"]
    threshold = summary["selected_thresholds"]
    lines = [
        "# FeedbackBench：冻结预训练策略的反馈与推理预算实测",
        "",
        f"本页由完整原始记录生成，生成时间：{summary['generated_at']}。"
        f"共完成 {counts['total_completed_episodes']} 个计划回合，其中覆盖诊断 {counts['coverage_episodes']}、"
        f"开发 {counts['development_episodes']}、保留实验 {counts['held_out_episodes']}。"
        "各阶段分别解释；重复方法、初态和开发回合不能合并成独立泛化样本或单一总成功率。integration smoke 不额外计数。",
        "",
        "## 完成范围与可追溯证据", "",
        "| 阶段 | 任务簇 | 完成 / 计划回合 | 原始协议 |", "|---|---:|---:|---|",
    ]
    for name, stem in STAGES:
        stage = stages[name]
        link = _link(output / stem / "protocol.json", destination)
        lines.append(f"| {stage['experiment_name']} | {stage['tasks']} | "
                     f"{stage['completed_episodes']} / {stage['planned_episodes']} | [协议]({link}) |")
    lines += ["", "## 覆盖诊断", "", "上游执行语义，每任务一次初态；不作为调参集，也不是上游论文等规模复现。", "",
              _method_table(stages["coverage"]["groups"]), "", "## 仅开发集确定的阈值", "",
              f"选择 `{selection['selected']}`：图像阈值 `{threshold['vision_threshold']}`，"
              f"夹爪阈值 `{threshold['gripper_threshold']}`。"
              f"是否启用预声明回退规则：{'是' if selection['fallback_used'] else '否'}。"
              "选择结果已按开发原始文件哈希、协议指纹及预声明规则复核，没有读取测试收益重新选择。",
              "", f"[开发选择证据]({_link(output / 'development_selection.json', destination)})", "",
              "## MINERVA 保留初态：质量与推理预算", "", _method_table(stages["methods"]["groups"]), "",
              "开发集选定的 adaptive 对 fixed8 是研究主线。下表完整列出所有预声明候选对 fixed8 的配对比较，"
              "方向均为候选减 fixed8；调用差为配对回合均值。", "",
              _contrast_table(comparisons["all_main_methods_vs_fixed8"]), "",
              "## 组件消融", "", "方向为关闭组件的变体减完整 adaptive；配对检查要求仅改变对应组件。", "",
              _contrast_table(comparisons["direct_ablations"]), "",
              "## 四步观测延迟", "", "分别报告同一方法的 delay4 减 clean。此表没有检验方法与条件的交互。", "",
              _contrast_table(comparisons["delay"]["comparisons"]), "",
              "## SmolVLA CPU 受限对照", "", _method_table(stages["smolvla"]["groups"]), "",
              "相对 fixed8 的同模型配对对照：", "", _contrast_table(comparisons["smolvla_vs_fixed8"]), "",
              "## SmolVLA flow 积分步数", ""]
    flow = comparisons["flow_steps"]
    lines += [f"固定 15 步且关闭时间集成；{flow['paired_tasks']} 个任务、{flow['paired_episodes']} 对回合。"
              f"5 减 10 次 flow 积分的任务宏成功率差为 {flow['macro_success_difference'] * 100:+.1f} 百分点，"
              f"任务簇 95% 区间 {_ci(flow)}。", "",
              "| 任务 | 初态 / 种子 | 成功 10 / 5 | 调用 10 / 5 | 模型秒数 10 / 5 |",
              "|---|---:|---:|---:|---:|"]
    for row in flow["per_pair"]:
        lines.append(f"| {row['suite']}:{row['task_id']} | {row['init_state_id']} / {row['seed']} | "
                     f"{int(row['success_10'])} / {int(row['success_5'])} | {row['calls_10']} / {row['calls_5']} | "
                     f"{row['model_seconds_10']:.2f} / {row['model_seconds_5']:.2f} |")
    lines += ["", "## 能支持与不能支持的结论", "",
              "这些数字描述冻结公开权重在已知任务、预声明初态和本机运行条件下的表现。"
              "MINERVA 使用任务 ID 条件，不称为 VLA；SmolVLA 小样本不能概括完整基准。"
              "项目没有重新训练模型，不声称方法首创、SOTA、未见任务或真机泛化。", "",
              "任务宏平均对每个任务等权。区间使用 2,000 次任务簇重采样，以现有初态和权重为条件，"
              "少量任务的区间不稳定，未做多重比较校正。零宽度区间可能仅反映已观察任务平均配对效应没有变异；"
              "不证明等效、非劣效或真实不确定性为零。", "",
              "成功提前结束会减少总调用数，因此同时报告每百步调用率。Windows 共享电脑的耗时为描述性结果；"
              "仿真在推理时暂停，delay4 是人为传感延迟，不是真实异步控制的计算延迟。", "",
              f"[机器可读完整统计及来源指纹]({_link(output / 'campaign_summary.json', destination)}) · "
              f"[源码冻结]({_link(output / 'campaign_freeze.json', destination)})"]
    report = output / "report.html"
    if report.is_file():
        lines += ["", f"[阶段报告与回放总览]({_link(report, destination)})"]
    return "\n".join(lines) + "\n"


def summarize_campaign(campaign_dir: str | Path, *, markdown_path: str | Path | None = None) -> Path:
    """Validate all six complete stages and write final JSON plus Markdown evidence."""
    output = Path(campaign_dir).resolve()
    destination = Path(markdown_path).resolve() if markdown_path is not None else output / "RESULTS.md"
    if destination.suffix.lower() != ".md":
        raise ValueError("Markdown destination must have a .md suffix")
    freeze, freeze_hash = _read_object(output / "campaign_freeze.json")
    status, status_hash = _read_object(output / "campaign_status.json")
    names = [name for name, _ in STAGES]
    if (freeze.get("stage_order") != names or freeze.get("integration_smoke_included") is not False
            or status.get("status") != "complete_through_requested_stage" or status.get("through") != "flow_steps"
            or any(status.get("stages", {}).get(name, {}).get("status") != "complete" for name in names)):
        raise ExperimentIntegrityError("final summary requires all six campaign stages completed; partial results refused")
    _verify_frozen_sources(freeze)
    data, stages, inputs, fingerprints = {}, {}, {}, {}
    for name, stem in STAGES:
        stage = load_experiment(output / stem)
        plan, plan_hash = _read_object(output / "frozen_plans" / f"{stem}.json")
        if stage.protocol.get("plan") != plan or stage.protocol["cases"] != expand_plan(plan):
            raise ExperimentIntegrityError(f"stage protocol/cases differ from frozen plan: {name}")
        provenance = {field: stage.protocol.get(field) for field in ("plan", "sources", "runtime_versions", "artifacts")}
        if stage.protocol.get("fingerprint") != digest_json(provenance):
            raise ExperimentIntegrityError(f"stage protocol fingerprint is invalid: {name}")
        stage_sources = stage.protocol.get("sources")
        if not isinstance(stage_sources, dict) or set(stage_sources) != set(RUNTIME_SOURCES):
            raise ExperimentIntegrityError(f"stage is missing frozen runtime source evidence: {name}")
        for source, checksum in stage_sources.items():
            if freeze["source_sha256"].get(f"src/embodied_research/{source}") != checksum:
                raise ExperimentIntegrityError(f"stage source differs from campaign freeze: {name}/{source}")
        result = summarize_experiment(stage)
        if (status["stages"][name].get("completed") != result["completed_episodes"]
                or status["stages"][name].get("total") != result["planned_episodes"]):
            raise ExperimentIntegrityError(f"stage counts differ from completed campaign status: {name}")
        data[name] = stage
        stages[name] = {**result, "directory": stem, "experiment_name": plan["name"],
                        "split": plan.get("split"), "frozen_provenance": provenance}
        inputs.update({f"{stem}/{path}": value for path, value in stage.input_sha256.items()})
        inputs[f"frozen_plans/{stem}.json"] = plan_hash
        fingerprints[name] = stage.protocol["fingerprint"]

    development_trials = {(row["suite"], row["task_id"], row["init_state_id"]) for row in data["development"].records}
    for name in names[2:]:
        trials = {(row["suite"], row["task_id"], row["init_state_id"]) for row in data[name].records}
        if development_trials & trials:
            raise ExperimentIntegrityError(f"development initial states leaked into held-out stage: {name}")

    selection, selection_hash = _read_object(output / "development_selection.json")
    dev_selection, dev_selection_hash = _read_object(output / "20_development/selection.json")
    development = data["development"]
    if development.protocol["plan"]["name"] != "development-calibration":
        raise ExperimentIntegrityError("threshold selection requires the declared development-calibration stage")
    expected_selection = {
        **choose_candidate(stages["development"]["groups"]),
        "development_protocol_fingerprint": development.protocol["fingerprint"],
        "development_inputs_sha256": digest_json(development.input_sha256),
        "rule": development.protocol["plan"]["selection_rule"], "test_results_read": False,
    }
    if selection != expected_selection or selection != dev_selection:
        raise ExperimentIntegrityError("development selection differs from frozen evidence or predeclared rule")
    selected_rows = [row for row in development.records if row["method"] == selection["selected"]]
    threshold_fields = ("vision_threshold", "gripper_threshold")
    thresholds = {field: selected_rows[0]["scheduler_config"][field] for field in threshold_fields}
    for stage_name in ("development", "methods", "delay", "smolvla"):
        for row in data[stage_name].records:
            if row["method"] == selection["selected"] or row["method"] in {"adaptive", "no_vision", "no_gripper", "no_ensemble"}:
                if {field: row["scheduler_config"].get(field) for field in threshold_fields} != thresholds:
                    raise ExperimentIntegrityError(f"selected development threshold changed in stage {stage_name}")
    main = _exact_comparison(stages["methods"]["paired_comparisons"], "minerva", "adaptive", "clean")
    ablations = stages["methods"]["ablation_comparisons"]
    if {row["method"] for row in ablations} != {"no_vision", "no_gripper", "no_ensemble"}:
        raise ExperimentIntegrityError("final summary requires all three direct component ablations")
    smol = [_exact_comparison(stages["smolvla"]["paired_comparisons"], "smolvla", method, "clean")
            for method in ("adaptive", "fixed15")]
    inputs.update({"campaign_freeze.json": freeze_hash, "campaign_status.json": status_hash,
                   "development_selection.json": selection_hash, "20_development/selection.json": dev_selection_hash})
    summary = {
        "schema_version": 1, "status": "complete", "generated_at": utc_now(), "stage_order": names,
        "counts": {"total_completed_episodes": sum(stage["completed_episodes"] for stage in stages.values()),
                   "coverage_episodes": stages["coverage"]["completed_episodes"],
                   "development_episodes": stages["development"]["completed_episodes"],
                   "held_out_episodes": sum(stages[name]["completed_episodes"] for name in names[2:]),
                   "integration_smoke_episodes_included": 0},
        "counts_scope": "stage-separated repeated-method episodes; never independent samples or a pooled success rate",
        "stages": stages, "development_selection": selection, "selected_thresholds": thresholds,
        "comparisons": {"main_adaptive_vs_fixed8": main,
                        "all_main_methods_vs_fixed8": stages["methods"]["paired_comparisons"],
                        "direct_ablations": ablations,
                        "delay": compare_observation_delay(data["methods"], data["delay"]),
                        "smolvla_vs_fixed8": smol,
                        "flow_steps": compare_flow_steps(data["smolvla"], data["flow_steps"])},
        "campaign_freeze": freeze, "stage_protocol_fingerprints": fingerprints, "input_sha256": inputs,
        "summary_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Frozen published weights, known tasks, no claim of unseen-task or real-robot generalization.",
                        "MINERVA is task-ID conditioned, not a VLA; SmolVLA is a limited CPU pilot.",
                        "Task-cluster intervals condition on observed trials; small samples and no multiplicity correction.",
                        "Zero-width intervals do not establish equivalence, noninferiority, or zero true uncertainty.",
                        "Episode calls depend on early success; shared-computer wall time is descriptive.",
                        "No novelty or SOTA claim; integration smoke is excluded from campaign counts."],
    }
    markdown = _markdown(summary, output, destination)
    # No final output exists until every stage, pairing, selection, and source check passes.
    write_json(output / "campaign_summary.json", summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(destination)
    return output / "campaign_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/campaign")
    parser.add_argument("--markdown", type=Path, help="Default: <campaign directory>/RESULTS.md")
    arguments = parser.parse_args()
    print(summarize_campaign(arguments.output, markdown_path=arguments.markdown))


if __name__ == "__main__":
    main()
