"""Post-hoc scheduler-log diagnostics; never a success mechanism or threshold tuner.

Example: python scripts/run.py scripts/analyze_triggers.py --stage outputs/campaign/30_methods
         --stage outputs/campaign/40_delay --stage outputs/campaign/50_smolvla
Outputs are saved inside each stage, alongside its raw evidence and report.
Only complete stage corpora are accepted unless --partial is explicitly given.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any
from urllib.parse import quote

import numpy as np

from embodied_research.experiment import write_json
from embodied_research.scheduling import SchedulerConfig
from embodied_research.statistics import ExperimentIntegrityError, load_experiment


REASONS = {"initial", "chunk_exhausted", "fixed_interval", "max_interval", "vision_change", "gripper_change"}
NOTE = ("事后日志诊断，仅描述实际重规划次数、触发组合和间隔。信号原因可以重叠；"
        "边际原因计数不能相加作为独立推理次数。此分析不证明某个信号导致成功或失败，"
        "组件贡献须参考预声明消融；不会修改已选择的阈值。")


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentIntegrityError(f"{label} must be an integer >= {minimum}")
    return value


def _integers(value: Any, label: str, minimum: int = 0) -> list[int]:
    if not isinstance(value, list):
        raise ExperimentIntegrityError(f"{label} must be a list")
    return [_integer(item, label, minimum) for item in value]


def _interval_summary(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values), "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "histogram": {str(key): count for key, count in sorted(Counter(values).items())},
    }


def validate_episode_log(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct requests and spans, rejecting contradictions in complete logs."""
    identifier = record["id"]
    if record.get("status") != "complete":
        raise ExperimentIntegrityError("trigger diagnostics accept complete episode records only")
    log = record.get("decisions")
    if not isinstance(log, dict):
        raise ExperimentIntegrityError(f"scheduler decisions required; upstream-only records are unsupported: {identifier}")
    config = record.get("scheduler_config")
    if not isinstance(config, dict) or log.get("config") != config:
        raise ExperimentIntegrityError(f"scheduler config/log mismatch: {identifier}")
    try:
        if asdict(SchedulerConfig(**config)) != config:
            raise ValueError("full resolved scheduler configuration required")
    except (TypeError, ValueError) as error:
        raise ExperimentIntegrityError(f"invalid scheduler configuration: {identifier}: {error}") from error
    steps = _integer(record.get("steps"), "record steps", 1)
    calls = _integer(record.get("inference_calls"), "record inference_calls", 1)
    if (_integer(log.get("steps"), "log steps") != steps
            or _integer(config.get("horizon"), "config horizon", 1) != record["horizon"]
            or log.get("horizon_reached") is not (steps >= record["horizon"])):
        raise ExperimentIntegrityError(f"scheduler step/horizon mismatch: {identifier}")
    if log.get("pending_inference") is not False:
        raise ExperimentIntegrityError(f"completed episode contains pending inference: {identifier}")
    events = log.get("events")
    if not isinstance(events, list) or len(events) != steps:
        raise ExperimentIntegrityError(f"one scheduler event per executed step required: {identifier}")
    marginal: Counter[str] = Counter()
    exclusive: Counter[str] = Counter()
    plans, lengths = [], []
    active_chunks: list[tuple[int, int]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or _integer(event.get("step"), "event step") != index:
            raise ExperimentIntegrityError(f"scheduler events must be consecutive from step zero: {identifier}")
        reasons = event.get("reasons")
        if (not isinstance(reasons, list) or any(not isinstance(reason, str) or reason not in REASONS for reason in reasons)
                or len(set(reasons)) != len(reasons)):
            raise ExperimentIntegrityError(f"invalid or duplicate trigger reasons: {identifier}")
        requested = bool(reasons)
        if event.get("requested") is not requested or event.get("chunk_received") is not requested:
            raise ExperimentIntegrityError(f"request/reasons/chunk receipt mismatch: {identifier}")
        if event.get("action_returned") is not True:
            raise ExperimentIntegrityError(f"completed event did not return an action: {identifier}")
        age = None if not plans else index - plans[-1]
        recorded_age = event.get("steps_since_plan")
        if recorded_age is not None:
            _integer(recorded_age, "steps_since_plan", 1)
        if recorded_age != age:
            raise ExperimentIntegrityError(f"steps_since_plan disagrees with request history: {identifier}")
        if index == 0 and reasons != ["initial"] or index > 0 and "initial" in reasons:
            raise ExperimentIntegrityError(f"initial trigger must occur alone at step zero: {identifier}")
        if any(reason in reasons for reason in ("initial", "chunk_exhausted", "fixed_interval")) and len(reasons) != 1:
            raise ExperimentIntegrityError(f"exclusive scheduler trigger combined with another reason: {identifier}")
        if "vision_change" in reasons and (config.get("mode") != "adaptive" or config.get("use_vision") is not True):
            raise ExperimentIntegrityError(f"vision trigger recorded with its signal disabled: {identifier}")
        if "gripper_change" in reasons and (config.get("mode") != "adaptive" or config.get("use_gripper") is not True):
            raise ExperimentIntegrityError(f"gripper trigger recorded with its signal disabled: {identifier}")
        if "fixed_interval" in reasons and config.get("mode") != "fixed":
            raise ExperimentIntegrityError(f"fixed trigger recorded in adaptive mode: {identifier}")
        if "max_interval" in reasons and config.get("mode") != "adaptive":
            raise ExperimentIntegrityError(f"adaptive maximum trigger recorded in fixed mode: {identifier}")
        if plans and index >= plans[-1] + lengths[-1] and reasons != ["chunk_exhausted"]:
            raise ExperimentIntegrityError(f"exhausted chunk did not cause a compulsory request: {identifier}")
        if "chunk_exhausted" in reasons and (not plans or index < plans[-1] + lengths[-1]):
            raise ExperimentIntegrityError(f"chunk_exhausted before the current chunk ended: {identifier}")
        for field in ("image_delta", "gripper_delta"):
            value = event.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ExperimentIntegrityError(f"invalid logged {field}: {identifier}")
        for field, enabled in (("image_delta", config["use_vision"]), ("gripper_delta", config["use_gripper"])):
            must_exist = index > 0 and config["mode"] == "adaptive" and enabled
            if (event.get(field) is not None) != must_exist:
                raise ExperimentIntegrityError(f"logged {field} presence disagrees with signal configuration: {identifier}")
        # Recheck decisions against their logged features, not against unobserved
        # sensor truth or alternate thresholds. This is an internal log audit.
        expected = []
        if index == 0:
            expected = ["initial"]
        elif index >= plans[-1] + lengths[-1]:
            expected = ["chunk_exhausted"]
        elif config["mode"] == "fixed":
            if age >= config["fixed_interval"]:
                expected = ["fixed_interval"]
        elif age >= config["min_interval"]:
            if age >= config["max_interval"]:
                expected.append("max_interval")
            if event.get("image_delta") is not None and event["image_delta"] >= config["vision_threshold"]:
                expected.append("vision_change")
            if event.get("gripper_delta") is not None and event["gripper_delta"] >= config["gripper_threshold"]:
                expected.append("gripper_change")
        if set(reasons) != set(expected):
            raise ExperimentIntegrityError(f"trigger reasons disagree with logged features/interval rules: {identifier}")
        if requested:
            length = _integer(event.get("chunk_length"), "received chunk_length", 1)
            plans.append(index)
            lengths.append(length)
            active_chunks.append((index, index + length))
            marginal.update(reasons)
            exclusive[" + ".join(sorted(reasons))] += 1
        elif "chunk_length" in event:
            raise ExperimentIntegrityError(f"unrequested event claims a received chunk length: {identifier}")
        active_chunks = [(start, end) for start, end in active_chunks if end > index]
        active_count = _integer(event.get("active_predictions"), "active_predictions", 1)
        used = _integer(event.get("used_predictions"), "used_predictions", 1)
        expected_used = len(active_chunks) if config.get("temporal_ensemble") else 1
        if active_count != len(active_chunks) or used != expected_used:
            raise ExperimentIntegrityError(f"active/used prediction counts disagree with chunk validity: {identifier}")
    if (_integer(log.get("inference_requests"), "inference_requests") != calls
            or _integer(log.get("chunks_received"), "chunks_received") != calls or len(plans) != calls):
        raise ExperimentIntegrityError(f"event requests disagree with episode inference count: {identifier}")
    if _integers(log.get("plan_steps"), "plan_steps") != plans:
        raise ExperimentIntegrityError(f"plan_steps disagree with request events: {identifier}")
    logged_counts = log.get("trigger_counts")
    if not isinstance(logged_counts, dict):
        raise ExperimentIntegrityError(f"trigger_counts must be an object: {identifier}")
    for reason, count in logged_counts.items():
        if reason not in REASONS:
            raise ExperimentIntegrityError(f"unknown trigger count reason: {identifier}")
        _integer(count, "trigger count", 1)
    if logged_counts != dict(marginal):
        raise ExperimentIntegrityError(f"trigger_counts disagree with event reasons: {identifier}")
    completed = [right - left for left, right in zip(plans, plans[1:])]
    final_span = steps - plans[-1]
    executed = [*completed, final_span]
    if (_integers(log.get("completed_replan_intervals"), "completed_replan_intervals", 1) != completed
            or _integers(log.get("executed_chunk_spans"), "executed_chunk_spans", 1) != executed
            or sum(executed) != steps or any(span > length for span, length in zip(executed, lengths))):
        raise ExperimentIntegrityError(f"replan intervals/executed spans disagree with steps or chunk lengths: {identifier}")
    rate = log.get("inference_calls_per_100_steps")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)
            or not math.isclose(rate, 100 * calls / steps, rel_tol=1e-10, abs_tol=1e-10)):
        raise ExperimentIntegrityError(f"logged call rate disagrees with steps/calls: {identifier}")
    return {
        **{key: record[key] for key in ("id", "model", "suite", "task_id", "task", "init_state_id", "seed", "condition", "method", "success")},
        "steps": steps, "inference_calls": calls, "inference_calls_per_100_steps": rate,
        "exclusive_reason_combinations": dict(sorted(exclusive.items())),
        "marginal_reason_counts": dict(sorted(marginal.items())),
        "plan_steps": plans, "completed_replan_intervals": completed, "executed_chunk_spans": executed,
        "final_censored_span": final_span,
        "completed_interval_summary": _interval_summary(completed),
        "executed_span_summary": _interval_summary(executed),
    }


def _aggregate(episodes: list[dict[str, Any]], *, task_level: bool) -> list[dict[str, Any]]:
    fields = ("model", "condition", "method", "suite", "task_id") if task_level else ("model", "condition", "method")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode[field] for field in fields)].append(episode)
    output = []
    for key, rows in sorted(groups.items()):
        marginal, exclusive = Counter(), Counter()
        completed, executed, censored = [], [], []
        for row in rows:
            marginal.update(row["marginal_reason_counts"])
            exclusive.update(row["exclusive_reason_combinations"])
            completed.extend(row["completed_replan_intervals"])
            executed.extend(row["executed_chunk_spans"])
            censored.append(row["final_censored_span"])
        calls, steps = sum(row["inference_calls"] for row in rows), sum(row["steps"] for row in rows)
        output.append({
            **dict(zip(fields, key)), "episodes": len(rows), "tasks": len({(row["suite"], row["task_id"]) for row in rows}),
            "steps": steps, "inference_calls": calls, "inference_calls_per_100_steps": 100 * calls / steps,
            "exclusive_reason_combinations": dict(sorted(exclusive.items())), "marginal_reason_counts": dict(sorted(marginal.items())),
            "completed_replan_intervals": _interval_summary(completed), "executed_chunk_spans": _interval_summary(executed),
            "final_censored_spans": _interval_summary(censored),
        })
    return output


def _plot(output: Path, summary: dict[str, Any]) -> str | None:
    groups = summary["groups"]
    if not groups:
        return None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = [f"{row['model']}/{row['condition']}/{row['method']}\n"
              f"tasks={row['tasks']}, episodes={row['episodes']}, calls={row['inference_calls']}" for row in groups]
    combinations = sorted({reason for row in groups for reason in row["exclusive_reason_combinations"]})
    fig, ax = plt.subplots(figsize=(11, max(4, .62 * len(groups) + 1.5)))
    offset = np.zeros(len(groups))
    colors = plt.get_cmap("tab20")
    for index, reason in enumerate(combinations):
        widths = np.asarray([100 * row["exclusive_reason_combinations"].get(reason, 0) / row["inference_calls"] for row in groups])
        ax.barh(labels, widths, left=offset, label=reason, color=colors(index % 20))
        offset += widths
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of inference requests (%) | exclusive combinations sum to 100%")
    prefix = "PARTIAL | " if summary["partial"] else ""
    ax.set_title(f"{prefix}{output.name} | post-hoc scheduler diagnostics\n"
                 "Observed trigger combinations; not a causal success analysis")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"trigger_combinations.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "figures/trigger_combinations.png"


def _report(output: Path, summary: dict[str, Any], figure: str | None) -> str:
    def escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    title = "部分事后诊断 · 尚未完成" if summary["partial"] else "重规划触发模式 · 事后诊断"
    text = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
    text += """<style>body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;background:#f2f6f8;color:#172d40;margin:0}
main{max-width:1400px;margin:auto;padding:36px 24px}section{background:white;border:1px solid #dce5ea;padding:24px;border-radius:12px;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce5ea;vertical-align:top}
.scroll{overflow:auto}img{width:100%;height:auto}a{color:#097c83}.note{background:#fff8e9;padding:14px;border-left:4px solid #b68123}</style><main>"""
    text += f"<h1>{title}</h1><p class=note>{NOTE}</p><p>已校验 {len(summary['episodes'])} 个完整回合；"
    text += f"排除未完成记录：{summary['missing_episodes']} 个缺失、{summary['error_episodes']} 个错误。</p>"
    if figure:
        text += f'<section><img src="{figure}" alt="按方法列示的互斥触发组合占推理请求百分比"><p><a href="{figure}">PNG</a> · '
        text += '<a href="figures/trigger_combinations.svg">SVG</a></p></section>'
    for heading, rows, task_level in (("按模型 / 条件 / 方法", summary["groups"], False),
                                       ("全部逐任务结果", summary["per_task"], True)):
        text += f"<section><h2>{heading}</h2><div class=scroll><table><thead><tr>"
        headers = ("组别", "任务 / 回合", "步数 / 调用", "调用 / 百步", "互斥原因组合（次数）", "边际原因（可重叠）",
                   "完整重规划间隔：数量 / 均值", "末段删失跨度均值")
        text += "".join(f"<th>{value}</th>" for value in headers) + "</tr></thead><tbody>"
        for row in rows:
            label = f"{row['model']}/{row['condition']}/{row['method']}"
            if task_level:
                label += f"/{row['suite']}:{row['task_id']}"
            intervals = row["completed_replan_intervals"]
            mean = "—" if intervals["mean"] is None else f"{intervals['mean']:.2f}"
            cells = (escape(label), f"{row['tasks']} / {row['episodes']}", f"{row['steps']} / {row['inference_calls']}",
                     f"{row['inference_calls_per_100_steps']:.2f}",
                     escape(json.dumps(row["exclusive_reason_combinations"], ensure_ascii=False)),
                     escape(json.dumps(row["marginal_reason_counts"], ensure_ascii=False)),
                     f"{intervals['count']} / {mean}", f"{row['final_censored_spans']['mean']:.2f}")
            text += "<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>"
        text += "</tbody></table></div></section>"
    text += ("<section><h2>间隔与证据口径</h2><p>completed_replan_intervals 只包含两次已观察推理之间的间隔，"
             "每回合最后一段没有观察到下一次推理，作为 final_censored_span 单列；不能当作完整重规划间隔。"
             "executed_chunk_spans 包含最后一段，其和必须等于已执行步数。间隔均值按实际间隔计算，"
             "不是任务等权估计；没有将日志事件当作独立实验样本做显著性检验。</p>"
             '<p><a href="trigger_diagnostics.json">全部逐回合统计、间隔分布与原始文件 SHA-256</a> · '
             '<a href="report.html">阶段实验报告</a></p>'
             "<details><summary>全部逐回合原始证据</summary><ul>")
    for row in summary["episodes"]:
        source = output / row["source_directory"] / "episodes" / row["id"] / "result.json"
        href = quote(Path(os.path.relpath(source, output)).as_posix(), safe="/")
        text += f'<li><a href="{href}">{escape(row["stage"] + "/" + row["id"])}</a>：'
        text += f'{row["inference_calls"]} 次调用，{row["steps"]} 步，最后删失跨度 {row["final_censored_span"]}</li>'
    return text + "</ul></details></section></main></html>"


def analyze_triggers(stage_dir: str | Path, *, partial: bool = False) -> Path:
    """Validate one stage and write descriptive diagnostics, never raw files."""
    sources = [Path(stage_dir).resolve()]
    episodes, stages, identities = [], {}, set()
    for source in sources:
        data = load_experiment(source, partial=partial)
        for row in data.records:
            identity = tuple(row[field] for field in ("model", "suite", "task_id", "init_state_id", "seed", "condition", "method"))
            if identity in identities:
                raise ExperimentIntegrityError("duplicate episode identity across selected stages")
            identities.add(identity)
            # Each diagnostic lives beside this stage's protocol and episodes.
            # Preserve portable references when the evidence directory is copied.
            episodes.append({**validate_episode_log(row), "stage": source.name, "source_directory": "."})
        stages[source.name] = {"planned_episodes": len(data.protocol["cases"]), "completed_episodes": len(data.records),
                               "missing_episodes": len(data.missing_ids), "error_episodes": len(data.errors),
                               "protocol_fingerprint": data.protocol.get("fingerprint"),
                               "sources": data.protocol.get("sources"), "input_sha256": data.input_sha256}
    summary = {
        "schema_version": 1, "analysis_type": "post_hoc_scheduler_diagnostic", "partial": partial,
        "note": NOTE, "stages": stages, "episodes": episodes,
        "groups": _aggregate(episodes, task_level=False), "per_task": _aggregate(episodes, task_level=True),
        "missing_episodes": sum(stage["missing_episodes"] for stage in stages.values()),
        "error_episodes": sum(stage["error_episodes"] for stage in stages.values()),
        "counting": {"exclusive_combinations": "each request contributes once; sums equal inference calls",
                     "marginal_reasons": "one request may contribute to multiple reasons; do not sum as calls",
                     "completed_replan_intervals": "between consecutive observed requests; excludes final censored span",
                     "executed_chunk_spans": "includes final censored span; sum equals executed steps"},
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = sources[0]
    figure = _plot(output, summary)
    report = _report(output, summary, figure)
    write_json(output / "trigger_diagnostics.json", summary)
    (output / "trigger_diagnostics.html").write_text(report, encoding="utf-8")
    return output / "trigger_diagnostics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, type=Path, help="Repeat for each stage to include")
    parser.add_argument("--partial", action="store_true", help="Explicitly incomplete diagnostics; invalid logs still rejected")
    arguments = parser.parse_args()
    for stage in arguments.stage:
        print(analyze_triggers(stage, partial=arguments.partial))


if __name__ == "__main__":
    main()
