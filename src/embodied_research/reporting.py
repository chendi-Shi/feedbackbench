"""Standalone Chinese research report with exported Matplotlib figures."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .statistics import (
    ExperimentData, compare_flow_steps, compare_observation_delay, load_experiment, summarize_experiment,
)


_ZERO_WIDTH_NOTE = (
    "零宽度任务簇区间（例如 [0, 0]）可能只是已观察任务的配对平均效应没有变异；"
    "双方在全部已测回合成功时就会出现这种情况。"
    "这不证明两种方法等效或非劣效，也不表示真实不确定性为零。"
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}%"


def _json_block(value: Any) -> str:
    return "<pre>" + _escape(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre>"


def _experiment_name(data: ExperimentData) -> str:
    plan = data.protocol.get("plan")
    if isinstance(plan, dict) and isinstance(plan.get("name"), str) and plan["name"].strip():
        return plan["name"].strip()
    name = data.protocol.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else data.output_dir.name


def _add_heatmap_context(data: ExperimentData, summary: dict[str, Any]) -> None:
    """Retain every planned axis, including wholly missing methods or tasks."""
    group_fields = ("model", "condition", "method")
    summary["heatmap_groups"] = [dict(zip(group_fields, values)) for values in sorted({
        tuple(row[field] for field in group_fields) for row in data.protocol["cases"]
    })]
    summary["heatmap_tasks"] = [{"suite": suite, "task_id": task_id} for suite, task_id in sorted({
        (row["suite"], row["task_id"]) for row in data.protocol["cases"]
    })]
    initial_states: dict[tuple[Any, ...], set[int]] = {}
    for row in data.records:
        key = tuple(row[field] for field in (*group_fields, "suite", "task_id"))
        initial_states.setdefault(key, set()).add(row["init_state_id"])
    for group in summary["groups"]:
        prefix = tuple(group[field] for field in group_fields)
        for task in group["per_task"]:
            task["initial_states"] = len(initial_states[(*prefix, task["suite"], task["task_id"])])


def _export_task_heatmap(root: Path, summary: dict[str, Any]) -> tuple[str, str, str] | None:
    """Color only observed per-task rates; missing cells have no numeric value."""
    columns = summary.get("heatmap_groups", summary["groups"])
    tasks = summary.get("heatmap_tasks", [])
    if not columns or not tasks:
        return None
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    lookup = {}
    for group in summary["groups"]:
        for task in group["per_task"]:
            lookup[(group["model"], group["condition"], group["method"], task["suite"], task["task_id"])] = task
    values = np.full((len(tasks), len(columns)), np.nan)
    for i, task in enumerate(tasks):
        for j, group in enumerate(columns):
            observed = lookup.get((group["model"], group["condition"], group["method"], task["suite"], task["task_id"]))
            if observed is not None:
                values[i, j] = 100 * observed["success_rate"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(max(6.5, 2 + 1.25 * len(columns)), max(3.8, 1.8 + 0.44 * len(tasks))))
    cmap = plt.get_cmap("viridis").with_extremes(bad="#e5e8ec")
    shown = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="none")
    for i, task in enumerate(tasks):
        for j, group in enumerate(columns):
            observed = lookup.get((group["model"], group["condition"], group["method"], task["suite"], task["task_id"]))
            if observed is None:
                label, color = "no data", "#596773"
            else:
                label = f"{100 * observed['success_rate']:.1f}%\n"
                label += f"i={observed['initial_states']}, e={observed['episodes']}"
                color = "black" if values[i, j] >= 45 else "white"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(columns)), [f"{row['method']}\n{row['model']}/{row['condition']}" for row in columns],
                  rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)), [f"{row['suite']}:{row['task_id']}" for row in tasks], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    name = summary.get("experiment_name", root.name)
    if summary.get("partial"):
        name = f"PARTIAL | {name}"
    ax.set_title(f"{name}\nAll planned tasks and methods | observed per-task success")
    ax.set_xlabel("i = distinct observed initial states; e = completed episodes\n"
                  "Gray / no data = no complete record; never counted as failure", fontsize=9)
    colorbar = fig.colorbar(shown, ax=ax, fraction=0.035, pad=0.03, ticks=[0, 25, 50, 75, 100])
    colorbar.set_label("Observed success (%)")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"task_method_success.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ("全部任务 × 方法：单任务成功率；i 为不同初态数，e 为完成回合数；灰格缺失不按失败计",
            "figures/task_method_success.svg", "figures/task_method_success.png")


def _export_contrast_figure(
    root: Path, comparisons: list[dict[str, Any]], name: str, axis_label: str, caption: str,
    *, experiment_name: str | None = None,
) -> tuple[str, str, str] | None:
    pairs = [row for row in comparisons if row["macro_success_difference"] is not None]
    if not pairs:
        return None
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(10, max(4, 0.65 * len(pairs) + 1.5)))
    labels = []
    for index, row in enumerate(pairs):
        interval = row["task_cluster_95ci"]
        labels.append(f"{row['model']}/{row['condition']}/{row['method']}\n"
                      f"(tasks={row['paired_tasks']}, paired episodes={row['paired_episodes']})")
        if interval is not None:
            ax.hlines(index, interval[0] * 100, interval[1] * 100, color="#097c83", linewidth=2)
        ax.scatter(row["macro_success_difference"] * 100, index, color="#097c83",
                   marker="o" if interval is not None else "x", s=45)
    ax.set_yticks(range(len(pairs)), labels)
    ax.axvline(0, color="#8d99a5", linestyle="--", linewidth=1)
    ax.set_xlim(-105, 105)
    ax.invert_yaxis()
    ax.set_xlabel(axis_label)
    ax.set_title(f"{experiment_name or root.name}\n"
                 "Task-cluster 95% bootstrap intervals | small samples, no multiplicity correction")
    ax.grid(axis="x", alpha=0.15)
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caption, f"figures/{name}.svg", f"figures/{name}.png"


def _place_budget_labels(ax: Any, point_labels: dict[tuple[float, float], list[str]]) -> list[Any]:
    """Place numbered labels in display space without moving measured points."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    figure = ax.figure
    figure.canvas.draw()  # Limits and transforms must be final before measuring text.
    renderer = figure.canvas.get_renderer()
    font = FontProperties(size=9)
    pixels_per_point = figure.dpi / 72
    bounds = ax.get_window_extent(renderer)
    points = {coordinate: ax.transData.transform(coordinate) for coordinate in point_labels}
    marker_radius = 7 * pixels_per_point
    markers = [Bbox.from_bounds(x - marker_radius, y - marker_radius, 2 * marker_radius, 2 * marker_radius)
               for x, y in points.values()]
    occupied, annotations = [], []
    # Nearby labels get staggered above/below or sideways. Ordering depends only
    # on coordinates, so identical data always produce the same placement.
    offsets = [(dx, dy) for level in range(12) for dy in (16 + 18 * level, -20 - 18 * level)
               for dx in (0, 18, -18, 36, -36, 54, -54, 72, -72)]
    # Prefer vertical staggering so densely packed points retain their horizontal
    # order and their leader lines do not swap neighboring point identities.
    offsets.sort(key=lambda pair: (pair[0] != 0, pair[0] ** 2 + pair[1] ** 2,
                                   pair[1] < 0, abs(pair[0]), pair[0]))
    for coordinate in sorted(point_labels):
        x, y = points[coordinate]
        text = ", ".join(point_labels[coordinate])
        width = renderer.get_text_width_height_descent(text, font, ismath=False)[0] + 6 * pixels_per_point
        height = 17 * pixels_per_point

        def available(center_x: float, center_y: float) -> Bbox | None:
            box = Bbox.from_bounds(center_x - width / 2, center_y - height / 2, width, height)
            if not (bounds.contains(box.x0 - 2, box.y0 - 2) and bounds.contains(box.x1 + 2, box.y1 + 2)):
                return None
            return None if any(box.overlaps(other) for other in (*occupied, *markers)) else box

        chosen = None
        for dx, dy in offsets:
            chosen = available(x + dx * pixels_per_point, y + dy * pixels_per_point)
            if chosen is not None:
                break
        if chosen is None:
            # Extremely crowded local neighborhoods may need a farther label;
            # scan free positions inside the axes and choose the nearest one.
            import numpy as np

            candidates = [(cx, cy) for cx in np.arange(bounds.x0 + width, bounds.x1 - width / 2, width + 3)
                          for cy in np.arange(bounds.y0 + height, bounds.y1 - height / 2, height + 3)]
            candidates.sort(key=lambda center: ((center[0] - x) ** 2 + (center[1] - y) ** 2, center))
            chosen = next((box for cx, cy in candidates if (box := available(cx, cy)) is not None), None)
        if chosen is None:
            raise ValueError("Budget plot has too many labels for a legible layout")
        occupied.append(chosen)
        center_x, center_y = (chosen.x0 + chosen.x1) / 2, (chosen.y0 + chosen.y1) / 2
        annotations.append(ax.annotate(
            text, coordinate, xytext=((center_x - x) / pixels_per_point, (center_y - y) / pixels_per_point),
            textcoords="offset points", ha="center", va="center", fontsize=9, zorder=5,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": .9},
            arrowprops={"arrowstyle": "-", "color": "#627c8b", "linewidth": .7, "shrinkA": 2, "shrinkB": 5},
        ))
    return annotations


def _plot_figures(root: Path, summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    groups = summary["groups"]
    heatmap = _export_task_heatmap(root, summary)
    if not groups:
        return [heatmap] if heatmap is not None else []
    # Kept lazy so corpus validation and statistics require only NumPy.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 180,
    })
    artifacts = [heatmap] if heatmap is not None else []
    experiment_name = summary.get("experiment_name", root.name)
    if summary.get("partial"):
        experiment_name = f"PARTIAL | {experiment_name}"

    def save(fig: Any, name: str, title: str) -> None:
        fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        artifacts.append((title, f"figures/{name}.svg", f"figures/{name}.png"))

    labels = [f"{row['model']} / {row['condition']} / {row['method']}\n"
              f"(tasks={row['tasks']}, episodes={row['episodes']})" for row in groups]
    height = max(4.0, 0.6 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    colors = ["#097c83" if "adaptive" in row["method"].lower() else "#516a82" for row in groups]
    bars = ax.barh(labels, [row["macro_task_success"] * 100 for row in groups], color=colors)
    ax.bar_label(bars, labels=[f"{r['macro_task_success'] * 100:.1f}%" for r in groups], padding=4)
    ax.set_xlim(0, 111)
    ax.set_xlabel("Task-macro success (%)")
    ax.set_title(f"{experiment_name}\nObserved task-macro success | completed trials, equal task weights")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    save(fig, "task_macro_success", "任务宏平均成功率（描述性统计）")

    fig, ax = plt.subplots(figsize=(10, 6))
    point_labels: dict[tuple[float, float], list[str]] = {}
    for index, (row, color) in enumerate(zip(groups, colors), start=1):
        x = row["mean_inference_calls"]
        y = row["macro_task_success"] * 100
        label = (f"{index}. {row['model']}/{row['condition']}/{row['method']}\n"
                 f"   tasks={row['tasks']}, episodes={row['episodes']}")
        ax.scatter(x, y, s=60, color=color, label=label)
        point_labels.setdefault((x, y), []).append(str(index))
    ax.set_xlabel("Mean model inference calls per episode")
    ax.set_ylabel("Task-macro success (%)")
    ax.set_ylim(-4, 109)
    ax.margins(x=0.2)
    ax.set_title(f"{experiment_name}\nObserved quality vs inference budget | completed trials")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8,
              title="Group / sample counts")
    ax.grid(alpha=0.15)
    _place_budget_labels(ax, point_labels)
    save(fig, "quality_budget", "成功率与实际推理预算（不能将不同预算直接视为方法收益）")

    pairs = [row for row in summary["paired_comparisons"] if row["macro_success_difference"] is not None]
    if pairs:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.65 * len(pairs) + 1.5)))
        labels = []
        for index, row in enumerate(pairs):
            value = row["macro_success_difference"] * 100
            interval = row["task_cluster_95ci"]
            labels.append(f"{row['model']}/{row['condition']}/{row['method']}\n"
                          f"(tasks={row['paired_tasks']}, paired episodes={row['paired_episodes']})")
            if interval is not None:
                ax.hlines(index, interval[0] * 100, interval[1] * 100, color="#097c83", linewidth=2)
            ax.scatter(value, index, color="#097c83", marker="o" if interval is not None else "x", s=45)
        ax.set_yticks(range(len(pairs)), labels)
        ax.axvline(0, color="#8d99a5", linestyle="--", linewidth=1)
        ax.set_xlim(-105, 105)
        ax.invert_yaxis()
        ax.set_xlabel("Paired task-macro success difference vs fixed8 (percentage points)")
        ax.set_title(f"{experiment_name}\n"
                     "Task-cluster 95% bootstrap intervals | X = fewer than 2 tasks, no interval")
        ax.grid(axis="x", alpha=0.15)
        save(fig, "paired_differences", "对固定 8 步基线的配对差异与任务簇区间")
    ablation = _export_contrast_figure(
        root, summary["ablation_comparisons"], "ablation_differences",
        "Paired task-macro success difference: variant minus adaptive (percentage points)",
        "关闭组件减完整 adaptive 的配对差异与任务簇区间",
        experiment_name=experiment_name,
    )
    if ablation is not None:
        artifacts.append(ablation)
    return artifacts


def _group_table(summary: dict[str, Any]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in (
            _escape(row["model"]), _escape(row["condition"]), _escape(row["method"]),
            f"{row['tasks']} / {row['episodes']}", _percent(row["macro_task_success"]),
            f"{row['successes']} / {row['episodes']}", _number(row["mean_inference_calls"], 1),
            _number(row["mean_steps"], 1), _number(row["inference_calls_per_100_steps"], 1),
            _number(row["mean_model_seconds"], 2), _number(row["total_wall_seconds"] / 60, 2),
        )) + "</tr>" for row in summary["groups"]
    )
    return '<div class="scroll"><table><thead><tr>' + "".join(
        f"<th>{label}</th>" for label in (
            "模型", "条件", "方法", "任务 / 回合", "任务宏成功率", "成功 / 回合", "均值调用数", "均值步数", "调用 / 百步",
            "均值模型耗时 / 秒", "累计墙钟 / 分",
        )
    ) + "</tr></thead><tbody>" + body + "</tbody></table></div>"


def _paired_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["paired_comparisons"]:
        interval = row["task_cluster_95ci"]
        ci = "不可估计" if interval is None else f"[{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}]"
        cells = (
            _escape(f"{row['model']} / {row['condition']} / {row['method']}"),
            f"{row['paired_tasks']} / {row['paired_episodes']}",
            _percent(row["macro_success_difference"], signed=True), ci,
            f"{row['candidate_wins']} / {row['baseline_wins']} / {row['ties']}",
            _percent(row["inference_call_reduction"]),
            f"{row['unmatched_candidate']} / {row['unmatched_baseline']}",
        )
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    if not rows:
        return "<p>现有完整回合中没有可以与 fixed8 配对的其他方法，故不计算方法差异。</p>"
    labels = ("模型 / 条件 / 方法", "任务 / 配对", "差异 / 百分点", "任务簇 95% CI", "胜 / 负 / 平",
              "调用数减少", "未配对候选 / 基线")
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in labels) + \
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _contrast_table(comparisons: list[dict[str, Any]]) -> str:
    rows = []
    for row in comparisons:
        interval = row["task_cluster_95ci"]
        ci = "不可估计" if interval is None else f"[{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}]"
        calls = row["mean_calls_difference"]
        cells = (
            _escape(f"{row['model']} / {row['condition']} / {row['method']}"),
            f"{row['paired_tasks']} / {row['paired_episodes']}",
            _percent(row["macro_success_difference"], signed=True), ci,
            "—" if calls is None else f"{calls:+.2f}",
            f"{row['candidate_wins']} / {row['baseline_wins']} / {row['ties']}",
            f"{row['unmatched_candidate']} / {row['unmatched_baseline']}",
        )
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    labels = ("模型 / 条件 / 方法", "任务 / 配对", "成功率差 / 百分点", "任务簇 95% CI",
              "每回合均值调用差", "胜 / 负 / 平", "未配对候选 / 参照")
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in labels) + \
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _timing_table(summary: dict[str, Any]) -> str:
    keys = ("policy_inference_seconds", "preprocess_seconds", "scheduler_seconds", "simulator_seconds", "wall_seconds")
    rows = []
    for row in summary["groups"]:
        cells = [_escape(f"{row['model']} / {row['condition']} / {row['method']}")]
        cells += [_number(row["mean_timings"][key], 3) for key in keys]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    labels = ("模型 / 条件 / 方法", "模型推理", "预处理", "控制辅助开销（含反归一化/队列/集成）", "仿真", "墙钟")
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in labels) + \
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _evidence(data: ExperimentData) -> str:
    records = {row["id"]: row for row in (*data.records, *data.errors)}
    rows = []
    videos = []
    for case in data.protocol["cases"]:
        record = records.get(case["id"])
        status = "missing" if record is None else record["status"]
        outcome = "—" if status != "complete" else ("成功" if record["success"] else "失败")
        label = _escape(f"{case['model']} / {case['suite']}:{case['task_id']} / {case['method']}")
        link = "—"
        if record is not None:
            href = quote(f"episodes/{record['id']}/result.json", safe="/")
            link = f'<a href="{href}">原始 JSON</a>'
            if record["status"] == "complete" and record.get("video"):
                video_href = quote(f"episodes/{record['id']}/{record['video']}", safe="/")
                link += f' · <a href="{video_href}">回放</a>'
                # Every recorded video is linked; displayed examples are the first
                # six in the frozen protocol order, never selected by success.
                if len(videos) < 6:
                    videos.append(f'<figure><img loading="lazy" src="{video_href}" alt="{label}">'
                                  f'<figcaption>{label} · 初态 {case["init_state_id"]} · {outcome}</figcaption></figure>')
        rows.append(f"<tr><td>{label}</td><td>{case['init_state_id']} / {case['seed']}</td>"
                    f"<td>{_escape(status)} / {outcome}</td><td>{link}</td></tr>")
    return (
        '<p><a href="protocol.json">冻结实验协议</a> · <a href="summary.json">统计结果与输入 SHA-256</a></p>'
        '<p>回放展示遵循协议顺序，最多展示前六条已保存录像；所有录像均在下表链接，不按成功挑选。</p>'
        '<div class="videos">' + "".join(videos) + "</div>"
        '<details><summary>逐回合记录（包括缺失与错误）</summary><div class="scroll"><table>'
        '<thead><tr><th>模型 / 任务 / 方法</th><th>初态 / 种子</th><th>状态 / 结果</th><th>证据</th></tr></thead>'
        '<tbody>' + "".join(rows) + "</tbody></table></div></details>"
    )


def build_report(output_dir: str | Path, partial: bool = False) -> Path:
    """Validate frozen raw files, export figures, and write an offline HTML report."""
    data = load_experiment(output_dir, partial=partial)
    summary = summarize_experiment(data)
    summary["experiment_name"] = _experiment_name(data)
    _add_heatmap_context(data, summary)
    figures = _plot_figures(data.output_dir, summary)
    title = "部分实验报告 · 尚未完成" if partial else "预训练机器人策略 · 实验报告"
    warnings = "".join(f"<p>{_escape(warning)}</p>" for warning in summary["warnings"])
    question = data.protocol.get("question") or "预训练机器人策略的推理调用预算与闭环任务成功率之间存在怎样的关系？"
    hypothesis = data.protocol.get("hypothesis")
    figure_html = "".join(
        f'<figure><img src="{svg}" alt="{_escape(label)}"><figcaption>{_escape(label)} · '
        f'<a href="{svg}">SVG</a> / <a href="{png}">PNG</a></figcaption></figure>'
        for label, svg, png in figures
    )
    limitations = data.protocol.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    limitation_html = "".join(f"<li>{_escape(item)}</li>" for item in limitations)
    metadata = {}
    configurations = {}
    for record in data.records:
        metadata.setdefault(record["model"], record["model_info"])
        key = f"{record['model']} / {record['condition']} / {record['method']}"
        configs = configurations.setdefault(key, [])
        if record["scheduler_config"] not in configs:
            configs.append(record["scheduler_config"])
    protocol_context = {key: value for key, value in data.protocol.items() if key != "cases"}
    html_text = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>""" + _escape(title) + """</title>
<style>
:root{color-scheme:light;--ink:#172d40;--muted:#587080;--teal:#097c83;--line:#dce5ea}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f2f6f8;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1280px;margin:auto;padding:44px 28px 72px}header{padding:28px 0;border-bottom:3px solid var(--teal)}
h1{font-size:clamp(26px,4vw,40px);line-height:1.3;margin:8px 0 16px}h2{margin:0 0 18px;font-size:23px}
p{margin:12px 0}section{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin-top:24px}
.eyebrow{letter-spacing:.14em;color:var(--teal);font-weight:700;font-size:13px}.muted,figcaption{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:18px}
.card b{display:block;font-size:28px;line-height:1.25}.card span{font-size:13px;color:var(--muted)}
.warning{background:#fff8e9;border-left:4px solid #b68123;padding:12px 20px;margin:20px 0}.warning:empty{display:none}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px 10px;border-bottom:1px solid var(--line)}
th{background:#edf4f6;white-space:nowrap}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
figure{margin:24px 0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:white}figcaption{font-size:13px;margin-top:8px}
.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.videos figure{margin:10px 0}
a{color:var(--teal);text-underline-offset:3px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf4f6;padding:18px;border-radius:8px;font-size:12px;line-height:1.55}
summary{cursor:pointer;font-weight:600;padding:10px 0}li{margin:8px 0}footer{padding-top:28px;font-size:13px;color:var(--muted)}
@media(max-width:750px){main{padding:20px 12px 40px}section{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}.videos{grid-template-columns:1fr}}
</style></head><body><main><header><div class="eyebrow">PRETRAINED ROBOT POLICIES / REPRODUCIBLE CPU STUDY</div>
<h1>""" + _escape(title) + "</h1><p>" + _escape(question) + "</p></header>"
    html_text += '<div class="cards">' + "".join(
        f'<div class="card"><b>{_escape(value)}</b><span>{_escape(label)}</span></div>'
        for value, label in (
            (f"{summary['completed_episodes']} / {summary['planned_episodes']}", "完成 / 计划回合"),
            (summary["tasks"], "已完成任务簇"), (len(summary["models"]), "实际评估模型"),
            (_number(summary["total_wall_seconds"] / 60, 1), "完成回合累计墙钟 / 分钟"),
        )
    ) + f'</div><div class="warning">{warnings}</div>'
    html_text += "<section><h2>研究问题与实验边界</h2>"
    if hypothesis is not None:
        html_text += "<p><strong>预声明假设：</strong>" + _escape(hypothesis) + "</p>"
    html_text += (
        "<p>本项目研究公开预训练策略在基准任务中的闭环执行与推理预算。"
        "已知任务的保留初态评估不等于新技能泛化，也不等于真机或通用 VLA 能力验证。"
        "调度规则是候选工程启发式，不声称算法首创或 SOTA。</p>"
        f"<p>当前数据：完成 {summary['completed_episodes']} 回合，错误 {summary['error_episodes']} 回合，"
        f"缺失 {summary['missing_episodes']} 回合。失败回合按原样保留；完整报告要求全部计划回合产生有效结果。</p>"
        "<details><summary>冻结协议与划分</summary>" + _json_block(protocol_context) + "</details></section>"
    )
    html_text += (
        "<section><h2>执行方法与可复现实现</h2>"
        "<p>预训练模型输出动作块；统一调度器决定何时重新推理以及如何使用仍有效的历史预测。"
        "fixed8 指本实验统一执行器的固定 8 步对照，不自动等同原论文默认评测结果。"
        "观察变化策略只使用当前与过去的相机和夹爪观测，不读取任务成功标志、奖励或真实物体位置。</p>"
        "<p>方法间共用配对初态、种子和模型权重；具体阈值、执行间隔、时间集成开关以及回合上限见实际配置。"
        "候选阈值应只在开发集确定，并在正式测试前冻结。</p>"
        "<details><summary>已执行方法的实际配置</summary>" + _json_block(configurations) + "</details></section>"
    )
    html_text += "<section><h2>成功率与推理成本</h2>" + _group_table(summary)
    html_text += (
        "<p class=\"muted\">任务宏成功率先在每个 (suite, task_id) 内求初态/种子平均，再等权平均任务。"
        "成功计数为回合计数；调用数与模型耗时为回合均值。累计墙钟为完成回合耗时之和，"
        "不代表包含安装、下载和全部进程启动的项目总工时。成功提前结束会减少回合调用数；"
        "因此另列实际执行步数与每百步调用率。不同推理预算的结果应作为质量—成本曲线解释。</p></section>"
    )
    html_text += "<section><h2>运行成本分解</h2>" + _timing_table(summary) + (
        "<p class=\"muted\">所有数值为每回合平均秒数，直接来自原始计时字段。"
        "墙钟包含该回合其他开销；此表不把推理调用次数视作不同模型之间等价的计算量。</p></section>"
    )
    html_text += "<section><h2>与固定 8 步基线的配对对照</h2>" + _paired_table(summary)
    html_text += (
        "<p class=\"muted\">按模型、条件、任务、初态和随机种子严格配对。差异为候选减基线。"
        "95% 区间用固定种子进行 2,000 次任务簇重采样，保持同任务试验一起出现；"
        "该区间以现有初态和模型权重为条件，不包括训练随机性或真机迁移的不确定性。"
        "少于两个任务不计算区间；少量任务的区间不能证明结论稳定。</p>"
        f'<p class="muted">{_ZERO_WIDTH_NOTE}</p></section>'
    )
    if summary["ablation_comparisons"]:
        html_text += "<section><h2>组件消融：关闭组件减完整 adaptive</h2>" + _contrast_table(summary["ablation_comparisons"])
        html_text += (
            "<p>每对试验仅关闭对应图像、夹爪或时间集成组件，其余调度配置、模型、环境及初始观察一致。"
            "成功率和调用数均按变体减完整 adaptive；负成功率差表示关闭组件后更差，负调用差表示调用更少。</p>"
            "<p class=\"muted\">区间按任务簇重采样，未做多重比较校正；这些预声明消融不用于挑选测试赢家。"
            "调用数差为配对回合均值，也受成功提前结束影响。小任务样本不能确证组件具有普遍收益。</p>"
            f'<p class="muted">{_ZERO_WIDTH_NOTE}</p></section>'
        )
    html_text += "<section><h2>可导出图表</h2>" + figure_html + "</section>"
    if (data.output_dir / "trigger_diagnostics.html").is_file():
        html_text += (
            '<section><h2>事后运行诊断</h2><p><a href="trigger_diagnostics.html">重规划触发组合与执行间隔</a>：'
            "该诊断核对实际调用次数和日志，不作为信号导致成功或失败的因果证据，组件贡献以消融对照为准。</p></section>"
        )
    html_text += "<section><h2>逐回合证据与回放</h2>" + _evidence(data) + (
        "<details><summary>配对初始图像与机器人状态一致性审计</summary>"
        + _json_block(summary["initial_state_parity"]) + "</details></section>"
    )
    html_text += "<section><h2>模型与限制</h2><ul>" + limitation_html + (
        "<li>完整性检查验证协议与记录一致，不替代模型实现审计或外部复现。</li>"
        "<li>若在同一保留测试集上选择阈值、方法或检查点，就不能将其继续视为未使用测试集。</li>"
        "</ul><details><summary>实际模型元数据</summary>" + _json_block(metadata) + "</details></section>"
    )
    html_text += "<footer>本页由冻结协议和原始试验记录生成；输入文件 SHA-256 位于 summary.json。图表使用 Matplotlib，无外部 CDN。</footer></main></body></html>"
    # All transformations above finish before report outputs are replaced. Raw
    # protocol/results are never edited by this module.
    (data.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    destination = data.output_dir / "report.html"
    destination.write_text(html_text, encoding="utf-8")
    return destination


def build_delay_report(baseline_dir: str | Path, delayed_dir: str | Path, output_dir: str | Path) -> Path:
    """Export strictly matched clean-to-delay contrasts for both declared methods."""
    baseline, delayed = load_experiment(baseline_dir), load_experiment(delayed_dir)
    summary = compare_observation_delay(baseline, delayed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure = _export_contrast_figure(
        output, summary["comparisons"], "delay_differences",
        "Paired task-macro success difference: delay4 minus clean (percentage points)",
        "同一方法的 delay4 减 clean 配对差异",
        experiment_name=f"{_experiment_name(delayed)} vs matched clean from {_experiment_name(baseline)}",
    )
    text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>观测延迟配对试验</title><style>body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#172d40;margin:0;background:#f2f6f8}
main{max-width:1300px;margin:auto;padding:40px 24px}section{background:white;padding:24px;border:1px solid #dce5ea;border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce5ea}
.scroll{overflow:auto}.warning{background:#fff8e9;border-left:4px solid #b68123;padding:12px}a{color:#097c83}figure img{width:100%}</style><main>
<h1>观测延迟：delay4 减 clean</h1><section>"""
    text += (
        "<p>分别比较 fixed8 与 adaptive 在四控制步观测延迟下的变化。"
        "每对模型、任务、初态、种子及调度配置一致；初始图像和本体状态通过严格核对。"
        "成功率差与调用差均为 delay4 减 clean，负值分别表示成功减少、调用减少。</p>"
        f'<p class="warning">{_escape(summary["warning"])}</p>' + _contrast_table(summary["comparisons"])
        + "<p>区间采用 2,000 次任务簇重采样，少于两个任务不计算区间；未做多重比较校正。"
        "不能根据两个方法的区间是否重叠，推断它们的延迟敏感性有显著差异；本表未检验方法与条件的交互。"
        "调用数为配对回合均值，受成功提前结束影响。</p>"
        f"<p>{_ZERO_WIDTH_NOTE}</p>"
    )
    if figure is not None:
        label, svg, png = figure
        text += (f'<figure><img src="{svg}" alt="{_escape(label)}"><figcaption>{_escape(label)} · '
                 f'<a href="{svg}">SVG</a> / <a href="{png}">PNG</a></figcaption></figure>')
    rows = []
    for comparison in summary["comparisons"]:
        for row in comparison["per_pair"]:
            links = []
            for source, identifier, label in ((baseline, row["baseline_id"], "clean"),
                                               (delayed, row["candidate_id"], "delay4")):
                source_path = source.output_dir / "episodes" / identifier / "result.json"
                href = quote(Path(os.path.relpath(source_path, output)).as_posix(), safe="/")
                links.append(f'<a href="{href}">{label}</a>')
            cells = (_escape(f"{comparison['model']} / {comparison['method']} / {row['suite']}:{row['task_id']}"),
                     f"{row['init_state_id']} / {row['seed']}",
                     f"{int(row['baseline_success'])} / {int(row['candidate_success'])}",
                     f"{row['baseline_calls']} / {row['candidate_calls']}", " · ".join(links))
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    text += ('<h2>全部配对证据</h2><div class="scroll"><table><thead><tr><th>模型 / 方法 / 任务</th>'
             '<th>初态 / 种子</th><th>成功 clean / delay4</th><th>调用 clean / delay4</th><th>原始结果</th>'
             '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
             '<p><a href="delay_comparison.json">配对结果与输入 SHA-256</a> · <a href="report.html">实验总览</a></p>'
             '</section></main></html>')
    (output / "delay_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    destination = output / "delay_comparison.html"
    destination.write_text(text, encoding="utf-8")
    return destination


def build_flow_report(baseline_dir: str | Path, candidate_dir: str | Path, output_dir: str | Path) -> Path:
    """Export the predeclared four-pair cross-stage flow-step pilot."""
    baseline = load_experiment(baseline_dir)
    candidate = load_experiment(candidate_dir)
    summary = compare_flow_steps(baseline, candidate)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in summary["per_pair"]:
        base_file = baseline.output_dir / "episodes" / row["baseline_id"] / "result.json"
        candidate_file = candidate.output_dir / "episodes" / row["candidate_id"] / "result.json"
        base_href = quote(Path(os.path.relpath(base_file, output)).as_posix(), safe="/")
        candidate_href = quote(Path(os.path.relpath(candidate_file, output)).as_posix(), safe="/")
        cells = (
            _escape(f"{row['suite']}:{row['task_id']} / {row['task']}"),
            f"{row['init_state_id']} / {row['seed']}",
            "成功" if row["success_10"] else "失败", "成功" if row["success_5"] else "失败",
            _percent(row["success_difference"], signed=True), f"{row['calls_10']} / {row['calls_5']}",
            f"{_number(row['model_seconds_10'])} / {_number(row['model_seconds_5'])}",
            f'<a href="{base_href}">10 步</a> · <a href="{candidate_href}">5 步</a>',
        )
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    interval = summary["task_cluster_95ci"]
    ci = "不可估计" if interval is None else f"[{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}] 百分点"
    headers = ("任务", "初态 / 种子", "10 步结果", "5 步结果", "差异 / 百分点", "调用数 10 / 5", "模型秒数 10 / 5", "原始证据")
    text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmolVLA flow 步数配对试验</title><style>body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#172d40;margin:0;background:#f2f6f8}
main{max-width:1300px;margin:auto;padding:40px 24px}section{background:white;padding:24px;border:1px solid #dce5ea;border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce5ea}
.scroll{overflow:auto}.warning{background:#fff8e9;border-left:4px solid #b68123;padding:12px}a{color:#097c83}</style><main>
<h1>SmolVLA：5 步与 10 步 flow 的配对比较</h1><section>"""
    text += (f"<p>{summary['paired_tasks']} 个任务、{summary['paired_episodes']} 对完整回合。"
             "固定 15 步执行且关闭时间集成；权重、模型配置、环境和配对初态一致，flow 积分步数为比较变量。</p>"
             f"<p>任务宏成功率差异（5 减 10）：{_percent(summary['macro_success_difference'], signed=True)} 百分点；"
             f"任务簇 bootstrap 95% 区间：{ci}。</p>"
             f'<p class="warning">{_escape(summary["warning"])}</p>')
    text += '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in headers)
    text += "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    text += (
        "<p>区间使用 2,000 次任务簇重采样，种子固定。模型耗时按完整回合统计，成功提前结束也会改变调用总数。</p>"
        f"<p>{_ZERO_WIDTH_NOTE}</p>"
        '<p><a href="flow_comparison.json">配对结果、校验覆盖与输入 SHA-256</a> · <a href="report.html">实验总览</a></p></section></main></html>'
    )
    (output / "flow_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    destination = output / "flow_comparison.html"
    destination.write_text(text, encoding="utf-8")
    return destination
