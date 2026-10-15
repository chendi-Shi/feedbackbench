"""Sealed, independently counted prospective full10 versus noise2 CPU results.

Only derived report exports are written. The runner verifies raw evidence without
importing Torch or a simulator. --partial is required for incomplete evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

from embodied_research.experiment import digest_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]
RESAMPLES, BOOTSTRAP_SEED = 2000, 20261015
METHODS = ("full10", "noise2")
EXPORT_SCHEMA = "feedbackbench.flow_confirmation_exports.v1"
EXPORT_FILES = (
    "report.json", "report.md", "groups.csv", "episodes.csv", "pairs.csv", "calls.csv",
    "figures/flow_success_budget.png", "figures/flow_success_budget.svg",
    "figures/flow_latency.png", "figures/flow_latency.svg",
)


def _support(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = _support("flow_report_cache_support", "summarize_cache_study.py")
IntegrityError = support.IntegrityError
sha, read_json, _require = support.sha, support.read_json, support._require


def load_study(output_dir, *, partial=False, root=ROOT):
    runner = _support("flow_report_runner_support", "run_flow_confirmation.py")
    return runner.verify_study(Path(output_dir), partial=partial, root=Path(root))


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _cluster_ci(values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) < 2:
        return None
    _require(np.isfinite(values).all(), "nonfinite task effect")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values), (RESAMPLES, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def _implementation_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in (
        "summarize_flow_confirmation.py", "summarize_cache_study.py", "run_flow_confirmation.py",
    )}


def summarize(study):
    protocol = study["protocol"]
    cases = protocol["spec"]["cases"]
    _require(len(cases) == 16 and {c["method"] for c in cases} == set(METHODS),
             "confirmation requires exactly the declared 16 two-method cases")
    _require({c["init_state_id"] for c in cases} == {41, 43}, "unexpected prospective initial states")
    _require({c["split"] for c in cases} == {"confirmation"}, "cannot mix earlier study episodes")
    groups = [support._group("confirmation", method, cases, study) for method in METHODS]
    comparison = support._compare("confirmation", "noise2", "full10", cases, study)
    _require(comparison["planned_pairs"] == 8, "confirmation must retain eight planned pairs")
    comparison["difference_direction"] = "noise2 minus full10"
    for pair in comparison["per_pair"]:
        a = study["records"][pair["candidate_id"]]
        b = study["records"][pair["baseline_id"]]
        for prefix, row in (("candidate", a), ("baseline", b)):
            pair[f"{prefix}_calls"] = len(row["calls"])
            pair[f"{prefix}_total_nfe"] = sum(c["nfe"] for c in row["calls"])
            pair[f"{prefix}_steps"] = row["record"]["steps"]
        pair["calls_difference"] = pair["candidate_calls"] - pair["baseline_calls"]
        pair["total_nfe_difference"] = pair["candidate_total_nfe"] - pair["baseline_total_nfe"]
    by_task = defaultdict(list)
    for pair in comparison["per_pair"]:
        by_task[pair["suite"], pair["task_id"]].append(pair)
    for task in comparison["per_task"]:
        pairs = by_task[task["suite"], task["task_id"]]
        task["calls_difference"] = _mean(p["calls_difference"] for p in pairs)
        task["total_nfe_difference"] = _mean(p["total_nfe_difference"] for p in pairs)
    # The shared arithmetic helper uses the original pilot's seed. Replace all
    # inference summaries with the prospective protocol's independently fixed seed.
    for source, mean_key, ci_key in (
        ("success_difference", "macro_success_difference", "success_task_cluster_95ci"),
        ("mean_call_seconds_difference", "macro_mean_call_seconds_difference", "latency_task_cluster_95ci_seconds"),
        ("calls_difference", "macro_calls_difference", "calls_task_cluster_95ci"),
        ("total_nfe_difference", "macro_total_nfe_difference", "total_nfe_task_cluster_95ci"),
    ):
        values = [r[source] for r in comparison["per_task"]]
        comparison[mean_key], comparison[ci_key] = _mean(values), _cluster_ci(values)
    episodes, calls = [], []
    for case in cases:
        row = study["records"].get(case["id"])
        episode = {**{k: case[k] for k in ("id", "model", "suite", "task_id", "init_state_id", "method", "split")},
                   "status": study["statuses"][case["id"]],
                   "result_path": f"episodes/{case['id']}/result.json"}
        if row is not None:
            episode.update({key: row["record"][key] for key in (
                "success", "steps", "horizon", "inference_calls", "wall_seconds", "policy_inference_seconds",
            )})
            episode.update(total_nfe=sum(c["nfe"] for c in row["calls"]),
                           result_sha256=row["result_sha256"], receipt_sha256=row["receipt_sha256"])
            calls.extend({"id": case["id"], "suite": case["suite"], "task_id": case["task_id"],
                          "init_state_id": case["init_state_id"], **call} for call in row["calls"])
        episodes.append(episode)
    return {
        "schema": "feedbackbench.flow_confirmation_report.v1", "generated_at": utc_now(),
        "status": "complete" if study["complete"] else "PARTIAL",
        "name": protocol["spec"]["name"], "question": protocol["spec"]["question"],
        "counts": {"planned": len(cases), **{state: list(study["statuses"].values()).count(state)
                   for state in ("complete", "missing", "unsealed", "error")}},
        "protocol_fingerprint": protocol["fingerprint"], "frozen_sources": protocol["sources"],
        "report_implementation_sha256": _implementation_hashes(),
        "evidence_sha256": study["evidence_sha256"], "origin": protocol.get("origin"),
        "model_artifacts": protocol["artifacts"]["models"], "runtime_versions": protocol["runtime_versions"],
        "pairing_audit": study["pairing_audit"], "groups": groups, "comparison": comparison,
        "episodes": episodes, "calls": calls,
        "uncertainty": {"unit": "equal-weight task-cluster mean of within-task paired effects",
                        "resamples": RESAMPLES, "seed": BOOTSTRAP_SEED, "planned_task_clusters": 4,
                        "warning": "Only four task clusters. Zero-width observed bootstrap intervals are not equivalence, noninferiority, or zero true uncertainty."},
        "interpretation_limits": [
            "The v1 init11 noise2 observation selected this hypothesis. These fresh prospective cases are counted separately; no pooling with v1.",
            "Initial states are held out from this project's earlier study only, not certified absent from upstream pretraining.",
            "Partial reports retain all planned cases and summarize only observed complete pairs; no missing outcomes are imputed.",
            "NFE counts action-head evaluations only. Current-observation VLM computation remains in both methods.",
            "Per-call p50/p95 pool calls descriptively. Paired latency compares episode means; later observations can diverge.",
            "Wall difference is restricted to both-success pairs, is outcome-conditioned, and has no confidence interval or overall speedup claim.",
            "All raw episode wall times remain visible, including full-horizon failures.",
            "First-ten action distance requires ten saved commands from both episodes. Mixed robot-action coordinates are not a quality metric or a full-chunk comparison.",
            "Shared CPU load, counterbalancing and synchronous paused simulation do not establish controlled throughput or real-time deployment.",
            "No foundation-model training, new-algorithm novelty, complete LIBERO, SOTA, unseen-task or real-robot claim.",
        ],
    }


def _csv_bytes(rows):
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def _csv_rows(report):
    return {"groups.csv": report["groups"], "episodes.csv": report["episodes"],
            "pairs.csv": report["comparison"]["per_pair"], "calls.csv": report["calls"]}


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _interval(value, *, percent=False):
    if value is None:
        return "—"
    return "[" + ", ".join(_fmt(x * (100 if percent else 1)) for x in value) + "]"


def _markdown(report):
    counts, comparison = report["counts"], report["comparison"]
    title = "完整结果" if report["status"] == "complete" else "PARTIAL：未完成，仅报告已封存证据"
    lines = [f"# {report['name']} — {title}", "", report["question"], "",
             f"已核验 **{counts['complete']}/{counts['planned']}** 回合；缺失 {counts['missing']}、"
             f"未封存 {counts['unsealed']}、运行错误 {counts['error']}。计划为 **8 组配对、4 个任务簇**。", "",
             "本次假设来自 v1 的探索结果：noise2 在共同 init11 的四个任务上成功。"
             "新试验预先固定 init41/43，不与 v1 合并分母，也不代表上游预训练未见这些初态。", "",
             "full10 与 noise2 都在当前观测条件下，从相同配对噪声覆盖完整 flow 时间区间，分别执行10步和2步。"
             "没有缓存读取、latent 续算或模型训练；两者每次生成50动作并执行前10个后重规划。", "",
             "## 成功与计算预算", "",
             "| 方法 | 完成 / 计划 | 成功 / 已完成 | 调用总数 | 总 NFE | NFE / 调用 |",
             "|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        lines.append(f"| {group['method']} | {group['completed_episodes']}/{group['planned_episodes']} | "
                     f"{group['successes']}/{group['completed_episodes']} | {group['replans']} | "
                     f"{group['total_nfe']} | {_fmt(group['mean_nfe_per_replan'])} |")
    lines += ["", "![成功与 action-head 预算](figures/flow_success_budget.png)", "",
              "NFE 仅计 action head；两种方法都执行当前观测的 VLM。总 NFE 同时受每调用预算和回合调用数影响。", "",
              "## 预声明配对比较：noise2 − full10", "",
              f"完成 **{comparison['completed_pairs']}/{comparison['planned_pairs']}** 配对，覆盖 "
              f"**{comparison['observed_task_clusters']}/4** 个任务簇。", "",
              "| 指标 | 任务等权配对差 | 任务簇 bootstrap 95% 区间 |",
              "|---|---:|---:|",
              f"| 成功差（百分点） | {_fmt(None if comparison['macro_success_difference'] is None else 100 * comparison['macro_success_difference'])} | {_interval(comparison['success_task_cluster_95ci'], percent=True)} |",
              f"| 每调用平均耗时差（秒） | {_fmt(comparison['macro_mean_call_seconds_difference'])} | {_interval(comparison['latency_task_cluster_95ci_seconds'])} |",
              f"| 回合调用数差 | {_fmt(comparison['macro_calls_difference'])} | {_interval(comparison['calls_task_cluster_95ci'])} |",
              f"| 回合总 NFE 差 | {_fmt(comparison['macro_total_nfe_difference'])} | {_interval(comparison['total_nfe_task_cluster_95ci'])} |", "",
              "先计算同任务两初态的配对差，再对四个任务等权平均；2,000次重采样，seed=20261015。"
              "仅四个任务簇，区间不稳定。区间含0或零宽都不证明等价、非劣或真实不确定性为0。", "",
              f"双方成功配对 **{comparison['both_success_pairs']}** 组，其回合 wall 平均差 "
              f"**{_fmt(comparison['mean_wall_difference_both_success_only'])} 秒**。"
              "这是按结果筛选后的描述量，不给置信区间，不作为总体加速倍数。全部失败与成功回合 wall 都保留在 CSV。", "",
              "![调用延迟描述](figures/flow_latency.png)", "",
              "调用池的分位数与配对回合均值采用不同权重，不能混用。共享主机负载及后续轨迹分化影响耗时；"
              "同步仿真在推理时暂停，不能从20 Hz控制频率推断本机实时执行。", "",
              "## 初始输入、噪声与动作诊断", "",
              "配对核对初始正面/手腕图像、机器人状态、环境与调度配置，并核对所有共有决策序号的噪声 SHA。"
              "后续观测允许因动作不同而分化；首10步比较的是后处理后、环境内部限幅前的动作命令。", "",
              f"首10步动作诊断有效 **{comparison['first_10_complete_pairs']}** 对、缺失 "
              f"**{comparison['first_10_missing_pairs']}** 对。混合七维坐标 RMSE 的回合均值为 "
              f"**{_fmt(comparison['mean_first_actions_rmse_all_dims'], 6)}**；逐通道指标见 pairs.csv。"
              "不足10步时标为缺失，仍保留成功结果；不将动作距离解释为任务质量或完整50动作块误差。", "",
              "## 全部计划回合", "",
              "| Suite / 初态 | 方法 | 状态 | 成功 | 步数 | 调用数 | 总 NFE |",
              "|---|---|---|---:|---:|---:|---:|"]
    for episode in report["episodes"]:
        outcome = "—" if "success" not in episode else str(int(episode["success"]))
        lines.append(f"| {episode['suite']} / {episode['init_state_id']} | {episode['method']} | "
                     f"{episode['status']} | {outcome} | {episode.get('steps', '—')} | "
                     f"{episode.get('inference_calls', '—')} | {episode.get('total_nfe', '—')} |")
    lines += ["", "## 证据与限制", "",
              "[协议](protocol.json) · [报告 JSON](report.json) · [分组](groups.csv) · [回合](episodes.csv) · "
              "[配对](pairs.csv) · [调用](calls.csv)", "",
              "报告绑定协议、原始证据、冻结源码和报告实现 SHA；exports.json 在全部导出完成后最后写入。"
              "只读校验会重新计算统计量、Markdown 与 CSV，并核对所有导出文件哈希。", "",
              "未完成数据不会补值或并入完整结论。已有成功/失败不能用于改选任务、初态、方法或提前终止。"
              "本项目不声称算法首创、完整LIBERO复现、SOTA、未见任务泛化或真机效果。", ""]
    return "\n".join(lines)


def _plots(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    groups = report["groups"]
    colors = ["#475569", "#0f766e"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, stem):
        for suffix in ("png", "svg"):
            fig.savefig(destination / f"{stem}.{suffix}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    success = [g["observed_macro_task_success"] for g in groups]
    axes[0].bar(METHODS, [0 if v is None else v for v in success], color=colors)
    axes[0].set(ylim=(0, 1.18), ylabel="Observed task-macro success", title="Success: completed episodes only")
    for index, group in enumerate(groups):
        axes[0].text(index, (success[index] or 0) + .04,
                     f"{group['successes']}/{group['completed_episodes']} completed\n{group['planned_episodes']} planned",
                     ha="center", fontsize=9)
    axes[1].bar(METHODS, [g["mean_nfe_per_replan"] or 0 for g in groups], color=colors)
    axes[1].set(ylim=(0, 12.4), ylabel="Action-head NFE per call", title="Budget does not imply retained success")
    for index, group in enumerate(groups):
        axes[1].text(index, (group["mean_nfe_per_replan"] or 0) + .4,
                     f"total NFE {group['total_nfe']}\n{group['replans']} calls", ha="center", fontsize=9)
    figure.suptitle(f"Prospective flow comparison · {report['status']} · 4 planned task clusters", fontsize=12)
    save(figure, "flow_success_budget")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    positions = np.arange(2)
    for offset, key, label, color in ((-.18, "p50", "p50", "#0f766e"), (.18, "p95", "p95", "#99cfc2")):
        axes[0].bar(positions + offset, [g["timings_seconds"]["end_to_end_seconds"][key] or 0 for g in groups],
                    width=.35, label=label, color=color)
    axes[0].set(xticks=positions, xticklabels=METHODS, ylabel="Seconds per call", title="Pooled calls: descriptive p50 / p95")
    axes[0].legend(frameon=False)
    pairs = report["comparison"]["per_pair"]
    values = [p["mean_call_seconds_difference"] for p in pairs]
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].scatter(range(len(values)), values, color="#0f766e", s=36)
    axes[1].set(ylabel="Noise2 − full10 seconds", xlabel="Completed paired initial-state block",
                title="Paired episode mean call latency", xticks=range(len(values)))
    if not pairs:
        axes[1].text(.5, .5, "No completed pairs", transform=axes[1].transAxes, ha="center")
    figure.suptitle("Shared CPU host; later observations may diverge; no real-time claim", fontsize=11)
    save(figure, "flow_latency")


def _validate_exports(output):
    output = Path(output)
    seal = read_json(output / "exports.json")
    _require(set(seal) == {"schema", "files"} and seal["schema"] == EXPORT_SCHEMA,
             "invalid flow export manifest")
    _require(set(seal["files"]) == set(EXPORT_FILES), "unexpected flow export inventory")
    for name, expected in seal["files"].items():
        _require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "invalid export digest")
        path = support._export_path(output, name)
        _require(sha(path) == expected, f"export hash mismatch: {name}")
    support._validate_markdown_resources(output)


def build_report(output_dir, *, partial=False, root=ROOT):
    output = Path(output_dir)
    report = summarize(load_study(output, partial=partial, root=root))
    _plots(output, report)
    for name, rows in _csv_rows(report).items():
        (output / name).write_bytes(_csv_bytes(rows))
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    write_json(output / "report.json", report)
    support._validate_markdown_resources(output)
    # Final write seals one coherent generation; it never changes raw receipts.
    write_json(output / "exports.json", {"schema": EXPORT_SCHEMA,
               "files": {name: sha(output / name) for name in EXPORT_FILES}})
    _validate_exports(output)
    return output / "report.json"


def validate_report(output_dir, *, partial=False, root=ROOT):
    """Recompute evidence/statistics/CSV/Markdown without importing plotting or models."""
    output = Path(output_dir)
    _validate_exports(output)
    actual = read_json(output / "report.json")
    expected = summarize(load_study(output, partial=partial, root=root))
    def without_time(value):
        return {k: v for k, v in value.items() if k != "generated_at"}
    _require(digest_json(without_time(actual)) == digest_json(without_time(expected)),
             "flow report differs from verified evidence or implementation")
    _require((output / "report.md").read_text(encoding="utf-8") == _markdown(actual),
             "flow Markdown differs from verified report")
    for name, rows in _csv_rows(expected).items():
        _require((output / name).read_bytes() == _csv_bytes(rows), f"flow CSV differs from evidence: {name}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/frontier/flow_confirmation")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_report(args.output, partial=args.partial)
        print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    else:
        print(build_report(args.output, partial=args.partial))


if __name__ == "__main__":
    main()
