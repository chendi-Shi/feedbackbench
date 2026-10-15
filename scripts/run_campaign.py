"""Run the frozen serial campaign; thresholds are selected on development only.

Launch with this project's environment, for example:
  python scripts/run.py scripts/run_campaign.py --through development

Re-running the same command resumes exact matching stage protocols. No stage
silently skips failed episodes, and held-out plans are frozen together before
the first held-out rollout. This module never imports or scores integration
smoke results, which overlap one coverage case and must not be double counted.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from embodied_research.experiment import digest_json, run_plan, utc_now, write_json
from embodied_research.reporting import build_report
from embodied_research.selection import select_development
from embodied_research.statistics import load_experiment


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ("coverage", "10_coverage"), ("development", "20_development"),
    ("methods", "30_methods"), ("delay", "40_delay"),
    ("smolvla", "50_smolvla"), ("flow_steps", "60_flow_steps"),
)
SOURCE_FILES = (
    "scripts/run_campaign.py", "scripts/make_plans.py",
    "src/embodied_research/selection.py", "src/embodied_research/experiment.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
    "src/embodied_research/simulator.py", "src/embodied_research/scheduling.py",
    "src/embodied_research/statistics.py",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _check_freeze(root: Path, output: Path) -> None:
    expected = {"schema_version": 1, "stage_order": [name for name, _ in STAGES],
                "source_sha256": _source_hashes(root),
                "integration_smoke_included": False,
                "counts_scope": "coverage diagnostic + development + held-out studies; distinct from smoke"}
    destination = output / "campaign_freeze.json"
    if destination.exists():
        if _read(destination) != expected:
            raise ValueError("Campaign source/selection/generator changed; preserve this run and use a new output directory")
    else:
        if any((output / stem / "protocol.json").exists() for _, stem in STAGES):
            raise ValueError("Cannot retrospectively freeze a campaign containing existing stage protocols")
        write_json(destination, expected)


def _generate_plans(root: Path, selected: str | None = None) -> None:
    command = [sys.executable, str(root / "scripts/make_plans.py")]
    if selected is not None:
        command += ["--selected", selected]
    subprocess.run(command, cwd=root, check=True)


def _freeze_plan(root: Path, output: Path, stem: str) -> Path:
    raw = (root / "plans" / f"{stem}.json").read_bytes()
    # Reject malformed generator output before persisting or running it.
    if not isinstance(json.loads(raw), dict):
        raise ValueError(f"Plan must be a JSON object: {stem}")
    destination = output / "frozen_plans" / f"{stem}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(f"Generated plan differs from the frozen campaign plan: {stem}")
    else:
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    return destination


def _select_and_freeze_test(root: Path, output: Path) -> dict[str, Any]:
    frozen_selection = output / "development_selection.json"
    development = output / "20_development"
    if frozen_selection.exists():
        previous = _read(frozen_selection)
        current = load_experiment(development)
        if (previous["development_protocol_fingerprint"] != current.protocol["fingerprint"]
                or previous["development_inputs_sha256"] != digest_json(current.input_sha256)):
            raise ValueError("Frozen development evidence changed; do not reselect against existing test results")
    selection = select_development(development)
    if frozen_selection.exists():
        if _read(frozen_selection) != selection:
            raise ValueError("Development selection changed after freeze")
    else:
        write_json(frozen_selection, selection)
    _check_freeze(root, output)
    _generate_plans(root, selection["selected"])
    _check_freeze(root, output)
    # Freeze every held-out plan at once, even if --through stops at development.
    # Recheck the base plans too: adding a selected threshold must not alter them.
    for _, stem in STAGES:
        _freeze_plan(root, output, stem)
    return selection


def _write_index(output: Path, status: dict[str, Any]) -> None:
    rows = []
    for name, stem in STAGES:
        record = status["stages"].get(name, {})
        state = record.get("status", "not_started")
        counts = f"{record.get('completed', '—')} / {record.get('total', '—')}"
        links = []
        if (output / stem / "report.html").is_file():
            links.append(f'<a href="{stem}/report.html">阶段报告</a>')
        if (output / stem / "protocol.json").is_file():
            links.append(f'<a href="{stem}/protocol.json">冻结协议</a>')
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(state)}</td>"
                    f"<td>{counts}</td><td>{' · '.join(links)}</td></tr>")
    flow = ('<p><a href="flow_comparison.html">SmolVLA 5 步与 10 步 flow 的跨阶段配对比较</a></p>'
            if (output / "flow_comparison.html").is_file() else "")
    delay = ('<p><a href="delay_comparison.html">fixed8 与 adaptive 的 clean / delay4 跨阶段配对比较</a></p>'
             if (output / "delay_comparison.html").is_file() else "")
    text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FeedbackBench 实验总览</title>
<style>body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#172d40;background:#f2f6f8;margin:0}
main{max-width:1050px;margin:auto;padding:40px 24px}section{background:white;padding:24px;border:1px solid #dce5ea;border-radius:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce5ea}a{color:#097c83}.note{color:#587080}</style>
<main><h1>FeedbackBench 实验总览</h1><section>"""
    text += f"<p>本次目标阶段：{html.escape(status['through'])}；状态：{html.escape(status['status'])}。</p>"
    text += (
        "<p>40 任务覆盖诊断、开发集调参、保留测试分别列示，不能将它们合成单一成功率。"
        "此总览不计入 integration smoke；其中一条 smoke 与覆盖诊断初态重复，不能额外计作独立试验。</p>"
        "<table><thead><tr><th>阶段</th><th>状态</th><th>完成 / 计划</th><th>证据</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>" + flow + delay
        + '<p><a href="campaign_freeze.json">生成器与选择规则源码冻结</a> · '
        '<a href="campaign_status.json">运行状态</a></p>'
    )
    if (output / "development_selection.json").is_file():
        text += '<p><a href="development_selection.json">仅使用开发数据的阈值选择证据</a></p>'
    if status.get("error"):
        text += "<p>停止原因：" + html.escape(status["error"]) + "</p>"
    text += '<p class="note">尚未完成的阶段没有补值或收益声明。每个完整阶段报告均重新核对原始回合与冻结协议。</p></section></main></html>'
    (output / "report.html").write_text(text, encoding="utf-8")


def run_campaign(output_dir: str | Path | None = None, *, through: str = "flow_steps",
                 project_root: str | Path | None = None) -> Path:
    """Run serial stages up to and including ``through``; return the index path."""
    root = Path(project_root or ROOT).resolve()
    output = Path(output_dir or root / "outputs/campaign").resolve()
    names = [name for name, _ in STAGES]
    if through not in names:
        raise ValueError(f"Unknown campaign stage {through!r}; choose {names}")
    output.mkdir(parents=True, exist_ok=True)
    _check_freeze(root, output)
    _generate_plans(root)
    _check_freeze(root, output)
    for _, stem in STAGES[:2]:
        _freeze_plan(root, output, stem)
    status_path = output / "campaign_status.json"
    status = _read(status_path) if status_path.exists() else {"schema_version": 1, "stages": {}}
    status.update(through=through, status="running", started_at=utc_now())
    status.pop("error", None)
    active_name = None
    active_directory = None
    try:
        for name, stem in STAGES[:names.index(through) + 1]:
            active_name, active_directory = name, output / stem
            _check_freeze(root, output)
            frozen_plan = _freeze_plan(root, output, stem)
            status["stages"][name] = {"status": "running"}
            write_json(status_path, status)
            _write_index(output, status)
            result = run_plan(frozen_plan, active_directory, root=root)
            if result["completed"] != result["total"]:
                raise RuntimeError(f"Stage {name} returned incomplete; cannot advance the campaign")
            build_report(active_directory, partial=False)
            status["stages"][name] = {"status": "complete", "completed": result["completed"],
                                       "total": result["total"], "finished_at": utc_now()}
            if name == "development":
                selected = _select_and_freeze_test(root, output)
                status["selected"] = selected["selected"]
            if name == "delay":
                from embodied_research.reporting import build_delay_report
                build_delay_report(output / "30_methods", output / "40_delay", output)
            if name == "flow_steps":
                from embodied_research.reporting import build_flow_report
                build_flow_report(output / "50_smolvla", output / "60_flow_steps", output)
            write_json(status_path, status)
            _write_index(output, status)
        status.update(status="complete_through_requested_stage", finished_at=utc_now())
        write_json(status_path, status)
        _write_index(output, status)
        return output / "report.html"
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        status.update(status="interrupted" if interrupted else "error", error=f"{type(error).__name__}: {error}",
                      finished_at=utc_now())
        if active_name is not None:
            status["stages"].setdefault(active_name, {})["status"] = status["status"]
        if not interrupted and active_directory is not None and (active_directory / "protocol.json").is_file():
            try:
                build_report(active_directory, partial=True)
            except Exception as report_error:
                status["partial_report_error"] = f"{type(report_error).__name__}: {report_error}"
        write_json(status_path, status)
        _write_index(output, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=[name for name, _ in STAGES], default="flow_steps")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/campaign")
    arguments = parser.parse_args()
    print(run_campaign(arguments.output, through=arguments.through))


if __name__ == "__main__":
    main()
