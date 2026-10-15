"""Regenerate plots from immutable completed records in a fresh interpreter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_research.reporting import build_delay_report, build_flow_report, build_report

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("10_coverage", "20_development", "30_methods", "40_delay", "50_smolvla", "60_flow_steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "outputs/campaign")
    parser.add_argument("--completed-only", action="store_true", help="Allow rebuilding completed stages during a running campaign")
    args = parser.parse_args()
    status = json.loads((args.campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if not args.completed_only and (status.get("through") != "flow_steps"
            or status.get("status") != "complete_through_requested_stage"):
        parser.error("Campaign is incomplete; use --completed-only for a clearly limited rebuild")
    completed = []
    for stage in STAGES:
        progress = args.campaign / stage / "progress.json"
        if progress.is_file() and json.loads(progress.read_text(encoding="utf-8")).get("finished"):
            print(build_report(progress.parent), flush=True)
            completed.append(stage)
    if {"30_methods", "40_delay"}.issubset(completed):
        print(build_delay_report(args.campaign / "30_methods", args.campaign / "40_delay", args.campaign))
    if {"50_smolvla", "60_flow_steps"}.issubset(completed):
        print(build_flow_report(args.campaign / "50_smolvla", args.campaign / "60_flow_steps", args.campaign))


if __name__ == "__main__":
    main()
