"""Small CLI: frozen protocol in, raw episodes and an auditable report out."""

import argparse
import json


def main():
    from .runtime import configure_platform
    configure_platform()
    parser = argparse.ArgumentParser(prog="feedbackbench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run/resume a declared paired experiment")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--limit", type=int)
    report = commands.add_parser("report", help="Report original records, without changing or excluding them")
    report.add_argument("--output", required=True)
    report.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        from .experiment import run_plan
        print(json.dumps(run_plan(args.plan, args.output, limit=args.limit), indent=2))
    else:
        from .reporting import build_report
        print(build_report(args.output, partial=args.partial))
