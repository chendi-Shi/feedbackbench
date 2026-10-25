"""Report the approved scheduling revision without changing parent reports."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_campaign_common import digest, now, read, write_json
from report_action_campaign import build_report as build_parent_report
from run_action_schedule_r1 import inspect_case, verify_revision


def build_report(revision_path, revision_sha):
    revision_path = Path(revision_path).resolve()
    revision, campaign, protocol = verify_revision(revision_path, revision_sha)
    parent = revision["parent"]
    completed, resource_missing, pending, audit, diagnostics = [], [], [], [], []
    for case in protocol["evaluation"]["cases"]:
        try:
            item = inspect_case(campaign / "evaluation" / case["case_id"], case,
                                parent["protocol_sha256"], parent["trained_manifest_sha256"],
                                revision_path, revision_sha)
        except (ValueError, RuntimeError, OSError) as error:
            # Partial/active/terminal-unsealed evidence is not a scientific result.
            # The supervisor separately uses strict inspect_case and cannot retry
            # these cases merely because this report preserves an unknown row.
            pending.append(case["case_id"])
            diagnostics.append({"case_id": case["case_id"], "error": repr(error),
                                "official_success": None, "automatic_retry_permitted": False})
            audit.append({"case_id": case["case_id"], "classification": "needs_audit"})
            continue
        category = item["classification"]
        if category == "completed":
            completed.append(case["case_id"])
        elif category == "resource_missing":
            resource_missing.append({"case_id": case["case_id"], "official_success": None,
                                     "attempt_count": item["attempt_count"],
                                     "reason": item.get("missing_reason", "three_resource_interruptions"),
                                     "attempts": item["attempts"]})
        elif category == "pending":
            pending.append(case["case_id"])
        else:
            raise ValueError("Unknown scheduling classification")
        audit.append({"case_id": case["case_id"], "classification": category,
                      "attempt_count": item["attempt_count"]})
    report = build_parent_report(campaign)
    if report["evaluation_completed"] != len(completed):
        raise ValueError("Parent scientific report and scheduling inventory differ")
    if len(completed) + len(resource_missing) + len(pending) != 280:
        raise ValueError("Original 280-case denominator changed")
    report.update(schema="feedbackbench.action_schedule_report.r1", generated_at=now(),
                  scheduling_revision_sha256=revision_sha,
                  scheduling_revision_id=revision["revision_id"],
                  resource_missing_cases=resource_missing, pending_cases=pending,
                  evaluation_resource_missing=len(resource_missing),
                  evaluation_pending=len(pending), case_audit=audit, case_audit_errors=diagnostics,
                  scheduling_finished=not pending,
                  original_study_complete=report["status"] == "complete")
    # Original reporter already withholds any contrast whose required pairs are absent.
    # Unknown resource cases remain unknown, not task failures or removed denominators.
    if resource_missing:
        report["status"] = "partial"
        report["original_study_complete"] = False
    report["limitations"].append(
        "Scheduling revision retains resource-exhausted cases as unknown in the original 280-case denominator. "
        "Finishing the remaining schedule does not complete the original study.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, type=Path)
    parser.add_argument("--revision-sha256", required=True)
    args = parser.parse_args(argv)
    result = build_report(args.revision, args.revision_sha256)
    # Fixed independent destination: never overwrite the original frozen report.
    write_json(args.revision.resolve().parent / "report.json", result)
    print({key: result[key] for key in ("status", "training_completed", "evaluation_completed",
           "evaluation_planned", "evaluation_resource_missing", "evaluation_pending", "scheduling_finished")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
