"""Verify the frozen reporting amendment and write its separate v5 report."""
from __future__ import annotations

import argparse
from pathlib import Path

from budget_campaign_common_v5 import canonical_hash, digest, owned_path, read_json, write_json
import budget_report_v5_r1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_reporting_revision(campaign):
    campaign = Path(campaign).resolve()
    repository = campaign.parents[2]
    revision_path = campaign / "reporting_r1/protocol.json"
    revision = read_json(revision_path)
    require(revision["schema"] == "feedbackbench.reporting_revision.v1", "Unknown reporting protocol schema")
    require(revision["revision_id"] == "v5-report-r1", "Unknown reporting revision")
    require(canonical_hash({k: v for k, v in revision.items() if k != "revision_fingerprint"}) == revision["revision_fingerprint"],
            "Reporting protocol fingerprint mismatch")
    protocol = read_json(campaign / "protocol.json")
    require(digest(campaign / "protocol.json") == revision["base_protocol_sha256"], "Original experiment protocol changed")
    require(protocol["campaign_fingerprint"] == revision["base_campaign_fingerprint"], "Original experiment fingerprint changed")
    for relative, expected in protocol["sources"].items():
        require(digest(owned_path(repository, relative)) == expected, f"Original experiment source changed: {relative}")
    for relative, expected in revision["sources"].items():
        require(digest(owned_path(repository, relative)) == expected, f"Reporting source changed: {relative}")
    require(digest(Path(__file__)) == revision["sources"]["scripts/run_budget_report_r1.py"], "Loaded reporting entry point differs")
    require(digest(Path(budget_report_v5_r1.__file__)) == revision["sources"]["scripts/budget_report_v5_r1.py"], "Loaded reporting implementation differs")
    for relative, record in revision["preserved_files"].items():
        path = owned_path(campaign, "reporting_r1", relative)
        require(digest(path) == record["sha256"] and path.stat().st_size == record["bytes"], f"Preserved revision evidence changed: {relative}")
    evidence = read_json(campaign / "reporting_r1/pre_revision_evidence.json")
    require(digest(campaign / "trained_manifest.json") == evidence["trained_manifest_sha256"], "Frozen trained manifest changed")
    for phase in ("collection", "evaluation"):
        for row in evidence[phase]:
            require(digest(owned_path(campaign, phase, row["directory"], "receipt.json")) == row["receipt_sha256"],
                    f"Pre-revision episode changed: {row['case_id']}")
    return revision


def generate_report(campaign):
    campaign = Path(campaign).resolve()
    revision = verify_reporting_revision(campaign)
    report = budget_report_v5_r1.generate_report(campaign)
    report["reporting_revision"] = {
        "id": revision["revision_id"],
        "fingerprint": revision["revision_fingerprint"],
        "protocol_sha256": digest(campaign / "reporting_r1/protocol.json"),
        "changed_after_test_started": True,
        "scope": "probability_boundary_validation_only",
        "experiment_campaign_fingerprint": revision["base_campaign_fingerprint"],
        "prior_results_rewritten": False,
        "completed_episodes_rerun": False,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = args.campaign.resolve()
    output = (args.output or campaign / "report_r1.json").resolve()
    require(output == campaign / "report_r1.json", "Only the campaign root report_r1.json output is permitted")
    report = generate_report(campaign)
    write_json(output, report)
    print(f"{report['status']}: {report['progress']['evaluation']['verified_complete']}/"
          f"{report['progress']['evaluation']['planned']} evaluation episodes; v5-report-r1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
