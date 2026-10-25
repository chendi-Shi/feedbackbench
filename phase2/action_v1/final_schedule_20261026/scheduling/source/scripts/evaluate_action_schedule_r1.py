"""Authorize one scheduling-revision attempt, then call the frozen worker.

Only the receipt boundary is decorated: the scheduling binding is written
immediately before the unchanged original seal inventories the result files.
The original output-root guard, model, simulation and result binding remain.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from action_campaign_common import ROOT, write_json
from run_action_schedule_r1 import require, verify_launch_authorization, verify_revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, type=Path)
    parser.add_argument("--revision-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    args = parser.parse_args(argv)
    revision, parent, protocol = verify_revision(args.revision, args.revision_sha256)
    output = args.output.resolve()
    cases = [case for case in protocol["evaluation"]["cases"] if case["case_id"] == args.case_id]
    require(len(cases) == 1, "Case is absent or ambiguous")
    evidence = verify_launch_authorization(args.revision, args.revision_sha256, args.case_id,
                                          output, args.authorization)
    # The supervisor created this attempt's .launch, but no result directory yet.
    # Audit every older attempt independently without treating this active launch
    # as an exited attempt. The authorization binds its exact next ordinal.
    case_dir = parent / "evaluation" / args.case_id
    ordinal = evidence["authorization"]["attempt_number"]
    previous_launches = sorted(case_dir.glob("attempt_*.launch"))
    require([p.name for p in previous_launches] == [f"attempt_{n:03d}.launch" for n in range(1, ordinal + 1)],
            "Attempt ordinal no longer matches the durable launch inventory")
    require(not list(case_dir.glob("attempt_*/receipt.json")), "A sealed episode must never be retried")
    from run_action_campaign import refuse_unsealed_complete
    refuse_unsealed_complete(case_dir)
    # verify_launch_authorization also rechecks every predecessor's complete
    # file inventory, including the original outcome and any scheduling seal.
    import evaluate_action_adaptation as original
    frozen_seal = original.seal

    def seal_with_schedule(directory, binding):
        require(Path(directory).resolve() == output, "Original worker sealed another output")
        require(binding == {"protocol_sha256": revision["parent"]["protocol_sha256"],
                            "trained_manifest_sha256": revision["parent"]["trained_manifest_sha256"],
                            "case_id": args.case_id}, "Original result binding changed")
        require(not (output / "schedule_binding.json").exists(), "Scheduling seal binding already exists")
        verify_revision(args.revision, args.revision_sha256)
        current_evidence = verify_launch_authorization(args.revision, args.revision_sha256,
            args.case_id, output, args.authorization, require_live=False)
        require(current_evidence == evidence, "Scheduling authorization changed before sealing")
        write_json(output / "schedule_binding.json", evidence["binding"])
        return frozen_seal(directory, binding)

    original.seal = seal_with_schedule
    try:
        return original.main(["--protocol", str(parent / "protocol.json"),
            "--protocol-sha256", revision["parent"]["protocol_sha256"],
            "--trained-manifest", str(parent / "trained_manifest.json"),
            "--trained-manifest-sha256", revision["parent"]["trained_manifest_sha256"],
            "--case-id", args.case_id, "--output", str(output)])
    finally:
        original.seal = frozen_seal


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform
    configure_platform()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    raise SystemExit(main())
