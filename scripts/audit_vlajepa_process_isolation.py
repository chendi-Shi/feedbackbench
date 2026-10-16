"""Audit sealed compact/isolated overlap evidence without policy/environment calls.

Run once all three overlapping isolated cases have sealed. Both complete study
matrices retain their own denominators. Existing audit output is never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": checksum}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_array(before, after):
    import numpy as np

    same_shape = before.shape == after.shape
    same_dtype = before.dtype == after.dtype
    exact = bool(same_shape and same_dtype and np.array_equal(before, after))
    numeric = same_shape and before.dtype.kind in "biuf" and after.dtype.kind in "biuf"
    return {
        "exact_equal": exact,
        "before_shape": list(before.shape), "after_shape": list(after.shape),
        "before_dtype": str(before.dtype), "after_dtype": str(after.dtype),
        "before_array_sha256": hashlib.sha256(np.ascontiguousarray(before).tobytes()).hexdigest(),
        "after_array_sha256": hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest(),
        "different_elements": int(np.count_nonzero(before != after)) if same_shape else None,
        "maximum_absolute_difference": float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64))))
        if numeric and before.size else (0.0 if numeric else None),
    }


def compare_npz(before, after, keys):
    import numpy as np

    with np.load(before, allow_pickle=False) as left, np.load(after, allow_pickle=False) as right:
        return {key: compare_array(left[key], right[key]) for key in keys}


def all_exact(comparisons):
    return all(row["exact_equal"] for row in comparisons.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/vlajepa-process-isolation-consistency-20261016.json")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Diagnostic output must stay in artifacts; original experiment outputs are read-only")
    if args.output.exists():
        raise FileExistsError("Existing diagnostic retained; choose an independently named output")
    from embodied_research.runtime import configure_platform
    configure_platform()
    import numpy as np
    from PIL import Image
    import run_vlajepa_compact as compact
    import run_vlajepa_isolated as isolated

    old_root = ROOT / "outputs/frontier/vlajepa_compact_closed_loop"
    new_root = ROOT / "outputs/frontier/vlajepa_isolated_closed_loop"
    old_runner, new_runner = compact.configured_runner(), isolated.configured_runner()
    old_study = old_runner.verify_study(old_root, partial=True)
    new_study = new_runner.verify_study(new_root, partial=True)
    workers = isolated.verify_workers(new_runner, new_root, new_study)
    old_protocol, new_protocol = old_runner.verify_protocol(old_root), new_runner.verify_protocol(new_root)
    old_cases = {case["suite"]: case for case in old_protocol["spec"]["cases"]}
    new_cases = {case["suite"]: case for case in new_protocol["spec"]["cases"]}
    overlaps = ("libero_spatial", "libero_object", "libero_goal")
    cases = []
    for suite in overlaps:
        before, after = old_root / "episodes" / old_cases[suite]["id"], new_root / "episodes" / new_cases[suite]["id"]
        if not all((directory / "receipt.json").is_file() for directory in (before, after)):
            raise ValueError("Wait for all three new overlap cases to seal before running this post-hoc audit")
        old_result = old_runner.verify_episode(before, old_cases[suite], old_protocol)
        new_result = new_runner.verify_episode(after, new_cases[suite], new_protocol)
        for key in ("task_id", "init_state_id", "seed", "condition", "method", "scheduler", "num_steps"):
            if old_cases[suite][key] != new_cases[suite][key]:
                raise ValueError("Overlap experimental parameters differ: " + key)
        initial_state = compare_array(np.asarray(old_result["initial_robot_state"], dtype=np.float32),
                                      np.asarray(new_result["initial_robot_state"], dtype=np.float32))
        images = {}
        for name in ("initial.png", "initial_wrist.png", "final.png"):
            with Image.open(before / name) as old_image, Image.open(after / name) as new_image:
                images[name] = compare_array(np.asarray(old_image), np.asarray(new_image))
        first_call = compare_npz(before / "call_0000.npz", after / "call_0000.npz",
                                 ("rng_state", "noise", "model_actions", "environment_actions", "input_robot_state"))
        trajectory = compare_npz(before / "trajectory.npz", after / "trajectory.npz",
                                 ("states", "actions", "final_state", "replan_steps"))
        old_refs, new_refs = read(before / "calls.json"), read(after / "calls.json")
        calls = []
        for index, (old_ref, new_ref) in enumerate(zip(old_refs, new_refs)):
            old_path, new_path = before / old_ref["record"], after / new_ref["record"]
            old_call, new_call = read(old_path), read(new_path)
            arrays = compare_npz(old_path.with_suffix(".npz"), new_path.with_suffix(".npz"),
                                 ("rng_state", "noise", "model_actions", "environment_actions", "input_robot_state"))
            calls.append({"call_index": index, "old_record": binding(old_path), "new_record": binding(new_path),
                          "old_arrays": binding(old_path.with_suffix(".npz")), "new_arrays": binding(new_path.with_suffix(".npz")),
                          "input_hashes_exact": old_call["input_sha256"] == new_call["input_sha256"],
                          "task_exact": old_call["task"] == new_call["task"], "arrays": arrays,
                          "all_arrays_exact": all_exact(arrays)})
        terminal_keys = ("success", "terminated", "truncated", "steps", "horizon", "inference_calls")
        terminal_exact = all(old_result[key] == new_result[key] for key in terminal_keys)
        exact = (initial_state["exact_equal"] and all_exact(images) and all_exact(first_call)
                 and all_exact(trajectory) and terminal_exact and len(old_refs) == len(new_refs)
                 and all(row["all_arrays_exact"] and row["input_hashes_exact"] and row["task_exact"] for row in calls))
        cases.append({"suite": suite, "all_compared_numerical_evidence_exact": exact,
                      "old_receipt": binding(before / "receipt.json"), "new_receipt": binding(after / "receipt.json"),
                      "old_receipt_files": read(before / "receipt.json")["files"],
                      "new_receipt_files": read(after / "receipt.json")["files"],
                      "initial_robot_state": initial_state, "image_pixels": images,
                      "first_call_arrays": first_call, "full_saved_trajectory": trajectory,
                      "old_terminal": {key: old_result[key] for key in terminal_keys},
                      "new_terminal": {key: new_result[key] for key in terminal_keys}, "terminal_exact": terminal_exact,
                      "old_forward_calls": len(old_refs), "new_forward_calls": len(new_refs), "call_comparisons": calls})
    if "torch" in sys.modules:
        raise RuntimeError("Read-only audit unexpectedly imported Torch")
    exact = all(case["all_compared_numerical_evidence_exact"] for case in cases)
    result = {
        "schema": "feedbackbench.process_isolation_consistency.v1",
        "status": "exact_overlap_consistency" if exact else "overlap_differences_observed",
        "audited_at": datetime.now(timezone.utc).isoformat(), "audit_source": binding(__file__),
        "scope": "Post-hoc consistency diagnostic on three overlapping sealed episodes; no model calls, training or new benchmark trials.",
        "torch_imported": False, "old_protocol": binding(old_root / "protocol.json"),
        "new_protocol": binding(new_root / "protocol.json"),
        "old_protocol_fingerprint": old_protocol["fingerprint"],
        "new_protocol_fingerprint": new_protocol["fingerprint"],
        "old_matrix": {key: old_study[key] for key in ("planned", "completed", "complete", "successes", "task_failures")},
        "new_matrix": {key: new_study[key] for key in ("planned", "completed", "complete", "successes", "task_failures")},
        "isolated_worker_bindings_verified": [
            {key: row[key] for key in ("path", "sha256", "pid", "process_create_time", "case_id", "status",
                                       "protocol_fingerprint", "script_sha256", "run_records")}
            for row in workers],
        "worker_receipt_proof": "verify_workers validates distinct PID/create-time identities and binds each complete episode's model_info.run_record to its worker's hashed run record; verify_episode validates the receipt and every bound episode file.",
        "all_three_overlaps_exact": exact, "cases": cases,
        "limitations": ["Three repeated known tasks/initial states; no independent-success or generalization claim.",
                        "Both original matrices keep their own denominators, failures and incomplete cases.",
                        "Differences are reported without retries, exclusion or filtering.",
                        "Timing, RSS, wall-clock timestamps, process identities, file metadata and renderer allocator state are not numerical-equality targets.",
                        "Exact saved observations/actions do not prove universal input equivalence or all unobserved simulator state equality.",
                        "No model invocation, environment step, receipt rewrite or original evidence modification occurs here."]}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(cases), "all_exact": exact,
                      "output": str(args.output), "torch_imported": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
