"""Read-only verification of an explicitly supplied public action-study snapshot.

No project checkout, private .runtime files, Git, raw demonstrations, Torch, or
model loading is required. NumPy is used only for numeric NPZ/statistical checks.
Public projection receipts cannot establish validity of the original full private
seals: omitted files and redacted original bytes are intentionally unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

sys.dont_write_bytecode = True
SCHEMA = "feedbackbench.action_publication.v1"
MAX_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def scan_private(content, label):
    # Decode all public text strictly; the private exporter performs local
    # identifier redaction. A portable verifier has no knowledge of that home.
    content.decode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def safe_name(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "Publication names must be portable relative paths")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in value.split("/")),
            "Unsafe publication path")
    require(all(not part.endswith((".", " ")) for part in path.parts), "Unsafe Windows path component")
    return value


def child(base, relative):
    path = Path(base) / safe_name(relative)
    # Never follow a symlink/junction outside a designated root.
    require(path.resolve().is_relative_to(Path(base).resolve()), "Path escapes its publication root")
    return path


def long_path(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def normal_path(path):
    value = str(path)
    return Path(value[4:] if value.startswith("\\\\?\\") else value).resolve()


def stable_bytes(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked evidence: {path.name}")
    require(path.stat().st_size <= MAX_BYTES, f"File exceeds public size limit: {path.name}")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Evidence changed while reading")
    require(expected is None or sha(content) == expected, f"Evidence hash differs: {path.name}")
    return content


def inspect_adapter(content, run, protocol_sha, binding_sha, run_id):
    """Inspect numeric NPZ only; never import torch or instantiate a policy."""
    import numpy as np

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        require(sum(info.file_size for info in archive.infolist()) < MAX_BYTES, "Oversized adapter contents")
        require(len({i.filename for i in archive.infolist()}) == len(archive.infolist()), "Duplicate NPZ members")
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        encoded = archive["manifest_utf8"]
        require(encoded.dtype == np.uint8 and encoded.ndim == 1, "Adapter manifest must be numeric UTF-8")
        manifest = json.loads(encoded.tobytes())
        scan_private(encoded.tobytes(), "adapter manifest")
        expected_meta = {"protocol_sha256": protocol_sha, "run_id": run_id,
                         "binding_sha256": binding_sha, "step": run["steps"]}
        require(manifest.get("metadata") == expected_meta, "Adapter is not bound to the final protocol/run/step")
        tensors = {name: archive[name] for name in archive.files if name != "manifest_utf8"}
        if run["mode"] == "lora":
            shapes = []
            names = sorted(f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"
                           for layer in range(32) for kind in ("q", "v"))
            modules = manifest.get("modules", [])
            require([m["name"] for m in modules] == names, "Wrong LoRA module selection")
            for name in names:
                layer = int(name.split("layers.")[1].split(".")[0])
                is_q = name.endswith("q_proj")
                shapes += [(8, 480 if is_q or layer % 2 == 0 else 320), (960 if is_q else 320, 8)]
            expected = {f"tensor_{i:03d}": shape for i, shape in enumerate(shapes)}
            require(manifest.get("schema") == "feedbackbench.action_expert_lora.v1"
                    and manifest.get("rank") == 8 and manifest.get("alpha") == 8
                    and manifest.get("dropout") == 0.0, "Wrong LoRA architecture")
            parameter_count = 552960
        else:
            require(run["mode"] == "output_head" and manifest.get("schema") == "feedbackbench.action_output_head.v1",
                    "Unsupported adapter family")
            expected, parameter_count = {"weight": (32, 480), "bias": (32,)}, 15392
        require(set(tensors) == set(expected) == set(manifest.get("tensor_sha256", {})), "Wrong adapter tensor set")
        for name, shape in expected.items():
            array = tensors[name]
            require(array.dtype == np.float32 and array.shape == shape and np.isfinite(array).all(), "Invalid adapter tensor")
            require(sha(array.tobytes()) == manifest["tensor_sha256"][name], "Adapter tensor digest changed")
        require(manifest["trainable_parameters"] == parameter_count
                and manifest.get("seed") == run["initialization_seed"], "Adapter parameter count or seed changed")
    return manifest


def inspect_simulation(trace, calls_raw, result):
    import numpy as np

    with zipfile.ZipFile(io.BytesIO(trace)) as archive:
        require(sum(info.file_size for info in archive.infolist()) < MAX_BYTES, "Oversized simulation trace")
        require(len({i.filename for i in archive.infolist()}) == len(archive.infolist()), "Duplicate simulation NPZ members")
    with np.load(io.BytesIO(trace), allow_pickle=False) as archive:
        steps = result["steps"]
        shapes = {"actions": (steps, 7), "states": (steps, 8), "next_states": (steps, 8),
                  "rewards": (steps,), "terminated": (steps,), "truncated": (steps,)}
        require(set(archive.files) == set(shapes), "Unexpected simulation arrays")
        for name, shape in shapes.items():
            array = archive[name]
            dtype = np.bool_ if name in ("terminated", "truncated") else np.float32
            require(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(), "Invalid simulation array")
        require(bool(archive["terminated"][-1]) == result["terminated"] and bool(archive["truncated"][-1]) == result["truncated"]
                and not archive["terminated"][:-1].any() and not archive["truncated"][:-1].any(), "Simulation terminal flags disagree")
        if steps > 1:
            require(np.array_equal(archive["next_states"][:-1], archive["states"][1:]), "Simulation state sequence is discontinuous")
    calls = [json.loads(line) for line in calls_raw.decode().splitlines() if line.strip()]
    require(len(calls) == result["decision_count"] and sum(c["actual_nfe"] for c in calls) == result["NFE"], "Call/NFE count differs")
    start = 0
    for i, call in enumerate(calls):
        require(call["decision"] == i and call["start_step"] == start and type(call["executed_steps"]) is int
                and 0 < call["executed_steps"] <= 10 and call["actual_nfe"] == 10, "Invalid native inference call record")
        for key in ("inference_seconds", "preprocess_seconds", "postprocess_seconds", "decision_wall_seconds"):
            require(isinstance(call[key], (int, float)) and math.isfinite(call[key]) and call[key] >= 0, "Invalid inference timing")
        token = f"action-eval-v1:{result['case']['task_uid']}:{result['case']['initial_state_id']}:{i}"
        require(call["noise_seed"] == int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little"), "Paired inference noise seed changed")
        start += call["executed_steps"]
    require(start == steps and sum(c["inference_seconds"] for c in calls) == result["inference_seconds"], "Calls do not reproduce trajectory length/timing")


def verify_contrasts(results, contrasts):
    """Reproduce the registered descriptive four-task bootstrap without model code."""
    import numpy as np

    rows = {(r["case"]["suite"], r["case"]["initial_state_id"], r["case"]["condition"], r["case"]["method_id"]):
            r["official_success"] for r in results}
    require(set(contrasts) == {f"lora_minus_{right}_{condition}" for right in ("frozen", "output_head")
                              for condition in ("clean", "brightness075")}, "Missing paired contrast")
    for key, contrast in contrasts.items():
        right_family, condition = key.removeprefix("lora_minus_").rsplit("_", 1)
        paired, per_task, missing = [], {}, False
        for suite in SUITES:
            differences = []
            for initial in (5, 15, 25, 35, 45):
                for seed in (17, 23, 47):
                    a = rows.get((suite, initial, condition, f"lora_s{seed}"))
                    right = "frozen" if right_family == "frozen" else f"output_head_s{seed}"
                    b = rows.get((suite, initial, condition, right))
                    if a is None or b is None:
                        missing = True
                        continue
                    delta = int(a) - int(b)
                    differences.append(delta)
                    paired.append({"suite": suite, "initial_state_id": initial, "seed": seed, "difference": delta})
            if differences:
                per_task[suite] = float(np.mean(differences))
        if missing:
            require(contrast == {"status": "pending_complete_pairs", "estimate": None}, "Incomplete paired data received an efficacy estimate")
            continue
        values = np.asarray(list(per_task.values()))
        random = np.random.default_rng(20261022)
        bootstrap = values[random.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        expected = {"status": "complete_pairs", "estimate": float(values.mean()),
                    "descriptive_task_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
                    "task_groups": 4, "initial_states_per_task": 5, "training_seeds": 3,
                    "per_task": per_task, "paired_differences": paired}
        require(all(contrast.get(k) == v for k, v in expected.items()), "Published paired contrast or task-bootstrap interval differs")


def validate_payload(contents):
    """Self-contained public audit; no raw data, models, or live study required."""
    manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
    require(manifest.get("schema") == SCHEMA, "Unknown publication manifest")
    require(set(contents) == set(manifest["files"]) | {"PUBLICATION_MANIFEST.json"}, "Public file inventory differs")
    for name, record in manifest["files"].items():
        safe_name(name)
        require(sha(contents[name]) == record["sha256"] and len(contents[name]) == record["bytes"], "Published content changed")
        if not name.endswith(".npz"):
            scan_private(contents[name], name)
    protocol = json.loads(contents["protocol.json"])
    require(protocol.get("frozen") is True and protocol.get("development_only") is not True
            and protocol.get("study_id") == "action-adaptation-v1", "Published protocol is not the formal study")
    require(sha(contents["protocol.json"]) == manifest["protocol_sha256"]
            == contents["protocol.sha256"].decode().strip(), "Published protocol binding changed")
    source_index = json.loads(contents["source_index.json"])
    require(set(source_index) == set(protocol["source_files"]), "Missing published source")
    for name, expected in protocol["source_files"].items():
        require(sha(contents[source_index[name]]) == expected, "Frozen source bytes changed")
    require(sha(contents["data_manifest.json"]) == protocol["data"]["manifest"]["sha256"], "Data split metadata changed")
    data = json.loads(contents["data_manifest.json"])
    require(sha(contents["data_sources.json"]) == data["source_manifest"]["sha256"], "Expert source metadata changed")
    for run in protocol["runs"].values():
        require(sha(contents["schedules/" + Path(run["schedule"]["path"]).name]) == run["schedule"]["sha256"], "Frozen schedule changed")
    for name, reference in (("prerequisites/probe.json", protocol["probe_evidence"]),
                            ("prerequisites/alignment.json", protocol["data"]["alignment_evidence"])):
        require(manifest["files"][name]["source_sha256"] == reference["sha256"], "Prerequisite source binding changed")
        require(json.loads(contents[name])["status"] == "passed", "Missing passed prerequisite")
    history = json.loads(contents["prerequisites/alignment_history/index.json"])["records"]
    require(any(item["status"] != "passed" for item in history.values()), "Prior diagnostic failures omitted")
    for original_sha, item in history.items():
        record = manifest["files"][item["public_path"]]
        require(record["source_sha256"] == original_sha and record["sha256"] == item["public_sha256"]
                and sha(contents[item["source_code"]]) == item["source_code_sha256"], "Historical alignment provenance changed")
        require(json.loads(contents[item["public_path"]])["status"] == item["status"], "Historical alignment result changed")
    pipeline = protocol["pipeline_evidence"]
    require(pipeline["excluded_from_formal_results"] is True, "Pipeline preflight counted as study results")
    for key in ("report", "protocol", "trained_manifest"):
        require(sha(contents[f"prerequisites/pipeline/{key}.json"]) == pipeline[key]["sha256"], "Pipeline evidence changed")
    historical_protocol = json.loads(contents["prerequisites/pipeline/protocol.json"])
    historical_index = json.loads(contents["prerequisites/pipeline/source_index.json"])
    require(set(historical_index) == set(historical_protocol["source_files"]), "Historical source coverage differs")
    for name, expected in historical_protocol["source_files"].items():
        require(sha(contents["prerequisites/pipeline/" + historical_index[name]]) == expected, "Historical source substitution")
    development = protocol["development_evaluation_evidence"]
    require(development["excluded_from_formal_results"] is True and development["success_was_not_a_gate"] is True,
            "Development episode used as an efficacy gate")
    for key in ("report", "protocol", "receipt"):
        target = "prerequisites/development/" + ("source_receipt.json" if key == "receipt" else key + ".json")
        require(sha(contents[target]) == development[key]["sha256"], "Development source binding changed")
    development_report = json.loads(contents["prerequisites/development/report.json"])
    require(development_report["excluded_from_formal_study"] is True and development_report["status"] == "complete",
            "Invalid development result")
    index = json.loads(contents["evidence_index.json"])
    report = json.loads(contents["report.json"])
    require(sha(contents["report.json"]) == manifest["report_sha256"], "Report binding changed")
    require(report["protocol_sha256"] == manifest["protocol_sha256"], "Report uses another protocol")
    require(report["training_completed"] == len(index["training"]) and report["training_planned"] == len(protocol["runs"]),
            "Training denominator changed")
    trained_manifest = None
    if report.get("trained_manifest_sha256") is not None:
        require(sha(contents["source_trained_manifest.json"]) == report["trained_manifest_sha256"], "Frozen trained-manifest binding changed")
        trained_manifest = json.loads(contents["source_trained_manifest.json"])
        require(trained_manifest["protocol_sha256"] == manifest["protocol_sha256"]
                and set(trained_manifest["runs"]) == set(index["training"]) == set(protocol["runs"]),
                "Evaluation requires every predeclared final adapter")
    for run_id, item in index["training"].items():
        receipt = json.loads(contents[item["source_receipt"]])
        require(canonical_hash({k: v for k, v in receipt.items() if k != "receipt_sha256"}) == receipt["receipt_sha256"],
                "Source training receipt selfhash changed")
        require(canonical_hash(receipt["binding"]) == receipt["binding_sha256"], "Source training binding changed")
        binding = receipt["binding"]
        run = protocol["runs"][run_id]
        require(binding["protocol_sha256"] == manifest["protocol_sha256"] and binding["run_id"] == run_id
                and binding["schedule_sha256"] == run["schedule"]["sha256"]
                and binding["data_manifest_sha256"] == protocol["data"]["manifest"]["sha256"]
                and binding["alignment_sha256"] == protocol["data"]["alignment_evidence"]["sha256"], "Training input binding differs")
        require(receipt["completed_steps"] == protocol["runs"][run_id]["steps"], "Nonfinal checkpoint exported")
        adapter = inspect_adapter(contents[item["adapter"]], protocol["runs"][run_id], manifest["protocol_sha256"], receipt["binding_sha256"], run_id)
        require(sha(contents[item["adapter"]]) == receipt["files"]["adapter.npz"], "Adapter differs from original receipt")
        summary = json.loads(contents[item["summary"]])
        require(summary["base_fingerprint"] == adapter["base_fingerprint"] and summary["completed_steps"] == receipt["completed_steps"],
                "Training summary differs from final adapter")
        require(summary["frozen_parameters_unchanged"] is True and summary["closed_loop_success_claim"] is False,
                "Training summary makes an unsupported base/success claim")
        if trained_manifest is not None:
            binding = trained_manifest["runs"][run_id]
            require(sha(contents[item["source_receipt"]]) == binding["receipt"]["sha256"]
                    and sha(contents[item["adapter"]]) == binding["adapter"]["sha256"]
                    and manifest["files"][item["summary"]]["source_sha256"] == binding["report"]["sha256"]
                    and binding["base_fingerprint"] == adapter["base_fingerprint"], "Public training evidence differs from frozen final manifest")
    cases = {case["case_id"]: case for case in protocol["evaluation"]["cases"]}
    require(set(index["evaluation"]) <= set(cases), "Unexpected evaluated case")
    results = []
    original_receipts = {r["sha256"] for r in report["sealed_receipts"]}
    require(not index["evaluation"] or trained_manifest is not None, "Evaluation lacks its frozen trained manifest")
    require(len(original_receipts) == len(index["evaluation"]) == report["evaluation_completed"], "Episode evidence count changed")
    # JSON sorts the evidence-index keys; preserve the frozen protocol order for
    # exactly the same floating-point timing accumulation as the study reporter.
    for case_id in cases:
        if case_id not in index["evaluation"]:
            continue
        item = index["evaluation"][case_id]
        result = json.loads(contents[item["report"]])
        receipt = json.loads(contents[item["source_receipt"]])
        require(sha(contents[item["source_receipt"]]) in original_receipts, "Unexpected source seal")
        require(receipt["binding"] == {"case_id": case_id, "protocol_sha256": manifest["protocol_sha256"],
                                      "trained_manifest_sha256": report["trained_manifest_sha256"]}, "Episode seal binding changed")
        require(sha(contents[item["report"]]) == receipt["files"]["report.json"]["sha256"], "Episode report differs from source seal")
        for key, source_name in (("actions", "actions.npz"), ("calls", "calls.jsonl")):
            require(sha(contents[item[key]]) == receipt["files"][source_name]["sha256"], "Simulation evidence differs from source seal")
        horizon = protocol["evaluation"]["horizons"][cases[case_id]["suite"]]
        require(result["case"] == cases[case_id] and result.get("status") == "complete"
                and type(result.get("official_success")) is bool and type(result.get("steps")) is int
                and 0 < result["steps"] <= horizon and result["horizon"] == horizon
                and type(result.get("terminated")) is bool and type(result.get("truncated")) is bool
                and (result["terminated"] or (result["truncated"] and result["steps"] == horizon)),
                "Incomplete or incorrectly bound result entered the denominator")
        require(result["protocol_sha256"] == manifest["protocol_sha256"]
                and result["trained_manifest_sha256"] == report["trained_manifest_sha256"], "Episode report protocol binding changed")
        inspect_simulation(contents[item["actions"]], contents[item["calls"]], result)
        results.append(result)
    require(report["evaluation_planned"] == len(cases) and set(report["missing_cases"]) == set(cases) - set(index["evaluation"]),
            "Missing cases or planned denominator changed")
    complete = len(index["training"]) == len(protocol["runs"]) and len(results) == len(cases)
    require(report["status"] == manifest["status"] == ("complete" if complete else "partial"), "False completion claim")
    expected_rate_keys = {(m["method_id"], condition) for m in protocol["evaluation"]["methods"] for condition in ("clean", "brightness075")}
    require({(r["method_id"], r["condition"]) for r in report["rates"]} == expected_rate_keys
            and len(report["rates"]) == len(expected_rate_keys), "Missing or duplicate method/condition rates")
    for rate in report["rates"]:
        selected = [r for r in results if (r["case"]["method_id"], r["case"]["condition"]) == (rate["method_id"], rate["condition"])]
        planned = sum((c["method_id"], c["condition"]) == (rate["method_id"], rate["condition"]) for c in cases.values())
        successes = sum(r["official_success"] for r in selected)
        expected = {"completed": len(selected), "planned": planned, "successes": successes, "failures": len(selected) - successes,
                    "observed_success_fraction": successes / len(selected) if selected else None,
                    "coverage_complete": len(selected) == planned,
                    "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected),
                    "inference_seconds": sum(r["inference_seconds"] for r in selected)}
        require(all(rate[k] == v for k, v in expected.items()), "Public outcomes do not reproduce reported rates/timings")
    verify_contrasts(results, report["contrasts"])
    for name in contents:
        if name.endswith("/PUBLIC_PROJECTION.json"):
            projection = json.loads(contents[name])
            require(projection["original_full_seal_verifiable_from_public_subset"] is False, "Misleading full-seal claim")
            require(sha(contents[projection["source_receipt"]]) == projection["source_receipt_sha256"], "Projection source changed")
            source = json.loads(contents[projection["source_receipt"]])
            require(projection["source_binding"] == source["binding"], "Projection binding differs from original receipt")
            require(set(projection["source_receipt_members"]) == set(projection["included"]), "Incomplete projection membership map")
            represented = {member for member in projection["source_receipt_members"].values() if member is not None}
            require(set(projection["omitted_source_files"]) == set(source["files"]) - represented,
                    "Projection omitted-file declaration differs from original receipt")
            for included, record in projection["included"].items():
                require(manifest["files"][included] == record, "Projection included file changed")
                member = projection["source_receipt_members"][included]
                if member is not None:
                    source_record = source["files"][member]
                    expected = source_record["sha256"] if isinstance(source_record, dict) else source_record
                    require(record["source_sha256"] == expected, "Projection membership claim differs from source receipt")
    required_projections = {"prerequisites/development/PUBLIC_PROJECTION.json"}
    required_projections.update(str(PurePosixPath(item["adapter"]).parent / "PUBLIC_PROJECTION.json") for item in index["training"].values())
    required_projections.update(str(PurePosixPath(item["report"]).parent / "PUBLIC_PROJECTION.json") for item in index["evaluation"].values())
    require(required_projections <= set(contents), "A public subset omits its explicit projection receipt")
    if manifest.get("supplemental_sources_sha256") is not None:
        verify_supplements(contents, manifest)
    return manifest


def verify_supplements(contents, manifest):
    raw = contents["supplemental_sources.json"]
    require(sha(raw) == manifest["supplemental_sources_sha256"], "Supplemental release binding changed")
    release = json.loads(raw)
    require(release.get("schema") == "feedbackbench.action_publication_supplements.v1"
            and release.get("outside_formal_source_inventory") is True
            and release.get("executed_in_archived_training_or_development") is False,
            "Later source release misrepresented as original execution source")
    require(set(release["groups"]) == {"public_reproduction", "public_verifier"}, "Unknown supplemental source group")
    reproduction = release["groups"]["public_reproduction"]
    source = json.loads(contents[reproduction["release_manifest"]])
    require(sha(contents[reproduction["release_manifest"]]) == reproduction["release_manifest_sha256"]
            and source.get("schema") == "feedbackbench.reproduction_source_release.v1"
            and source.get("postdates_formal_protocol_freeze") is True
            and source.get("executed_in_archived_development_run") is False
            and source.get("modifies_existing_frozen_protocols") is False, "Reproduction release provenance changed")
    require(set(source["files"]) == set(reproduction["files"]), "Reproduction release source coverage differs")
    for group in release["groups"].values():
        for original, record in group["files"].items():
            safe_name(original)
            require(sha(contents[record["target"]]) == record["sha256"]
                    and manifest["files"][record["target"]]["category"] == "supplemental_source", "Supplemental source changed")
            if group is reproduction:
                require(record["sha256"] == source["files"][original]
                        and len(contents[record["target"]]) == source["byte_sizes"][original], "Reproduction source release differs")

def verify_snapshot(directory, expected_sha256):
    """Verify published bytes/semantics; return explicit scope, never write files."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") is not None,
            "An explicit lowercase manifest SHA256 is required")
    directory = normal_path(directory)
    require(directory.is_dir(), "Snapshot directory is missing")
    filesystem_root = long_path(directory)
    raw = stable_bytes(filesystem_root / "PUBLICATION_MANIFEST.json", expected_sha256)
    manifest = json.loads(raw)
    require(manifest.get("schema") == SCHEMA and isinstance(manifest.get("files"), dict), "Unknown snapshot schema")
    require(len(manifest["files"]) <= 5000, "Unexpected public inventory size")
    require(all(isinstance(record.get("bytes"), int) and 0 <= record["bytes"] <= MAX_BYTES
                for record in manifest["files"].values()), "Invalid declared public file size")
    require(sum(record["bytes"] for record in manifest["files"].values()) <= MAX_TOTAL_BYTES,
            "Public snapshot exceeds the bounded verification memory limit")
    names = list(manifest["files"])
    require(len({name.casefold() for name in names}) == len(names), "Case-colliding public paths")
    require("PUBLICATION_MANIFEST.json" not in names, "Manifest may not recursively list itself")
    contents = {name: stable_bytes(child(filesystem_root, name), record["sha256"])
                for name, record in manifest["files"].items()}
    contents["PUBLICATION_MANIFEST.json"] = raw
    actual = {p.relative_to(filesystem_root).as_posix() for p in filesystem_root.rglob("*") if p.is_file()}
    require(actual == set(contents), "Unexpected or missing file in snapshot")
    validate_payload(contents)
    report = json.loads(contents["report.json"])
    return {"status": "verified", "manifest_sha256": sha(raw), "snapshot_id": manifest["snapshot_id"],
            "study_status": report["status"], "protocol_sha256": manifest["protocol_sha256"], "files": len(contents),
            "training_completed": report["training_completed"], "training_planned": report["training_planned"],
            "evaluation_completed": report["evaluation_completed"], "evaluation_planned": report["evaluation_planned"],
            "writes": 0, "model_loaded": False, "original_full_private_seals_verified": False,
            "historical_git_checkout_verified": False,
            "scope": "Published byte hashes, frozen bindings, public projections, numeric adapters/traces, outcomes, timings, and paired task-bootstrap statistics.",
            "limitations": ["Source SHA values for redacted/omitted evidence identify provenance; original private bytes cannot be reconstructed or verified here.",
                            "No Git history or earlier publication files are verified by this standalone command.",
                            "Simulation is not replayed; reported official success predicates are not independently re-evaluated by this file audit.",
                            "The caller must obtain the expected manifest SHA from a trusted publication reference."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_snapshot(args.snapshot, args.manifest_sha256)
        require(not args.require_complete or result["study_status"] == "complete", "Snapshot is partial")
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
