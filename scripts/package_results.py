"""Package completed evidence, preserving raw results and redacting diagnostic copies."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

from embodied_research.experiment import digest_json, expand_plan
from embodied_research.statistics import load_experiment

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("10_coverage", "20_development", "30_methods", "40_delay", "50_smolvla", "60_flow_steps")
DIRECTORIES = ("src", "scripts", "tests", "docs", "plans", ".github", "outputs/campaign",
               "outputs/smoke-v1", "outputs/simulator-smoke", "outputs/replays", "outputs/checkpoints/before_smol")
REQUIRED_FILES = (
    "README.md", "pyproject.toml", "constraints-runtime.txt", "requirements-runtime.in",
    "requirements-lock.txt", "Run.ps1", ".gitignore", "artifacts/hardware.json",
    "artifacts/smolvla_profile.json", "artifacts/policy_smoke_observation.npz",
    "artifacts/unit-tests.xml", "artifacts/policy-tests.xml", "artifacts/smolvla-loading-tests.xml",
    "artifacts/runtime_metadata_audit.json", "artifacts/pip-check.txt",
    "assets/libero/asset_manifest.json", "models/minerva/download_manifest.json",
    "models/smolvla/download_manifest.json", "models/smolvlm_assets/download_manifest.json",
)
TEXT_SUFFIXES = {".json", ".xml", ".txt", ".log", ".csv", ".tsv", ".md", ".html", ".svg", ".py", ".toml", ".yaml", ".yml", ".ps1"}
PRIVATE_PATH = re.compile(r"(?i)(?:\b[A-Z]:[\\/]+Users[\\/]+|/(?:home|Users)/)[^\"'<>\r\n]+")
HOST_ATTRIBUTE = re.compile(r'\bhostname\s*=\s*["\'][^"\']+["\']', re.IGNORECASE)
HOST_KEYS = {"hostname", "computername"}
PROBE_MANIFESTS = {"models/vlajepa/manifest.json", "models/vlajepa/manifest.pending.json"}
INT8_VARIANT = "cpu_qwen_dynamic_int8_bf16_embedding_fp32_action"
INT8_INPUT_REVISIONS = {
    "lerobot_revision": "ae2b57f12223c8c5184fcd814eb63cd9ce7de203",
    "policy_revision": "735d9f692981e286ade093b5046627eda876e5d0",
    "qwen_metadata_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
}
VLAJEPA_CHECKPOINT = {"path": "models/vlajepa/policy/model.safetensors", "bytes": 6163212550,
                     "sha256": "b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _safe_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-project bundle input: {name}")
    return path


def validate_episode_files(directory: Path, result: dict) -> None:
    """Require usable trajectory and camera evidence, not only a result JSON."""
    import numpy as np
    from PIL import Image

    for name in ("result.json", "trajectory.npz", "initial.png", "initial_wrist.png", "final.png"):
        _safe_file(directory.resolve(), name)
    if result.get("record_video"):
        if result.get("video") != "rollout.gif":
            raise ValueError(f"Recorded video is not declared: {result['id']}")
        video = _safe_file(directory.resolve(), "rollout.gif")
        with Image.open(video) as image:
            if image.format != "GIF" or image.size != (result["environment"]["image_size"],) * 2:
                raise ValueError(f"Invalid recorded video: {result['id']}")
            image.verify()
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        actions, states, final = data["actions"], data["states"], data["final_state"]
        if (actions.shape != (result["steps"], 7) or states.shape != (result["steps"], 8)
                or final.shape != (8,) or not result["steps"]
                or not all(np.isfinite(value).all() for value in (actions, states, final))):
            raise ValueError(f"Invalid saved trajectory: {result['id']}")
        np.testing.assert_array_equal(states[0], np.asarray(result["initial_robot_state"], dtype=states.dtype))
        np.testing.assert_array_equal(final, np.asarray(result["final_robot_state"], dtype=final.dtype))
    size = result["environment"]["image_size"]
    for name in ("initial.png", "initial_wrist.png", "final.png"):
        with Image.open(directory / name) as image:
            pixels = np.asarray(image.convert("RGB"))
            if pixels.shape != (size, size, 3):
                raise ValueError(f"Invalid saved camera dimensions: {result['id']}/{name}")
            if name == "initial.png" and hashlib.sha256(pixels.tobytes()).hexdigest() != result["initial_image_sha256"]:
                raise ValueError(f"Initial camera hash differs from result: {result['id']}")


def validate_html_resources(document: Path, root: Path) -> None:
    """Fail closed on missing relative report links, including exported figures."""
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in {"src", "href"} or not value:
                    continue
                reference = urlsplit(value)
                if reference.scheme or reference.netloc or not reference.path:
                    continue
                target = (document.parent / unquote(reference.path)).resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file():
                    raise ValueError(f"Missing or out-of-project report resource: {document.name}: {value}")

    Links().feed(document.read_text(encoding="utf-8"))


def validate_campaign(root: Path) -> tuple[dict, dict]:
    campaign = root / "outputs/campaign"
    status = _read(campaign / "campaign_status.json")
    if status.get("status") != "complete_through_requested_stage" or status.get("through") != "flow_steps":
        raise ValueError("A research bundle requires every declared campaign stage to finish")
    freeze = _read(campaign / "campaign_freeze.json")
    for name, expected in freeze["source_sha256"].items():
        if _hash(_safe_file(root, name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    counts, stages = {}, {}
    for stage in STAGES:
        data = load_experiment(campaign / stage)
        protocol = data.protocol
        provenance = {key: protocol[key] for key in ("plan", "sources", "runtime_versions", "artifacts")}
        if digest_json(provenance) != protocol["fingerprint"]:
            raise ValueError(f"Protocol fingerprint differs from its source material: {stage}")
        if protocol["plan"] != _read(campaign / "frozen_plans" / f"{stage}.json"):
            raise ValueError(f"Stage plan differs from campaign freeze: {stage}")
        if protocol["cases"] != expand_plan(protocol["plan"]):
            raise ValueError(f"Declared cases differ from frozen plan: {stage}")
        for name, checksum in protocol["sources"].items():
            if _hash(_safe_file(root, f"src/embodied_research/{name}")) != checksum:
                raise ValueError(f"Stage source differs from bundled code: {stage}/{name}")
        summary = _read(campaign / stage / "summary.json")
        if summary.get("partial") is not False or summary.get("input_sha256") != data.input_sha256:
            raise ValueError(f"Stage report is partial or stale: {stage}")
        _safe_file(root, f"outputs/campaign/{stage}/report.html")
        for record in data.records:
            validate_episode_files(campaign / stage / "episodes" / record["id"], record)
        counts[stage] = len(data.records)
        stages[stage] = data
    selection = _read(campaign / "development_selection.json")
    development = stages["20_development"]
    if (selection["development_protocol_fingerprint"] != development.protocol["fingerprint"]
            or selection["development_inputs_sha256"] != digest_json(development.input_sha256)):
        raise ValueError("Development selection is not bound to the bundled evidence")
    for name, first, second, second_key in (
        ("delay", "30_methods", "40_delay", "delayed_input_sha256"),
        ("flow", "50_smolvla", "60_flow_steps", "candidate_input_sha256"),
    ):
        comparison = _read(campaign / f"{name}_comparison.json")
        if (comparison["baseline_input_sha256"] != stages[first].input_sha256
                or comparison[second_key] != stages[second].input_sha256):
            raise ValueError(f"Cross-stage report is stale: {name}")
        _safe_file(root, f"outputs/campaign/{name}_comparison.html")
    _safe_file(root, "outputs/campaign/report.html")
    for document in campaign.rglob("*.html"):
        validate_html_resources(document, root)
    assets = _read(root / "assets/libero/asset_manifest.json")
    for data in stages.values():
        evidence = data.protocol["artifacts"]
        if digest_json(assets) != evidence["assets_manifest_sha256"]:
            raise ValueError("Bundled asset manifest differs from the evaluated snapshot")
        for name, expected in evidence["models"].items():
            manifest = _read(root / "models" / name / "download_manifest.json")
            actual = {"revision": manifest["revision"], "files": {
                item["file"]: item["sha256"] for item in manifest["files"]}}
            if actual != expected:
                raise ValueError(f"Bundled model manifest differs from evaluated weights: {name}")
    return counts, {stage: data.protocol["fingerprint"] for stage, data in stages.items()}


def _sanitize(value):
    if isinstance(value, str):
        return PRIVATE_PATH.sub("<REDACTED_LOCAL_PATH>", value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: "<REDACTED_HOST>" if key.lower() in HOST_KEYS else _sanitize(item)
                for key, item in value.items()}
    return value


def _contains_host_key(value):
    if isinstance(value, dict):
        return any(key.lower() in HOST_KEYS or _contains_host_key(item) for key, item in value.items())
    return isinstance(value, list) and any(_contains_host_key(item) for item in value)


def archive_content(relative: str, raw: bytes) -> tuple[bytes, bool]:
    """Redact only diagnostic archive copies; never rewrite formal evidence."""
    path = Path(relative)
    diagnostic_json = (path.name == "runtime_audit.json"
                       or relative in {"outputs/simulator-smoke/smoke.json", "artifacts/runtime_metadata_audit.json",
                                       "outputs/frontier/profile.json", "outputs/frontier/profile.progress.json",
                                       "outputs/frontier/cache_study/compatibility.json",
                                       "outputs/frontier/flow_confirmation/compatibility.json"})
    if diagnostic_json:
        parsed = json.loads(raw)
        sanitized = _sanitize(parsed)
        if sanitized != parsed:
            return (json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n").encode(), True
    if path.suffix == ".xml" and path.parts[0] == "artifacts":
        document = ET.fromstring(raw)
        changed = False
        for element in document.iter():
            for key, value in element.attrib.items():
                replacement = "<REDACTED_HOST>" if key.lower() == "hostname" else _sanitize(value)
                changed |= replacement != value
                element.set(key, replacement)
            for field in ("text", "tail"):
                value = getattr(element, field)
                if value is not None:
                    replacement = _sanitize(value)
                    changed |= replacement != value
                    setattr(element, field, replacement)
        if changed:
            return ET.tostring(document, encoding="utf-8", xml_declaration=True), True
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = raw.decode("utf-8-sig")
        json_host = (path.suffix.lower() == ".json" and not diagnostic_json
                     and _contains_host_key(json.loads(text)))
        if PRIVATE_PATH.search(text) or HOST_ATTRIBUTE.search(text) or json_host:
            raise ValueError(f"Host-identifying text found outside redactable diagnostics: {relative}")
    return raw, False


def validate_artifact_identity(root: Path, artifacts: dict) -> None:
    """Bind the exported small manifests to evaluated identities, without loading weights."""
    assets = _read(_safe_file(root, "assets/libero/asset_manifest.json"))
    if (digest_json(assets) != artifacts["assets_manifest_sha256"]
            or assets["revision"] != artifacts["assets_revision"]):
        raise ValueError("Frontier asset identity changed")
    if set(artifacts["models"]) != {"smolvla", "smolvlm_assets"}:
        raise ValueError("Frontier model identity requires both SmolVLA and SmolVLM manifests")
    for model, expected in artifacts["models"].items():
        manifest = _read(_safe_file(root, f"models/{model}/download_manifest.json"))
        files = {item["file"]: item["sha256"] for item in manifest["files"]}
        actual = {"revision": manifest["revision"], "files": files}
        if len(files) != len(manifest["files"]) or actual != expected:
            raise ValueError(f"Frontier model identity changed: {model}")


def _validate_quantized_probe(root, relative, record, reference, audit):
    """Check the separate changed-numerics probe without importing model code."""
    if "variant" not in record and record["status"] != "passed_quantized_forward":
        return
    if record.get("variant") != INT8_VARIANT:
        raise ValueError(f"Unsupported or missing quantized probe variant: {relative}")
    changes = record.get("numeric_changes")
    if (not isinstance(changes, list) or not changes
            or any(not isinstance(change, str) or not change.strip() for change in changes)):
        raise ValueError(f"Quantized probe must disclose numeric changes: {relative}")
    if "script_snapshot" not in record:
        raise ValueError(f"Quantized probe lacks variant source snapshot: {relative}")
    base = record.get("base_script")
    if not isinstance(base, dict) or base.get("path") != "scripts/probe_vlajepa.py":
        raise ValueError(f"Quantized probe lacks original-source binding: {relative}")
    stem = relative.removesuffix(".json")
    reference(base.get("snapshot_path"), stem + ".base_source.py", base.get("sha256"))
    if (type(record.get("inference_executed")) is not bool
            or type(record.get("forward_started")) is not bool
            or type(record.get("forward_calls")) is not int
            or record["forward_calls"] not in (0, 1)
            or record["forward_calls"] != int(record["inference_executed"])
            or (record["inference_executed"] and not record["forward_started"])
            or record["status"] == "passed_real_forward"
            or (record["status"] == "passed_quantized_forward" and not record["inference_executed"])):
        raise ValueError(f"Quantized forward-stage flags disagree: {relative}")
    audit.update(variant=record["variant"], numeric_changes=changes,
                 source_snapshot=record["script_snapshot"], base_script=base,
                 parity_established=False, task_quality_evaluated=False,
                 forward_started=record["forward_started"], forward_calls=record["forward_calls"],
                 successful_quantized_forward=record["status"] == "passed_quantized_forward")
    audit["limitations"].append(
        "Separate changed-numerics CPU INT8 feasibility variant. Even a completed one-observation "
        "forward establishes neither original-model parity nor closed-loop task quality.")
    if record["status"] != "passed_quantized_forward":
        audit["limitations"].append(
            "No successful quantized forward is established; a failed or interrupted stage may include an attempted forward.")
        return
    import numpy as np

    required = {"manifest", "schema", "loading", "actions", "observation", "quantized_forward_audit"}
    if not required <= record.keys():
        raise ValueError(f"Quantized forward lacks bound evidence: {relative}")
    if "snapshot_path" not in record["manifest"]:
        raise ValueError(f"Quantized forward requires immutable inputs snapshot: {relative}")
    manifest = _read(_safe_file(root, record["manifest"]["snapshot_path"]))
    checkpoint = [entry for entry in manifest.get("files", [])
                  if isinstance(entry, dict) and entry.get("path") == VLAJEPA_CHECKPOINT["path"]]
    if (record["manifest"].get("path") != "models/vlajepa/manifest.json"
            or record["manifest"].get("checkpoint_complete") is not True
            or manifest.get("status") != "verified"
            or any(manifest.get(key) != value for key, value in INT8_INPUT_REVISIONS.items())
            or any(record.get(key) != INT8_INPUT_REVISIONS[key] for key in ("lerobot_revision", "policy_revision"))
            or len(checkpoint) != 1
            or any(checkpoint[0].get(key) != value for key, value in VLAJEPA_CHECKPOINT.items())):
        raise ValueError(f"Quantized forward inputs lack the complete pinned checkpoint identity: {relative}")
    for field, expected in (("forward_calls", 1), ("flow_steps", 4), ("batch_size", 1)):
        if type(record.get(field)) is not int or record[field] != expected:
            raise ValueError(f"Quantized forward scope differs: {relative}: {field}")
    loading = _read(_safe_file(root, record["loading"]["path"]))
    for field in ("all_required_inference_keys_consumed", "all_source_tensors_finite", "native_tied_word_embeddings"):
        if loading.get(field) is not True:
            raise ValueError(f"Quantized loading audit is incomplete: {relative}: {field}")
    for field, expected in (("linear_modules", 300), ("linear_elements", 1811939328),
                            ("theoretical_parameter_payload_bytes", 3072709660)):
        if type(loading.get(field)) is not int or loading[field] != expected:
            raise ValueError(f"Quantized loading inventory differs: {relative}: {field}")
    summary = {key: value for key, value in loading.items() if key != "tensors"}
    repeated = {key: value for key, value in record["loading"].items() if key not in {"path", "sha256"}}
    if digest_json(summary) != digest_json(repeated):
        raise ValueError(f"Quantized loading summaries disagree: {relative}")
    forward = record["quantized_forward_audit"]
    calls = forward.get("module_calls") if isinstance(forward, dict) else None
    if (not isinstance(calls, dict) or len(calls) != 300
            or forward.get("distinct_modules_executed") != 300
            or forward.get("all_observed_inputs") != "CPU float32"
            or any(type(value) is not int or value < 1 for value in calls.values())):
        raise ValueError(f"Quantized executed-module audit differs: {relative}")
    if record["actions"].get("shape") != [1, 7, 7]:
        raise ValueError(f"Quantized action metadata shape differs: {relative}")
    with np.load(_safe_file(root, record["actions"]["path"]), allow_pickle=False) as arrays:
        if set(arrays.files) != {"model_actions", "environment_actions"}:
            raise ValueError(f"Quantized action inventory differs: {relative}")
        for name in arrays.files:
            array = arrays[name]
            if array.shape != (1, 7, 7) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Quantized actions are invalid: {relative}: {name}")
        first = np.asarray(record["actions"].get("first_environment_action"))
        if (first.shape != (7,) or not np.isfinite(first).all()
                or not np.allclose(first, arrays["environment_actions"][0, 0], rtol=0, atol=1e-7)):
            raise ValueError(f"Quantized first-action summary differs: {relative}")


def frontier_probe_files(root: Path) -> tuple[set[Path], list[dict]]:
    """Export bounded probe references and download metadata, never model payloads."""
    files = {_safe_file(root, path.relative_to(root).as_posix())
             for path in (root / "artifacts/frontier").glob("*")
             if path.is_file() and path.suffix in {".json", ".npz", ".xml"}}
    audits = []
    for document in sorted(files):
        if document.suffix != ".json":
            continue
        record = _read(document)
        if not isinstance(record, dict) or "script_sha256" not in record or "status" not in record:
            continue
        relative = document.relative_to(root).as_posix()
        stem = relative.removesuffix(".json")

        def reference(name, expected_name, checksum):
            if name != expected_name or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError(f"Unsafe or unbound probe reference: {relative}")
            target = _safe_file(root, name)
            if _hash(target) != checksum:
                raise ValueError(f"Probe reference hash mismatch: {relative}: {name}")
            files.add(target)
            return name

        inference_executed = record.get("inference_executed")
        if "variant" not in record and "inference_executed" not in record:
            if record["status"] in {"blocked_preflight", "passed_schema_only"}:
                inference_executed = False
            elif record["status"] == "passed_real_forward":
                inference_executed = True
        audit = {"record": relative, "status": record["status"],
                 "inference_executed": inference_executed, "limitations": []}
        if "script_snapshot" in record:
            reference(record["script_snapshot"], stem + ".source.py", record["script_sha256"])
        else:
            audit["limitations"].append(
                "No retained source snapshot. This record alone does not establish a real "
                "forward pass or parity with the current probe script.")
        if "variant" not in record and inference_executed is None:
            timing = record.get("timings", {}).get("forward_seconds")
            native_source = root / "scripts/probe_vlajepa.py"
            if ("script_snapshot" in record and native_source.is_file()
                    and record["script_sha256"] == _hash(_safe_file(root, "scripts/probe_vlajepa.py"))
                    and type(timing) in (int, float) and math.isfinite(timing) and timing >= 0):
                audit.update(inference_executed=True, forward_returned_from_stage_evidence=True)
                audit["limitations"].append(
                    "The source-bound forward timing proves return from predict_action_chunk only; "
                    "it does not establish validated actions or a passed probe.")
            else:
                audit["limitations"].append(
                    "Inference execution is unknown: this legacy record has no explicit flag or source-bound returned-forward evidence.")
        if record["status"] == "passed_real_forward":
            if not {"script_snapshot", "manifest", "schema", "loading", "actions", "observation"} <= record.keys():
                raise ValueError(f"Real-forward probe lacks bound evidence: {relative}")
        if "manifest" in record:
            manifest = record["manifest"]
            if (not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str)
                    or manifest["path"] not in PROBE_MANIFESTS):
                raise ValueError(f"Unsafe probe manifest reference: {relative}")
            origin, checksum = manifest["path"], manifest.get("sha256")
            if "snapshot_path" in manifest:
                audit["manifest_archive_path"] = reference(manifest["snapshot_path"], stem + ".inputs.json", checksum)
                # A later download may supersede the live manifest. The immutable
                # inputs snapshot remains authoritative for this particular probe.
                if (root / origin).is_file():
                    original = _safe_file(root, origin)
                    if _hash(original) == checksum:
                        files.add(original)
            else:
                audit["manifest_archive_path"] = reference(origin, origin, checksum)
        for field, suffix in (("schema", ".schema.json"), ("loading", ".loading.json"), ("actions", ".actions.npz")):
            if field in record:
                item = record[field]
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid probe {field} reference: {relative}")
                reference(item.get("path"), stem + suffix, item.get("sha256"))
        if "observation" in record:
            item = record["observation"]
            if not isinstance(item, dict):
                raise ValueError(f"Invalid probe observation reference: {relative}")
            reference(item.get("path"), "artifacts/policy_smoke_observation.npz", item.get("sha256"))
        _validate_quantized_probe(root, relative, record, reference, audit)
        stopped_path = root / (stem + ".memory_stop.json")
        if stopped_path.is_file():
            stopped = _read(stopped_path)
            if (stopped.get("status") != "stopped_memory_guard"
                    or record["status"] in {"passed_real_forward", "passed_quantized_forward"}):
                raise ValueError(f"Probe memory-stop evidence contradicts status: {relative}")
            files.add(_safe_file(root, stem + ".memory_stop.json"))
            audit["effective_status"] = "stopped_memory_guard"
            audit["memory_stop"] = {"path": stem + ".memory_stop.json", "sha256": _hash(stopped_path)}
            audit["limitations"].append("Original record retains its last stage; the separate memory-guard stop is the terminal outcome.")
        audits.append(audit)
    # A completed download is useful provenance even when RAM preflight blocks
    # inference before the probe can bind a manifest. Never collect its payloads.
    name = "models/vlajepa/manifest.json"
    if (root / name).is_file():
        manifest_path = _safe_file(root, name)
        metadata = _read(manifest_path)
        files.add(manifest_path)
        audits.append({"record": name, "kind": "download_metadata",
                       "download_status": metadata.get("status"), "inference_executed": False,
                       "limitations": ["Download metadata only; it does not establish a real forward pass. "
                                       "A probe's bound inputs snapshot remains authoritative for that probe."]})
    name = "artifacts/vlajepa-real-probe-console.log"
    if (root / name).is_file():
        files.add(_safe_file(root, name))
    return files, audits


def validate_optional_retrieval_analysis(root: Path, output: Path) -> None:
    """An optional diagnostic must be complete and current if either export exists."""
    paths = [output / f"retrieval_analysis.{suffix}" for suffix in ("json", "csv")]
    if any(path.exists() for path in paths):
        for path in paths:
            _safe_file(root, path.relative_to(root).as_posix())
        from analyze_cache_retrieval import validate_analysis

        validate_analysis(output, partial=False, root=root)


def validate_frontier(root: Path) -> tuple[dict, dict]:
    """Validate complete new study plus exactly four preserved baseline stages.

    This never relabels the old 248-case campaign complete. Both source freezes
    and the checkpoint's individual raw-file hashes must still match.
    """
    from run_cache_study import SOURCES, study_spec, verify_cache_binding, verify_complete
    from summarize_cache_study import validate_report

    campaign = root / "outputs/campaign"
    freeze = _read(campaign / "campaign_freeze.json")
    for name, checksum in freeze["source_sha256"].items():
        if _hash(_safe_file(root, name)) != checksum:
            raise ValueError(f"Frozen baseline source changed: {name}")
    checkpoint = _read(campaign / "resume_audit.json")
    for name, checksum in checkpoint["raw_episode_sha256"].items():
        if _hash(_safe_file(root, name)) != checksum:
            raise ValueError(f"Preserved baseline raw evidence changed: {name}")
    receipt = _read(root / "outputs/checkpoints/before_smol/stop_receipt.json")
    if receipt["status"] != "stopped" or receipt["audit_sha256"] != _hash(campaign / "resume_audit.json"):
        raise ValueError("Baseline stop receipt is missing or mismatched")
    counts, fingerprints, stages = {}, {}, {}
    assets = _read(root / "assets/libero/asset_manifest.json")
    for stage, expected_count in zip(STAGES[:4], (40, 32, 128, 32), strict=True):
        data = load_experiment(campaign / stage)
        if data.partial or len(data.records) != expected_count or data.errors:
            raise ValueError(f"Preserved baseline stage incomplete: {stage}")
        summary = _read(campaign / stage / "summary.json")
        if summary["partial"] or summary["input_sha256"] != data.input_sha256:
            raise ValueError(f"Stale baseline report: {stage}")
        protocol = data.protocol
        provenance = {key: protocol[key] for key in ("plan", "sources", "runtime_versions", "artifacts")}
        if digest_json(provenance) != protocol["fingerprint"]:
            raise ValueError(f"Baseline protocol fingerprint changed: {stage}")
        if (protocol["plan"] != _read(campaign / "frozen_plans" / f"{stage}.json")
                or protocol["cases"] != expand_plan(protocol["plan"])):
            raise ValueError(f"Baseline declared plan/cases changed: {stage}")
        for name, checksum in protocol["sources"].items():
            if Path(name).name != name or _hash(_safe_file(root, f"src/embodied_research/{name}")) != checksum:
                raise ValueError(f"Baseline stage source changed: {stage}/{name}")
        artifacts = protocol["artifacts"]
        if digest_json(assets) != artifacts["assets_manifest_sha256"]:
            raise ValueError(f"Baseline asset identity changed: {stage}")
        for model, expected in artifacts["models"].items():
            manifest = _read(_safe_file(root, f"models/{model}/download_manifest.json"))
            actual = {"revision": manifest["revision"], "files": {item["file"]: item["sha256"] for item in manifest["files"]}}
            if actual != expected:
                raise ValueError(f"Baseline model identity changed: {stage}/{model}")
        for record in data.records:
            validate_episode_files(campaign / stage / "episodes" / record["id"], record)
        counts[stage] = expected_count
        fingerprints[stage] = data.protocol["fingerprint"]
        stages[stage] = data
    selection = _read(campaign / "development_selection.json")
    development = stages["20_development"]
    if (selection["development_protocol_fingerprint"] != development.protocol["fingerprint"]
            or selection["development_inputs_sha256"] != digest_json(development.input_sha256)):
        raise ValueError("Baseline development selection changed")
    delay = _read(campaign / "delay_comparison.json")
    if (delay["baseline_input_sha256"] != stages["30_methods"].input_sha256
            or delay["delayed_input_sha256"] != stages["40_delay"].input_sha256):
        raise ValueError("Baseline delay comparison is stale")
    for stage in STAGES[4:]:
        # A protocol-only directory is preparation; any other evidence contradicts
        # the bundle's explicit claim that these original stages were unattempted.
        if any(path.is_file() and path.relative_to(campaign / stage).as_posix() != "protocol.json"
               for path in (campaign / stage).rglob("*")):
            raise ValueError(f"Original stage now contains an attempt: {stage}")
    output = root / "outputs/frontier/cache_study"
    protocol = _read(output / "protocol.json")
    provenance = {key: protocol[key] for key in ("spec", "sources", "runtime_versions", "artifacts")}
    if (digest_json(provenance) != protocol["fingerprint"] or protocol["spec"] != study_spec()
            or set(protocol["sources"]) != set(SOURCES)):
        raise ValueError("Frontier study protocol differs from its frozen sources/design")
    for name, checksum in protocol["sources"].items():
        if _hash(_safe_file(root, name)) != checksum:
            raise ValueError(f"Frozen frontier source changed: {name}")
    validate_artifact_identity(root, protocol["artifacts"])
    binding = _read(output / "frozen_cache/binding.json")
    verify_cache_binding(output, protocol, binding)
    for case in protocol["spec"]["cases"]:
        record = verify_complete(output / "episodes" / case["id"], case, protocol["fingerprint"])
        if record is None:
            raise ValueError(f"Frontier study incomplete: {case['id']}")
        validate_episode_files(output / "episodes" / case["id"], record)
    counts["frontier_development"] = 4
    counts["frontier_heldout"] = 24
    fingerprints["frontier_cache_study"] = protocol["fingerprint"]
    validate_report(output, root=root)
    validate_optional_retrieval_analysis(root, output)
    _safe_file(root, "outputs/frontier/cache_study/report.md")
    _safe_file(root, "docs/RESULTS.md")
    for document in campaign.rglob("*.html"):
        validate_html_resources(document, root)
    return counts, fingerprints


def validate_continuation(root: Path) -> tuple[dict, dict, dict]:
    """Require the complete prospective study while preserving earlier denominators."""
    from run_flow_confirmation import verify_study
    from summarize_flow_confirmation import validate_report

    counts, fingerprints = validate_frontier(root)
    output = root / "outputs/frontier/flow_confirmation"
    study = verify_study(output, partial=False, root=root)
    report = validate_report(output, partial=False, root=root)
    protocol = study["protocol"]
    cases = protocol["spec"]["cases"]
    comparison = report["comparison"]
    if (not study["complete"] or report["status"] != "complete" or len(study["records"]) != 16
            or report["counts"] != {"planned": 16, "complete": 16, "missing": 0, "unsealed": 0, "error": 0}
            or comparison["completed_pairs"] != 8 or comparison["observed_task_clusters"] != 4
            or {case["init_state_id"] for case in cases} != {41, 43}
            or {case["method"] for case in cases} != {"full10", "noise2"}):
        raise ValueError("Continuation requires the complete 16-episode, 8-pair, 4-task prospective study")
    for row in study["records"].values():
        validate_episode_files(output / "episodes" / row["case"]["id"], row["record"])
    _safe_file(root, "docs/FLOW_CONFIRMATION_PROTOCOL.md")
    counts = {**counts, "flow_confirmation": 16}
    fingerprints = {**fingerprints, "flow_confirmation": protocol["fingerprint"]}
    scope = {"episodes": 16, "paired_initial_states": 8, "task_clusters": 4,
             "initial_state_ids": [41, 43], "methods": ["full10", "noise2"],
             "protocol_fingerprint": protocol["fingerprint"], "hypothesis_origin": protocol["origin"],
             "earlier_evidence_role": "v1 selected this hypothesis exploratorily; earlier episodes are not pooled into this stage",
             "upstream_training_overlap": "Fresh to this project only; upstream training exclusion is not established"}
    return counts, fingerprints, scope


def validate_optional_flow_replay(root: Path) -> list[dict]:
    """Index the one optional diagnostic without adding another scored episode."""
    relative = "outputs/replays/flow-spatial43-noise2"
    output = root / relative
    if not output.exists():
        return []
    audit_path = output / "flow_replay_audit.json"
    audit = _read(audit_path) if audit_path.is_file() else None
    status = audit.get("status", "missing_status") if audit is not None else "missing_audit"
    entry = {"path": relative, "recorded_status": status,
             "excluded_from_benchmark_counts": True, "new_scored_episodes": 0,
             "exact_replay_verified": False,
             "limitations": ["Saved-action diagnostic only; not another policy rollout or success-rate observation.",
                             "Replay does not establish the cause of host timing variation."]}
    if audit is not None:
        entry["audit_sha256"] = _hash(_safe_file(root, f"{relative}/flow_replay_audit.json"))
    if status == "passed_exact_replay":
        from replay_flow_episode import validate_replay

        checked = validate_replay(output, root=root)
        entry.update(exact_replay_verified=True, original=checked["original"],
                     sources_sha256=checked["sources_sha256"],
                     new_policy_inference_calls=checked["new_policy_inference_calls"])
    else:
        entry["limitations"].append(
            "Failed or incomplete diagnostic retained as raw evidence; no trajectory, image, or terminal-flag parity is established.")
    return [entry]


def package(root: Path, destination: Path, *, frontier: bool = False, continuation: bool = False) -> dict:
    root, destination = root.resolve(), destination.resolve()
    frontier = frontier or continuation
    directories = (*DIRECTORIES, "outputs/frontier") if frontier else DIRECTORIES
    if destination.exists():
        raise FileExistsError("Preserve the existing bundle; choose another destination")
    if any(destination.is_relative_to(root / directory) for directory in directories):
        raise ValueError("Bundle destination must be outside source and evidence directories")
    continuation_scope = None
    continuation_diagnostics = []
    if continuation:
        counts, fingerprints, continuation_scope = validate_continuation(root)
        continuation_diagnostics = validate_optional_flow_replay(root)
    else:
        counts, fingerprints = validate_frontier(root) if frontier else validate_campaign(root)
    files = set()
    for directory in directories:
        for path in (root / directory).rglob("*"):
            if not continuation and any(path.is_relative_to(root / name) for name in (
                    "outputs/frontier/flow_confirmation", "outputs/replays/flow-spatial43-noise2")):
                continue
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".tmp", ".part"}:
                files.add(_safe_file(root, path.relative_to(root).as_posix()))
    for name in REQUIRED_FILES:
        files.add(_safe_file(root, name))
    probe_evidence = []
    if frontier:
        for name in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml"):
            files.add(_safe_file(root, name))
        for name in ("artifacts/frontier-unit-tests.xml",):
            if (root / name).is_file():
                files.add(_safe_file(root, name))
        if continuation:
            for name in ("artifacts/continuation-unit-tests.xml", "artifacts/continuation-report-tests.xml",
                         "artifacts/continuation-targeted-tests.xml", "artifacts/continuation-packaging-tests.xml",
                         "artifacts/flow-confirmation-tests.xml", "artifacts/flow-replay-tests.xml",
                         "artifacts/continuation-vlajepa-int8-tests.xml"):
                if (root / name).is_file():
                    files.add(_safe_file(root, name))
        probe_files, probe_evidence = frontier_probe_files(root)
        files.update(probe_files)
    manifest = {"format": "FeedbackBench completed research bundle", "schema_version": 2,
                "episode_counts_by_stage": counts, "protocol_fingerprints_by_stage": fingerprints,
                "excluded": ["model weights", "simulation assets", "Python environments",
                             "third-party source checkout (bootstrap fetches pinned commit)", "private service records"],
                "reproduce": ["python scripts/bootstrap.py --download",
                              "python scripts/run.py scripts/run_campaign.py --output outputs/reproduction"],
                "view": "Open outputs/campaign/report.html for the bundled evidence",
                "rebuild_stage_reports": [f"python scripts/run.py -m embodied_research report --output outputs/campaign/{stage}"
                                          for stage in STAGES],
                "privacy": {"scope": "Only diagnostic copies have local home paths and hostnames redacted; formal results/protocols are unchanged",
                            "hashes": "sha256 describes archive bytes; source_sha256 binds the unmodified local original",
                            "diagnostic_hash_fields": "Existing hashes inside redacted diagnostics still describe their original source material",
                            "redacted_files": []}, "files": {}}
    if frontier:
        manifest.update(
            format="FeedbackBench frontier study and preserved baseline evidence bundle",
            schema_version=3,
            probe_evidence=probe_evidence,
            completed_scope="28-case ActionCache transfer pilot plus four complete MINERVA stages (232 episodes)",
            unfinished_original_plan={"50_smolvla": 12, "60_flow_steps": 4,
                                      "status": "never attempted; superseded research priority, not completed"},
            reproduce=["python scripts/bootstrap.py --download",
                       "python scripts/run.py scripts/run_cache_study.py --output outputs/reproduction-cache"],
            view="Open docs/RESULTS.md and outputs/frontier/cache_study/report.md",
            rebuild_stage_reports=[f"python scripts/run.py -m embodied_research report --output outputs/campaign/{s}"
                                   for s in STAGES[:4]],
        )
    if continuation:
        manifest.update(
            format="FeedbackBench prospective continuation and preserved earlier evidence bundle",
            schema_version=4,
            completed_scope="16 prospective flow-confirmation episodes; 28 separate ActionCache pilot episodes; 232 preserved MINERVA episodes",
            continuation_scope=continuation_scope,
            continuation_diagnostics=continuation_diagnostics,
            separate_scopes={"minerva_baseline": 232, "action_cache_v1": 28, "flow_confirmation": 16,
                             "pooled_success_rate": "not defined across studies or models"},
            view="Open outputs/frontier/flow_confirmation/report.md, docs/FLOW_CONFIRMATION_PROTOCOL.md, and docs/RESULTS.md",
            reproduce=["python scripts/bootstrap.py --download",
                       "python scripts/run.py scripts/run_flow_confirmation.py --output outputs/reproduction-flow-confirmation"],
            quantized_probe_scope="Optional one-observation changed-numerics feasibility only; successful forward is not required, and establishes neither original-model parity nor task quality",
        )
    transformed = {}
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        source_hash = _hash(path)
        source_size = path.stat().st_size
        if path.suffix.lower() in TEXT_SUFFIXES:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source_hash:
                raise ValueError(f"Bundle input changed while reading: {relative}")
            content, redacted = archive_content(relative, raw)
            if redacted:
                transformed[relative] = content
                manifest["privacy"]["redacted_files"].append(relative)
            checksum, size = hashlib.sha256(content).hexdigest(), len(content)
        else:
            checksum, size = source_hash, source_size
        manifest["files"][relative] = {"sha256": checksum, "bytes": size,
                                       "source_sha256": source_hash, "source_bytes": source_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            created = True
            for path in sorted(files):
                relative = path.relative_to(root).as_posix()
                entry = manifest["files"][relative]
                if relative in transformed:
                    if _hash(path) != entry["source_sha256"]:
                        raise ValueError(f"Bundle input changed before archiving: {relative}")
                    archive.writestr("feedbackbench/" + relative, transformed[relative])
                    continue
                info = zipfile.ZipInfo("feedbackbench/" + relative)
                info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".gif", ".npz"} else zipfile.ZIP_DEFLATED
                checksum = hashlib.sha256()
                with path.open("rb") as source, archive.open(info, "w") as target:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        target.write(chunk)
                if checksum.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Bundle input changed while archiving: {relative}")
            archive.writestr("feedbackbench/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(destination) as archive:
            damaged = archive.testzip()
            if damaged:
                raise RuntimeError(f"Archive CRC validation failed: {damaged}")
    except BaseException:
        if created:
            destination.unlink()  # Only this invocation's newly created, failed archive.
        raise
    return {"destination": str(destination), "files": len(files), "bytes": destination.stat().st_size,
            "episode_counts_by_stage": counts, "redacted_diagnostic_files": manifest["privacy"]["redacted_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--frontier", action="store_true", help="Require the complete 28-case study and preserve the incomplete original 248-case plan honestly")
    scope.add_argument("--continuation", action="store_true", help="Require the separate complete 16-case prospective study plus all preserved frontier evidence")
    args = parser.parse_args()
    output = args.output or ROOT / "deliverables" / ("feedbackbench-continuation.zip" if args.continuation else "feedbackbench.zip")
    print(json.dumps(package(ROOT, output, frontier=args.frontier, continuation=args.continuation), indent=2))


if __name__ == "__main__":
    main()
