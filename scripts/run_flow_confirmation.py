"""Prospective sixteen-episode full10/noise2 confirmation on fresh initial states.

The existing cache pilot and all of its frozen sources stay unchanged.  This
runner imports Torch and simulator code only after a complete, pure preflight.
Partial and failed attempts are retained and cannot be automatically retried.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickletools
import re
import traceback
import zipfile

import numpy as np

from embodied_research.experiment import (
    artifact_provenance, digest_json, episode_id, run_episode, utc_now, versions, write_json,
)
from embodied_research.scheduling import SchedulerConfig

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (280, 280, 300, 520), strict=True))
INITIAL_STATES = (41, 43)
METHODS = ("full10", "noise2")
SOURCES = (
    "scripts/run_flow_confirmation.py", "scripts/run.py", "src/embodied_research/__init__.py",
    "src/embodied_research/cache_policy.py", "src/embodied_research/cache_sampler.py",
    "src/embodied_research/action_cache.py", "src/embodied_research/experiment.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
    "src/embodied_research/simulator.py", "src/embodied_research/scheduling.py",
    "scripts/summarize_flow_confirmation.py", "scripts/summarize_cache_study.py",
    "docs/FLOW_CONFIRMATION_PROTOCOL.md",
)
ORIGIN_PATHS = {
    "campaign_freeze": "outputs/campaign/campaign_freeze.json",
    "v1_protocol": "outputs/frontier/cache_study/protocol.json",
    "v1_report": "outputs/frontier/cache_study/report.json",
    "v1_exports": "outputs/frontier/cache_study/exports.json",
}
PROTOCOL_FIELDS = ("schema", "spec", "sources", "runtime_versions", "artifacts", "origin", "initial_states")
TIMINGS = ("preprocess_seconds", "preprocess_since_last_call_seconds", "vlm_seconds",
           "vlm_transformer_seconds", "vision_prefix_seconds", "input_preparation_seconds",
           "key_features_seconds", "prepare_seconds", "action_head_seconds", "latent_export_seconds",
           "sampling_seconds", "projection_lookup_seconds", "postprocess_seconds", "inference_seconds",
           "prediction_seconds", "end_to_end_seconds", "amortized_end_to_end_seconds")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=nonfinite)


def _relative(root, name):
    _require(isinstance(name, str) and not Path(name).is_absolute() and not Path(name).drive
             and "\\" not in name and ":" not in name,
             "Unsafe project-relative path")
    path = (Path(root) / name).resolve()
    _require(path.is_relative_to(Path(root).resolve()), "Path escapes project")
    return path


def cases_for_study():
    cases = []
    for task_index, suite in enumerate(SUITES):
        for initial_index, initial in enumerate(INITIAL_STATES):
            offset = (task_index + initial_index) % 2
            for method in METHODS[offset:] + METHODS[:offset]:
                case = {
                    "model": "smolvla", "suite": suite, "task_id": 1, "init_state_id": initial,
                    "seed": 1000, "condition": "clean", "method": method,
                    "num_steps": 10 if method == "full10" else 2,
                    "scheduler": {"mode": "fixed", "fixed_interval": 10, "temporal_ensemble": False},
                    "record_video": True, "split": "confirmation",
                }
                case["id"] = episode_id(case)
                cases.append(case)
    return cases


def study_spec():
    return {
        "name": "SmolVLA prospective full10 versus noise2 confirmation v2",
        "question": "Does two-step sampling from fresh noise retain success while reducing flow NFE on fresh initial states?",
        "primary": {"methods": list(METHODS), "pairs": 8, "task_clusters": 4,
                    "initial_states": list(INITIAL_STATES),
                    "metrics": ["success", "nfe_per_replan", "seconds_per_replan"]},
        "origin_scope": "Hypothesis motivated by completed v1; v1 outcomes are exploratory and are not pooled with confirmation.",
        "freshness": {"previous_project_initial_states": [0, 1, 2, 11, 29],
                      "scope": "New relative to our studies, not proven unseen in upstream pretraining."},
        "execution": {"threads": 2, "interop_threads": 1, "image_size": 360, "warmup_calls": 2,
                      "warmup": "First newly executed case observation; reset again before its scored episode.",
                      "replan_interval": 10, "temporal_ensemble": False, "simulation_control_hz": 20,
                      "noise_pairing": "One identical-shape noise draw per decision after resetting seed1000 per episode.",
                      "termination": "Official success or original full horizon; no wall-clock early exit.",
                      "suite_horizons": HORIZONS,
                      "order": "Rotate first method by (task_index + initial_index) modulo2; four blocks start each method."},
        "limitations": ["Four known pretrained tasks and eight paired initial states, not a full benchmark.",
                        "Small sample cannot establish equivalence or noninferiority.",
                        "Shared-host synchronous CPU timings do not demonstrate real-time deployment.",
                        "No caches, training, GPU, paid cloud, or real robot."],
        "cases": cases_for_study(),
    }


def verify_origins(root=ROOT):
    """Check both old source freezes and the completed hypothesis-origin exports."""
    root = Path(root)
    documents = {key: read_json(root / name) for key, name in ORIGIN_PATHS.items()}
    old, v1 = documents["campaign_freeze"], documents["v1_protocol"]
    for mapping in (old["source_sha256"], v1["sources"]):
        _require(bool(mapping), "Empty predecessor source freeze")
        for name, expected in mapping.items():
            _require(file_sha(_relative(root, name)) == expected, f"Predecessor frozen source changed: {name}")
    v1_fields = ("spec", "sources", "runtime_versions", "artifacts")
    _require(digest_json({key: v1[key] for key in v1_fields}) == v1["fingerprint"], "v1 fingerprint changed")
    report = documents["v1_report"]
    _require(report["status"] == "complete" and report["counts"]["complete"] == 28
             and report["counts"]["planned"] == 28, "Hypothesis origin must be completed v1")
    _require(report["protocol_fingerprint"] == v1["fingerprint"], "v1 report/protocol mismatch")
    exports = documents["v1_exports"]
    _require(exports["schema"] == "feedbackbench.cache_study_exports.v1", "Unsupported v1 exports")
    _require("report.json" in exports["files"], "v1 report is not exported")
    origin_root = root / "outputs/frontier/cache_study"
    for name, expected in exports["files"].items():
        _require(file_sha(_relative(origin_root, name)) == expected, f"v1 export changed: {name}")
    return {"files": {name: file_sha(root / name) for name in ORIGIN_PATHS.values()},
            "v1_protocol_fingerprint": v1["fingerprint"],
            "use": "Hypothesis origin only; no predecessor rollout contributes to confirmation denominators."}, v1


def initial_archive_shape(path):
    """Inspect the published NumPy-array torch ZIP without executing pickle code.

    The pinned files use protocol2, a two-dimensional little-endian float64
    ndarray and one latin1 payload. Only structural opcode data are read.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(sorted(names) == ["archive/data.pkl", "archive/version"], "Unsupported initial-state archive")
        payload = archive.read("archive/data.pkl")
    operations = list(pickletools.genops(payload))
    _require(operations[0][0].name == "PROTO" and operations[0][1] == 2,
             "Initial-state pickle protocol differs")
    globals_ = [argument for opcode, argument, _ in operations if opcode.name == "GLOBAL"]
    _require(globals_ in (["numpy.core.multiarray _reconstruct", "numpy ndarray", "_codecs encode", "numpy dtype"],
                         ["numpy._core.multiarray _reconstruct", "numpy ndarray", "_codecs encode", "numpy dtype"]),
             "Unsupported initial-state ndarray globals")
    compact = [(opcode.name, argument) for opcode, argument, _ in operations
               if opcode.name not in ("BINPUT", "LONG_BINPUT")]
    first_mark = next((index for index, (name, _) in enumerate(compact) if name == "MARK"), None)
    _require(first_mark is not None, "Initial-state array shape missing")
    shape_ops = compact[first_mark + 1:first_mark + 5]
    _require(len(shape_ops) == 4 and shape_ops[0][1] == 1 and shape_ops[-1][0] == "TUPLE2"
             and all(name in ("BININT1", "BININT2", "BININT", "INT") for name, _ in shape_ops[:3]),
             "Initial-state ndarray shape structure differs")
    shape = tuple(value for _, value in shape_ops[1:3])
    _require(all(type(value) is int and 0 < value < 100000 for value in shape), "Invalid initial-state dimensions")
    strings = [argument for opcode, argument, _ in operations if opcode.name == "BINUNICODE"]
    _require("f8" in strings and "<" in strings and "latin1" in strings, "Initial states require little-endian float64")
    expected_bytes = math.prod(shape) * 8
    values = [value for value in strings if len(value) == expected_bytes]
    _require(len(values) == 1, "Initial-state payload length differs from shape")
    array = np.frombuffer(values[0].encode("latin1"), dtype="<f8")
    _require(np.isfinite(array).all(), "Nonfinite published initial state")
    _require(operations[-1][0].name == "STOP", "Truncated initial-state pickle")
    return shape


def initial_state_manifest():
    distribution = importlib.metadata.distribution("hf-libero")
    root = Path(os.path.abspath(distribution.locate_file("libero/libero")))
    map_path = root / "benchmark/libero_suite_task_map.py"
    benchmark_path = root / "benchmark/__init__.py"

    def literal_assignment(path, name):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        matches = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        _require(len(matches) == 1, f"Ambiguous benchmark declaration: {name}")
        return ast.literal_eval(matches[0].value)

    task_map = literal_assignment(map_path, "libero_task_map")
    _require(literal_assignment(benchmark_path, "task_orders")[0] == list(range(10)),
             "Default benchmark task order changed")
    rows = []
    for suite in SUITES:
        task_name = task_map[suite][1]
        _require(re.fullmatch(r"[A-Za-z0-9_]+", task_name) is not None, "Unsafe task filename")
        relative = f"init_files/{suite}/{task_name}.pruned_init"
        path = root / relative
        shape = initial_archive_shape(path)
        _require(max(INITIAL_STATES) < shape[0], f"Declared initial state outside published range: {suite}")
        rows.append({"suite": suite, "task_id": 1, "task_name": task_name,
                     "package_relative_path": f"libero/libero/{relative}", "sha256": file_sha(path),
                     "shape": list(shape), "count": shape[0]})
    return {"distribution": "hf-libero", "version": distribution.version,
            "inspection": "zipfile+pickletools structural inspection; no pickle execution, Torch import or environment",
            "task_map_sha256": file_sha(map_path), "benchmark_source_sha256": file_sha(benchmark_path),
            "tasks": rows}


def make_protocol(root=ROOT):
    root = Path(root)
    origin, previous = verify_origins(root)
    initial_states = initial_state_manifest()
    artifacts = artifact_provenance(root, ["smolvla"])
    _require(artifacts == previous["artifacts"], "Model/upstream/assets differ from v1")
    body = {"schema": "feedbackbench.flow_confirmation_protocol.v1", "spec": study_spec(),
            "sources": {name: file_sha(root / name) for name in SOURCES}, "runtime_versions": versions(),
            "artifacts": artifacts, "origin": origin, "initial_states": initial_states}
    return {**body, "fingerprint": digest_json(body)}


def verify_protocol(protocol, root=ROOT):
    _require(all(key in protocol for key in (*PROTOCOL_FIELDS, "fingerprint")), "Incomplete confirmation protocol")
    _require(protocol["schema"] == "feedbackbench.flow_confirmation_protocol.v1", "Unsupported confirmation protocol")
    _require(protocol["spec"] == study_spec(), "Confirmation design differs from frozen runner")
    _require(digest_json({key: protocol[key] for key in PROTOCOL_FIELDS}) == protocol["fingerprint"],
             "Confirmation protocol fingerprint changed")
    _require(set(protocol["sources"]) == set(SOURCES), "Confirmation source inventory changed")
    for name, expected in protocol["sources"].items():
        _require(file_sha(_relative(root, name)) == expected, f"Frozen confirmation source changed: {name}")
    origin, previous = verify_origins(root)
    _require(protocol["origin"] == origin, "Hypothesis origin binding changed")
    _require(protocol["artifacts"] == previous["artifacts"], "Confirmation artifact identities differ from v1")
    rows = protocol["initial_states"]["tasks"]
    _require(len(rows) == 4 and {row["suite"] for row in rows} == set(SUITES), "Initial-state task denominator changed")
    for row in rows:
        _require(row["task_id"] == 1 and row["count"] == row["shape"][0] > max(INITIAL_STATES)
                 and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is not None,
                 "Invalid published initial-state metadata")


def _exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze_protocol(output, root=ROOT):
    """Pure preflight; does not import Torch or create a simulator."""
    output = Path(output)
    protocol = make_protocol(root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        saved = read_json(path)
        verify_protocol(saved, root)
        _require(saved["fingerprint"] == protocol["fingerprint"], "Code/config/dependencies/inputs changed; refusing resume")
        return saved
    _require(not any(output.iterdir()), "Unbound existing study output; refusing to replace")
    saved = {**protocol, "created_at": utc_now()}
    _exclusive_json(path, saved)
    return saved


def verify_complete(directory, case, fingerprint):
    directory = Path(directory)
    if not directory.exists():
        return None
    _require(directory.is_dir(), "Episode path must be a directory")
    result_path, receipt_path = directory / "result.json", directory / "receipt.json"
    if not result_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Partial attempt retained; automatic retry forbidden: {case['id']}")
    result, receipt = read_json(result_path), read_json(receipt_path)
    _require(result.get("status") == "complete" and result.get("protocol_fingerprint") == fingerprint,
             f"Failed or incompatible record: {case['id']}")
    _require(all(result.get(key) == value for key, value in case.items()), "Case identity/configuration changed")
    children = list(directory.iterdir())
    _require(all(path.is_file() and not path.is_symlink() for path in children), "Unexpected episode subdirectory/link")
    actual = {path.name for path in children if path.name != "receipt.json"}
    _require(actual == set(receipt["files"]), "Episode receipt inventory changed")
    required = {"attempt.json", "result.json", "calls.json", "trajectory.npz", "initial.png", "initial_wrist.png", "final.png"}
    if case["record_video"]:
        required.add("rollout.gif")
    _require(required <= actual, "Required episode artifact missing")
    for name, expected in receipt["files"].items():
        _require(Path(name).name == name and not any(char in name for char in "/\\:"), "Unsafe receipt path")
        _require(file_sha(directory / name) == expected, f"Episode artifact hash mismatch: {name}")
    attempt = read_json(directory / "attempt.json")
    _require(attempt["case"] == case and attempt["protocol_fingerprint"] == fingerprint, "Attempt binding differs")
    return result


def seal_episode(directory):
    directory = Path(directory)
    _exclusive_json(directory / "receipt.json", {
        "files": {path.name: file_sha(path) for path in sorted(directory.iterdir()) if path.is_file()},
        "sealed_at": utc_now(),
    })


def _validate_record(directory, case, record, protocol):
    from PIL import Image

    calls = read_json(directory / "calls.json")
    _require(record["hypothesis_origin_sha256"] == digest_json(protocol["origin"]),
             "Episode hypothesis-origin binding differs")
    steps = record["steps"]
    _require(type(steps) is int and 0 < steps <= HORIZONS[case["suite"]], "Invalid episode steps")
    _require(record["horizon"] == HORIZONS[case["suite"]], "Original full horizon changed")
    expected_scheduler = asdict(SchedulerConfig(**case["scheduler"], horizon=record["horizon"]))
    _require(record["scheduler_config"] == expected_scheduler, "Execution scheduler differs from protocol")
    expected_environment = {"suite": case["suite"], "task_id": case["task_id"], "horizon": record["horizon"],
                            "control_hz": 20, "control_mode": "relative", "hard_reset": True,
                            "settle_steps": 10, "image_size": 360, "images_flipped_hw_once": True,
                            "object_state_exposed_to_policy": False}
    _require(all(record["environment"].get(key) == value for key, value in expected_environment.items()),
             "Environment differs from confirmation protocol")
    _require(type(record["success"]) is bool and type(record["terminated"]) is bool
             and type(record["truncated"]) is bool, "Invalid termination flags")
    _require(record["terminated"] or record["truncated"], "Episode ended without official terminal or horizon")
    _require(record["success"] or steps == record["horizon"], "Unsuccessful rollout ended early")
    expected_steps = list(range(0, steps, 10))
    _require(record["inference_calls"] == len(calls) == len(expected_steps), "Inference count disagrees with fixed execution")
    model = record["model_info"]
    checkpoint = protocol["artifacts"]["models"]["smolvla"]
    _require((model["revision"], model["model_sha256"]) ==
             (checkpoint["revision"], checkpoint["files"]["model.safetensors"]), "Policy checkpoint identity changed")
    meta = model["cache_policy"]
    _require(meta["mode"] == case["method"] and meta["call_log"] == calls and meta["cache"] is None,
             "Policy/call evidence differs or unexpected cache")
    for index, call in enumerate(calls):
        _require(call["call_index"] == index and call["observation_count"] == expected_steps[index] + 1,
                 "Decision/observation indices disagree")
        _require(call["mode"] == case["method"] and call["nfe"] == case["num_steps"]
                 and type(call["nfe"]) is int and call["hit"] is None, "NFE/mode differs from no-cache inference")
        _require(re.fullmatch(r"[a-f0-9]{64}", call["noise_sha256"]) is not None, "Invalid noise fingerprint")
        _require(all(type(call.get(key)) in (int, float) and math.isfinite(call[key]) and call[key] >= 0
                     for key in TIMINGS), "Invalid call timing")
    _require(meta["total_nfe"] == sum(call["nfe"] for call in calls), "Total NFE differs from calls")
    _require(math.isfinite(record["wall_seconds"]) and record["wall_seconds"] >= 0, "Invalid rollout wall time")
    _require(math.isclose(record["policy_inference_seconds"], sum(call["inference_seconds"] for call in calls),
                         rel_tol=1e-9, abs_tol=1e-7), "Episode/call inference timing differs")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        actions, states = trajectory["actions"].copy(), trajectory["states"].copy()
        _require(actions.shape == (steps, 7) and states.shape == (steps, 8)
                 and np.isfinite(actions).all() and np.isfinite(states).all(), "Invalid action/state trajectory")
        _require(np.array_equal(trajectory["replan_steps"], expected_steps), "Trajectory decision steps differ")
        _require(np.allclose(states[0], record["initial_robot_state"], rtol=0, atol=1e-6), "Initial state differs")
    with Image.open(directory / "initial.png") as image:
        image_digest = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
    _require(image_digest == record["initial_image_sha256"], "Initial RGB fingerprint differs")
    receipt = read_json(directory / "receipt.json")
    return {"case": case, "record": record, "calls": calls, "actions": actions,
            "result_sha256": file_sha(directory / "result.json"),
            "receipt_sha256": file_sha(directory / "receipt.json"), "artifact_sha256": receipt["files"]}


def verify_study(output, partial=False, root=ROOT):
    """Read-only report/packaging consumer; no weights or environments required.

    Returns protocol, records[id]={case,record,calls,actions,result_sha256,
    receipt_sha256,artifact_sha256}, statuses, complete, evidence_sha256 and
    pairing_audit. Partial explicitly exposes missing/error/unsealed attempts.
    """
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    verify_protocol(protocol, root)
    cases = protocol["spec"]["cases"]
    case_ids = {case["id"] for case in cases}
    for path in (output / "episodes").glob("*"):
        _require(path.is_dir() and path.name in case_ids, "Unplanned episode artifact")
    records, statuses = {}, {}
    for case in cases:
        directory = output / "episodes" / case["id"]
        if not directory.exists():
            statuses[case["id"]] = "missing"
            continue
        if not (directory / "receipt.json").exists():
            result = read_json(directory / "result.json") if (directory / "result.json").exists() else {}
            statuses[case["id"]] = "error" if result.get("status") == "error" else "unsealed"
            continue
        record = verify_complete(directory, case, protocol["fingerprint"])
        records[case["id"]] = _validate_record(directory, case, record, protocol)
        statuses[case["id"]] = "complete"
    complete = len(records) == len(cases)
    if not partial and not complete:
        raise RuntimeError(f"Confirmation incomplete: {len(records)}/{len(cases)}; use partial explicitly")
    blocks = defaultdict(list)
    for row in records.values():
        case = row["case"]
        blocks[(case["suite"], case["task_id"], case["init_state_id"])].append(row)
    paired_blocks = aligned = 0
    for rows in blocks.values():
        if len(rows) < 2:
            continue
        first, second = rows
        paired_blocks += 1
        _require(first["record"]["initial_image_sha256"] == second["record"]["initial_image_sha256"],
                 "Paired initial RGB differs")
        _require(first["artifact_sha256"]["initial_wrist.png"] == second["artifact_sha256"]["initial_wrist.png"],
                 "Paired initial wrist image differs")
        _require(first["record"]["environment"] == second["record"]["environment"]
                 and first["record"]["scheduler_config"] == second["record"]["scheduler_config"],
                 "Paired environment/scheduler differs")
        _require(np.allclose(first["record"]["initial_robot_state"], second["record"]["initial_robot_state"],
                             rtol=0, atol=1e-6), "Paired initial robot state differs")
        for one, two in zip(first["calls"], second["calls"]):
            _require(one["noise_sha256"] == two["noise_sha256"], "Paired decision noise differs")
            aligned += 1
    evidence = {"protocol.json": file_sha(output / "protocol.json")}
    for case_id, row in records.items():
        evidence[f"episodes/{case_id}/receipt.json"] = row["receipt_sha256"]
        evidence.update({f"episodes/{case_id}/{name}": value for name, value in row["artifact_sha256"].items()})
    return {"protocol": protocol, "records": records, "statuses": statuses, "complete": complete,
            "evidence_sha256": evidence, "pairing_audit": {"paired_initial_blocks": paired_blocks,
            "same_ordinal_noise_pairs_checked": aligned, "proprio_atol": 1e-6,
            "scope": "Identical initial front/wrist RGB, proprioception, environment, scheduler and shared decision noise; subsequent observations may diverge."}}


def _execution_runtime(root):
    from embodied_research.runtime import compatibility_audit, configure_runtime
    configure_runtime(root)
    import torch
    from embodied_research.cache_policy import CachePolicyAdapter
    from embodied_research.policies import PolicyAdapter
    from embodied_research.simulator import LiberoSimulator
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return PolicyAdapter, CachePolicyAdapter, LiberoSimulator, compatibility_audit


def run(output, *, limit=None, freeze_only=False, root=ROOT, runtime_factory=None, episode_runner=None):
    if limit is not None:
        _require(type(limit) is int and limit > 0, "limit must be a positive integer")
    root, output = Path(root), Path(output)
    protocol = freeze_protocol(output, root)
    cases = protocol["spec"]["cases"]
    # Reject every prior partial/error before model imports, including cases later in the schedule.
    for case in cases:
        verify_complete(output / "episodes" / case["id"], case, protocol["fingerprint"])
    verify_study(output, partial=True, root=root)
    if freeze_only:
        return {"status": "frozen", "total": len(cases), "fingerprint": protocol["fingerprint"]}
    lock = output / "active.lock"
    _exclusive_json(lock, {"pid": os.getpid(), "created_at": utc_now()})
    simulator = adapter = None
    simulator_key = None
    newly_completed = 0
    warmed = False
    try:
        remaining = [case for case in cases if not (output / "episodes" / case["id"] / "receipt.json").exists()]
        if remaining:
            adapter_class, policy_class, simulator_class, audit = (runtime_factory or _execution_runtime)(root)
        for ordinal, case in enumerate(cases, start=1):
            directory = output / "episodes" / case["id"]
            if directory.exists():
                continue
            if limit is not None and newly_completed >= limit:
                break
            directory.mkdir(parents=True)
            _exclusive_json(directory / "attempt.json", {"case": case, "protocol_fingerprint": protocol["fingerprint"],
                                                        "started_at": utc_now()})
            write_json(output / "progress.json", {"current": case, "ordinal": ordinal, "total": len(cases),
                                                  "updated_at": utc_now()})
            print(f"[{ordinal}/{len(cases)}] {case['id']} (confirmation)", flush=True)
            try:
                if adapter is None:
                    adapter = adapter_class("smolvla", num_steps=10, temporal_ensemble_coeff=None)
                key = case["suite"], case["task_id"]
                if key != simulator_key:
                    if simulator is not None:
                        simulator.close()
                    simulator = simulator_class(*key, project_root=root, image_size=360)
                    simulator_key = key
                if not warmed:
                    warm = simulator.reset(seed=case["seed"], init_state_id=case["init_state_id"])
                    adapter.reset(seed=case["seed"])
                    for _ in range(2):
                        adapter.observe(warm)
                        adapter.predict_chunk()
                    warmed = True
                policy = policy_class(adapter, case["method"])
                result = (episode_runner or run_episode)(simulator, policy, case, directory)
                result["protocol_fingerprint"] = protocol["fingerprint"]
                result["hypothesis_origin_sha256"] = digest_json(protocol["origin"])
                _exclusive_json(directory / "calls.json", policy.call_log)
                _exclusive_json(directory / "result.json", result)
                seal_episode(directory)
                newly_completed += 1
                print(f"  success={result['success']} steps={result['steps']} calls={result['inference_calls']} "
                      f"wall={result['wall_seconds']:.1f}s", flush=True)
                policy.reset(seed=case["seed"])
            except Exception as error:
                if not (directory / "result.json").exists():
                    _exclusive_json(directory / "result.json", {**case, "status": "error", "error": str(error),
                        "traceback": traceback.format_exc(), "protocol_fingerprint": protocol["fingerprint"],
                        "finished_at": utc_now()})
                raise
        study = verify_study(output, partial=True, root=root)
        completed = list(study["records"])
        write_json(output / "progress.json", {"completed_ids": completed, "completed": len(completed),
                   "total": len(cases), "finished": study["complete"], "updated_at": utc_now()})
        if remaining:
            write_json(output / "compatibility.json", audit())
        return {"completed": len(completed), "total": len(cases)}
    finally:
        if simulator is not None:
            simulator.close()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/frontier/flow_confirmation")
    parser.add_argument("--show-design", action="store_true")
    parser.add_argument("--freeze-only", action="store_true", help="Validate and freeze without importing Torch or environments")
    parser.add_argument("--limit", type=int, help="Stop after this many NEW complete episodes; denominator remains16")
    args = parser.parse_args()
    if args.show_design:
        print(json.dumps(study_spec(), indent=2))
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        print(json.dumps(run(args.output, limit=args.limit, freeze_only=args.freeze_only), indent=2))


if __name__ == "__main__":
    main()
