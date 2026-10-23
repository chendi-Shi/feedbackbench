"""Fetch the four fixed official expert sources with bounded, resumable ranges.

--download starts missing ranges. --resume explicitly permits continuing existing
partial ranges. No request retries, no automatic full-file fallback or deletion. Default invocation
prints the pinned source plan without network access. Audit files and the exclusive
lock are private runtime outputs shared with the original acquisition helper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_research.action_demo_data import sha256_file, write_json  # noqa: E402

ORIGINAL_PRIVATE_SCRIPT_SHA256 = "6301819a43c0a116b0fc3aa8fffbee3350407803492cc6712f20257ee65fc38c"

REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
REPO = "yifengzhu-hf/LIBERO-datasets"
NETWORK_CAP = 4 * 1024 ** 3
DISK_CAP = 8 * 1024 ** 3
WIDTH = 3 * 1024 ** 2
SOURCES = [
    {"suite": "libero_spatial", "task_id": 0,
     "task_name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
     "bytes": 508779600, "sha256": "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead",
     "path": "data/action_adaptation/probe_v1/task0.hdf5", "existing_probe_source": True},
    {"suite": "libero_object", "task_id": 0,
     "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
     "bytes": 780145352, "sha256": "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d",
     "path": "data/action_adaptation/study_v1/libero_object.hdf5"},
    {"suite": "libero_goal", "task_id": 0, "task_name": "open_the_middle_drawer_of_the_cabinet",
     "bytes": 702223367, "sha256": "20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737",
     "path": "data/action_adaptation/study_v1/libero_goal.hdf5"},
    {"suite": "libero_10", "task_id": 0,
     "task_name": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
     "bytes": 1467540242, "sha256": "3173ebc709b064683956b757ae8737909457d730aa305b23c5735292e59066af",
     "path": "data/action_adaptation/study_v1/libero_10.hdf5"},
]


def source_manifest():
    return {
        "schema": "feedbackbench.action_adaptation_sources.v1", "revision": REVISION, "repo_id": REPO,
        "selection": "task_order_index=0, task_id=0 in all four suites, chosen before phase2 results",
        "split_intent": {"development": [0], "train": list(range(1, 41)), "validation": list(range(41, 50))},
        "split_guard": "Inspect actual episodes; missing demo_0..49 is an error, never replace tasks",
        "sources": [{**s, "task_uid": f"{s['suite']}:0", "repo": REPO, "revision": REVISION,
                     "language": (s["task_name"].split("SCENE2_", 1)[1] if s["suite"] == "libero_10"
                                  else s["task_name"]).replace("_", " "),
                     "upstream_path": f"{s['suite']}/{s['task_name']}_demo.hdf5"} for s in SOURCES],
        "network_payload_limit_bytes": NETWORK_CAP, "disk_limit_bytes": DISK_CAP,
        "original_unique_file_bytes": sum(s["bytes"] for s in SOURCES),
        "probe_budget_unchanged": True, "new_phase2_budget_is_separate_authorization": True,
    }


def inspect_metadata(source):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{source['suite']}?limit=100"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Oversized metadata")
    expected_path = f"{source['suite']}/{source['task_name']}_demo.hdf5"
    matches = [r for r in json.loads(raw) if r.get("path") == expected_path]
    if len(matches) != 1:
        raise ValueError("Missing/ambiguous official fixed task")
    entry = matches[0]
    if (entry.get("size") != source["bytes"] or entry.get("lfs", {}).get("oid") != source["sha256"]
            or entry.get("lfs", {}).get("size") != source["bytes"]):
        raise ValueError("Official metadata changed from reviewed fixed source")
    return {"entry": entry, "metadata_bytes": len(raw), "metadata_sha256": hashlib.sha256(raw).hexdigest()}


def download_source(source, state, persist, *, resume, workers):
    target = ROOT / source["path"]
    if target.exists():
        if target.stat().st_size != source["bytes"] or sha256_file(target) != source["sha256"]:
            raise ValueError("Existing source differs; refusing overwrite")
        state.update({"status": "verified_existing", "sha256": source["sha256"]})
        persist()
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    directory = target.parent / (source["suite"] + "_ranges")
    directory.mkdir(exist_ok=True)
    ranges = []
    for index, start in enumerate(range(0, source["bytes"], WIDTH)):
        end = min(source["bytes"] - 1, start + WIDTH - 1)
        complete = directory / f"{index:04d}.bin"
        partial = directory / f"{index:04d}.part"
        if complete.exists() and partial.exists():
            raise RuntimeError("Ambiguous range pieces")
        existing = complete if complete.exists() else partial
        retained = existing.stat().st_size if existing.exists() else 0
        if not 0 <= retained <= end - start + 1:
            raise RuntimeError("Piece exceeds its fixed range")
        if retained and not resume:
            raise RuntimeError("Existing pieces need explicit --resume")
        if complete.exists() and retained != end - start + 1:
            raise RuntimeError("Completed range has wrong size")
        record = {"index": index, "start": start, "end": end, "retained_bytes": retained,
                  "retained_sha256": sha256_file(existing) if retained else None,
                  "received_this_attempt": 0, "status": "complete" if complete.exists() else "pending"}
        ranges.append(record)
    state.update({"status": "downloading", "ranges": ranges, "workers": workers})
    persist()
    guard = threading.RLock()
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{source['suite']}/{source['task_name']}_demo.hdf5"

    def fetch(record):
        if record["status"] == "complete":
            return
        partial = directory / f"{record['index']:04d}.part"
        complete = directory / f"{record['index']:04d}.bin"
        first = record["start"] + record["retained_bytes"]
        end = record["end"]
        received = 0
        try:
            if first <= end:
                request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{end}",
                                                               "Accept-Encoding": "identity"})
                with guard:
                    record["stage"] = "connecting"
                    persist()
                with urllib.request.urlopen(request, timeout=180) as response:
                    with guard:
                        record.update({"http_status": response.status,
                                       "content_range": response.headers.get("Content-Range"),
                                       "content_length": response.headers.get("Content-Length"),
                                       "response_host": urllib.parse.urlsplit(response.geturl()).hostname})
                        persist()
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {first}-{end}/{source['bytes']}"
                            or response.headers.get("Content-Length") != str(end - first + 1)
                            or response.headers.get("Content-Encoding") not in (None, "identity")):
                        raise RuntimeError("Server did not honor exact range; body refused")
                    record["stage"] = "reading_body"
                    with partial.open("ab") as stream:
                        if stream.tell() != record["retained_bytes"]:
                            raise RuntimeError("Prefix changed before append")
                        last = 0.0
                        while received < end - first + 1:
                            block = response.read1(min(65536, end - first + 1 - received))
                            if not block:
                                raise RuntimeError("Truncated range")
                            stream.write(block)
                            received += len(block)
                            with guard:
                                record["received_this_attempt"] = received
                                if time.monotonic() - last > 5:
                                    persist()
                                    last = time.monotonic()
            if partial.stat().st_size != end - record["start"] + 1:
                raise RuntimeError("Range final size incorrect")
            digest = sha256_file(partial)
            partial.replace(complete)
            with guard:
                record.update({"status": "complete", "sha256": digest})
                persist()
        except BaseException as error:
            with guard:
                record.update({"status": "needs_attention", "error": f"{type(error).__name__}: {error}",
                               "received_this_attempt": received,
                               "preserved_bytes": partial.stat().st_size if partial.exists() else 0})
                persist()
            raise

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, r) for r in ranges]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise RuntimeError(f"{len(errors)} ranges need attention; no request retried: {errors[0]}")
    state["status"] = "assembling"
    persist()
    assembly = target.with_suffix(".assembled.part")
    digest = hashlib.sha256()
    total = 0
    with assembly.open("xb") as output:
        for record in ranges:
            piece = directory / f"{record['index']:04d}.bin"
            with piece.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    total += len(block)
                    digest.update(block)
                    output.write(block)
    if total != source["bytes"] or digest.hexdigest() != source["sha256"]:
        raise RuntimeError("Assembled official source SHA256/size mismatch")
    assembly.replace(target)
    state.update({"status": "verified", "sha256": digest.hexdigest(), "bytes": total})
    persist()


def main():
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    directory = ROOT / "data/action_adaptation/study_v1"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "source_manifest.json"
    manifest = source_manifest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen phase2 source manifest changed")
    else:
        write_json(manifest_path, manifest)
    if not args.download:
        print(json.dumps(manifest, indent=2))
        return
    audits = ROOT / ".runtime/action_phase2_download"
    audits.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    audit_path = audits / (stamp + ".json")
    if audit_path.exists():
        raise FileExistsError("Unique audit collision")
    attempt_count = len(list(audits.glob("*.json"))) + 1
    projected_payload = sum(s["bytes"] for s in SOURCES) + 6 * 1024 ** 2 + attempt_count * 2 * 1024 ** 2
    projected_disk = 2 * sum(s["bytes"] for s in SOURCES) + 128 * 1024 ** 2
    if projected_payload > NETWORK_CAP or projected_disk > DISK_CAP:
        raise RuntimeError("Phase2 resource cap would be exceeded")
    if psutil.virtual_memory().available < 1024 ** 3:
        raise RuntimeError("Less than 1 GiB available RAM; do not compete with other projects")
    actual = sum(p.stat().st_size for p in (ROOT / "data/action_adaptation").rglob("*") if p.is_file())
    if actual > DISK_CAP or shutil.disk_usage(directory).free < max(0, projected_disk - actual) + 512 * 1024 ** 2:
        raise RuntimeError("Insufficient bounded disk headroom")
    process = psutil.Process()
    owner = {"pid": process.pid, "create_time": process.create_time(), "script_sha256": sha256_file(Path(__file__))}
    lock = ROOT / ".runtime/action_phase2_download.lock"
    with lock.open("x") as stream:
        json.dump(owner, stream)
    state = {"status": "running", "owner": owner, "started_unix": time.time(),
             "source_manifest_sha256": sha256_file(manifest_path), "sources": [],
             "original_private_script_sha256": ORIGINAL_PRIVATE_SCRIPT_SHA256,
             "maximum_budgeted_payload": projected_payload, "maximum_budgeted_disk": projected_disk,
             "network_payload_cap": NETWORK_CAP, "disk_cap": DISK_CAP, "automatic_retry": False}

    def persist():
        state["updated_unix"] = time.time()
        write_json(audit_path, state)

    try:
        persist()
        for source in SOURCES:
            record = {"suite": source["suite"], "path": source["path"], "expected_bytes": source["bytes"],
                      "expected_sha256": source["sha256"], "status": "metadata_check"}
            state["sources"].append(record)
            persist()
            record["official_metadata"] = inspect_metadata(source)
            persist()
            download_source(source, record, persist, resume=args.resume, workers=args.workers)
        state.update({"status": "complete", "finished_unix": time.time()})
        persist()
        print(json.dumps({"status": "complete", "audit": str(audit_path), "manifest": str(manifest_path)}))
    except BaseException as error:
        state.update({"status": "needs_attention", "error": f"{type(error).__name__}: {error}"})
        persist()
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text()) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
