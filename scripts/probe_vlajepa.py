"""Isolated, pinned VLA-JEPA CPU feasibility probe; never a success benchmark.

``fetch`` downloads source, tokenizer metadata and the robot policy, not Qwen
base weights. Dependencies are pure-Python wheels in a separate target overlay.
``audit`` is stdlib/metadata only. ``run`` is deliberately an explicit command.
No existing environment, MINERVA source, or original policy files are modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import struct
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models" / "vlajepa"
SOURCE = ROOT / "frontier" / "vendor" / "lerobot"
OVERLAY = ROOT / ".runtime" / "frontier_deps"
REVISION = "ae2b57f12223c8c5184fcd814eb63cd9ce7de203"
POLICY = "lerobot/VLA-JEPA-LIBERO"
POLICY_REVISION = "735d9f692981e286ade093b5046627eda876e5d0"
QWEN = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
SKIPPED_PREFIXES = ("model.video_encoder.", "model.video_predictor.")
POLICY_SHA256 = "b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c"
POLICY_BYTES = 6_163_212_550
DEPENDENCIES = {
    "diffusers": ("0.39.0", "912aca51b5787365110806e984d5555735bf8a461073bb8459029d0bca7870ef"),
    "importlib_metadata": ("8.7.1", "5a1f80bf1daa489495071efbb095d75a634cf28a8bc299581244063b53176151"),
}
RUNTIME_PACKAGES = ("torch", "torchvision", "transformers", "safetensors", "accelerate", "numpy", "psutil")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FeedbackBench-VLAJEPA-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_remaining_ranges(url, destination, size, sha256, connections):
    """Fetch disjoint suffix ranges, preserving the already downloaded prefix."""
    if not 1 <= connections <= 4:
        raise ValueError("Checkpoint download requires one to four connections")
    destination = Path(destination)
    if destination.exists() or connections == 1:
        return download(url, destination, size, sha256)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > size:
        raise ValueError("Partial checkpoint exceeds pinned size")
    if prefix == size:
        return download(url, destination, size, sha256)
    span = math.ceil((size - prefix) / connections)
    segments = [(start, min(start + span, size)) for start in range(prefix, size, span)]

    def fetch_range(bounds):
        start, end = bounds
        path = destination.with_name(f"{destination.name}.range-{start}-{end}")
        for attempt in range(8):
            # Re-read the persisted length after every transient failure. Never
            # append a repeated response prefix or restart an entire segment.
            done = path.stat().st_size if path.exists() else 0
            if done > end - start:
                raise ValueError("Invalid existing range length")
            if done == end - start:
                return path
            headers = {"Range": f"bytes={start + done}-{end - 1}", "User-Agent": "FeedbackBench-VLAJEPA-audit/1"}
            request = urllib.request.Request(url + f"?download=true&range_start={start + done}", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start + done}-{end - 1}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not return the exact pinned checkpoint byte range")
                    last_log = time.monotonic()
                    with path.open("ab" if done else "wb") as handle:
                        while block := response.read(1024 * 1024):
                            if handle.tell() + len(block) > end - start:
                                raise ValueError("Range response exceeded its declared boundary")
                            handle.write(block)
                            if time.monotonic() - last_log > 30:
                                print(f"Range {start}: {handle.tell() / (end - start):.1%}", flush=True)
                                last_log = time.monotonic()
                        handle.flush()
                        os.fsync(handle.fileno())
                if path.stat().st_size != end - start:
                    raise ConnectionError("Incomplete range response; partial bytes preserved")
                return path
            except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                    raise
                if attempt == 7:
                    raise
                retained = path.stat().st_size if path.exists() else 0
                print(f"Range {start}: {type(error).__name__}; retained {retained} bytes; retry {attempt + 1}/7", flush=True)
                time.sleep(min(2 ** (attempt + 1), 20))

    with ThreadPoolExecutor(max_workers=connections) as pool:
        paths = list(pool.map(fetch_range, segments))
    # Append completed ranges to the existing prefix only after every range succeeds.
    # A crash here leaves a usable larger prefix, and a later attempt only fetches its suffix.
    with partial.open("ab") as output:
        for path in paths:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    result = download(url, destination, size, sha256)
    for path in paths:
        path.unlink()  # Exact newly-created range files only; never recursive.
    return result


def git_blob_digest(path):
    size = Path(path).stat().st_size
    result = hashlib.sha1(f"blob {size}\0".encode())
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, destination, size, sha256=None, blob_sha1=None):
    """Stream one immutable object, resuming only an exact HTTP byte range."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"Existing file has wrong size: {relative(destination)}")
    else:
        partial = destination.with_name(destination.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError("Partial file exceeds declared object size")
        if offset < size:
            headers = {"User-Agent": "FeedbackBench-VLAJEPA-audit/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor resume range; partial file preserved")
                with partial.open("ab" if offset else "wb") as handle:
                    last_log = time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        if handle.tell() + len(block) > size:
                            raise ValueError("Response exceeds pinned file size")
                        handle.write(block)
                        if size > 100_000_000 and time.monotonic() - last_log > 30:
                            print(f"Downloading {destination.name}: {handle.tell() / size:.1%}", flush=True)
                            last_log = time.monotonic()
                    handle.flush()
                    os.fsync(handle.fileno())
        if partial.stat().st_size != size:
            raise ValueError("Incomplete object; partial file preserved for a later retry")
        if sha256 and digest(partial) != sha256:
            raise ValueError("Downloaded object SHA256 mismatch; partial file preserved")
        if blob_sha1 and git_blob_digest(partial) != blob_sha1:
            raise ValueError("Downloaded source Git blob hash mismatch")
        os.replace(partial, destination)
    actual = digest(destination)
    if sha256 and actual != sha256:
        raise ValueError("Existing object SHA256 mismatch")
    if blob_sha1 and git_blob_digest(destination) != blob_sha1:
        raise ValueError("Existing source Git blob hash mismatch")
    return {"path": relative(destination), "bytes": size, "sha256": actual, "git_blob_sha1": blob_sha1}


def overlay_paths():
    return [OVERLAY / f"{name}-{version}" for name, (version, _) in DEPENDENCIES.items()]


def activate_overlay():
    for path in reversed(overlay_paths()):
        if path.is_dir():
            sys.path.insert(0, str(path))


def dependency_audit():
    from packaging.requirements import Requirement

    activate_overlay()
    versions = {}
    for name in (*RUNTIME_PACKAGES, *DEPENDENCIES):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    failures = []
    for name, (version, _) in DEPENDENCIES.items():
        if versions[name] != version:
            failures.append(f"{name} requires pinned {version}; found {versions[name]}")
            continue
        for text in metadata.requires(name) or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
                versions[requirement.name] = installed
                if installed not in requirement.specifier:
                    failures.append(f"Unsatisfied {text}: {installed}")
            except metadata.PackageNotFoundError:
                failures.append(f"Missing dependency {text}")
    return {"versions": versions, "failures": failures}


def fetch_dependencies():
    records = []
    for name, (version, expected_hash) in DEPENDENCIES.items():
        data = read_json_url(f"https://pypi.org/pypi/{name}/{version}/json")
        requires_dist = data["info"]["requires_dist"]
        files = [f for f in data["urls"] if f["filename"].endswith("-py3-none-any.whl")]
        if len(files) != 1 or files[0]["digests"]["sha256"] != expected_hash:
            raise ValueError("Pinned PyPI wheel metadata changed")
        entry = files[0]
        wheel = OVERLAY / "wheels" / entry["filename"]
        record = download(entry["url"], wheel, entry["size"], expected_hash)
        target = OVERLAY / f"{name}-{version}"
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or ".data" in path.parts:
                    raise ValueError("Unsafe or non-pure wheel layout")
                if member.is_dir():
                    continue
                output = target.joinpath(*path.parts)
                data = archive.read(member)
                if output.exists():
                    if digest(output) != hashlib.sha256(data).hexdigest():
                        raise ValueError("Existing overlay differs from pinned wheel")
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("xb") as handle:
                        handle.write(data)
        record.update(package=name, version=version, requires_dist=requires_dist)
        records.append(record)
    result = dependency_audit()
    if result["failures"]:
        raise RuntimeError(json.dumps(result))
    return {"wheels": records, "audit": result}


def fetch_assets(connections=1):
    if (ASSETS / "manifest.json").is_file():
        verify_manifest()
        print("Existing immutable VLA-JEPA manifest verified; no new download needed.", flush=True)
        return
    pending = ASSETS / "manifest.pending.json"
    if pending.is_file():
        verify_manifest(schema_only=True)
        print("Verified cached source and metadata; resuming checkpoint only...", flush=True)
        return finish_checkpoint(json.loads(pending.read_text(encoding="utf-8")), connections)
    manifest = {"created_at": utc_now(), "lerobot_revision": REVISION, "policy_revision": POLICY_REVISION,
                "qwen_metadata_revision": QWEN_REVISION, "scope": "CPU feasibility only; no base Qwen weights", "files": []}
    print("Fetching pinned tokenizer and policy metadata...", flush=True)
    for repo, revision, folder in [(QWEN, QWEN_REVISION, "qwen"), (POLICY, POLICY_REVISION, "policy")]:
        data = read_json_url(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
        if data["sha"] != revision:
            raise ValueError("HF resolved an unexpected revision")
        for entry in data["siblings"]:
            name = entry["rfilename"]
            if name == ".gitattributes" or (repo == QWEN and name.endswith(".safetensors")):
                continue
            if name == "model.safetensors":
                weight_entry = entry
                continue
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
            manifest["files"].append(download(url, ASSETS / folder / name, entry["size"], (entry.get("lfs") or {}).get("sha256")))
    print("Fetching immutable official source (7.6 MB)...", flush=True)
    tree = read_json_url(f"https://api.github.com/repos/huggingface/lerobot/git/trees/{REVISION}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Incomplete Git tree")
    entries = [x for x in tree["tree"] if x["type"] == "blob" and
               (x["path"].startswith("src/lerobot/") or x["path"] in ("LICENSE", "pyproject.toml"))]

    def fetch_source(entry):
        url = f"https://raw.githubusercontent.com/huggingface/lerobot/{REVISION}/{entry['path']}"
        return download(url, SOURCE / entry["path"], entry["size"], blob_sha1=entry["sha"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest["files"].extend(pool.map(fetch_source, entries))
    manifest["dependencies"] = fetch_dependencies()
    json_write(ASSETS / "manifest.pending.json", manifest)
    if weight_entry["lfs"]["sha256"] != POLICY_SHA256 or weight_entry["size"] != POLICY_BYTES:
        raise ValueError("Unexpected checkpoint digest")
    return finish_checkpoint(manifest, connections)


def finish_checkpoint(manifest, connections):
    print("Fetching the single 6.16 GB robot checkpoint; Qwen base weights are excluded...", flush=True)
    url = f"https://huggingface.co/{POLICY}/resolve/{POLICY_REVISION}/model.safetensors"
    manifest["files"].append(download_remaining_ranges(url, ASSETS / "policy" / "model.safetensors",
                                                       POLICY_BYTES, POLICY_SHA256, connections))
    manifest["completed_at"] = utc_now()
    manifest["status"] = "verified"
    json_write(ASSETS / "manifest.json", manifest)
    print("VLA-JEPA source and checkpoint manifest verified.", flush=True)


def read_safetensors_header(path):
    with Path(path).open("rb") as handle:
        size_data = handle.read(8)
        if len(size_data) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", size_data)[0]
        if not 0 < size <= 4_000_000:
            raise ValueError("Unexpected safetensors header size")
        return json.loads(handle.read(size))


def verify_manifest(schema_only=False):
    path = ASSETS / "manifest.json"
    if schema_only and not path.is_file():
        path = ASSETS / "manifest.pending.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"lerobot_revision": REVISION, "policy_revision": POLICY_REVISION,
                "qwen_metadata_revision": QWEN_REVISION}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest revision mismatch")
    if not schema_only and manifest.get("status") != "verified":
        raise ValueError("Checkpoint download has not been verified")
    records = manifest["files"] + manifest["dependencies"]["wheels"]
    for record in records:
        entry = PurePosixPath(record["path"])
        if entry.is_absolute() or ".." in entry.parts:
            raise ValueError("Unsafe manifest path")
        target = ROOT.joinpath(*entry.parts)
        if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
            raise ValueError(f"Provenance verification failed: {record['path']}")
    # The independently extracted overlay must still be byte-identical to each wheel.
    for record in manifest["dependencies"]["wheels"]:
        target = OVERLAY / f"{record['package']}-{record['version']}"
        with zipfile.ZipFile(ROOT / record["path"]) as wheel:
            for member in wheel.infolist():
                if not member.is_dir() and digest(target / member.filename) != hashlib.sha256(wheel.read(member)).hexdigest():
                    raise ValueError("Dependency overlay differs from its verified wheel")
    return {"path": relative(path), "sha256": digest(path), "files_verified": len(records),
            "checkpoint_complete": manifest.get("status") == "verified"}


def validate_schema(header, model_shapes):
    checkpoint = {key: value for key, value in header.items() if key != "__metadata__"}
    skipped = {key: value for key, value in checkpoint.items() if key.startswith(SKIPPED_PREFIXES)}
    active = {key: value for key, value in checkpoint.items() if key not in skipped}
    missing = sorted(set(model_shapes) - set(active))
    unexpected = sorted(set(active) - set(model_shapes))
    mismatched = {key: {"checkpoint": active[key]["shape"], "model": list(model_shapes[key])}
                  for key in active.keys() & model_shapes.keys()
                  if list(model_shapes[key]) != active[key]["shape"]}
    if missing or unexpected or mismatched:
        raise ValueError(json.dumps({"missing": missing, "unexpected": unexpected, "shape_mismatch": mismatched}))
    return {"active": active, "skipped": skipped,
            "active_elements": sum(math.prod(value["shape"]) for value in active.values()),
            "skipped_elements": sum(math.prod(value["shape"]) for value in skipped.values())}


class MemoryGuard:
    """Only this process can exit; no external PID is ever signalled."""

    def __init__(self, output, max_rss_gib, min_available_gib):
        import psutil

        self.psutil = psutil
        self.process = psutil.Process(os.getpid())
        self.identity = (os.getpid(), self.process.create_time())
        self.output = output
        self.max_rss = max_rss_gib * 1024**3
        self.min_available = min_available_gib * 1024**3
        self.peak_rss = 0
        self.minimum_available = None
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self.stop_event.wait(1.0):
            rss = self.process.memory_info().rss
            available = self.psutil.virtual_memory().available
            self.peak_rss = max(self.peak_rss, rss)
            self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
            if len(self.samples) < 7200:
                self.samples.append({"utc": utc_now(), "rss_bytes": rss, "available_bytes": available})
            if rss > self.max_rss or available < self.min_available:
                current = self.psutil.Process(os.getpid())
                if (current.pid, current.create_time()) != self.identity:
                    return
                record = {"status": "stopped_memory_guard", "utc": utc_now(), "pid": current.pid,
                          "process_create_time": current.create_time(), "rss_bytes": rss,
                          "available_bytes": available, "limits": {"max_rss_bytes": self.max_rss,
                          "min_available_bytes": self.min_available}, "action": "exit own process only"}
                stop_path = self.output.with_name(self.output.stem + ".memory_stop.json")
                try:
                    json_write(stop_path, record)
                finally:
                    # os._exit cannot target any other process, even after a PID is reused.
                    os._exit(72)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        return {"peak_rss_bytes": self.peak_rss, "minimum_available_bytes": self.minimum_available,
                "sampling_seconds": 1.0, "samples": self.samples}


def isolate_official_source():
    if any(name == "lerobot" or name.startswith("lerobot.") for name in sys.modules):
        raise RuntimeError("Refusing to mix an already imported LeRobot implementation")
    activate_overlay()
    sys.path.insert(0, str(SOURCE / "src"))
    # Preserve the existing audited platform-only workaround; do not expose MINERVA.
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform

    configure_platform()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def audit_buffers(policy, torch):
    """Prove nonpersistent values come from the official deterministic constructors."""
    results = []
    for module_name, module in policy.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers[name]
            if value is None:
                continue
            key = f"{module_name}.{name}"
            if value.is_meta:
                raise ValueError(f"Nonpersistent buffer remained meta: {key}")
            if key == "model._action_token_ids_t":
                expected = torch.tensor(policy.model.action_token_ids, dtype=torch.long)
                rule = "official expanded tokenizer action-token IDs"
            elif module.__class__.__name__ == "Qwen3VLVisionRotaryEmbedding" and name == "inv_freq":
                expected = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
                rule = "official vision rotary constructor formula"
            elif module.__class__.__name__ == "Qwen3VLTextRotaryEmbedding" and name in ("inv_freq", "original_inv_freq"):
                if module.rope_type != "default":
                    raise ValueError("Unaudited non-default rotary initialization")
                expected, _ = module.compute_default_rope_parameters(module.config, device=torch.device("cpu"))
                rule = "official text rotary constructor formula"
            else:
                raise ValueError(f"Unaudited nonpersistent buffer: {key}")
            if not torch.equal(value, expected):
                raise ValueError(f"Deterministic buffer value differs: {key}")
            results.append({"name": key, "shape": list(value.shape), "dtype": str(value.dtype), "rule": rule})
    return results


def build_meta_policy():
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.initialization import no_init_weights
    from unittest.mock import patch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    config = PreTrainedConfig.from_pretrained(ASSETS / "policy", local_files_only=True,
                                             cli_overrides=["--device=cpu"])
    if config.type != "vla_jepa" or config.num_inference_timesteps != 4 or config.chunk_size != 7:
        raise ValueError("Unexpected pinned policy contract")
    if config.jepa_tubelet_size != 2 or config.num_video_frames != 8:
        raise ValueError("Training-only module omission would change prompt construction")
    config.enable_world_model = False
    config.qwen_model_name = str(ASSETS / "qwen")
    config.dtype = torch.bfloat16
    config.device = "cpu"
    qwen_config = Qwen3VLConfig.from_pretrained(ASSETS / "qwen", local_files_only=True)
    calls = []

    def architecture_only(path, **kwargs):
        if Path(path) != ASSETS / "qwen" or kwargs != {"torch_dtype": torch.bfloat16}:
            raise ValueError("Unexpected Qwen constructor call")
        calls.append("Qwen3VLForConditionalGeneration.from_pretrained -> same class._from_config")
        return Qwen3VLForConditionalGeneration._from_config(qwen_config, dtype=torch.bfloat16)

    with init_empty_weights(include_buffers=False), no_init_weights():
        with patch.object(Qwen3VLForConditionalGeneration, "from_pretrained", side_effect=architecture_only):
            policy = VLAJEPAPolicy(config)
    if len(calls) != 1 or any(not parameter.is_meta for parameter in policy.parameters()):
        raise ValueError("Construction unexpectedly materialized model parameters")
    return policy, {"constructor_changes": calls, "world_model_enabled": False,
                    "reason": "training-only modules never called by official predict_action",
                    "buffers": audit_buffers(policy, torch)}


def tensor_is_finite(tensor, torch):
    flat = tensor.reshape(-1)
    return all(bool(torch.isfinite(flat[start:start + 1_000_000]).all())
               for start in range(0, flat.numel(), 1_000_000))


def assert_equal_tensor_payloads(checkpoint, first, second):
    """Check native alias equality without faulting both large tensors into mmap RSS."""
    if first["shape"] != second["shape"] or first["dtype"] != second["dtype"]:
        raise ValueError("Native tied checkpoint tensors have different metadata")
    a_start, a_end = first["data_offsets"]
    b_start, b_end = second["data_offsets"]
    if a_end - a_start != b_end - b_start:
        raise ValueError("Native tied checkpoint tensors have different byte lengths")
    with Path(checkpoint).open("rb") as first_stream, Path(checkpoint).open("rb") as second_stream:
        base = struct.unpack("<Q", first_stream.read(8))[0] + 8
        first_stream.seek(base + a_start)
        second_stream.seek(base + b_start)
        remaining = a_end - a_start
        while remaining:
            size = min(1024 * 1024, remaining)
            a = first_stream.read(size)
            b = second_stream.read(size)
            if len(a) != size or len(b) != size or a != b:
                raise ValueError("Pinned Qwen requests tied embeddings but checkpoint copies differ")
            remaining -= size


def load_inference_tensors(policy, schema, checkpoint):
    import torch
    from safetensors import safe_open

    # _from_config / init_empty_weights can suppress tying. Follow the pinned
    # Qwen config explicitly, only after proving both stored tensors are equal.
    embedding_key = "model.qwen.model.model.language_model.embed_tokens.weight"
    head_key = "model.qwen.model.lm_head.weight"
    shared = bool(policy.model.qwen.model.config.tie_word_embeddings)
    records = []
    if shared:
        assert_equal_tensor_payloads(checkpoint, schema["active"][embedding_key], schema["active"][head_key])
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        for key, entry in schema["active"].items():
            target_dtype = torch.bfloat16 if key.startswith("model.qwen.") else torch.float32
            if shared and key == head_key:
                records.append({"name": key, "shape": entry["shape"], "dtype": str(target_dtype),
                                "elements": math.prod(entry["shape"]), "alias": embedding_key,
                                "equality": "all stored bytes equal to canonical tensor; original config ties them"})
                continue
            tensor = stream.get_tensor(key)
            if list(tensor.shape) != entry["shape"] or not tensor_is_finite(tensor, torch):
                raise ValueError(f"Invalid checkpoint tensor: {key}")
            tensor = tensor.to(dtype=target_dtype)
            parent_name, leaf = key.rsplit(".", 1)
            module = policy.get_submodule(parent_name)
            if leaf in module._parameters:
                module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
            elif leaf in module._buffers:
                module._buffers[leaf] = tensor
            else:
                raise ValueError(f"State key does not name a parameter or buffer: {key}")
            records.append({"name": key, "shape": entry["shape"], "dtype": str(target_dtype),
                            "checkpoint_dtype": entry["dtype"], "elements": tensor.numel()})
        if shared:
            policy.model.qwen.model.lm_head.weight = policy.model.qwen.model.model.language_model.embed_tokens.weight
    if any(tensor.is_meta for tensor in policy.state_dict().values()):
        raise ValueError("Loading left uninitialized meta tensors")
    policy.requires_grad_(False).eval()
    return {"tensors": records, "native_tied_word_embeddings": shared,
            "unique_parameters": sum(p.numel() for p in policy.parameters()),
            "unique_parameter_bytes": sum(p.numel() * p.element_size() for p in policy.parameters()),
            "all_loaded_tensors_finite": True, "buffers": audit_buffers(policy, torch)}


def run_probe(args):
    import faulthandler
    import psutil

    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a previous probe; choose a new --output")
    record = {"status": "preflight", "started_at": utc_now(), "script_sha256": digest(__file__),
              "scope": "one real LIBERO observation, batch 1, four flow steps; not closed-loop success",
              "lerobot_revision": REVISION, "policy_revision": POLICY_REVISION,
              "platform": sys.platform, "python": platform.python_version(), "threads": args.threads,
              "theoretical_untied_inference_weight_bytes": 5_498_123_292, "timings": {}}
    available = psutil.virtual_memory().available
    minimum = (1.5 if args.schema_only else args.min_free_gib) * 1024**3
    record["preflight"] = {"available_bytes": available, "required_available_bytes": minimum}
    if available < minimum:
        record.update(status="blocked_preflight", reason="Insufficient available RAM; no model constructed")
        json_write(args.output, record)
        return 3
    json_write(args.output, record)
    source_snapshot = args.output.with_name(args.output.stem + ".source.py")
    with source_snapshot.open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    record["script_snapshot"] = relative(source_snapshot)
    faulthandler.dump_traceback_later(120, repeat=True)
    guard = MemoryGuard(args.output, args.max_rss_gib, args.min_runtime_free_gib)
    guard.start()
    try:
        record["manifest"] = verify_manifest(args.schema_only)
        manifest_snapshot = args.output.with_name(args.output.stem + ".inputs.json")
        with manifest_snapshot.open("xb") as handle:
            handle.write((ROOT / record["manifest"]["path"]).read_bytes())
        record["manifest"]["snapshot_path"] = relative(manifest_snapshot)
        record["dependencies"] = dependency_audit()
        if record["dependencies"]["failures"]:
            raise RuntimeError("Dependency audit failed")
        isolate_official_source()
        record["status"] = "importing_runtime"
        json_write(args.output, record)
        started = time.perf_counter()
        import torch
        import lerobot
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        if not Path(lerobot.__file__).resolve().is_relative_to(SOURCE):
            raise RuntimeError("LeRobot import came from the wrong source tree")
        record["timings"]["imports_seconds"] = time.perf_counter() - started
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(123)
        record.update(status="building_meta_schema", implementation=relative(Path(sys.modules[VLAJEPAPolicy.__module__].__file__)))
        json_write(args.output, record)
        started = time.perf_counter()
        policy, construction = build_meta_policy()
        record["timings"]["meta_construction_seconds"] = time.perf_counter() - started
        record["construction"] = construction
        record["attention_implementation"] = {
            "text": policy.model.qwen.model.config.text_config._attn_implementation,
            "vision": policy.model.qwen.model.config.vision_config._attn_implementation,
        }
        checkpoint = ASSETS / "policy" / "model.safetensors"
        if args.schema_only and not checkpoint.is_file():
            checkpoint = checkpoint.with_name(checkpoint.name + ".part")
        schema = validate_schema(read_safetensors_header(checkpoint),
                                 {key: tuple(value.shape) for key, value in policy.state_dict().items()})
        schema_path = args.output.with_name(args.output.stem + ".schema.json")
        json_write(schema_path, schema)
        record["schema"] = {"path": relative(schema_path), "sha256": digest(schema_path),
                            "active_elements": schema["active_elements"], "skipped_elements": schema["skipped_elements"]}
        if args.schema_only:
            record.update(status="passed_schema_only", inference_executed=False,
                          note="Meta architecture only. This is not a pretrained forward result.")
            return 0
        if psutil.virtual_memory().available < args.min_free_gib * 1024**3:
            record.update(status="blocked_preflight", reason="RAM dropped below threshold after imports; no weights loaded")
            return 3
        record["status"] = "loading_checkpoint"
        json_write(args.output, record)
        started = time.perf_counter()
        loading = load_inference_tensors(policy, schema, checkpoint)
        record["timings"]["load_tensors_seconds"] = time.perf_counter() - started
        loading_path = args.output.with_name(args.output.stem + ".loading.json")
        json_write(loading_path, loading)
        record["loading"] = {key: value for key, value in loading.items() if key != "tensors"}
        record["loading"].update(path=relative(loading_path), sha256=digest(loading_path))
        record["rss_after_load_bytes"] = psutil.Process().memory_info().rss
        record["status"] = "preprocessing_real_observation"
        json_write(args.output, record)
        import numpy as np
        from contextlib import ExitStack
        from functools import wraps
        from unittest.mock import patch
        from lerobot.processor import load_pretrained_policy_processors

        with np.load(args.observation, allow_pickle=False) as saved:
            observation = {key: saved[key].copy() for key in saved.files}
        observation["task"] = str(observation["task"].item())
        for key in policy.config.input_features:
            value = observation[key]
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite real observation")
            if "images" in key and (value.ndim != 3 or value.shape[0] != 3 or value.min() < 0 or value.max() > 1):
                raise ValueError("Expected pre-oriented CHW RGB [0,1] from the real simulator snapshot")
            observation[key] = torch.from_numpy(value)
        record["observation"] = {"path": relative(args.observation), "sha256": digest(args.observation),
                                 "task": observation["task"], "camera_orientation": "snapshot already flipped by official LIBERO preparation; no additional flip"}
        started = time.perf_counter()
        preprocess, postprocess = load_pretrained_policy_processors(
            ASSETS / "policy", preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        batch = preprocess(observation)
        record["timings"]["preprocess_seconds"] = time.perf_counter() - started
        record["status"] = "running_single_real_forward"
        json_write(args.output, record)
        sections = {}

        def timed(name, function):
            @wraps(function)
            def wrapper(*inputs, **kwargs):
                start = time.perf_counter()
                try:
                    return function(*inputs, **kwargs)
                finally:
                    entry = sections.setdefault(name, {"calls": 0, "seconds": 0.0})
                    entry["calls"] += 1
                    entry["seconds"] += time.perf_counter() - start
            return wrapper

        started = time.perf_counter()
        with ExitStack() as stack:
            for obj, attribute, name in [(policy.model.qwen, "build_inputs", "qwen_preprocessing"),
                                         (policy.model, "_qwen_last_decoder_hidden", "qwen_forward"),
                                         (policy.model.action_model, "predict_action", "flow_action_head")]:
                stack.enter_context(patch.object(obj, attribute, timed(name, getattr(obj, attribute))))
            with torch.inference_mode():
                actions = policy.predict_action_chunk(batch)
        record["timings"]["forward_seconds"] = time.perf_counter() - started
        record["timings"]["sections"] = sections
        expected_shape = (1, 7, 7)
        if tuple(actions.shape) != expected_shape or not torch.isfinite(actions).all():
            raise ValueError("Invalid action chunk from real pretrained inference")
        started = time.perf_counter()
        environment_actions = postprocess(actions)
        record["timings"]["postprocess_seconds"] = time.perf_counter() - started
        if tuple(environment_actions.shape) != expected_shape or not torch.isfinite(environment_actions).all():
            raise ValueError("Invalid postprocessed actions")
        artifact = args.output.with_name(args.output.stem + ".actions.npz")
        with artifact.open("xb") as handle:
            np.savez(handle, model_actions=actions.detach().float().numpy(),
                     environment_actions=environment_actions.detach().float().numpy())
        record.update(status="passed_real_forward", inference_executed=True, forward_calls=1,
                      flow_steps=4, batch_size=1, actions={"path": relative(artifact), "sha256": digest(artifact),
                      "shape": list(actions.shape), "first_environment_action": environment_actions[0, 0].tolist()})
        return 0
    except Exception as error:
        def portable_error(text):
            for host_path, token in [(str(ROOT), "<project>"), (sys.prefix, "<environment>"),
                                     (sys.base_prefix, "<python>"), (str(Path.home()), "<user>")]:
                text = text.replace(host_path, token).replace(host_path.replace("\\", "/"), token)
            return text

        record.update(status="failed", error_type=type(error).__name__,
                      error=portable_error(str(error)), traceback=portable_error(traceback.format_exc()))
        print(record["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        record["memory"] = guard.finish()
        record["finished_at"] = utc_now()
        json_write(args.output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "audit", "run"))
    parser.add_argument("--observation", type=Path, default=ROOT / "artifacts/policy_smoke_observation.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/frontier/vlajepa_probe.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=6.0)
    parser.add_argument("--max-rss-gib", type=float, default=7.5)
    parser.add_argument("--min-runtime-free-gib", type=float, default=0.5)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--connections", type=int, choices=(1, 2, 4), default=1,
                        help="Fetch the one checkpoint using disjoint HTTP ranges; never downloads Qwen weights")
    args = parser.parse_args()
    if args.threads < 1 or args.min_free_gib < 6 or args.max_rss_gib <= 0 or args.min_runtime_free_gib <= 0:
        parser.error("threads must be positive; materialization requires at least 6 GiB available")
    if args.command == "fetch":
        fetch_assets(args.connections)
    elif args.command == "audit":
        print(json.dumps(dependency_audit(), indent=2))
    else:
        return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
