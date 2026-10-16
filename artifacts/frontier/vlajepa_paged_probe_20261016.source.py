"""Separate CPU dynamic-INT8 VLA-JEPA feasibility variant, never a parity benchmark.

Qwen Linear weights are INT8 and activations are dynamically quantized inside
PyTorch's CPU kernel; its external inputs/outputs are FP32. The shared token
embedding stays BF16 in storage and casts lookup outputs to FP32. Remaining
Qwen tensors and the complete four-step action head are FP32. This is a changed
numeric implementation, not the official BF16 model or a task-success result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import faulthandler
from functools import wraps
import importlib.util
import math
from pathlib import Path
import platform
import struct
import sys
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/probe_vlajepa.py"
BASE_SPEC = importlib.util.spec_from_file_location("feedbackbench_vlajepa_base", BASE_PATH)
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)
EMBEDDING = "model.qwen.model.model.language_model.embed_tokens.weight"
HEAD = "model.qwen.model.lm_head.weight"
EXPECTED_LINEAR_COUNT = 300
EXPECTED_LINEAR_ELEMENTS = 1_811_939_328
EXPECTED_PAYLOAD_BYTES = 3_072_709_660
MIN_FREE_GIB = 4.25
MAX_RSS_GIB = 5.0
MIN_RUNTIME_FREE_GIB = 0.5


def cpu_fp32_context(device_type, dtype=None):
    """Explicitly disable the official BF16 autocast for this FP32-input variant."""
    if device_type != "cpu":
        raise ValueError("The dynamic-INT8 variant is CPU-only")
    return nullcontext()


def checkpoint_layout(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated checkpoint header")
        header_bytes = struct.unpack("<Q", raw)[0]
        if not 0 < header_bytes <= 4_000_000:
            raise ValueError("Unexpected checkpoint header size")
    return 8 + header_bytes


def read_tensor_bytes(path, entry, data_start):
    """One owned bytearray per tensor, never a long-lived full-file mmap."""
    widths = {"BF16": 2, "F32": 4}
    if entry["dtype"] not in widths:
        raise ValueError("Unsupported checkpoint tensor dtype")
    start, end = entry["data_offsets"]
    expected = math.prod(entry["shape"]) * widths[entry["dtype"]]
    if start < 0 or end - start != expected or data_start + end > Path(path).stat().st_size:
        raise ValueError("Invalid checkpoint tensor bounds")
    payload = bytearray(expected)
    with Path(path).open("rb", buffering=0) as handle:
        handle.seek(data_start + start)
        view = memoryview(payload)
        offset = 0
        while offset < expected:
            read = handle.readinto(view[offset:min(offset + 4 * 1024 * 1024, expected)])
            if not read:
                raise ValueError("Truncated checkpoint tensor")
            offset += read
    return payload


def read_tensor(path, entry, data_start, dtype, torch):
    payload = read_tensor_bytes(path, entry, data_start)
    native_dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[entry["dtype"]]
    tensor = torch.frombuffer(payload, dtype=native_dtype).reshape(entry["shape"])
    if not base.tensor_is_finite(tensor, torch):
        raise ValueError("Nonfinite checkpoint tensor")
    return tensor.to(dtype=dtype)


def assign_tensor(policy, key, tensor, torch):
    parent, leaf = key.rsplit(".", 1)
    module = policy.get_submodule(parent)
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor
    else:
        raise ValueError(f"State key has no declared parameter/buffer: {key}")


def cast_embedding_output_to_float(module, inputs, output):
    return output.float()


def quantize_one_linear(module, weight, bias, torch):
    """Construct only this FP32 layer; never materialize the complete FP32 model."""
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization import default_dynamic_qconfig

    if type(module) is not torch.nn.Linear:
        raise ValueError("Only exact native nn.Linear modules may be replaced")
    if weight.dtype != torch.float32 or (bias is not None and bias.dtype != torch.float32):
        raise ValueError("Dynamic quantization requires FP32 source weight and bias")
    if list(weight.shape) != [module.out_features, module.in_features]:
        raise ValueError("Linear shape differs from declared architecture")
    if (module.bias is None) != (bias is None):
        raise ValueError("Linear bias differs from declared architecture")
    if bias is not None and list(bias.shape) != [module.out_features]:
        raise ValueError("Linear bias shape differs from declared architecture")
    layer = torch.nn.Linear(module.in_features, module.out_features, bias=bias is not None,
                            device="meta", dtype=torch.float32)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    layer.eval()
    layer.qconfig = default_dynamic_qconfig
    quantized = DynamicLinear.from_float(layer)
    if quantized._packed_params.dtype != torch.qint8:
        raise ValueError("Dynamic Linear did not produce INT8 packed weights")
    return quantized


def load_quantized_tensors(policy, schema, checkpoint):
    import torch
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

    active = schema["active"]
    if not policy.model.qwen.model.config.tie_word_embeddings:
        raise ValueError("Pinned shared token embeddings unexpectedly disabled")
    base.assert_equal_tensor_payloads(checkpoint, active[EMBEDDING], active[HEAD])
    modules = {name: module for name, module in policy.named_modules()
               if name.startswith("model.qwen.") and type(module) is torch.nn.Linear
               and name != HEAD.removesuffix(".weight")}
    elements = sum(math.prod(active[name + ".weight"]["shape"]) for name in modules)
    if len(modules) != EXPECTED_LINEAR_COUNT or elements != EXPECTED_LINEAR_ELEMENTS:
        raise ValueError("Quantized Linear inventory differs from pinned architecture")
    linear_keys = {name + ".weight" for name in modules}
    linear_keys |= {name + ".bias" for name, module in modules.items() if module.bias is not None}
    data_start = checkpoint_layout(checkpoint)
    consumed = set()
    records = []
    # The token embedding remains owned BF16 bytes: no 1.24-GB FP32 copy is made.
    for key, entry in active.items():
        if key in linear_keys or key == HEAD:
            continue
        dtype = torch.bfloat16 if key == EMBEDDING else torch.float32
        tensor = read_tensor(checkpoint, entry, data_start, dtype, torch)
        assign_tensor(policy, key, tensor, torch)
        consumed.add(key)
        records.append({"name": key, "shape": entry["shape"], "dtype": str(dtype),
                        "elements": tensor.numel(), "checkpoint_dtype": entry["dtype"]})
        del tensor
    for index, (name, module) in enumerate(modules.items(), 1):
        key = name + ".weight"
        weight = read_tensor(checkpoint, active[key], data_start, torch.float32, torch)
        bias_key = name + ".bias"
        bias = read_tensor(checkpoint, active[bias_key], data_start, torch.float32, torch) if module.bias is not None else None
        quantized = quantize_one_linear(module, weight, bias, torch)
        parent, leaf = name.rsplit(".", 1)
        policy.get_submodule(parent)._modules[leaf] = quantized
        qweight = quantized.weight()
        records.append({"name": key, "shape": active[key]["shape"], "dtype": str(qweight.dtype),
                        "elements": qweight.numel(), "checkpoint_dtype": active[key]["dtype"],
                        "quantization": "PyTorch default_dynamic_qconfig", "qscheme": str(qweight.qscheme()),
                        "scale": qweight.q_scale(), "zero_point": qweight.q_zero_point()})
        consumed.add(key)
        if bias is not None:
            consumed.add(bias_key)
            records.append({"name": bias_key, "shape": active[bias_key]["shape"],
                            "dtype": "torch.float32", "elements": bias.numel(),
                            "checkpoint_dtype": active[bias_key]["dtype"]})
        del weight, bias, qweight, quantized
        if index % 25 == 0:
            print(f"Quantized Qwen Linear {index}/{len(modules)}", flush=True)
    policy.model.qwen.model.lm_head.weight = policy.get_parameter(EMBEDDING)
    consumed.add(HEAD)
    records.append({"name": HEAD, "shape": active[HEAD]["shape"], "dtype": "torch.bfloat16",
                    "alias": EMBEDDING, "equality": "all stored bytes equal; original config ties them"})
    if consumed != set(active):
        raise ValueError("Loader did not consume exactly the required inference state")
    if any(value.is_meta for value in list(policy.parameters()) + list(policy.buffers())):
        raise ValueError("Quantized model retains an uninitialized meta tensor")
    embedding = policy.get_submodule(EMBEDDING.removesuffix(".weight"))
    embedding.register_forward_hook(cast_embedding_output_to_float)
    if policy.model.qwen.model.dtype != torch.float32 or policy.model.qwen.model.model.visual.dtype != torch.float32:
        raise ValueError("Qwen/vision dtype discovery must resolve to FP32, not the BF16 token table")
    if embedding.weight.dtype != torch.bfloat16 or policy.model.qwen.model.lm_head.weight is not embedding.weight:
        raise ValueError("BF16 token embedding storage is not natively shared")
    if any(not isinstance(policy.get_submodule(name), DynamicLinear) for name in modules):
        raise ValueError("Qwen Linear replacement inventory mismatch")
    if any(p.dtype != torch.float32 for p in policy.model.action_model.parameters()):
        raise ValueError("Action head must remain complete FP32")
    policy.requires_grad_(False).eval()
    payload_bytes = EXPECTED_LINEAR_ELEMENTS + sum(p.numel() * p.element_size() for p in policy.parameters())
    # Dynamic Linear biases are packed, so they are absent from parameters().
    bias_elements = sum(math.prod(active[key]["shape"]) for key in linear_keys if key.endswith(".bias"))
    payload_bytes += 4 * bias_elements
    if payload_bytes != EXPECTED_PAYLOAD_BYTES:
        raise ValueError("Quantized parameter payload differs from pinned inventory")
    return {"tensors": records, "all_required_inference_keys_consumed": True,
            "all_source_tensors_finite": True, "native_tied_word_embeddings": True,
            "linear_modules": len(modules), "linear_elements": elements,
            "linear_bias_elements": bias_elements, "theoretical_parameter_payload_bytes": payload_bytes,
            "payload_excludes": ["packed-kernel metadata/alignment", "runtime/activations", "temporary conversion buffers"],
            "largest_linear_fp32_weight_bytes": max(math.prod(active[name + ".weight"]["shape"]) * 4 for name in modules),
            "quantized_engine": torch.backends.quantized.engine,
            "supported_engines": torch.backends.quantized.supported_engines,
            "buffers": base.audit_buffers(policy, torch)}


def snapshot(path, destination):
    with Path(destination).open("xb") as handle:
        handle.write(Path(path).read_bytes())
    return {"path": base.relative(destination), "sha256": base.digest(destination)}


def invoke_recorded_forward(output, record, function):
    """Persist attempted/completed inference independently from later validation."""
    record.update(forward_started=True, forward_started_at=base.utc_now())
    base.json_write(output, record)
    started = time.perf_counter()
    try:
        result = function()
    except Exception:
        record["timings"]["forward_attempt_seconds"] = time.perf_counter() - started
        raise
    record["timings"]["forward_seconds"] = time.perf_counter() - started
    record.update(inference_executed=True, forward_calls=1, forward_returned_at=base.utc_now())
    base.json_write(output, record)
    return result


def run(args):
    import psutil

    output = args.output
    stem = output.with_suffix("")
    companions = [output, *(Path(str(stem) + suffix) for suffix in
                  (".source.py", ".base_source.py", ".inputs.json", ".schema.json", ".loading.json", ".actions.npz", ".memory_stop.json"))]
    if any(path.exists() for path in companions):
        raise FileExistsError("Refusing to overwrite any previous probe evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    own = snapshot(__file__, Path(str(stem) + ".source.py"))
    parent = snapshot(BASE_PATH, Path(str(stem) + ".base_source.py"))
    record = {"status": "preflight", "variant": "cpu_qwen_dynamic_int8_bf16_embedding_fp32_action",
              "started_at": base.utc_now(), "script_sha256": own["sha256"], "script_snapshot": own["path"],
              "base_script": {"path": "scripts/probe_vlajepa.py", "sha256": parent["sha256"],
                              "snapshot_path": parent["path"]}, "inference_executed": False,
              "forward_started": False, "forward_calls": 0,
              "scope": "one real LIBERO observation; changed numeric implementation, not parity or task success",
              "lerobot_revision": base.REVISION, "policy_revision": base.POLICY_REVISION,
              "platform": sys.platform, "python": platform.python_version(), "threads": args.threads,
              "timings": {}, "theoretical_parameter_payload_bytes": EXPECTED_PAYLOAD_BYTES,
              "numeric_changes": ["300 Qwen Linear INT8 weights; CPU kernel dynamically quantizes activations; external FP32",
                                  "shared BF16 token embedding storage, lookup outputs cast FP32",
                                  "other Qwen tensors FP32; official Qwen CPU BF16 autocast disabled",
                                  "complete action head FP32, original four flow steps", "training-only world-model omitted as in base probe"],
              "guard": {"min_free_gib_before_import_and_loading": MIN_FREE_GIB,
                        "max_rss_gib": MAX_RSS_GIB, "min_runtime_free_gib": MIN_RUNTIME_FREE_GIB,
                        "reason": "separate 2.862-GiB theoretical quantized payload; original BF16 probe guards unchanged"}}
    available = psutil.virtual_memory().available
    record["preflight"] = {"available_bytes": available, "required_available_bytes": MIN_FREE_GIB * 1024**3}
    base.json_write(output, record)
    if available < MIN_FREE_GIB * 1024**3:
        record.update(status="blocked_preflight", reason="Insufficient available RAM for the separately specified quantized variant")
        base.json_write(output, record)
        return 3
    guard = base.MemoryGuard(output, MAX_RSS_GIB, MIN_RUNTIME_FREE_GIB)
    guard.start()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        record["manifest"] = base.verify_manifest()
        inputs = snapshot(ROOT / record["manifest"]["path"], Path(str(stem) + ".inputs.json"))
        record["manifest"]["snapshot_path"] = inputs["path"]
        record["dependencies"] = base.dependency_audit()
        if record["dependencies"]["failures"]:
            raise ValueError("Pinned dependency audit failed")
        base.isolate_official_source()
        record["status"] = "importing_runtime"
        base.json_write(output, record)
        started = time.perf_counter()
        import torch
        import lerobot
        import lerobot.policies.vla_jepa.modeling_vla_jepa as official

        if not Path(lerobot.__file__).resolve().is_relative_to(base.SOURCE):
            raise ValueError("Unexpected LeRobot source")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(123)
        record["timings"]["imports_seconds"] = time.perf_counter() - started
        record["status"] = "building_meta_schema"
        base.json_write(output, record)
        started = time.perf_counter()
        policy, record["construction"] = base.build_meta_policy()
        record["timings"]["meta_construction_seconds"] = time.perf_counter() - started
        checkpoint = base.ASSETS / "policy/model.safetensors"
        schema = base.validate_schema(base.read_safetensors_header(checkpoint),
                                      {key: tuple(value.shape) for key, value in policy.state_dict().items()})
        schema_path = Path(str(stem) + ".schema.json")
        base.json_write(schema_path, schema)
        record["schema"] = {"path": base.relative(schema_path), "sha256": base.digest(schema_path),
                            "active_elements": schema["active_elements"], "skipped_elements": schema["skipped_elements"]}
        available = psutil.virtual_memory().available
        record["pre_materialization"] = {"available_bytes": available, "required_available_bytes": MIN_FREE_GIB * 1024**3}
        if available < MIN_FREE_GIB * 1024**3:
            record.update(status="blocked_preflight", reason="RAM below quantized threshold after imports; no weights loaded")
            return 3
        record["status"] = "streaming_quantized_checkpoint"
        base.json_write(output, record)
        started = time.perf_counter()
        loading = load_quantized_tensors(policy, schema, checkpoint)
        record["timings"]["load_and_quantize_seconds"] = time.perf_counter() - started
        loading_path = Path(str(stem) + ".loading.json")
        base.json_write(loading_path, loading)
        record["loading"] = {key: value for key, value in loading.items() if key != "tensors"}
        record["loading"].update(path=base.relative(loading_path), sha256=base.digest(loading_path))
        record["rss_after_load_bytes"] = psutil.Process().memory_info().rss
        record["attention_implementation"] = {"text": policy.model.qwen.model.config.text_config._attn_implementation,
                                               "vision": policy.model.qwen.model.config.vision_config._attn_implementation}
        record["status"] = "preprocessing_real_observation"
        base.json_write(output, record)
        import numpy as np
        from lerobot.processor import load_pretrained_policy_processors

        with np.load(args.observation, allow_pickle=False) as saved:
            observation = {key: saved[key].copy() for key in saved.files}
        observation["task"] = str(observation["task"].item())
        for key in policy.config.input_features:
            value = observation[key]
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite observation")
            if "images" in key and (value.ndim != 3 or value.shape[0] != 3 or value.min() < 0 or value.max() > 1):
                raise ValueError("Expected pre-oriented CHW RGB [0,1]")
            observation[key] = torch.from_numpy(value)
        record["observation"] = {"path": base.relative(args.observation), "sha256": base.digest(args.observation),
                                 "task": observation["task"], "camera_orientation": "already prepared by official LIBERO adapter; no additional flip"}
        started = time.perf_counter()
        preprocess, postprocess = load_pretrained_policy_processors(
            base.ASSETS / "policy", preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        batch = preprocess(observation)
        record["timings"]["preprocess_seconds"] = time.perf_counter() - started
        record["status"] = "running_single_quantized_forward"
        base.json_write(output, record)
        sections = {}
        quantized_calls = {}

        def check_linear_input(name):
            def hook(module, inputs):
                if len(inputs) != 1 or inputs[0].dtype != torch.float32 or inputs[0].device.type != "cpu":
                    raise ValueError(f"Quantized Linear received a non-FP32 CPU activation: {name}")
                quantized_calls[name] = quantized_calls.get(name, 0) + 1
            return hook

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

        with ExitStack() as stack:
            stack.enter_context(patch.object(official, "get_autocast_context", cpu_fp32_context))
            from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

            for name, module in policy.named_modules():
                if isinstance(module, DynamicLinear):
                    hook = module.register_forward_pre_hook(check_linear_input(name))
                    stack.callback(hook.remove)
            for obj, attr, name in [(policy.model.qwen, "build_inputs", "qwen_preprocessing"),
                                    (policy.model, "_qwen_last_decoder_hidden", "qwen_forward"),
                                    (policy.model.action_model, "predict_action", "flow_action_head")]:
                stack.enter_context(patch.object(obj, attr, timed(name, getattr(obj, attr))))
            with torch.inference_mode():
                actions = invoke_recorded_forward(output, record, lambda: policy.predict_action_chunk(batch))
        record["timings"]["sections"] = sections
        record["quantized_forward_audit"] = {"module_calls": quantized_calls,
                                             "all_observed_inputs": "CPU float32",
                                             "distinct_modules_executed": len(quantized_calls)}
        if len(quantized_calls) != EXPECTED_LINEAR_COUNT:
            raise ValueError("Real forward did not execute the complete expected Qwen Linear inventory")
        if tuple(actions.shape) != (1, 7, 7) or actions.dtype != torch.float32 or not torch.isfinite(actions).all():
            raise ValueError("Invalid quantized action chunk")
        environment_actions = postprocess(actions)
        if tuple(environment_actions.shape) != (1, 7, 7) or not torch.isfinite(environment_actions).all():
            raise ValueError("Invalid postprocessed quantized actions")
        artifact = Path(str(stem) + ".actions.npz")
        with artifact.open("xb") as handle:
            np.savez(handle, model_actions=actions.detach().float().numpy(),
                     environment_actions=environment_actions.detach().float().numpy())
        record.update(status="passed_quantized_forward", inference_executed=True, forward_calls=1, flow_steps=4,
                      batch_size=1, actions={"path": base.relative(artifact), "sha256": base.digest(artifact),
                      "shape": list(actions.shape), "first_environment_action": environment_actions[0, 0].tolist()})
        return 0
    except Exception as error:
        def portable(text):
            for host_path, token in [(str(ROOT), "<project>"), (sys.prefix, "<environment>"),
                                     (sys.base_prefix, "<python>"), (str(Path.home()), "<user>")]:
                text = text.replace(host_path, token).replace(host_path.replace("\\", "/"), token)
            return text

        record.update(status="failed", error_type=type(error).__name__, error=portable(str(error)),
                      traceback=portable(traceback.format_exc()))
        print(record["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        record["memory"] = guard.finish()
        record["finished_at"] = base.utc_now()
        base.json_write(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, default=ROOT / "artifacts/policy_smoke_observation.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/frontier/vlajepa_int8_probe_20261015.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
