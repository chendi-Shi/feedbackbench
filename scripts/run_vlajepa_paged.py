"""Separate paged-embedding VLA-JEPA diagnostic; never changes the frozen owned variant."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
VARIANT = "cpu_qwen_dynamic_int8_paged_bf16_embedding_fp32_action"
MIN_FREE_GIB = 3.5
DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_paged_closed_loop"
DEFAULT_PROBE = ROOT / "artifacts/frontier/vlajepa_paged_probe_20261016.json"
REFERENCE = ROOT / "artifacts/frontier/vlajepa_int8_probe_20261015.json"
OBSERVATION_SHA = "114121b20b8103cfcdcf4906ac1938e6fc13ab1da9f45801bef586557687a0aa"
ACTION_SHA = "d6556d8a6fdf1c26245a25e8f7404bb121743f171f5d7b0eddbbf0f981595e15"
EXTRA_SOURCES = ("scripts/run_vlajepa_paged.py", "scripts/vlajepa_paged_embedding.py",
                 "docs/VLAJEPA_PAGED_PROTOCOL.md")


def original_runner():
    spec = importlib.util.spec_from_file_location("isolated_paged_base", ROOT / "scripts/run_vlajepa_closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes(runner):
    return {name: runner.file_sha(ROOT / name) for name in (*runner.SOURCES, *EXTRA_SOURCES)}


def separate_output(output):
    output = Path(output).resolve()
    old = (ROOT / "outputs/frontier/vlajepa_closed_loop").resolve()
    if not output.is_relative_to(ROOT) or output.is_relative_to(old) or old.is_relative_to(output):
        raise ValueError("Paged evidence must use a separate directory inside this project")
    return output


def validate_paging(audit):
    # The helper owns buffer validation; this consumer additionally binds the
    # actual model table and prevents an owned loading audit passing as paged.
    expected = {"runtime_mapping_count": 1, "embedding_read_attempts": 1,
                "expected_runtime_mapping_count_when_loaded": 1,
                "payload_bytes": 622329856, "elements": 311164928, "data_start": 216056,
                "dtype": "BF16", "shape": [151936, 2048],
                "checkpoint_data_offsets": [932694542, 1555024398],
                "tensor_file_range": [932910598, 1555240454], "access": "mmap.ACCESS_COPY",
                "embedding_materialized": True, "all_embedding_values_finite": True,
                "validation_mappings_closed_before_runtime_mapping": True,
                "complete_vocabulary_preserved": True, "dtype_conversion": False,
                "storage_lifetime": "torch.frombuffer storage retains mmap owner"}
    for name, value in expected.items():
        actual = audit.get(name)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("Paged embedding audit differs: " + name)
    granularity = audit.get("allocation_granularity_bytes")
    if type(granularity) is not int or granularity <= 0:
        raise ValueError("Invalid mapping granularity")
    start, end = expected["tensor_file_range"]
    aligned = start // granularity * granularity
    if (audit.get("mapping_file_range") != [aligned, end]
            or audit.get("mapping_length_bytes") != end - aligned
            or audit.get("tensor_offset_in_mapping") != start - aligned):
        raise ValueError("Mapped checkpoint range differs")
    chunk = audit.get("validation_chunk_elements")
    if type(chunk) is not int or not 0 < chunk <= 1000000:
        raise ValueError("Invalid bounded validation chunk")
    if audit.get("validation_mapping_count") != (expected["elements"] + chunk - 1) // chunk:
        raise ValueError("Complete embedding finite validation missing")
    if (type(audit.get("fallback_tensor_reads")) is not int or audit["fallback_tensor_reads"] <= 0
            or audit.get("measured_resident_saving_bytes") is not None
            or audit.get("theoretical_owned_allocation_avoided_bytes") != expected["payload_bytes"]):
        raise ValueError("Paged allocation/fallback evidence differs")


def run_probe(output):
    """Replay the original one-observation calculation, with only storage changed."""
    from vlajepa_paged_embedding import paged_embedding_loader
    runner = original_runner()
    quant, base = runner.quant, runner.base
    output = separate_output(output)
    runner.require(output.is_relative_to(ROOT) and output != REFERENCE, "Use a new project probe path")
    plan_path = output.with_suffix(".plan.json")
    runner.require(not output.exists() and not plan_path.exists(), "Previous probe evidence is immutable")
    reference = runner.read_json(REFERENCE)
    observation = runner.project_file(reference["observation"]["path"])
    reference_actions = runner.project_file(reference["actions"]["path"])
    runner.require(reference["status"] == "passed_quantized_forward", "Reference forward did not pass")
    runner.require(runner.file_sha(observation) == OBSERVATION_SHA, "Reference observation changed")
    runner.require(runner.file_sha(reference_actions) == ACTION_SHA, "Reference actions changed")
    plan = {"schema": "feedbackbench.vlajepa_paged_probe_plan.v1", "frozen_at": base.utc_now(),
            "variant": VARIANT, "sources": source_hashes(runner), "seed": 123, "threads": 2,
            "reference": {"path": base.relative(REFERENCE), "sha256": runner.file_sha(REFERENCE)},
            "observation_sha256": OBSERVATION_SHA, "reference_actions_sha256": ACTION_SHA,
            "comparison": "np.array_equal on both model_actions and environment_actions; no tolerance",
            "minimum_available_gib": MIN_FREE_GIB, "maximum_rss_gib": 5.5,
            "minimum_runtime_available_gib": 0.5,
            "scope": "one known observation; storage equivalence and feasibility, not task success"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    checkpoint = base.ASSETS / "policy/model.safetensors"
    embedding = base.read_safetensors_header(checkpoint)[quant.EMBEDDING]
    write = base.json_write
    audit = {}
    execution = {"action_head_nfe": 0, "tied_parameter_identity": False}
    load = quant.load_quantized_tensors

    def instrumented_load(policy, schema, checkpoint):
        loading = load(policy, schema, checkpoint)
        weight = policy.get_parameter(quant.EMBEDDING)
        execution.update(tied_parameter_identity=policy.model.qwen.model.lm_head.weight is weight,
                         embedding_shape=list(weight.shape), embedding_dtype=str(weight.dtype))

        def count_step(module, inputs):
            execution["action_head_nfe"] += 1

        handle = policy.model.action_model.model.register_forward_pre_hook(count_step)
        stack.callback(handle.remove)
        return loading

    def annotated_write(path, value):
        if Path(path) == output:
            value.update(variant=VARIANT, paged_plan={"path": base.relative(plan_path),
                         "sha256": runner.file_sha(plan_path)}, paging=dict(audit), paged_execution=dict(execution))
            value["guard"]["reason"] = "Separate exploratory paged-embedding budget; not a proven memory bound"
            value["storage_change"] = "Complete BF16 embedding uses private file mapping; FP32 lookup output and all values preserved"
            if value.get("status") == "passed_quantized_forward":
                value["status"] = "forward_complete_comparison_pending"
        return write(path, value)

    with ExitStack() as stack:
        audit = stack.enter_context(paged_embedding_loader(quant, checkpoint, embedding))
        stack.enter_context(patch.object(quant, "MIN_FREE_GIB", MIN_FREE_GIB))
        stack.enter_context(patch.object(quant, "MAX_RSS_GIB", 5.5))
        stack.enter_context(patch.object(quant, "load_quantized_tensors", instrumented_load))
        stack.enter_context(patch.object(base, "json_write", annotated_write))
        code = quant.run(SimpleNamespace(output=output, observation=observation, threads=2))
    if code != 0:
        return code
    result = runner.read_json(output)
    try:
        import numpy as np
        validate_paging(result["paging"])
        runner.require(result["paged_execution"]["action_head_nfe"] == 4 and result["paged_execution"]["tied_parameter_identity"] is True,
                       "Actual action-head execution or embedding alias differs")
        runner.require(result["loading"]["quantized_engine"] == reference["loading"]["quantized_engine"], "Quantized engine differs")
        runner.require(source_hashes(runner) == plan["sources"], "Probe sources changed during execution")
        comparison = {}
        current_actions = runner.project_file(result["actions"]["path"])
        with np.load(reference_actions, allow_pickle=False) as before, np.load(current_actions, allow_pickle=False) as after:
            runner.require(set(before.files) == set(after.files) == {"model_actions", "environment_actions"}, "Action keys differ")
            for name in before.files:
                a, b = before[name], after[name]
                same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
                comparison[name] = {"exact_equal": bool(same), "different_elements": int(np.count_nonzero(a != b)),
                                    "maximum_absolute_difference": float(np.max(np.abs(a-b)))}
        result["comparison"] = comparison
        runner.require(all(item["exact_equal"] for item in comparison.values()), "Paged forward differs from the reference")
        result["status"] = "passed_paged_forward_exact_reference"
        result["scope"] = "One known observation exactly matches owned INT8 actions; no environment execution or BF16 parity"
        write(output, result)
        return 0
    except Exception as error:
        result.update(status="failed_reference_comparison", comparison_error=runner.portable_error(error))
        write(output, result)
        raise


def verify_probe(runner, path=DEFAULT_PROBE):
    result = runner.read_json(path)
    runner.require(result.get("status") == "passed_paged_forward_exact_reference", "A successful paired paged forward is required")
    validate_paging(result["paging"])
    plan_path = runner.project_file(result["paged_plan"]["path"])
    runner.require(runner.file_sha(plan_path) == result["paged_plan"]["sha256"], "Probe plan changed")
    plan = runner.read_json(plan_path)
    for key, expected in {"variant": VARIANT, "seed": 123, "threads": 2, "minimum_available_gib": MIN_FREE_GIB,
                          "maximum_rss_gib": 5.5, "minimum_runtime_available_gib": 0.5,
                          "observation_sha256": OBSERVATION_SHA, "reference_actions_sha256": ACTION_SHA}.items():
        runner.require(type(plan.get(key)) is type(expected) and plan[key] == expected, "Probe plan differs: " + key)
    runner.require(plan["sources"] == source_hashes(runner), "Paired probe was run with different sources")
    runner.require(result["variant"] == VARIANT and plan["minimum_available_gib"] == MIN_FREE_GIB, "Probe variant/budget differs")
    runner.require(result.get("inference_executed") is True and result.get("forward_calls") == 1
                   and result.get("threads") == 2 and result.get("flow_steps") == 4, "Probe execution evidence differs")
    execution = result.get("paged_execution", {})
    runner.require(execution.get("action_head_nfe") == 4 and execution.get("tied_parameter_identity") is True
                   and execution.get("embedding_shape") == [151936, 2048]
                   and execution.get("embedding_dtype") == "torch.bfloat16", "Probe model execution differs")
    coverage = result["quantized_forward_audit"]
    runner.require(coverage["distinct_modules_executed"] == 300 and len(coverage["module_calls"]) == 300
                   and all(type(count) is int and count > 0 for count in coverage["module_calls"].values()), "Incomplete actual Qwen coverage")
    runner.require(result["guard"]["min_free_gib_before_import_and_loading"] == MIN_FREE_GIB
                   and result["guard"]["max_rss_gib"] == 5.5 and result["guard"]["min_runtime_free_gib"] == 0.5,
                   "Probe guard differs")
    runner.require(result["observation"]["sha256"] == OBSERVATION_SHA
                   and runner.file_sha(runner.project_file(result["observation"]["path"])) == OBSERVATION_SHA, "Probe observation changed")
    runner.require(all(result["comparison"][name]["exact_equal"] is True for name in ("model_actions", "environment_actions")),
                   "Reference equality not established")
    # Recompute the comparison, rather than trusting summary booleans.
    import numpy as np
    reference = runner.read_json(runner.project_file(plan["reference"]["path"]))
    runner.require(result["loading"]["quantized_engine"] == reference["loading"]["quantized_engine"], "Quantized engine differs")
    runner.require(runner.file_sha(runner.project_file(plan["reference"]["path"])) == plan["reference"]["sha256"], "Reference record changed")
    with np.load(runner.project_file(reference["actions"]["path"]), allow_pickle=False) as before, np.load(runner.project_file(result["actions"]["path"]), allow_pickle=False) as after:
        runner.require(runner.file_sha(runner.project_file(reference["actions"]["path"])) == ACTION_SHA, "Reference arrays changed")
        runner.require(runner.file_sha(runner.project_file(result["actions"]["path"])) == result["actions"]["sha256"], "Paged arrays changed")
        for name in ("model_actions", "environment_actions"):
            runner.require(before[name].dtype == after[name].dtype and np.array_equal(before[name], after[name]), "Recomputed reference comparison failed")
    return {"path": runner.base.relative(path), "sha256": runner.file_sha(path), "plan_sha256": runner.file_sha(plan_path)}


def configured_runner(probe=DEFAULT_PROBE):
    from vlajepa_paged_embedding import paged_embedding_loader
    runner = original_runner()
    proof = verify_probe(runner, probe)
    original_cases, original_spec = runner.cases_for_study, runner.study_spec
    original_load, original_verify = runner.quant.load_quantized_tensors, runner.verify_episode
    runner.VARIANT, runner.MIN_FREE_GIB = VARIANT, MIN_FREE_GIB
    runner.SOURCES = (*runner.SOURCES, *EXTRA_SOURCES)

    def cases():
        rows = original_cases()
        for row in rows:
            row["model"] = "vlajepa_paged"
            row["split"] = "paged_closed_loop_feasibility"
            row["id"] = runner.episode_id(row)
        return rows

    def spec():
        value = original_spec()
        value["name"] = "VLA-JEPA paged BF16 embedding / dynamic INT8 four-task feasibility"
        value["paired_probe"] = proof
        value["embedding_storage"] = {"dtype": "BF16", "shape": [151936, 2048], "bytes": 622329856,
                                      "mapping": "copy_on_write", "lookup_output": "FP32", "lm_head_tied": True}
        value["limitations"].extend(["Separate paging experiment; does not complete the frozen owned-embedding stage",
                                     "3.5 GiB gate is an exploratory budget, not an established hardware minimum",
                                     "One-observation exact action equality does not prove all-input equivalence"])
        return value

    def load(policy, schema, checkpoint):
        with paged_embedding_loader(runner.quant, checkpoint, schema["active"][runner.quant.EMBEDDING]) as audit:
            result = original_load(policy, schema, checkpoint)
        validate_paging(audit)
        weight = policy.get_parameter(runner.quant.EMBEDDING)
        runner.require(list(weight.shape) == [151936, 2048] and str(weight.dtype) == "torch.bfloat16", "Mapped model table differs")
        runner.require(policy.model.qwen.model.lm_head.weight is weight, "Mapped embedding is not tied")
        result["paging"] = {**audit, "embedding_shape": list(weight.shape), "embedding_dtype": str(weight.dtype),
                            "tied_parameter_identity": True, "checkpoint_sha256": runner.base.POLICY_SHA256}
        return result

    def verify(directory, case, protocol):
        result = original_verify(directory, case, protocol)
        if result is not None:
            loading = runner.read_json(runner.project_file(result["model_info"]["loading"]["path"]))
            audit = loading.get("paging", {})
            validate_paging(audit)
            runner.require(audit.get("embedding_shape") == [151936, 2048] and audit.get("embedding_dtype") == "torch.bfloat16"
                           and audit.get("tied_parameter_identity") is True and audit.get("checkpoint_sha256") == runner.base.POLICY_SHA256,
                           "Paged model loading evidence differs")
        return result

    runner.cases_for_study, runner.study_spec = cases, spec
    runner.quant.load_quantized_tensors, runner.verify_episode = load, verify
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "freeze", "run", "report", "validate"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "probe":
        return run_probe(args.output or DEFAULT_PROBE)
    output = separate_output(args.output or DEFAULT_OUTPUT)
    runner = configured_runner(args.probe)
    if args.command == "validate":
        result = runner.verify_study(output, partial=args.partial)
        result.pop("results")
    elif args.command == "report":
        result = runner.write_report(output)
    else:
        result = runner.run(output, freeze_only=args.command == "freeze", limit=args.limit)
        if "completed" in result:
            result = runner.write_report(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 3 if result.get("status") == "blocked_preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
