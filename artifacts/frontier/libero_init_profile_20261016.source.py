"""One environment-only memory diagnostic; no policy load or policy actions."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/frontier/libero_init_profile_20261016.json"
ARRAYS = OUTPUT.with_suffix(".npz")
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("environment_profile_base", ROOT / "scripts/probe_vlajepa.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def main():
    import psutil

    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("Prior environment diagnostic is immutable")
    process = psutil.Process()
    record = {"schema": "feedbackbench.libero_init_profile.v1", "started_at": base.utc_now(),
              "scope": "One environment initialization/reset; no policy loaded and no policy actions executed",
              "configuration": {"suite": "libero_spatial", "task_id": 1, "init_state_id": 41,
                                "seed": 1000, "image_size": 360, "hard_reset": True,
                                "official_internal_settle_steps": 10, "explicit_reset_calls": 1,
                                "explicit_policy_steps": 0},
              "guard": {"maximum_rss_gib": 5.5, "minimum_runtime_available_gib": 0.5,
                        "sampling_seconds": 1, "exit_scope": "own process only"},
              "source": {"path": base.relative(__file__), "sha256": base.digest(__file__)},
              "snapshots": [], "render_context_events": [], "status": "starting"}
    begun = time.perf_counter()
    guard = base.MemoryGuard(OUTPUT, 5.5, 0.5)
    guard.start()
    simulator = None

    def persist():
        base.json_write(OUTPUT, record)

    def sample(stage, **details):
        value = {"stage": stage, "utc": base.utc_now(), "elapsed_seconds": time.perf_counter() - begun,
                 "rss_bytes": process.memory_info().rss, "available_bytes": psutil.virtual_memory().available,
                 **details}
        record["snapshots"].append(value)
        record["status"] = stage
        persist()
        print(json.dumps(value), flush=True)
        return value

    def scalar(obj, name):
        value = getattr(obj, name, None)
        return None if value is None else int(value)

    try:
        sample("before_runtime_imports")
        from embodied_research.runtime import configure_runtime, configure_libero_paths
        configure_runtime(ROOT)
        base.isolate_official_source()
        configure_libero_paths(ROOT)
        from embodied_research.simulator import LiberoSimulator
        import lerobot.envs.libero as official_env
        import robosuite.utils.binding_utils as bindings
        import numpy as np
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        record["runtime_modules"] = {
            "environment": {"path": base.relative(official_env.__file__), "sha256": base.digest(official_env.__file__)},
            "simulator": {"path": "src/embodied_research/simulator.py", "sha256": base.digest(ROOT / "src/embodied_research/simulator.py")},
            "frozen_runtime": {"path": "src/embodied_research/runtime.py", "sha256": base.digest(ROOT / "src/embodied_research/runtime.py")}}
        sample("after_runtime_imports")
        original = bindings.MjRenderContext._set_mujoco_context_and_buffers

        def measured_context(context):
            index = 1 + sum(item["phase"] == "before" for item in record["render_context_events"])
            model = context.model._model
            config = {"model_offwidth": scalar(model.vis.global_, "offwidth"),
                      "model_offheight": scalar(model.vis.global_, "offheight"),
                      "model_offsamples": scalar(model.vis.quality, "offsamples"),
                      "model_shadowsize": scalar(model.vis.quality, "shadowsize"),
                      "model_ntex": scalar(model, "ntex"), "model_ntexdata": scalar(model, "ntexdata"),
                      "model_nbuffer": scalar(model, "nbuffer"), "data_narena": scalar(context.data._data, "narena")}
            before = sample("before_mjr_context", context_index=index, **config)
            record["render_context_events"].append({**before, "phase": "before"})
            persist()
            original(context)
            after = sample("after_mjr_context", context_index=index,
                           offWidth=scalar(context.con, "offWidth"), offHeight=scalar(context.con, "offHeight"),
                           offSamples=scalar(context.con, "offSamples"), shadowSize=scalar(context.con, "shadowSize"))
            record["render_context_events"].append({**after, "phase": "after"})
            persist()

        with patch.object(bindings.MjRenderContext, "_set_mujoco_context_and_buffers", measured_context):
            sample("before_environment_construction")
            simulator = LiberoSimulator("libero_spatial", 1, project_root=ROOT, image_size=360)
            if type(simulator.env) is not official_env.LiberoEnv:
                raise ValueError("Unexpected environment implementation")
            sample("after_environment_construction")
            observation = simulator.reset(seed=1000, init_state_id=41)
            sample("after_official_reset")
            frame = simulator.frame().copy()
            wrist = simulator.frame("image2").copy()
            payload = {**{key: value for key, value in observation.items() if key != "task"},
                       "task": np.asarray(observation["task"]), "main_frame": frame, "wrist_frame": wrist}
            with ARRAYS.open("xb") as stream:
                np.savez_compressed(stream, **payload)
            def array_hash(value):
                return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            record["reset_evidence"] = {"arrays": {"path": base.relative(ARRAYS), "sha256": base.digest(ARRAYS)},
                                        "task": observation["task"], "simulator_steps": simulator.steps,
                                        "metadata": simulator.metadata(),
                                        "array_sha256": {key: array_hash(value) for key, value in payload.items()},
                                        "robot_state": observation["observation.state"].tolist()}
            sample("after_evidence_saved")
            gc.collect()
            sample("after_gc_with_environment_alive")
            simulator.close()
            simulator = None
            sample("after_environment_closed")
            gc.collect()
            sample("after_close_gc")
        record["status"] = "complete_environment_only_diagnostic"
        return 0
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        return 1
    finally:
        if simulator is not None:
            simulator.close()
        record["memory"] = guard.finish()
        record["finished_at"] = base.utc_now()
        record["elapsed_seconds"] = time.perf_counter() - begun
        persist()
        print(json.dumps({"status": record["status"], "output": base.relative(OUTPUT)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
