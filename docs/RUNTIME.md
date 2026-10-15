# Tested runtime and reproducibility

The project uses its own Python 3.12 environment. The primary workspace's
environment and other projects' source code are unchanged. The user-authorized
temporary pause of another model service is documented below. `requirements-lock.txt`
records the primary environment's installed distributions, while source, model and asset revisions are
verified separately. The Windows CPU environment is the tested configuration;
the lock is not a claim that every platform has been validated.

## Windows compatibility

`runtime.py` checks the exact robosuite 1.4.0 source hash before changing three
platform statements in memory: the MuJoCo DLL location, selection of GLFW instead
of unsupported EGL, and the Unix-only log destination. Installed simulator source
remains intact. No dynamics, controller, observation, action limits, task assets,
initial states or success predicate is patched. Each run saves the exact changes.

The environment lives under `%LOCALAPPDATA%/FeedbackBench/`, with a project
`.venv` directory junction. Launch with `python scripts/run.py ...` or `Run.ps1`:
they preserve the short lexical path needed by LIBERO's long task filenames.
This avoids changing the Windows registry and avoids environment writes through
OneDrive. Project-local bytecode writes are disabled by the launcher.

The bootstrap also runs `repair_top_level_metadata.py` for six audited packages
whose missing `top_level.txt` declarations cause thousands of filesystem checks
during Transformers imports. The script uses this interpreter's own inference
rule, verifies each name against existing package files, and creates only absent
metadata declarations. Package code and original wheel RECORDs remain unchanged;
`artifacts/runtime_metadata_audit.json` records all contents and source hashes.
Run the script without `--apply` for a preview that leaves environment metadata
unchanged and still saves an audit record in the project.

## Dependency metadata exceptions

`pip check` reports four expected declaration mismatches:

| Requesting distribution | Declared requirement | Tested replacement / reason |
|---|---|---|
| hf-libero 0.1.4 | hf-egl-probe | EGL probe is Linux-specific; Windows uses GLFW |
| robomimic 0.2.0 | egl-probe | EGL probe is Linux-specific; Windows uses GLFW |
| hf-libero 0.1.4 | opencv-python | opencv-python-headless supplies the same `cv2` module |
| robosuite 1.4.0 | opencv-python | opencv-python-headless supplies the same `cv2` module |

Installing both OpenCV distributions would overlap the same module. These
exceptions are explicit, not presented as a clean dependency check. Actual
official-state reset, rendering, physics and preprocessing are exercised by
`scripts/smoke_simulator.py`. The successful local smoke and its exact audit are
under `outputs/simulator-smoke/`.

## Model loading

SmolVLA uses the complete frozen policy checkpoint, including its VLM. Only the
pinned backbone configuration and tokenizer assets are needed separately. The
official Transformers `no_init_weights` context skips random values that the
strict checkpoint load would immediately overwrite. An independent loading
audit verifies all 788 state tensors and four nonpersistent rotary buffers.
CPU float32 evaluation uses no quantization or weight changes.

Cold import/load time, warmed inference latency, simulation time and episode
wall time are different measurements. The report keeps them separate. Runs on
this shared laptop do not constitute a controlled hardware throughput benchmark.

During coverage and part of the main comparison on 2026-10-14, an unrelated
Ollama service repeatedly reloaded its model and consumed CPU and memory. The
user authorized a temporary pause; the service was stopped, then restarted by
another task and paused again. Main-comparison available memory fell to about
1 GB before recovering to about 3.8 GB. Episode wall times therefore span
different background load. Do not interpret that variation as a change in
model or scheduling efficiency; inference-call counts and the recorded
simulation outcomes remain separately auditable.

The user later authorized pausing the other project's evaluation and automatic
service recovery. That project retained its results and stopped its model
service during the main comparison. This resource coordination did not change
the frozen robot policies, cases, thresholds, or control-step time limits.

## Isolated frontier probe

The ActionCache transfer study uses the existing SmolVLA runtime and frozen
weights. Its two Torch worker threads and one inter-op thread are recorded in
the study protocol. The earlier single-observation CPU profile used four inter-op
threads; its timings are not substituted for closed-loop study measurements.

The separate VLA-JEPA probe fetches a pinned LeRobot source tree under
`frontier/vendor/lerobot/` and exactly two additional pure-Python distributions
under `.runtime/frontier_deps/`: `diffusers==0.39.0` and
`importlib_metadata==8.7.1`. The probe adds these paths only to its own process;
it does not install them into or replace the primary environment. Wheel hashes,
source revisions and the full checkpoint identity are recorded in its input
manifest. This separate dependency overlay is not represented by
`requirements-lock.txt`; `scripts/probe_vlajepa.py fetch` reproduces it.

The probe requires at least 6 GiB available RAM before materialization and has
an explicit process-memory guard. Its meta-only construction, actual pretrained
forward pass and any blocked/failed attempt are distinct recorded states.
Only a `passed_real_forward` record with bound actions establishes real
inference; it still does not establish a completed LIBERO task. See
[the model boundary notes](FRONTIER_LANDSCAPE.md) and the final probe artifacts.
