# VLA-JEPA LIBERO-10 single-task resource diagnostic

Planned **1** episode; completed **0**; official successes **0**; completed task failures **0**.
Completion and task success are separate. Missing, unsealed and error statuses remain in report.json.

This independent one-case protocol uses a 5.75 GiB preflight gate, 6 GiB RSS cap and 0.5 GiB runtime available-memory floor.
Earlier successes and interrupted attempts are separate evidence, not a combined four-of-four result.

| Suite | Official success | Steps / horizon | Policy calls | Action-head NFE | Inference seconds | Episode wall seconds |
|---|---:|---:|---:|---:|---:|---:|

Simulation pauses during inference. The 20 Hz simulated control frequency is not model throughput. Episode wall time excludes model loading and GIF encoding.
Shared-machine timings and one-second RSS samples do not establish stable throughput or instantaneous peak memory.

## Run and memory evidence

- [Run record](runs/run_0001.json): `running_episodes`; sampled peak RSS `None` bytes; minimum available system memory `None` bytes.
- [Run record](runs/run_0001.memory_stop.json): `stopped_memory_guard`; sampled peak RSS `None` bytes; minimum available system memory `None` bytes.

## Limitations

- changed numerics, no original BF16 comparison or quantization quality attribution
- different checkpoint, action interval and source from SmolVLA; not a matched comparison
- simulation pauses during inference; no real robot or real-time control claim
- all successes, task failures, incomplete and runtime errors retained separately
- Separate paging experiment; does not complete the frozen owned-embedding stage
- One-observation exact action equality does not prove all-input equivalence
- Independent successor to the interrupted paged-only attempt; all earlier failures retained
- One selected previously used task and initial state; exploratory resource feasibility, not generalization
- Independent one-case protocol and budget; cannot combine previous successes into a four-of-four experiment
- Every earlier interrupted attempt remains retained separately; no automatic retry of unsealed evidence
- Frozen engine stdout may display [1/4]; this protocol and all saved reports/progress plan exactly one case
- Frozen LIBERO reset may swallow repeated native-warning exceptions; a hung reset requires external timeout
