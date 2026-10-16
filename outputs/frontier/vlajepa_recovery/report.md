# Interrupted VLA-JEPA trajectory recovery

Independent new benchmark episodes: **0**.
Planned recovery: **1**; completed: **0**; official task successes: **0**; full-horizon task failures: **0**.
The original interrupted attempt remains unsealed. This recovery does not form a new four-of-four experiment.

## Run and loading evidence

- [Run](runs/run_0001.json): `blocked_preflight`; loading Nones; memory `{}`.

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
- Recovery of an existing interrupted trajectory, not a new independent evaluation sample
- Exact observable robot states and both camera hashes do not prove equality of unrecorded full MuJoCo internal state
- Prefix replay and one repeated boundary forward are included in recovery wall time; original timing is not synthesized
