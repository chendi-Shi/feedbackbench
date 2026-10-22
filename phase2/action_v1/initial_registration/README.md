# Action adaptation: immutable evidence snapshot

Status: **partial**. Training 0/6; evaluation 0/280. This is a snapshot, not live status.

[Frozen protocol](protocol.json) · [Results](report.json) · [Evidence index](evidence_index.json) · [Source index](source_index.json) · [Reproduction boundaries](REPRODUCTION.md) · [Publication manifest](PUBLICATION_MANIFEST.json)

Public projection receipts identify every included file and the original private receipt hash. Original full seals require omitted private files and cannot be verified from this subset. Adapters support inference with separately obtained pinned base weights; optimizer/RNG state is not published.

Four fixed tasks, five paired initial states, and three training seeds support only limited same-domain conclusions. Pretraining overlap is unknown. Observation/action alignment is checked; full dynamics replay is not verified. A decreasing training loss or partial evaluation does not establish better task success. Sources are archived under source/ at their original relative paths; this archive is not a standalone runnable environment.
