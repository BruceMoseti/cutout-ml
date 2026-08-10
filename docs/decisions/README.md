# Architecture decision records

One file per decision that was expensive to make and would be expensive to reverse.
Each record states the context that forced a choice, the choice, the alternatives that
were seriously considered and rejected, and the consequences — including the bad ones.

A decision belongs here when a reader of the code would otherwise reasonably ask "why
is it done this way?" and the answer takes more than a comment. Decisions that are
obvious in hindsight, or that are cheap to change, do not need a record.

Records are append-only. A superseded decision keeps its file and gains a
`Status: superseded by ADR-NNN` line rather than being edited into agreement with the
present.

| ADR | Decision | Status |
|---|---|---|
| [001](ADR-001-model-registry.md) | Models are declared in a registry and loaded by name | Accepted |
| [002](ADR-002-queues.md) | Redis broker with three queues and separate CPU/GPU workers | Accepted |
| [003](ADR-003-video-output.md) | Three video output modes; alpha capability is probed, not assumed | Accepted |
| [004](ADR-004-synthetic-dataset.md) | Evaluate on a procedurally generated dataset | Accepted |
| [005](ADR-005-onnx-runtime.md) | ONNX Runtime as the second serving runtime | Accepted |
| [006](ADR-006-storage-layout.md) | Server-generated random storage keys behind a narrow interface | Accepted |
| [007](ADR-007-idempotency.md) | Idempotency keys and a job/run split instead of exactly-once delivery | Accepted |
| [008](ADR-008-no-committed-weights.md) | No model weights in git | Accepted |
