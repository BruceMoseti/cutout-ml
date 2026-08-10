# Contributing

## Setup

Requires Python 3.12+, Node 20+, ffmpeg, and — for anything touching the API or worker —
PostgreSQL 15+ and Redis 7+.

```bash
make venv install install-web     # virtualenv, Python extras, npm ci
cp .env.example .env              # then edit
make migrate                      # alembic upgrade head
make doctor                       # what actually works on this machine
```

`make doctor` is the first thing to run when something is confusing. Most "why is this slow
/ why did that fail" questions here have one of three answers — no GPU, no ffmpeg, no
onnxruntime — and it prints all three.

The learned models need checkpoints, which are not committed
([ADR-008](docs/decisions/ADR-008-no-committed-weights.md)):

```bash
make weights                      # trains tiny/small/base/u2net-lite, then re-exports ONNX
```

The classical and trivial baselines need nothing, so the API, both pipelines, the CLI, the
web console and the whole test suite work immediately after a clone.

## The checks that must pass

```bash
make check       # ruff lint + ruff format --check + mypy + pytest
```

CI runs exactly these plus the frontend and a Docker build. Individually:

```bash
make lint                                  # ruff check . && ruff format --check .
make typecheck                             # mypy
make test                                  # pytest -m 'not integration'
make test-integration                      # needs Postgres, Redis, ffmpeg
cd apps/web && npm run lint && npm run typecheck && npm run test && npm run build
```

Formatting is `ruff format` with a 100-column line length. Do not argue with it; run
`make fmt`.

## Conventions

**Commits are [Conventional Commits](https://www.conventionalcommits.org/):** `feat:`,
`fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`, `style:`, with an optional scope
(`feat(video):`). The subject says what changed; **the body says why**. A commit whose body
restates the diff in prose has wasted the reader's time — the diff already says what.

**Comments explain intent or a constraint, never mechanics.** `# increment the counter`
above `i += 1` is noise. `# kombu.Queue objects, not dicts: Celery reads .name off each
entry in apply_async` is worth its line, because the next person to touch it will otherwise
reintroduce the bug. Most of the comments in this repository are of the second kind and new
ones should be too.

**Docstrings on modules and non-obvious functions.** Module docstrings carry the design
rationale — what the module is for, what alternative was rejected, what will bite you.

**Type annotations everywhere.** mypy runs with `disallow_untyped_defs`. Tests are relaxed
but still checked.

**Never fabricate a number.** No performance claim goes into a docstring, a comment or a
document unless it was measured on the machine described next to it. If something cannot be
measured in a given environment — TensorRT and CUDA cannot be measured on the CPU-only
machine the committed results came from — say "not measured on this hardware" rather than
estimating. Every table in `docs/benchmarks.md` and the README is *generated* from a
committed results JSON, which is what makes this enforceable rather than aspirational.

## Tests

- `tests/` for unit tests. No network, no database, no broker; must run in seconds.
- Tests marked `@pytest.mark.integration` may use Postgres, Redis and ffmpeg.
- `@pytest.mark.slow` for anything over a few seconds; `@pytest.mark.onnx` for anything
  needing onnxruntime.

**Determinism is required.** A flaky test is worse than no test, because it trains people
to re-run CI. Seed every RNG, never assert on wall-clock durations, and never assert on
floating-point equality without a tolerance. If a test is slow, make it smaller — do not
delete it and do not mark it `xfail`.

Where a numerical function has a hand-computable answer, assert against the hand-computed
value rather than against the implementation's own output. A test that asserts the code does
what the code does will pass through any refactor, including a wrong one.

## Adding a model

1. Adapter subclassing `SegmentationModel` or `TorchSegmentationModel`, implementing
   `preprocess`, `predict`, `postprocess`.
2. `register(ModelSpec(...))` in `models/registry.py`, with the **licence and upstream
   source filled in honestly**.
3. A `BenchmarkCase` in `benchmarks/run.py`.
4. `make bench` and commit the regenerated results JSON together with the code.

Nothing in the API, the pipelines or the worker changes. That is the point of
[ADR-001](docs/decisions/ADR-001-model-registry.md).

## Changing the dataset generator

`make eval-data` verifies a content fingerprint over the committed eval-set manifest, and CI
runs it. If you change generated pixels, that check fails — deliberately, because every
committed accuracy number was measured on the old pixels. The fix is not to update the
fingerprint:

1. Bump `GENERATOR_VERSION` in `datasets/manifest.py`.
2. `scripts/eval_data.py --write` to regenerate the manifest.
3. Re-run `make weights && make bench`.
4. Commit all of it together.

## Database changes

```bash
.venv/bin/alembic revision --autogenerate -m "add thing"
```

**Read the generated migration before committing it.** Autogenerate does not detect renames
(it emits a drop plus an add, which discards data), misses server-default changes, and will
happily generate a `DROP` for anything created outside Alembic. Every migration needs a
working `downgrade()`.

## Architecture decisions

If a change is expensive to reverse or a reviewer would reasonably ask "why this way?", add
a record to `docs/decisions/`. Context, Decision, Alternatives considered, Consequences —
including the bad consequences. An ADR that lists only benefits is marketing. Records are
append-only: supersede rather than edit.

## Pull requests

Small and focused. One reason to exist per PR. Fill in the template — particularly *how you
verified it*, which should name the command you ran, not "tested locally". If behaviour
changed, update the docs in the same PR.

## Benchmarks

Do not run the full suite in a PR; it is minutes of CPU and the numbers would be measured
on shared CI hardware under unknown contention, which makes them worse than no numbers.
`benchmarks.yml` is manual and scheduled for that reason. When you do commit new results,
commit the JSON and the regenerated markdown together, and say in the commit body what
hardware produced them.
