# Benchmark result archive

Every file here is the raw output of one `benchmarks/run.py` invocation, written
verbatim and never edited. `docs/benchmarks.md` and the README table are rendered from
the **most recent** file; the older ones are kept because deleting a measurement because
a later one is more flattering is how benchmark suites start lying.

Runs are not interchangeable. A file records the thread count, the git commit, the
hardware and the per-case machine load precisely so that two files can be compared
without guessing, and comparing across runs without reading those fields will produce
nonsense. The runs below differ by up to 298x on the same model, the same weights and even
the same thread count - `threadscale-eager-t8` reads 2202.3 ms in the run taken under full
load and 7.4 ms in the current, idle one - for reasons that have nothing to do with the
model.

| Run id | Threads | Cases | Wall clock | Commit | Peak external load | Status |
|---|---|---|---|---|---|---|
| `20260810T071109Z-e0c34c05` | 1, pinned | 28 | 299 s | `33263e89` (clean) | 0.04 / 8 cores | **current** - rendered into the docs |
| `20260810T054025Z-1385dc69` | 1, pinned | 28 | 304 s | `545d0b78` (clean) | 1.9 / 8 cores | superseded, see below |
| `20260810T050712Z-6314f8b1` | 1, pinned | 28 | 411 s | `57f368fe` (clean) | 8.0 / 8 cores | superseded, see below |
| `20260810T043848Z-8ff4b22d` | not pinned (8) | 20 | 2316 s | `008dc139` (dirty) | 8.0 / 8 cores | superseded, see below |

The current run is the first with no contended case at all: every one of its 27 timed cases
sampled an idle machine, so no figure in `docs/benchmarks.md` carries a `†`.

## Why `20260810T054025Z` is superseded

Its thread handling and its quiet-machine share make it the closest of the older runs to
the current one - 19 of 27 cases sampled quiet - but its `cutoutnet-base` row is measured
against a checkpoint that has no training record behind it. An earlier attempt at that
architecture's run was killed at epoch 10, leaving weights on disk (`055eef63...`, IoU
0.8595) that no `training/runs/*.json` describes. The run was repeated to completion and
the current file measures the finished checkpoint (`8c7acbb0...`, IoU 0.8615), recorded in
`training/runs/cutoutnet-base-20260810T065055Z.json`.

A checkpoint whose training history does not exist is not a result, however good its
number looks, which is why that row was re-measured rather than kept.

## Why `20260810T050712Z` is superseded

Its thread handling is the same as the current run's, but every one of its 27 timed cases
was measured while a neighbouring training job held all eight cores: external demand peaked
at 7.95 of 8. The harness recorded that per case and marked every latency row accordingly,
which is honest but leaves no unqualified timing in the table.

It is nonetheless the most informative of the superseded files, because it is the same
sweep as the current run under the opposite conditions. Its single-thread rows agree with
the current run's to within 1% (`threadscale-eager-t1`: 20.5 ms against 20.7 ms) while its
eight-thread PyTorch row is 298x slower (2202.3 ms against 7.4 ms). That pair is the
evidence behind the suite's single-threaded default, and it is why this file is kept rather
than deleted for being unflattering.

It is also measured against an earlier `cutoutnet-base` checkpoint (`da100e41...`, IoU
0.8549) than the current one (`8c7acbb0...`, IoU 0.8615), because training continued
after it ran.

## Why `20260810T043848Z` is superseded

It was measured before the harness controlled intra-op threads, so PyTorch was given one
thread per core on a machine whose one-minute load average ranged from 11.5 to 25.2 across
the run - on 8 cores - with external demand at 7.99 of 8 while the timings were taken. Its
PyTorch rows are consequently dominated by barrier waits: `cutoutnet` reads 2417.9 ms
there against 31.4 ms in the current run, and its ONNX rows are not comparable to its
PyTorch rows at all, because ONNX Runtime sized its own pool independently. The
`Thread scaling` section of `docs/benchmarks.md` measures that effect directly.

Two further reasons not to quote it: its working tree was dirty, so its numbers are not
attributable to a commit, and its `cutoutnet-base` accuracy came from an early checkpoint
(`624c76b9...`, IoU 0.7252) rather than the finished one the current run measures
(`8c7acbb0...`, IoU 0.8615), which is why that row alone reads as though the 4.3M-parameter
model were worse than the 1.1M one.

Both files are kept, unmodified, because each is a real measurement of a real
configuration - and because the differences between the three are the most useful thing in
this directory. They are the evidence for why the harness pins threads and why it records
machine load per case.

## `experiments/`

Not suite reports. Each file there is the output of a targeted experiment answering one
question that came up while reading the tables, written by a script in `benchmarks/` and
carrying the same environment and per-case load provenance. They live in a subdirectory
because the renderer treats every `.json` directly in this directory as a suite report and
would otherwise publish an experiment as the latest results.

- `order-effect-*.json` (`benchmarks/order_effect.py`) - how much a case's position within
  one process changes its measured latency. It is the evidence behind the repeatability
  note in `docs/benchmarks.md`, which would otherwise have to guess at a cause.

## Verifying a published number

Every figure in the README and in `docs/benchmarks.md` is rendered from one of these
files. To check one:

```bash
python -m cutoutml.benchmarks.render_report benchmarks/results/<run-id>.json
git diff --exit-code README.md docs/benchmarks.md   # no diff => the docs match the data
```

Accuracy figures are reproducible bit-for-bit from a checkpoint digest: the
`Checkpoint provenance` table in `docs/benchmarks.md` lists the SHA-256 of the weights
behind every accuracy row. Where a digest is unchanged between runs, the IoU is
identical to all printed digits; where it differs, the weights were retrained. Latency
figures are not reproducible in that sense and are not claimed to be - each carries the
machine load measured while it was taken.
