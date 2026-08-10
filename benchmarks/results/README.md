# Benchmark result archive

Every file here is the raw output of one `benchmarks/run.py` invocation, written
verbatim and never edited. `docs/benchmarks.md` and the README table are rendered from
the **most recent** file; the older ones are kept because deleting a measurement because
a later one is more flattering is how benchmark suites start lying.

Runs are not interchangeable. A file records the thread count, the git commit, the
hardware and the per-case machine load precisely so that two files can be compared
without guessing, and comparing across runs without reading those fields will produce
nonsense. The two runs below differ by up to 68x on the same model and the same weights,
for reasons that have nothing to do with the model.

| Run id | Threads | Cases | Wall clock | Commit | Status |
|---|---|---|---|---|---|
| `20260810T050712Z-6314f8b1` | 1, pinned | 28 | 411 s | `57f368fe` (clean) | **current** - rendered into the docs |
| `20260810T043848Z-8ff4b22d` | not pinned (8) | 20 | 2316 s | `008dc139` (dirty) | superseded, see below |

## Why `20260810T043848Z` is superseded

It was measured before the harness controlled intra-op threads, so PyTorch was given one
thread per core on a machine that already had ~14 runnable threads across 8 cores. Its
PyTorch rows are consequently dominated by barrier waits: `cutoutnet` reads 2417.9 ms
there against 35.6 ms in the current run, and its ONNX rows are not comparable to its
PyTorch rows at all, because ONNX Runtime sized its own pool independently. The
`Thread scaling` section of `docs/benchmarks.md` measures that effect directly.

Two further reasons not to quote it: its working tree was dirty, so its numbers are not
attributable to a commit, and its `cutoutnet-base` accuracy came from a checkpoint
(`624c76b9...`) that has since been retrained (`da100e41...`).

The file is kept, unmodified, because it is a real measurement of a real configuration -
and because the difference between the two runs is the most useful thing in this
directory. It is the evidence for why the harness pins threads now.

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
