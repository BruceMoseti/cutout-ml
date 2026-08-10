# ADR-004: Evaluate on a procedurally generated dataset

Status: Accepted

## Context

Accuracy numbers need ground truth. For segmentation and matting the standard sets are
DUTS (salient object detection), DIS5K (high-resolution dichotomous segmentation) and
AM-2K (animal matting). All three are the right thing to evaluate on and none of them can
be part of this repository's default path:

- They are hundreds of megabytes to tens of gigabytes. Committing them is out of the
  question and downloading them is not always possible — the machine these benchmarks were
  produced on could not reach the hosts.
- Their licences are research-use or unstated. Redistributing them, even as a subset, is
  not clearly permitted.
- A benchmark that cannot be reproduced by a reader is an anecdote. "Download 20 GB from a
  link that may have rotted" is a reproduction story that fails in practice.

The alternative — evaluate on nothing, and report only latency — throws away the more
interesting half of the comparison. Latency without accuracy cannot answer "is the 4.3M
parameter model worth 3x the compute of the 1.1M one", which is the question the benchmark
table exists to answer.

## Decision

Evaluate on a **procedurally generated dataset** (`cutoutml.datasets.synthetic`) that
ships as code plus a committed manifest, and treat real datasets as a first-class but
opt-in path (`cutoutml.datasets.real`, `--dataset-root /path/to/DUTS`).

The generator is designed to be *hard in the ways that matter*, not merely to exist:

1. **Fractional alpha.** Boundary pixels are genuinely partially transparent, and a
   configurable fraction of shapes get an additional Gaussian edge softening that
   simulates motion blur or shallow depth of field. This is what makes MAE more
   informative than IoU on this set — a model producing correct-but-mushy edges is
   punished by MAE and not by IoU.
2. **Distractors.** A second shaded shape is composited into the background and
   **excluded from the ground truth**. This is the single most important element of the
   design. Without it, a model scores well by segmenting "the thing that is not the
   backdrop", and saliency baselines look artificially strong.
3. **Colour collisions.** With probability 0.3 the foreground colour is pulled toward the
   background's mean, so colour alone cannot separate them.
4. **Large translation.** Objects move by up to ±27% of the frame. This is deliberately
   wide, and the cost of narrowing it is measured rather than asserted:
   `benchmarks/center_prior.py` re-generates the eval split at six translation ranges and
   scores the learning-free baselines on each. Centring every object lifts a fixed centred
   ellipse from 0.4382 to 0.5948 IoU and GrabCut seeded from a centred rectangle from
   0.6493 to 0.8306, while `trivial-ones` — which cannot benefit from a centre prior —
   moves the other way, 0.3590 to 0.3397. With a tighter range the benchmark would
   therefore be measuring how well each model has memorised a centre prior. The sweep is
   committed under `benchmarks/results/experiments/center-prior-*.json`.
5. **Photographic degradation.** Brightness/contrast jitter, blur, JPEG re-encode and
   noise. Blur is applied to the composite *and* the alpha together, because a blurred
   photograph genuinely has a blurred matte.
6. **Coverage rejection.** Samples whose foreground covers less than 3% or more than 75%
   of the frame are resampled, because per-image IoU on a nearly-empty or nearly-full mask
   is statistically useless.

**Determinism is the property that makes this work.** `sample(index)` is a pure function
of `(master_seed, split_offset, index)`, seeded through `numpy.random.SeedSequence` with a
spawn key rather than `seed + index` — naive additive seeding makes adjacent samples share
large parts of their bit stream. Splits get disjoint seed offsets, so train and test
cannot overlap.

**The manifest is committed, the images are not.** `datasets/synthetic-eval.json` records
the generator version, master seed, per-split counts, every generation parameter, and a
SHA-256 **content fingerprint** over the first eight samples. `make eval-data` regenerates
and compares. A mismatch means the pixels changed — a different OpenCV resampling default,
a NumPy RNG change, an accidental generator edit — and therefore that the committed
accuracy numbers describe a different dataset than the current checkout produces. That
fails CI, because the alternative is an accuracy column that drifts silently.

**Calibration rows are mandatory.** The benchmark table always includes `trivial-ones`
(predict foreground everywhere) and `trivial-center` (a fixed ellipse, ignoring the
image). IoU is only interpretable against what predicting *nothing* achieves: on the
shipped eval split the foreground covers 35.9% of the frame, so "predict everything"
already scores 0.3590 IoU. Any model that does not clearly beat those rows has learned
nothing.

## Alternatives considered

**Download DUTS/DIS5K in CI and evaluate on that.** The right answer if the network
allows it. Rejected as the *default*: it makes the benchmark unreproducible whenever the
host is unreachable or the licence changes, and it was not reachable here. It is fully
supported as an option — `RealSegmentationDataset` handles DUTS, DIS5K, AM-2K and a flat
`images/`+`masks/` layout, and the harness takes it with no other changes.

**Commit a small hand-picked subset of a real dataset.** Rejected on licence grounds, and
because a 50-image subset has confidence intervals wide enough to make model ranking
meaningless.

**Scrape and label a handful of images by hand.** Rejected: hand-drawn mattes have no
fractional alpha, so they would systematically reward hard-edged predictions and make the
matting metrics meaningless. Also unreproducible by a reader.

**Report latency only.** Rejected: it removes the ability to say anything about whether
extra capacity buys accuracy, which is the main question the table answers.

## Consequences

Good:

- The full benchmark, including accuracy, reproduces from a clean checkout with no
  downloads: `make weights && make bench`.
- The generator can be made harder on purpose. The distractor and colour-collision
  mechanisms exist because early versions were too easy — a saliency baseline scored
  suspiciously well, which was informative.
- The fingerprint turns "the eval set changed" from an invisible drift into a build
  failure.
- Sweeping a generation parameter is a controlled experiment on segmentation difficulty,
  which real datasets do not permit.

Bad, and accepted — and stated wherever numbers appear:

- **Absolute accuracy numbers are not comparable to published DUTS/DIS5K results.** They
  are internally comparable (same data, same budget, same harness) and nothing more.
  `docs/benchmarks.md` says so, the manifest's `notes` field says so, and the rendered
  tables carry the dataset id.
- Synthetic shapes are not photographs. A model tuned on this set is not a product-grade
  background remover, and no claim is made that it is. The gap is a real limitation, not
  a rounding error.
- The generator is code that can have bugs, and a generator bug is indistinguishable from
  a model result unless someone looks at the images. `scripts/eval_data.py --dump` exists
  for exactly that.
- Bumping `GENERATOR_VERSION` invalidates every committed number and requires re-running
  the suite. That cost is intentional friction.
