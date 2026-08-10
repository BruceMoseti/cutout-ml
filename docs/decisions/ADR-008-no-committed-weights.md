# ADR-008: No model weights in git

Status: Accepted

## Context

The repository needs checkpoints to do anything interesting. Four exist:
`cutoutnet-tiny.pt` (~0.6 MB), `cutoutnet-small.pt` (~4.7 MB), `cutoutnet-base.pt`
(~18 MB), plus the exported `cutoutnet-small.onnx` (~4.6 MB) and, if trained,
`u2net-lite.pt`. All are produced in this repository from a seeded procedural dataset.

The tempting option is to commit the small ones. A 4.7 MB file is not obviously a problem,
and it makes a fresh clone work immediately: `GET /v1/models` reports the default model as
available, `cutoutml segment photo.jpg` produces a cutout, and the benchmark suite runs
without a training step.

The cost is not the 4.7 MB. It is that git stores every version of it forever. Weights
change on every retrain, and a retrain is the normal way to work on this project — the
capacity sweep in the benchmark table exists precisely because the models get retrained.
Ten retrains of three checkpoints is a repository that is a few hundred megabytes of
history for a few tens of kilobytes of useful diff, and history cannot be pruned without
rewriting it.

There is a second, sharper problem. Upstream weights are not all redistributable.
Published U^2-Net weights are Apache-2.0, which permits redistribution, but they are hosted
off-PyPI on Google Drive and HuggingFace mirrors. Official BiRefNet code is MIT while
*some* third-party BiRefNet fine-tunes are non-commercial. A repository that commits some
weights invites the assumption that all weights in it are equally free to use, and that
assumption is wrong in a way that matters legally.

## Decision

**No binary weights, exported graphs or engines are tracked.** `.gitignore` excludes
`*.pt`, `*.pth`, `*.onnx`, `*.engine` and `*.plan` with no exceptions.

What is committed instead:

- **The training code and its seed.** `scripts/train_suite.sh` trains every CPU-feasible
  architecture on one identical budget, from the deterministic dataset described in ADR-004.
- **The training run records.** `training/runs/*.json` — the per-epoch losses, validation
  IoU/MAE, samples per second, and the full hyperparameter set of each run that produced a
  checkpoint. These are small, textual, diff-friendly and are the provenance for the
  accuracy column in `docs/benchmarks.md`.
- **The eval-set manifest** with a content fingerprint, so a reader can confirm they are
  measuring on the same data.
- **A fetch path for upstream weights.** `cutoutml.models.download_weights` names the
  source, the licence and the homepage for every external checkpoint, and downloads it on
  request. It never runs implicitly.
- **Per-model licence metadata in the registry**, surfaced through `GET /v1/models` and
  `docs/models.md`, so the licence of a checkpoint travels with the model rather than living
  in a README.

`make weights` produces everything the committed benchmark suite needs, then re-exports the
ONNX graph.

## Alternatives considered

**Commit the small checkpoints, gitignore the large ones.** The previous state of this
repository. Rejected: the cutoff is arbitrary, the history still grows on every retrain of
the committed one, and it creates exactly the "some weights are here so all weights must be
fine" ambiguity described above.

**Git LFS.** The standard answer, and it does keep the working tree small. Rejected here:
LFS requires server-side support and a client extension, so a plain `git clone` on a host
without LFS gives you pointer files where weights should be — a failure mode that is more
confusing than an honest absence. It also does not solve the licence question.

**GitHub Releases as the weight store.** A good answer, and the one to adopt if this project
ever has releases. Rejected for now because it makes the weights depend on a hosting
platform's API rather than on code in the repository, and because the checkpoints here are
cheap enough to reproduce that a download is not clearly better than a training run.
`download_weights.py` is already the right shape to point at a release asset.

**Train on first use, transparently.** Rejected: a background-removal request that silently
kicks off a multi-minute training run is a worse experience than an error saying what to run.

## Consequences

Good:

- History stays textual and small. A retrain changes a run JSON, not a blob.
- No licensing ambiguity: no checkpoint in this repository is redistributed by it, and every
  external one is named with its licence and source.
- The reproduction story is stronger, not weaker. A committed binary is unverifiable — it may
  or may not be what the committed training code produces. A committed *training run* plus a
  deterministic dataset is checkable.

Bad, and accepted:

- **A fresh clone cannot serve the learned models.** This is the real cost. It is mitigated
  rather than eliminated:
  - The classical and trivial baselines require no weights, so the API, both pipelines, the
    CLI and the web console are all functional and testable immediately after a clone.
  - `GET /v1/models` reports `weights_available: false` per model *before* a job is
    submitted, and `cutoutml doctor` reports it up front.
  - Each adapter's `weights_hint()` says exactly what to run, including the measured cost —
    for example, that training `u2net-full` (44M parameters) needs a GPU and is not worth
    attempting on CPU.
- Reproducing the benchmark table takes a training run first. On 8 CPU cores that is tens of
  minutes for the small models and hours for the full suite, which is documented in
  `docs/benchmarks.md` rather than glossed over.
- Retrained weights are not bit-identical across BLAS builds and thread counts, so a
  reader's numbers will be close to, not equal to, the committed ones. The committed run
  JSONs make the difference visible.
