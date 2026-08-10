# Models, weights, licences and attribution

Every model is a `ModelSpec` in `src/cutoutml/models/registry.py`. The licence, upstream
source and weight provenance of each are declared there, exposed by
`GET /v1/models/catalogue`, and reproduced below.

**No weights are committed to this repository.** See
[ADR-008](decisions/ADR-008-no-committed-weights.md) for why, and "Obtaining weights"
below for what to run instead.

## The registry

| Model | Architecture | Input | Runtime | Weights | Code licence | Weight licence |
|---|---|---|---|---|---|---|
| `cutoutnet` | CutoutNet-small, ~1.1M params | 256×256 | PyTorch | trained in-repo | MIT (original) | MIT (original) |
| `cutoutnet-tiny` | CutoutNet-tiny, ~0.12M params | 256×256 | PyTorch | trained in-repo | MIT (original) | MIT (original) |
| `cutoutnet-base` | CutoutNet-base, ~4.3M params | 256×256 | PyTorch | trained in-repo | MIT (original) | MIT (original) |
| `cutoutnet-onnx` | CutoutNet-small, exported | 256×256 | ONNX Runtime | exported in-repo | MIT (original) | MIT (original) |
| `u2net` | U²-Net full, ~44M params | 320×320 | PyTorch | published, converted from ONNX | Apache-2.0 (reimplemented) | Apache-2.0 (upstream) |
| `u2netp` | U²-Net-P, ~1.1M params | 320×320 | PyTorch | published, converted from ONNX | Apache-2.0 (reimplemented) | Apache-2.0 (upstream) |
| `u2net-onnx` | U²-Net full graph | 320×320 | ONNX Runtime | published, downloaded | n/a (upstream graph) | Apache-2.0 (upstream) |
| `u2netp-onnx` | U²-Net-P graph | 320×320 | ONNX Runtime | published, downloaded | n/a (upstream graph) | Apache-2.0 (upstream) |
| `u2net-lite` | U²-Net-P, ~1.1M params | 256×256 | PyTorch | trained in-repo | Apache-2.0 (reimplemented) | MIT (in-repo) |
| `birefnet` | BiRefNetCompact | 512×512 | PyTorch | not bundled | MIT (reimplemented) | see below |
| `classical` | GrabCut from a centred rectangle | 320×320 | OpenCV | none needed | MIT | n/a |
| `classical-saliency` | Spectral-residual saliency + Otsu | 320×320 | OpenCV | none needed | MIT | n/a |
| `classical-saliency-grabcut` | Saliency trimap → GrabCut | 320×320 | OpenCV | none needed | MIT | n/a |
| `trivial-center` | Fixed centred ellipse | 320×320 | NumPy | none needed | MIT | n/a |
| `trivial-ones` | Foreground everywhere | 320×320 | NumPy | none needed | MIT | n/a |
| `tensorrt` | CutoutNet-small engine | 256×256 | TensorRT | built from ONNX | MIT (original) | MIT (original) |

"Original" means designed and implemented in this repository. "Reimplemented" means the
architecture is described in a published paper and the code here is an independent
implementation, not a copy.

## Attribution, per architecture

### CutoutNet (original)

Designed for this project: a depthwise-separable convolutional encoder, an FPN-lite decoder
with lateral connections, and a learned alpha refinement head. Three widths — tiny, small
and base — trained on an identical data budget so that the trio isolates capacity from
everything else.

MIT, code and weights. Not derived from any pretrained model.

### U²-Net (Qin et al.)

- Paper: *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*,
  Qin, Zhang, Huang, Dehghan, Zaiane, Jagersand, Pattern Recognition 2020.
- Upstream: <https://github.com/xuebinqin/U-2-Net>, Apache-2.0.

The implementation in `src/cutoutml/models/u2net/arch.py` is **independent** — written from
the paper's description of the RSU blocks and nested U-structure — but is deliberately
**shape-compatible** with the official checkpoints. `U2NetAdapter` remaps upstream key
names on load, so official `u2net.pth` and `u2netp.pth` files work.

Shape compatibility is enforced rather than asserted. It was in fact broken until
`from_onnx` refused to convert: the decoder stage widths had been derived from the encoder
table, which does not match the published architecture (`stage4d` is `RSU4(1024,128,256)`,
not the `RSU4(1024,*,512)` that mirroring produces, and `stage1d` uses a 16-channel
bottleneck where its paired encoder stage uses 32). Because the adapter loads with
`strict=False` to tolerate upstream key renaming, loading the official checkpoint into the
broken architecture would have skipped the mismatched tensors and run inference on random
weights with nothing but a log warning. The widths are now transcribed, and the conversion
tests fail if any tensor does not fit.

**The published weights are obtainable here, and are what `u2net` and `u2netp` run.** The
authors distribute `u2net.pth` via Google Drive and the `.pth` mirrors are on HuggingFace,
which is blocked in some networks — but an ONNX export of the same Apache-2.0 weights is
redistributed from a GitHub release, which is reachable. `make weights-pretrained` fetches
those graphs and converts them:

```bash
make weights-pretrained     # downloads u2net.onnx + u2netp.onnx, writes u2net.pt + u2netp.pt
```

The conversion is not a repackaging. The graphs were exported with constant folding on, so
each `Conv → BatchNorm` pair has collapsed into one biased convolution, the BatchNorm
statistics no longer exist, and the parameter names of all 112 folded convolutions are gone
— they appear as numeric temporaries. Only `side1..side6` and `outconv`, which have no
BatchNorm after them, keep their names, so the name-based remapping used for a real `.pth`
cannot work. `cutoutml.models.u2net.from_onnx` instead pairs ONNX `Conv` nodes with the
module's convolutions positionally (both sequences are in execution order; the PyTorch one
is recovered by running the module under forward hooks rather than by trusting construction
order), then proves the pairing three ways: pairwise shapes, the seven convolutions that
kept their names landing where their names say, and numerical parity against onnxruntime.
Measured parity is **1.4e-7** for the full model and **1.5e-6** for lite, against a 1e-4
tolerance — roughly three orders of magnitude finer than one 8-bit alpha level, so the
difference cannot survive quantisation to a PNG.

Those two figures are not written down here from memory. Because the weights themselves are
not committed, the conversion also writes a record that is:
[`models/conversions/u2net.json`](../models/conversions/u2net.json) and
[`u2netp.json`](../models/conversions/u2netp.json) carry the parity figure, the tolerance it
was checked against, the SHA-256 of the source graph and of the checkpoint produced, the
convolution count, and the onnxruntime and torch versions that produced the comparison.
They are to a converted checkpoint what `training/runs/*.json` is to a trained one.

The resulting checkpoints have their BatchNorms set to exact identities, because the
convolutions have absorbed them. That makes them equivalent in `eval()` but **unsuitable for
fine-tuning**: the BatchNorms are neutral, not calibrated, and would start re-learning
statistics for activations that have already been scaled. Each file records this, its source
digest and its licence in a provenance dict stored inside the checkpoint as well, because a
sidecar is what goes missing when weights are copied between machines and U²-Net weights of
unknown origin are a licensing problem as much as a reproducibility one.

The conversion is byte-reproducible: converting one graph twice to the same path yields the
same checkpoint digest. That matters because the benchmark suite records the SHA-256 of the
weights behind every accuracy row and the archive index reads a changed digest as changed
weights — so the conversion timestamp lives in the record, never in the checkpoint. A test
pins both halves of that.

- `u2net` / `u2netp` run the authors' weights, Apache-2.0, with **real accuracy figures**.
- `u2net-onnx` / `u2netp-onnx` run the same weights under onnxruntime. Because parity is
  verified, the PyTorch and ONNX rows differ only by runtime — which is the only reason
  comparing them says anything.
- `u2net-lite` remains the separate, in-repo trained U²-Net-P. Those weights are MIT.
- Accuracy from the published weights is **not comparable** to the in-repo runs. They were
  trained on DUTS, a real-photograph saliency dataset, and are evaluated here on a
  synthetic eval set. See [docs/benchmarks.md](benchmarks.md), where this shows up as the
  pretrained models scoring *below* a small model trained in-repo on the eval set's own
  distribution — a domain-shift result, not a quality ranking.

### Preprocessing fidelity

U²-Net's reference pipeline rescales to `[0, 1]`, divides each image by its own maximum
intensity, and then applies `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`. The
max division was previously skipped here on the grounds that it is a no-op for any image
containing a saturated pixel. That is true of most photographs but false of this eval set:
**40 of its 64 test images peak below 255, and the dimmest peaks at 155** — so the
pretrained weights were being fed a compressed dynamic range they were never trained on.
Those two counts are asserted in `tests/test_u2net_weights.py` rather than written down
here, because the eval set is procedurally generated and a change to the generator would
otherwise falsify this paragraph silently. The division is now implemented, computed from
the source pixels rather than the letterboxed canvas so that constant padding cannot set
the maximum.

Because preprocessing is not part of an ONNX artefact, the ONNX specs carry the same
requirement explicitly via `intensity_scaling: "max"`. They also declare
`output_activation: "sigmoid"`, because these graphs apply the sigmoid internally and a
second one would compress every mask towards 0.5 — visibly softer edges, several IoU
points, and no error anywhere.

### BiRefNet (Zheng et al.)

- Paper: *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*,
  Zheng, Peng et al., CAAI AIR 2024.
- Upstream: <https://github.com/ZhengPeng7/BiRefNet>, code MIT.

`BiRefNetCompact` is **architecture-inspired**, not a port: a localisation module plus a
reconstruction module with inner (source-pixel) and outer (gradient) references. It is
**not weight-compatible** with official BiRefNet checkpoints, which are built on a Swin
transformer backbone whose tensor shapes do not match. Downloading them would produce a
checkpoint that cannot load, so `download_weights` offers no URL for it and says why
rather than pretending.

**Licence warning, and it is a real one:** official BiRefNet *code* is MIT, but **some
third-party fine-tuned BiRefNet checkpoints are released under non-commercial terms**.
Anyone using a BiRefNet checkpoint from a third party is responsible for checking its
specific licence. This repository ships none.

### GrabCut

- Paper: *"GrabCut": interactive foreground extraction using iterated graph cuts*,
  Rother, Kolmogorov, Blake, SIGGRAPH 2004.
- Implementation: OpenCV's `cv2.grabCut` (Apache-2.0), seeded here from a centred
  rectangle. The seeding and post-processing are original.

### Spectral-residual saliency

- Paper: *Saliency Detection: A Spectral Residual Approach*, Hou & Zhang, CVPR 2007.
- Implemented directly on top of NumPy/OpenCV FFT primitives.

### Third-party libraries

PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), OpenCV (Apache-2.0 for 4.5+), Pillow
(MIT-CMU), SciPy (BSD-3-Clause), FastAPI (MIT), SQLAlchemy (MIT), Celery (BSD-3-Clause),
ONNX Runtime (MIT), ffmpeg (LGPL-2.1+/GPL depending on build — invoked as a subprocess,
not linked).

## Obtaining weights

### Train them (the default path, works offline)

```bash
make weights          # tiny, small, base, u2net-lite, then re-export ONNX
make train            # just the default cutoutnet-small
scripts/train_suite.sh cutoutnet-tiny        # one architecture
```

Training uses the deterministic procedural dataset
([ADR-004](decisions/ADR-004-synthetic-dataset.md)), so a rerun is reproducible up to BLAS
non-determinism. Every run writes a record to `training/runs/*.json` — per-epoch losses,
validation IoU and MAE, throughput and the full hyperparameter set — and those records
**are** committed. They are what connects a benchmark row to a training history: the
`Checkpoint provenance` table in [docs/benchmarks.md](benchmarks.md) gives the SHA-256 of
the weights behind each accuracy figure, and a run record names the checkpoint it wrote. The
accuracy numbers themselves are not in the training records and are not meant to match them
— a record reports validation IoU over 192 held-out samples during training, while the
benchmark measures the finished checkpoint over the 64-sample test split with the serving
preprocessing and refinement stack attached.

Measured cost of the three committed runs, all on the same 8-core CPU with 8 intra-op
threads, at the suite's fixed budget of 2048 samples/epoch for 14 epochs at 256 px and batch
16:

| Run | Params | Median s/epoch | Range | Median samples/s | Wall clock | Best val IoU |
|---|---|---|---|---|---|---|
| `cutoutnet-tiny` | 0.12M | 242 | 171–680 | 8.5 | 71 min | 0.8079 |
| `cutoutnet-small` | 1.14M | 153 | 144–214 | 13.4 | 38 min | 0.8265 |
| `cutoutnet-base` | 4.34M | 174 | 172–271 | 11.8 | 42 min | 0.8434 |

**Those wall clocks do not rank the architectures, and the table is here to make that
visible rather than to hide it.** The smallest network is the slowest of the three, which
cannot be a property of the network: the `tiny` run's epochs range over a factor of four
against `small`'s factor of 1.5, so it was sharing the machine. Unlike the benchmark harness,
the trainer does not sample external CPU load, so there is no per-epoch evidence to attach
and no honest way to correct these figures — they are what those runs cost on that machine
on that day, and nothing more. Holding the sample budget fixed is what makes the *accuracy*
column comparable across the three; the seconds column is not comparable and should not be
read as though it were.

`u2net-lite` is trainable by the same command and is in `make weights`, but no run of it is
committed and none has been measured; its benchmark case is recorded as skipped for missing
weights. `u2net` (full) and `birefnet` are excluded from the suite by default — registered
and trainable, but a useful run needs a GPU.

### Download upstream weights

```bash
python -m cutoutml.models.download_weights --list
make weights-pretrained                                  # the route that works here
python -m cutoutml.models.download_weights --model u2net # the authors' .pth, needs HuggingFace
```

The downloader performs a real streamed download with SHA-256 verification where the
canonical hash is known, atomic replacement, and a clear diagnosis naming every host it
tried when none is reachable. It prints the licence before downloading. It never runs
implicitly — nothing in the API, the worker or the test suite fetches weights.

**Reachability is a real constraint, not a hypothetical.** In the environment this
repository was built in, `huggingface.co` is blocked at the network layer, and that is where
the official U²-Net and BiRefNet `.pth` checkpoints are mirrored. Rather than accept that as
the end of the matter, the downloader routes around it: the same U²-Net weights are
redistributed as an ONNX export from a GitHub release, which is reachable, and `from_onnx`
converts them back into a verified PyTorch checkpoint. That is why `u2net` has real accuracy
numbers here and `birefnet` does not — the BiRefNet checkpoints are not shape-compatible
with this repository's reimplementation, so no amount of network access would help, and its
benchmark row stays latency-only rather than being estimated.

### Which weights the benchmark suite needs

| Benchmark row | Needs |
|---|---|
| `trivial-*`, `classical-*` | nothing |
| `cutoutnet-*`, `u2net-lite` | `make weights` |
| `cutoutnet-onnx` | `make weights` (the ONNX export is its last step) |
| `u2net-pretrained`, `u2netp-pretrained`, and their `-onnx` pairs | `make weights-pretrained` |
| `birefnet-compact-randominit` | nothing — random weights, latency only |
| `tensorrt` | CUDA + TensorRT. Not measured on this hardware. |

A case whose checkpoint is missing is reported as `status: skipped` with the reason
attached, never silently downgraded to random weights — so a missing training run cannot be
misread as a bad accuracy number.

## Adding a model

1. Write an adapter subclassing `SegmentationModel` (or `TorchSegmentationModel`, which
   provides preprocessing, letterboxing, autocast and ONNX export) implementing
   `preprocess`, `predict` and `postprocess`.
2. `register(ModelSpec(...))` with the licence and upstream source filled in honestly.
3. Add a `BenchmarkCase` to `benchmarks/run.py` and re-run `make bench`.

Nothing in the API, the pipelines, the worker or the harness needs to change. See
[ADR-001](decisions/ADR-001-model-registry.md).
