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
| `u2net` | U²-Net full, ~44M params | 320×320 | PyTorch | not bundled | Apache-2.0 (reimplemented) | Apache-2.0 (upstream) |
| `u2net-lite` | U²-Net-P, ~1.1M params | 256×256 | PyTorch | trained in-repo | Apache-2.0 (reimplemented) | MIT (in-repo) or Apache-2.0 (upstream) |
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

- `u2net-lite` (U²-Net-P, ~1.1M params) is trained in this repository. Those weights are
  MIT.
- `u2net` (full, ~44M params) is not trained here: a useful run needs a GPU, so no
  checkpoint is shipped and **no accuracy figure is claimed**. Its benchmark row is
  latency-only, with random weights and accuracy explicitly `n/a`.
- Official pretrained weights are Apache-2.0 and may be downloaded — see below. Accuracy
  from official weights is **not comparable** to the in-repo runs, which see a different
  (synthetic) dataset.

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
**are** committed. They are the provenance for the accuracy column in
[docs/benchmarks.md](benchmarks.md).

Measured cost on the 8-core CPU these docs were written on, at the suite's fixed budget of
2048 samples/epoch for 14 epochs at 256 px: roughly 200–250 s per epoch for
`cutoutnet-tiny`. The per-step costs at batch 8 were 261 ms (tiny), 667 ms (small), 422 ms
(base) and 1194 ms (u2net-lite), so `u2net-lite` takes about four times as long as `tiny`
for the same number of samples. Holding the budget fixed rather than the wall clock is the
only way the cross-architecture comparison means anything, and this is what it costs.

`u2net` (full) and `birefnet` are excluded from the suite by default. They are registered
and trainable, but a useful run needs a GPU.

### Download upstream weights

```bash
python -m cutoutml.models.download_weights --list
python -m cutoutml.models.download_weights --model u2net
```

The downloader performs a real streamed download with SHA-256 verification where the
canonical hash is known, atomic replacement, and a clear diagnosis naming every host it
tried when none is reachable. It prints the licence before downloading. It never runs
implicitly — nothing in the API, the worker or the test suite fetches weights.

**Reachability is a real constraint, not a hypothetical.** In the environment this
repository was built in, `huggingface.co` is blocked at the network layer, and that is where
the official U²-Net and BiRefNet checkpoints are mirrored. The downloader handles that
case rather than assuming it away, which is also why the benchmark suite reports
`u2net` and `birefnet` as latency-only rows instead of omitting them or estimating numbers
for them.

### Which weights the benchmark suite needs

| Benchmark row | Needs |
|---|---|
| `trivial-*`, `classical-*` | nothing |
| `cutoutnet-*`, `u2net-lite` | `make weights` |
| `cutoutnet-onnx` | `make weights` (the ONNX export is its last step) |
| `u2net-full-randominit`, `birefnet-compact-randominit` | nothing — random weights, latency only |
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
