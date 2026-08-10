<!-- GENERATED FILE - do not edit by hand.
     Produced by `python -m cutoutml.benchmarks.render_report`
     from benchmarks/results/20260810T043848Z-8ff4b22d.json. -->

# Benchmarks

## Environment

- **Hardware**: Intel(R) Xeon(R) Processor, 8 vCPU (8 physical cores), 47 GB RAM, no GPU (CPU-only)
- **GPU**: none  <-- all numbers below are CPU-only; no GPU was available on this machine
- **OS / Python**: Linux 6.12.94+ (x86_64) / Python 3.12.3
- **PyTorch threads**: 8
- **Git commit**: `008dc139f7ca` on `cursor/cutoutml-platform-3514` (**working tree dirty** - numbers are not attributable to this commit)
- **Libraries**: celery 5.6.3, fastapi 0.141.1, numpy 2.5.2, onnx 1.22.0, onnxruntime 1.28.0, opencv-python-headless 5.0.0.93, pillow 12.3.0, scipy 1.18.0, sqlalchemy 2.0.51, torch 2.13.0+cpu
- **Run id**: `20260810T043848Z-8ff4b22d` (2026-08-10T04:38:48Z, 2316.28 s wall clock)

## Dataset

- **Dataset id**: `synthetic-v1.0.0-seed20240817`
- **Generator**: `cutoutml.datasets.synthetic` v1.0.0
- **Master seed**: `20240817`
- **Resolution**: [256, 256]
- **Content fingerprint**: `24b52899998fa9920c972c194876e3f6...` (first 8 samples)
- **Splits**: test=64
- **Harness**: 3 warmup + 20 timed repetitions per case, 64 accuracy samples

## Results

| Model | Runtime | Precision | Batch | IoU | MAE | F-beta | Boundary F1 | p50 ms/img | p95 ms/img | img/s | Peak RSS | Model size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trivial-ones | numpy | fp32 | 1 | 0.3590 | 0.6410 | 0.4142 | 0.0000 | 0.12 | 0.13 | 8419.3 | 332.4 MiB | n/a |
| trivial-center | numpy | fp32 | 1 | 0.4382 | 0.2644 | 0.6167 | 0.0959 | 0.54 | 0.62 | 1708.3 | 332.9 MiB | n/a |
| classical-saliency | opencv+numpy | fp32 | 1 | 0.1508 | 0.3772 | 0.3130 | 0.1325 | 0.87 | 0.92 | 1139.9 | 334.0 MiB | n/a |
| classical | opencv+numpy | fp32 | 1 | 0.6614 | 0.1455 | 0.7744 | 0.5766 | 274.33 | 309.41 | 3.6 | 334.7 MiB | n/a |
| classical-saliency-grabcut | opencv+numpy | fp32 | 1 | 0.1570 | 0.3693 | 0.3145 | 0.2901 | 411.88 | 435.50 | 2.5 | 334.7 MiB | n/a |
| cutoutnet-tiny | pytorch-eager | fp32 | 1 | 0.8241 | 0.0693 | 0.8936 | 0.7564 | 2028.82 | 2598.32 | 0.5 | 354.8 MiB | 0.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 2417.92 | 3022.73 | 0.4 | 392.2 MiB | 4.5 MiB |
| cutoutnet-base | pytorch-eager | fp32 | 1 | 0.7252 | 0.1111 | 0.8237 | 0.5468 | 2921.18 | 3392.39 | 0.3 | 369.0 MiB | 16.8 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 19.85 | 37.69 | 46.3 | 423.3 MiB | 4.4 MiB |
| u2netp | pytorch-eager | fp32 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 3430.95 | 4476.26 | 0.3 | 481.6 MiB | 4.6 MiB |
| u2netp-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 233.01 | 302.56 | 4.2 | 875.4 MiB | 4.4 MiB |
| u2net | pytorch-eager | fp32 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 4133.99 | 4377.58 | 0.3 | 749.2 MiB | 168.2 MiB |
| u2net-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 427.77 | 624.53 | 2.2 | 1376.7 MiB | 167.8 MiB |
| birefnet random-init | pytorch-eager | fp32 | 1 | n/a * | n/a * | n/a * | n/a * | 2384.42 | 3084.29 | 0.4 | 761.6 MiB | 11.8 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 2159.30 | 3086.75 | 0.4 | 897.9 MiB | 4.5 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 8 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 433.13 | 475.81 | 2.3 | 946.1 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 4 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 1022.00 | 1143.33 | 1.0 | 946.4 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 8 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 272.51 | 319.11 | 3.6 | 946.4 MiB | 4.5 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 8 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 11.20 | 18.21 | 84.8 | 1186.6 MiB | 4.4 MiB |

`n/a *` = accuracy not measurable for this row: the network ran with **random weights** so that latency could still be benchmarked without downloadable checkpoints. Latency in those rows is real; accuracy is meaningless.

## Runtime comparison

The same weights at the same batch size under PyTorch eager, `torch.compile` (Inductor) and ONNX Runtime, so the difference between the
rows is attributable to the runtime and nothing else. `Codegen s` is the
one-off tracing and compilation cost, which the timed loop excludes.

| Model | Batch | Runtime | Compiled | Codegen s | p50 ms/img | img/s | vs eager |
|---|---|---|---|---|---|---|---|
| cutoutnet | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 19.85 | 46.3 | 121.80x |
| cutoutnet | 1 | pytorch-compile:inductor:default | yes | 20.5 | 2159.30 | 0.4 | 1.12x |
| cutoutnet | 1 | pytorch-eager | - | n/a | 2417.92 | 0.4 | 1.00x |
| cutoutnet | 8 | onnxruntime:CPUExecutionProvider | - | n/a | 11.20 | 84.8 | 24.33x |
| cutoutnet | 8 | pytorch-eager | - | n/a | 272.51 | 3.6 | 1.00x |
| cutoutnet | 8 | pytorch-compile:inductor:default | yes | 26.9 | 433.13 | 2.3 | 0.63x |
| u2net | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 427.77 | 2.2 | 9.66x |
| u2net | 1 | pytorch-eager | - | n/a | 4133.99 | 0.3 | 1.00x |
| u2netp | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 233.01 | 4.2 | 14.72x |
| u2netp | 1 | pytorch-eager | - | n/a | 3430.95 | 0.3 | 1.00x |

## Checkpoint provenance

| Model | Weights | SHA-256 |
|---|---|---|
| cutoutnet | `cutoutnet-small.pt` | `7877d96d498a0631...` |
| cutoutnet-base | `cutoutnet-base.pt` | `624c76b94b727e3b...` |
| cutoutnet-onnx | `cutoutnet-small.onnx` | `45540e5ef2f1e94d...` |
| cutoutnet-tiny | `cutoutnet-tiny.pt` | `3fe10d23bcf4a0b3...` |
| u2net | `u2net.pt` | `46c41386f871b54c...` |
| u2net-onnx | `u2net.onnx` | `8d10d2f3bb75ae3b...` |
| u2netp | `u2netp.pt` | `8a1241a929abde40...` |
| u2netp-onnx | `u2netp.onnx` | `309c8469258dda74...` |

## Per-stage timing breakdown

Where the wall clock actually goes for one image. Useful because the model
is frequently not the bottleneck - preprocessing and alpha refinement are
resolution-dependent while inference is fixed at the letterboxed size.

| Model | Preprocess ms | Inference ms | Postprocess ms | Refine ms | Cold start s |
|---|---|---|---|---|---|
| trivial-ones | 1.34 | 0.19 | 19.94 | 0.10 | 0.000 |
| trivial-center | 1.34 | 0.62 | 3.64 | 0.10 | 0.000 |
| classical-saliency | 0.40 | 3.76 | 9.58 | 0.09 | 0.000 |
| classical | 0.53 | 278.07 | 6.75 | 0.11 | 0.000 |
| classical-saliency-grabcut | 0.54 | 392.30 | 20.62 | 0.11 | 0.000 |
| cutoutnet-tiny | 4.53 | 1591.55 | 16.98 | 0.17 | 0.023 |
| cutoutnet | 2.46 | 2657.70 | 6.22 | 0.18 | 0.300 |
| cutoutnet-base | 0.89 | 2364.76 | 4.60 | 0.16 | 0.904 |
| cutoutnet-onnx ONNX/CPU | 0.77 | 29.06 | 8.88 | 0.10 | 0.033 |
| u2netp | 1.66 | 5000.58 | 4.30 | 0.15 | 0.246 |
| u2netp-onnx ONNX/CPU | 1.58 | 215.67 | 0.21 | 0.14 | 0.058 |
| u2net | 5.18 | 4438.47 | 14.61 | 0.16 | 2.202 |
| u2net-onnx ONNX/CPU | 5.23 | 421.10 | 0.19 | 0.14 | 0.227 |
| birefnet random-init | 4.49 | 1792.63 | 14.02 | 0.17 | 0.123 |
| cutoutnet compiled | 0.83 | 2180.69 | 14.35 | 0.18 | 0.218 |
| cutoutnet compiled | 2.20 | 391.25 | 1.32 | 0.09 | 0.267 |
| cutoutnet | 0.83 | 576.22 | 1.12 | 0.10 | 0.289 |
| cutoutnet | 0.82 | 294.05 | 1.06 | 0.09 | 0.235 |
| cutoutnet-onnx ONNX/CPU | 0.78 | 14.96 | 1.08 | 0.08 | 0.042 |

## Full accuracy metrics

| Model | IoU | Dice | MAE | F-beta | max F-beta | S-measure | Boundary F1 | BER | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial-ones | 0.3590 | 0.5087 | 0.6410 | 0.4142 | 0.4142 | 0.1856 | 0.0000 | 0.5000 | 0.3590 | 1.0000 |
| trivial-center | 0.4382 | 0.5989 | 0.2644 | 0.6167 | 0.6180 | 0.5698 | 0.0959 | 0.2729 | 0.6472 | 0.6155 |
| classical-saliency | 0.1508 | 0.2537 | 0.3772 | 0.3130 | 0.3373 | 0.3351 | 0.1325 | 0.4684 | 0.4352 | 0.2003 |
| classical | 0.6614 | 0.7637 | 0.1455 | 0.7744 | 0.7824 | 0.7489 | 0.5766 | 0.1465 | 0.8412 | 0.7812 |
| classical-saliency-grabcut | 0.1570 | 0.2531 | 0.3693 | 0.3145 | 0.3410 | 0.3435 | 0.2901 | 0.4612 | 0.4482 | 0.1995 |
| cutoutnet-tiny | 0.8241 | 0.8916 | 0.0693 | 0.8936 | 0.9161 | 0.8778 | 0.7564 | 0.0707 | 0.8994 | 0.9065 |
| cutoutnet | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| cutoutnet-base | 0.7252 | 0.8146 | 0.1111 | 0.8237 | 0.8673 | 0.8091 | 0.5468 | 0.1206 | 0.8540 | 0.8243 |
| cutoutnet-onnx ONNX/CPU | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| u2netp | 0.6380 | 0.6967 | 0.1388 | 0.7205 | 0.7597 | 0.7405 | 0.6306 | 0.1788 | 0.8912 | 0.6800 |
| u2netp-onnx ONNX/CPU | 0.6380 | 0.6967 | 0.1388 | 0.7205 | 0.7597 | 0.7405 | 0.6306 | 0.1788 | 0.8912 | 0.6800 |
| u2net | 0.6974 | 0.7543 | 0.1221 | 0.7758 | 0.8109 | 0.7792 | 0.7111 | 0.1535 | 0.8894 | 0.7406 |
| u2net-onnx ONNX/CPU | 0.6974 | 0.7543 | 0.1221 | 0.7758 | 0.8109 | 0.7792 | 0.7111 | 0.1535 | 0.8894 | 0.7406 |
| cutoutnet compiled | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| cutoutnet compiled | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| cutoutnet | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| cutoutnet | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |
| cutoutnet-onnx ONNX/CPU | 0.8544 | 0.9098 | 0.0573 | 0.9114 | 0.9283 | 0.8953 | 0.7965 | 0.0590 | 0.9154 | 0.9215 |

## Skipped / failed cases

| Case | Status | Reason |
|---|---|---|
| `u2net-lite-fp32` | skipped | No weights found for model 'u2net-lite'. Expected a checkpoint at /agent/cutout-ml/models/u2net/u2net-lite.pt. Train it here: `scripts/train_suite.sh u2net-lite` (~1 hour on 8 CPU cores). Alternatively the authors publis |

## Methodology

### Why single-run timings are misleading

A number like "37 ms" from one `time.perf_counter()` pair around one forward pass is
close to useless, for four reasons that all apply on the machine these numbers came
from:

1. **The first call is not representative.** PyTorch and oneDNN choose convolution
   algorithms lazily and cache them; onnxruntime builds an execution plan; CUDA creates
   a context and autotunes. The first inference is routinely 2-50x the steady-state
   cost. The harness runs warmup iterations and *discards* them, reporting the first
   iteration separately as `first_inference_ms` and model load as
   `cold_start_seconds`.

2. **CUDA is asynchronous.** `model(x)` returns before the GPU has finished. Timing it
   without `torch.cuda.synchronize()` measures the launch overhead - often producing
   "0.4 ms" for work that takes 20 ms. The harness synchronises before starting and
   before stopping the clock. (On the CPU-only machine used here this is a no-op, but
   the code path is the same one a GPU run would take.)

3. **The distribution has a long right tail.** Frequency scaling, other tenants on a
   shared cloud VM, page faults and GC produce outliers. A mean absorbs them; a p99
   exposes them. The harness reports p50/p95/p99/mean/stddev/min/max, and the stddev is
   the number to look at first: if it is large relative to p50, the machine was not
   quiet and no other figure in the row should be trusted.

4. **Batch size changes the meaning of "latency".** At batch 1 you measure
   *responsiveness*; at batch 8 you measure *throughput*, and per-image latency
   improves while the latency any individual request experiences gets worse. Both are
   reported, and per-image figures are always explicitly per-image.

### What is measured

- **Latency**: wall clock around `model.predict(tensor)` only - preprocessing and
  encoding are excluded here and reported separately in the stage breakdown, because
  they scale with the source image size rather than the model.
- **Throughput**: `batch_size / mean_latency`. For video, frames/s equals images/s
  because frames go through the identical path.
- **Peak RSS**: process resident set size after the run, from `psutil`. It includes the
  interpreter and loaded libraries (~250 MB for PyTorch), so compare *differences*
  between rows, not absolute values.
- **Peak VRAM**: `torch.cuda.max_memory_allocated`, or `null` off-GPU.
- **Cold start**: wall clock of `model.load()` - weight loading, device transfer and
  graph/session construction. This is what a scale-from-zero request pays.
- **Model size**: on-disk checkpoint/graph size, or the in-memory parameter size when
  weights are random.

### Runtimes compared, and how a failure is reported

Three runtimes execute the *same* trained weights:

- **PyTorch eager** - the reference. Convolutions already go through oneDNN, which is
  why the compiled speedup below is smaller than a GPU reader might expect.
- **`torch.compile` (Inductor)** - traces the graph and generates C++. Two things make
  this easy to report dishonestly, so both are handled explicitly: the first call costs
  seconds to tens of seconds (recorded separately as `Codegen s` and excluded from the
  timed loop), and the compile can *fail* at runtime on a machine without a C++
  toolchain. A failure falls back to eager and is printed as `FAILED` with the exception,
  never as a compiled row - which is why the Runtime column comes from the harness rather
  than from the model adapter.
- **ONNX Runtime (CPU execution provider)** - a genuinely different implementation of
  the same graph. The export is asserted to compute the same function to within 2e-3 in
  `tests/test_registry.py`, so a runtime row cannot silently be a different model.

**TensorRT is implemented but unmeasured.** The adapter exists and is type-checked, but
building an engine requires a CUDA GPU, and no row is published for it. That is a gap,
not a result.

### Accuracy metrics

`IoU`, `Dice`, `MAE`, `F-beta` (beta^2 = 0.3, the salient-object-detection
convention), `max/mean F-beta` swept over 255 thresholds, `S-measure`, `Boundary F1`
within a 3 px tolerance, `BER`, precision and recall. Definitions and the reasoning
for each are in `src/cutoutml/core/metrics.py`.

Two metrics deserve attention because they disagree usefully:

- **IoU vs Boundary F1.** A star-shaped mask with thin spikes can score high IoU (most
  of the area is right) while Boundary F1 collapses (the spikes are wrong). Boundary F1
  is what correlates with "does this cutout look good".
- **IoU vs MAE.** IoU thresholds at 0.5 and is blind to how confident the model is.
  MAE uses the continuous alpha, so a model that produces correct-but-mushy soft edges
  is punished by MAE and not by IoU. For matting, MAE is the more honest number.

### Calibration references

The table includes deliberately content-blind rows (`trivial-ones`, `trivial-center`).
They exist because IoU is only interpretable relative to what predicting *nothing*
achieves: on a set where the foreground covers ~35% of the frame, "predict everything"
already scores 0.35 IoU. Any row that does not clearly beat those has learned nothing.
`classical` (GrabCut from a centred rectangle) is the strongest non-learned baseline
and is the number a learned model has to beat to be worth its weights.

### Honest limitations

- **The eval set is synthetic.** See
  [`docs/decisions/ADR-004-synthetic-dataset.md`](decisions/ADR-004-synthetic-dataset.md).
  Absolute numbers here are **not comparable to published DUTS/DIS5K results**. The
  same harness runs unchanged on real data via `cutoutml.datasets.real` - pass
  `--dataset-root /path/to/DUTS`.
- **No GPU was available.** Every measurement is CPU-only. The fp16/TensorRT code paths
  are implemented and type-checked but unmeasured here; rows for them are absent rather
  than estimated.
- **Random-weight rows measure architecture cost, not quality.** U^2-Net and BiRefNet
  need pretrained checkpoints that could not be downloaded, so their rows show real
  latency with `n/a` accuracy.

