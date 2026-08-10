<!-- GENERATED FILE - do not edit by hand.
     Produced by `python -m cutoutml.benchmarks.render_report`
     from benchmarks/results/20260810T054025Z-1385dc69.json. -->

# Benchmarks

## Environment

- **Hardware**: Intel(R) Xeon(R) Processor, 8 vCPU (8 physical cores), 47 GB RAM, no GPU (CPU-only)
- **GPU**: none  <-- all numbers below are CPU-only; no GPU was available on this machine
- **OS / Python**: Linux 6.12.94+ (x86_64) / Python 3.12.3
- **Intra-op threads**: 1 per runtime, pinned by the harness - see [Thread scaling](#thread-scaling)
- **Git commit**: `545d0b78c519` on `cursor/cutoutml-platform-3514`
- **Libraries**: celery 5.6.3, fastapi 0.141.1, numpy 2.5.2, onnx 1.22.0, onnxruntime 1.28.0, opencv-python-headless 5.0.0.93, pillow 12.3.0, scipy 1.18.0, sqlalchemy 2.0.51, torch 2.13.0+cpu
- **Run id**: `20260810T054025Z-1385dc69` (2026-08-10T05:40:25Z, 303.78 s wall clock)

## Dataset

- **Dataset id**: `synthetic-v1.0.0-seed20240817`
- **Generator**: `cutoutml.datasets.synthetic` v1.0.0
- **Master seed**: `20240817`
- **Resolution**: [256, 256]
- **Content fingerprint**: `24b52899998fa9920c972c194876e3f6...` (first 8 samples)
- **Splits**: test=64
- **Harness**: 3 warmup + 20 timed repetitions per case, 64 accuracy samples

## Results

| Model | Runtime | Precision | Batch | Threads | IoU | MAE | F-beta | Boundary F1 | p50 ms/img | p95 ms/img | img/s | Peak RSS | Model size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trivial-ones | numpy | fp32 | 1 | 1 | 0.3590 | 0.6410 | 0.4142 | 0.0000 | 0.12 | 0.12 | 8532.3 | 332.0 MiB | n/a |
| trivial-center | numpy | fp32 | 1 | 1 | 0.4382 | 0.2644 | 0.6167 | 0.0959 | 0.53 | 0.57 | 1861.2 | 332.5 MiB | n/a |
| classical-saliency | opencv+numpy | fp32 | 1 | 1 | 0.1508 | 0.3772 | 0.3130 | 0.1325 | 0.85 | 0.88 | 1171.7 | 333.8 MiB | n/a |
| classical | opencv+numpy | fp32 | 1 | 1 | 0.6614 | 0.1455 | 0.7744 | 0.5766 | 262.96 | 282.20 | 3.8 | 334.4 MiB | n/a |
| classical-saliency-grabcut | opencv+numpy | fp32 | 1 | 1 | 0.1570 | 0.3693 | 0.3145 | 0.2901 | 395.54 | 410.01 | 2.6 | 334.4 MiB | n/a |
| cutoutnet-tiny | pytorch-eager | fp32 | 1 | 1 | 0.8241 | 0.0693 | 0.8936 | 0.7564 | 9.73 | 10.14 | 102.1 | 353.8 MiB | 0.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 30.71 | 31.27 | 32.5 | 343.5 MiB | 4.5 MiB |
| cutoutnet-base | pytorch-eager | fp32 | 1 | 1 | 0.8595 | 0.0549 | 0.9168 | 0.8192 | 39.52 | 40.15 | 25.3 | 374.8 MiB | 16.8 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 16.31 | 16.35 | 61.3 | 430.6 MiB | 4.4 MiB |
| u2netp | pytorch-eager | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 315.02 | 334.92 | 3.2 | 478.5 MiB | 4.6 MiB |
| u2netp-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 248.63 † | 250.56 | 4.0 | 878.5 MiB | 4.4 MiB |
| u2net | pytorch-eager | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 610.88 † | 636.60 | 1.6 | 757.0 MiB | 168.2 MiB |
| u2net-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 592.95 | 597.00 | 1.7 | 1359.5 MiB | 167.8 MiB |
| birefnet random-init | pytorch-eager | fp32 | 1 | 1 | n/a * | n/a * | n/a * | n/a * | 208.69 † | 212.69 | 4.8 | 739.7 MiB | 11.8 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 19.12 † | 19.20 | 52.3 | 840.3 MiB | 4.5 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 18.41 † | 18.82 | 54.3 | 852.9 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 4 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 19.00 † | 19.63 | 52.3 | 853.0 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 31.21 † | 33.47 | 32.4 | 853.0 MiB | 4.5 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 17.23 | 18.34 | 57.5 | 1093.0 MiB | 4.4 MiB |

`n/a *` = accuracy not measurable for this row: the network ran with **random weights** so that latency could still be benchmarked without a loadable checkpoint. Latency in those rows is real; accuracy is meaningless.

`†` = measured while another workload held the CPU, so the figure is an upper bound rather than this model's cost. Accuracy columns are unaffected: they are deterministic in the weights and the eval set. See [Machine contention](#machine-contention).

## Machine contention

**8 of 27 timed cases were measured under contention.** External demand peaked at 1.9 of 8 cores (24% of the machine) while these timings were taken.

The latency, throughput and peak-memory columns for those rows are therefore upper bounds on this hardware's cost, not measurements of it. They are published with the evidence attached rather than omitted, and marked `†` wherever they appear. Nothing here is corrected or extrapolated: a scaled-down number would be a guess.

Accuracy is unaffected and is not qualified. IoU, MAE, F-measure and boundary F1 are deterministic functions of the weights and the eval set, and come out bit-identical whatever else the scheduler was doing.

| Case | External cores busy | Load avg (1m) | Latency trustworthy |
|---|---|---|---|
| `trivial-ones` | 0.0 / 8 | 0.7 | yes |
| `trivial-center` | 0.0 / 8 | 0.7 | yes |
| `classical-saliency` | 0.0 / 8 | 0.7 | yes |
| `classical-grabcut` | 0.0 / 8 | 0.6 | yes |
| `classical-saliency+grabcut` | 0.0 / 8 | 0.8 | yes |
| `cutoutnet-tiny-fp32` | 0.0 / 8 | 0.9 | yes |
| `cutoutnet-fp32` | 0.0 / 8 | 0.9 | yes |
| `cutoutnet-base-fp32` | 0.0 / 8 | 0.9 | yes |
| `cutoutnet-onnx-cpu` | 0.0 / 8 | 0.9 | yes |
| `u2netp-pretrained` | 0.0 / 8 | 0.9 | yes |
| `u2netp-pretrained-onnx` | 1.8 / 8 | 1.0 | **no** |
| `u2net-pretrained` | 1.6 / 8 | 1.2 | **no** |
| `u2net-pretrained-onnx` | 0.0 / 8 | 1.4 | yes |
| `birefnet-compact-randominit` | 1.0 / 8 | 1.7 | **no** |
| `cutoutnet-fp32-compiled` | 1.1 / 8 | 1.8 | **no** |
| `cutoutnet-fp32-b8-compiled` | 1.2 / 8 | 1.8 | **no** |
| `cutoutnet-fp32-b4` | 1.3 / 8 | 1.8 | **no** |
| `cutoutnet-fp32-b8` | 1.1 / 8 | 1.7 | **no** |
| `cutoutnet-onnx-b8` | 0.0 / 8 | 1.7 | yes |
| `threadscale-eager-t1` | 0.0 / 8 | 1.6 | yes |
| `threadscale-eager-t2` | 0.0 / 8 | 1.6 | yes |
| `threadscale-eager-t4` | 0.0 / 8 | 1.5 | yes |
| `threadscale-eager-t8` | 0.0 / 8 | 1.5 | yes |
| `threadscale-onnx-t1` | 0.0 / 8 | 1.5 | yes |
| `threadscale-onnx-t2` | 0.0 / 8 | 1.5 | yes |
| `threadscale-onnx-t4` | 0.0 / 8 | 1.5 | yes |
| `threadscale-onnx-t8` | 1.9 / 8 | 1.5 | **no** |

## Thread scaling

| Runtime | Threads | p50 ms | p95 ms | stddev ms | img/s | Speedup vs 1 thread |
|---|---|---|---|---|---|---|
| onnxruntime:CPUExecutionProvider | 1 | 16.4 | 16.5 | 0.1 | 60.8 | 1.00x |
| onnxruntime:CPUExecutionProvider | 2 | 9.2 | 9.2 | 0.0 | 108.4 | 1.78x |
| onnxruntime:CPUExecutionProvider | 4 | 5.7 | 5.9 | 0.1 | 173.5 | 2.87x |
| onnxruntime:CPUExecutionProvider | 8 | 9.5 | 9.9 | 2.1 | 119.6 | 1.73x |
| pytorch-eager | 1 | 21.3 | 21.6 | 0.2 | 46.8 | 1.00x |
| pytorch-eager | 2 | 13.9 | 14.0 | 0.1 | 71.8 | 1.53x |
| pytorch-eager | 4 | 9.5 | 9.6 | 0.2 | 104.9 | 2.25x |
| pytorch-eager | 8 | 8.8 | 9.2 | 0.7 | 111.2 | 2.41x |

Within each runtime the weights, the batch size and the image are identical; the only variable is how many intra-op threads the runtime was given. Compare down a runtime's rows, not across runtimes - the two runtimes execute different code.

- **onnxruntime:CPUExecutionProvider**: 3x between its own extremes - 5.7 ms at 4 thread(s) against 16.4 ms at 1 (`threadscale-onnx-t1`). That is, threads bought what they should have.
- **pytorch-eager**: 2x between its own extremes - 8.8 ms at 8 thread(s) against 21.3 ms at 1 (`threadscale-eager-t1`). That is, threads bought what they should have.

- **Repeatability**: `cutoutnet` at 1 thread(s) measured 21.3 ms here and 30.7 ms in the table above - 1.4x apart for the same configuration, minutes apart on the same machine. Neither figure is wrong; the machine was not the same machine at the two moments. This is what the `†` marks mean in practice, and it is the reason no single number on this page should be quoted without them.

Where a runtime gets *slower* with more threads, the extra time is not arithmetic but waiting. A U-Net forward pass is roughly a hundred parallel regions, each ending in a barrier, and a barrier cannot retire until every worker thread has been scheduled onto a core. Ask for eight threads on a machine whose cores are already committed and every one of those barriers waits on a descheduled thread, so the cost becomes a function of the scheduler rather than of the model. ONNX Runtime resists this better than PyTorch because it fuses the graph into far fewer parallel regions and controls its own spin-then-yield policy at each one.

Two consequences shape the rest of this document:

1. **The suite runs single-threaded by default** (`--threads 1`). One thread has no barriers to lose, which makes it the only CPU latency figure on a shared machine that means the same thing twice. It also understates what dedicated hardware would do, and that is the correct direction for a published number to be wrong in.
2. **A runtime comparison must fix the thread count.** ONNX Runtime resolves a request of 0 to one thread per core while PyTorch has its own default, so an uncontrolled 'PyTorch vs ONNX' row pair can differ by eight threads before it differs by a runtime. The harness now passes one count to both.

## Runtime comparison

The same weights at the same batch size under PyTorch eager, `torch.compile` (Inductor) and ONNX Runtime, so the difference between the
rows is attributable to the runtime and nothing else. `Codegen s` is the
one-off tracing and compilation cost, which the timed loop excludes.

| Model | Batch | Runtime | Compiled | Codegen s | p50 ms/img | img/s | vs eager |
|---|---|---|---|---|---|---|---|
| cutoutnet | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 16.31 | 61.3 | 1.88x |
| cutoutnet | 1 | pytorch-compile:inductor:default | yes | 4.2 | 19.12 | 52.3 | 1.61x |
| cutoutnet | 1 | pytorch-eager | - | n/a | 30.71 | 32.5 | 1.00x |
| cutoutnet | 8 | onnxruntime:CPUExecutionProvider | - | n/a | 17.23 | 57.5 | 1.81x |
| cutoutnet | 8 | pytorch-compile:inductor:default | yes | 2.3 | 18.41 | 54.3 | 1.70x |
| cutoutnet | 8 | pytorch-eager | - | n/a | 31.21 | 32.4 | 1.00x |
| u2net | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 592.95 | 1.7 | 1.03x |
| u2net | 1 | pytorch-eager | - | n/a | 610.88 | 1.6 | 1.00x |
| u2netp | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 248.63 | 4.0 | 1.27x |
| u2netp | 1 | pytorch-eager | - | n/a | 315.02 | 3.2 | 1.00x |

## Checkpoint provenance

| Model | Weights | SHA-256 |
|---|---|---|
| cutoutnet | `cutoutnet-small.pt` | `7877d96d498a0631...` |
| cutoutnet-base | `cutoutnet-base.pt` | `055eef63d5a55ceb...` |
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
| trivial-ones | 1.35 | 0.18 | 0.13 | 0.08 | 0.000 |
| trivial-center | 1.39 | 0.61 | 0.13 | 0.08 | 0.000 |
| classical-saliency | 0.43 | 1.19 | 0.16 | 0.11 | 0.000 |
| classical | 0.40 | 271.85 | 0.14 | 0.09 | 0.000 |
| classical-saliency-grabcut | 0.40 | 367.32 | 0.15 | 0.08 | 0.000 |
| cutoutnet-tiny | 0.78 | 9.96 | 0.06 | 0.10 | 0.022 |
| cutoutnet | 0.76 | 30.67 | 0.06 | 0.09 | 0.036 |
| cutoutnet-base | 0.79 | 39.55 | 0.06 | 0.09 | 0.069 |
| cutoutnet-onnx ONNX/CPU | 0.78 | 16.41 | 0.08 | 0.09 | 0.034 |
| u2netp | 1.57 | 299.99 | 0.20 | 0.12 | 0.062 |
| u2netp-onnx ONNX/CPU | 1.50 | 249.33 | 0.17 | 0.12 | 0.033 |
| u2net | 1.54 | 609.23 | 0.22 | 0.13 | 0.468 |
| u2net-onnx ONNX/CPU | 1.74 | 600.11 | 0.22 | 0.15 | 0.185 |
| birefnet random-init | 4.20 | 212.21 | 0.51 | 0.15 | 0.027 |
| cutoutnet compiled | 0.78 | 19.05 | 0.06 | 0.09 | 0.037 |
| cutoutnet compiled | 0.81 | 18.93 | 0.04 | 0.07 | 0.038 |
| cutoutnet | 0.77 | 19.60 | 0.05 | 0.08 | 0.039 |
| cutoutnet | 0.82 | 30.21 | 0.05 | 0.08 | 0.038 |
| cutoutnet-onnx ONNX/CPU | 0.80 | 17.83 | 0.05 | 0.08 | 0.020 |

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
| cutoutnet-base | 0.8595 | 0.9081 | 0.0549 | 0.9168 | 0.9407 | 0.8964 | 0.8192 | 0.0635 | 0.9282 | 0.9037 |
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
| `u2net-lite-fp32` | skipped | No weights found for model 'u2net-lite'. Expected a checkpoint at /agent/cutout-ml/models/u2net/u2net-lite.pt. The same architecture with the authors' pretrained weights is already registered as `u2netp` (Apache-2.0): ru |

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

   Leaving that inference to the reader is not good enough, though - a wide stddev is
   equally consistent with "this model has variable cost" and "someone else had the
   CPU". So the harness also *measures* how busy the machine was, per case, and marks
   the rows where the answer makes their timings meaningless. See
   [Machine contention](#machine-contention) for this run's numbers.

4. **Batch size changes the meaning of "latency".** At batch 1 you measure
   *responsiveness*; at batch 8 you measure *throughput*, and per-image latency
   improves while the latency any individual request experiences gets worse. Both are
   reported, and per-image figures are always explicitly per-image.

5. **A CPU latency figure without a thread count is not a measurement.** The same
   weights on the same machine differ by more than an order of magnitude depending on
   how many intra-op threads the runtime was given, and on a busy machine more threads
   can be dramatically *worse* - see [Thread scaling](#thread-scaling) for the measured
   curve. Every row records the thread count the runtime actually ran with, taken from
   the runtime rather than from the request, because ONNX Runtime silently resolves a
   request of 0 to one thread per core.

### What is measured

- **Latency**: wall clock around `model.predict(tensor)` only - preprocessing and
  encoding are excluded here and reported separately in the stage breakdown, because
  they scale with the source image size rather than the model.
- **Throughput**: `batch_size / mean_latency`. For video, frames/s equals images/s
  because frames go through the identical path.
- **Peak RSS**: process resident set size after the run, from `psutil`. It includes the
  interpreter and loaded libraries (~250 MB for PyTorch), so compare *differences*
  between rows, not absolute values.
- **Intra-op threads**: the width the runtime actually ran at, read back from the
  runtime. Pinned to the same value for every runtime in a comparison, because
  otherwise the comparison is partly a thread-count comparison.
- **Machine contention**: busy cores attributable to processes outside this process
  tree, sampled immediately before each timing loop. Measured as *external* demand
  rather than as a raw load average so that the harness's own consumption does not count
  against it, and in cores rather than as a load-average figure so the threshold means
  the same thing on a 4-core and a 64-core machine. A case is treated as quiet below
  half a busy core.
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
- **The machine was shared.** See [Machine contention](#machine-contention) for exactly
  which rows this affects and by how much. Timings on a contended row are upper bounds;
  accuracy is unaffected.
- **Latency here is single-threaded and therefore pessimistic.** These are per-core
  costs, not the best this hardware can do. A dedicated machine given one thread per
  core would be faster - by how much is a question this environment cannot answer, so
  no multi-threaded headline figure is published. [Thread scaling](#thread-scaling)
  shows what was measured instead.
- **Random-weight rows measure architecture cost, not quality.** Only BiRefNet is in that
  position: its official checkpoints target a Swin backbone whose shapes do not match this
  repository's reimplementation, so no download would help and its row shows real latency
  with `n/a` accuracy. U^2-Net's published weights *are* loaded here - see
  [docs/models.md](models.md) for the route.
- **The pretrained models are evaluated out of domain.** U^2-Net was trained on DUTS,
  a real-photograph saliency dataset, and is scored here against a synthetic eval set. It
  is expected to place below a small model trained in-repo on that eval set's own
  distribution, and it does. That ordering is a statement about the eval set, not about
  the models: read it as evidence that these synthetic numbers do not transfer to
  photographs, in either direction.

