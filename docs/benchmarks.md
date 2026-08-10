<!-- GENERATED FILE - do not edit by hand.
     Produced by `python -m cutoutml.benchmarks.render_report`
     from benchmarks/results/20260810T071109Z-e0c34c05.json. -->

# Benchmarks

## Environment

- **Hardware**: Intel(R) Xeon(R) Processor, 8 vCPU (8 physical cores), 47 GB RAM, no GPU (CPU-only)
- **GPU**: none  <-- all numbers below are CPU-only; no GPU was available on this machine
- **OS / Python**: Linux 6.12.94+ (x86_64) / Python 3.12.3
- **Intra-op threads**: 1 per runtime, pinned by the harness - see [Thread scaling](#thread-scaling)
- **Git commit**: `33263e89e939` on `cursor/cutoutml-platform-3514`
- **Libraries**: celery 5.6.3, fastapi 0.141.1, numpy 2.5.2, onnx 1.22.0, onnxruntime 1.28.0, opencv-python-headless 5.0.0.93, pillow 12.3.0, scipy 1.18.0, sqlalchemy 2.0.51, torch 2.13.0+cpu
- **Run id**: `20260810T071109Z-e0c34c05` (2026-08-10T07:11:09Z, 299.13 s wall clock)

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
| trivial-ones | numpy | fp32 | 1 | 1 | 0.3590 | 0.6410 | 0.4142 | 0.0000 | 0.11 | 0.12 | 8772.9 | 331.7 MiB | n/a |
| trivial-center | numpy | fp32 | 1 | 1 | 0.4382 | 0.2644 | 0.6167 | 0.0959 | 0.60 | 0.63 | 1650.9 | 332.1 MiB | n/a |
| classical-saliency | opencv+numpy | fp32 | 1 | 1 | 0.1508 | 0.3772 | 0.3130 | 0.1325 | 0.94 | 0.99 | 1055.3 | 333.5 MiB | n/a |
| classical | opencv+numpy | fp32 | 1 | 1 | 0.6614 | 0.1455 | 0.7744 | 0.5766 | 263.17 | 283.14 | 3.8 | 334.2 MiB | n/a |
| classical-saliency-grabcut | opencv+numpy | fp32 | 1 | 1 | 0.1570 | 0.3693 | 0.3145 | 0.2901 | 396.96 | 411.30 | 2.6 | 334.2 MiB | n/a |
| cutoutnet-tiny | pytorch-eager | fp32 | 1 | 1 | 0.8241 | 0.0693 | 0.8936 | 0.7564 | 10.02 | 11.00 | 98.2 | 353.5 MiB | 0.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 31.37 | 31.78 | 31.6 | 346.7 MiB | 4.5 MiB |
| cutoutnet-base | pytorch-eager | fp32 | 1 | 1 | 0.8615 | 0.0508 | 0.9183 | 0.8247 | 40.24 | 40.34 | 24.8 | 376.0 MiB | 16.8 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 16.66 | 16.72 | 60.0 | 428.9 MiB | 4.4 MiB |
| u2netp | pytorch-eager | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 276.65 | 301.13 | 3.6 | 455.8 MiB | 4.6 MiB |
| u2netp-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 250.14 | 250.86 | 4.0 | 905.8 MiB | 4.4 MiB |
| u2net | pytorch-eager | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 612.60 | 616.12 | 1.6 | 775.1 MiB | 168.2 MiB |
| u2net-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 594.36 | 595.57 | 1.7 | 1368.6 MiB | 167.8 MiB |
| birefnet random-init | pytorch-eager | fp32 | 1 | 1 | n/a * | n/a * | n/a * | n/a * | 223.98 | 228.57 | 4.5 | 748.8 MiB | 11.8 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 18.87 | 19.09 | 52.9 | 849.4 MiB | 4.5 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 17.84 | 17.89 | 56.1 | 862.3 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 4 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 19.19 | 19.33 | 52.0 | 862.4 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 28.74 | 29.16 | 34.7 | 862.4 MiB | 4.5 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 16.83 | 16.86 | 59.4 | 1102.4 MiB | 4.4 MiB |

`n/a *` = accuracy not measurable for this row: the network ran with **random weights** so that latency could still be benchmarked without a loadable checkpoint. Latency in those rows is real; accuracy is meaningless.

## Machine contention

Every case was measured on a quiet machine: external demand never exceeded 0.0 of 8 cores. The latency figures are this hardware's.

## Thread scaling

| Runtime | Threads | p50 ms | p95 ms | stddev ms | img/s | Speedup vs 1 thread |
|---|---|---|---|---|---|---|
| onnxruntime:CPUExecutionProvider | 1 | 16.4 | 16.7 | 0.1 | 60.9 | 1.00x |
| onnxruntime:CPUExecutionProvider | 2 | 9.2 | 9.6 | 0.2 | 107.8 | 1.79x |
| onnxruntime:CPUExecutionProvider | 4 | 5.9 | 6.5 | 0.3 | 166.5 | 2.75x |
| onnxruntime:CPUExecutionProvider | 8 | 4.6 | 5.1 | 0.2 | 216.0 | 3.55x |
| pytorch-eager | 1 | 20.7 | 20.8 | 0.1 | 48.3 | 1.00x |
| pytorch-eager | 2 | 13.4 | 14.1 | 0.3 | 73.9 | 1.54x |
| pytorch-eager | 4 | 9.2 | 9.4 | 0.1 | 108.3 | 2.24x |
| pytorch-eager | 8 | 7.4 | 8.0 | 0.7 | 131.3 | 2.80x |

Within each runtime the weights, the batch size and the image are identical; the only variable is how many intra-op threads the runtime was given. Compare down a runtime's rows, not across runtimes - the two runtimes execute different code.

- **onnxruntime:CPUExecutionProvider**: 4x between its own extremes - 4.6 ms at 8 thread(s) against 16.4 ms at 1 (`threadscale-onnx-t1`). That is, threads bought what they should have.
- **pytorch-eager**: 3x between its own extremes - 7.4 ms at 8 thread(s) against 20.7 ms at 1 (`threadscale-eager-t1`). That is, threads bought what they should have.

- **Repeatability**: `cutoutnet` at 1 thread(s) measured 20.7 ms here and 31.4 ms in the table above - 1.5x apart for the same configuration. Both rows sampled an idle machine, so contention does not account for it. What differs is where each sat in the run: latency here depends measurably on what ran earlier in the same process. `benchmarks/order_effect.py` isolates that effect - timing this configuration after a larger model, on a quiet machine, reproduces the faster figure with a standard deviation under 0.2 ms - and its result is archived under `benchmarks/results/experiments/`. Compare rows within this sweep, which ran back to back, rather than against the table above.

No runtime regressed with more threads in this run, so nothing here needs explaining away - but the mechanism that makes wide runs unsafe to publish from a shared machine is worth stating, because it is why the harness pins a count. A U-Net forward pass is roughly a hundred parallel regions, each ending in a barrier, and a barrier cannot retire until every worker thread has been scheduled onto a core. Ask for eight threads on a machine whose cores are already committed and every one of those barriers waits on a descheduled thread, so the cost becomes a function of the scheduler rather than of the model. ONNX Runtime resists this better than PyTorch because it fuses the graph into far fewer parallel regions and controls its own spin-then-yield policy at each one. Earlier runs of this same suite, taken while a neighbouring job held all eight cores, show exactly that regression; they are kept in `benchmarks/results/` for the comparison.

Two consequences shape the rest of this document:

1. **The suite runs single-threaded by default** (`--threads 1`). Threads did pay off in this run, so the default is not a claim that they cannot: it is that a one-thread figure is the only one that does not silently encode the core count of whichever machine took it, and the only one whose reproducibility does not depend on that machine staying idle. It understates what dedicated hardware would do, and that is the correct direction for a published number to be wrong in.
2. **A runtime comparison must fix the thread count.** ONNX Runtime resolves a request of 0 to one thread per core while PyTorch has its own default, so an uncontrolled 'PyTorch vs ONNX' row pair can differ by eight threads before it differs by a runtime. The harness now passes one count to both.

## Runtime comparison

The same weights at the same batch size under PyTorch eager, `torch.compile` (Inductor) and ONNX Runtime, so the difference between the
rows is attributable to the runtime and nothing else. `Codegen s` is the
one-off tracing and compilation cost, which the timed loop excludes.

| Model | Batch | Runtime | Compiled | Codegen s | p50 ms/img | img/s | vs eager |
|---|---|---|---|---|---|---|---|
| cutoutnet | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 16.66 | 60.0 | 1.88x |
| cutoutnet | 1 | pytorch-compile:inductor:default | yes | 4.2 | 18.87 | 52.9 | 1.66x |
| cutoutnet | 1 | pytorch-eager | - | n/a | 31.37 | 31.6 | 1.00x |
| cutoutnet | 8 | onnxruntime:CPUExecutionProvider | - | n/a | 16.83 | 59.4 | 1.71x |
| cutoutnet | 8 | pytorch-compile:inductor:default | yes | 2.3 | 17.84 | 56.1 | 1.61x |
| cutoutnet | 8 | pytorch-eager | - | n/a | 28.74 | 34.7 | 1.00x |
| u2net | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 594.36 | 1.7 | 1.03x |
| u2net | 1 | pytorch-eager | - | n/a | 612.60 | 1.6 | 1.00x |
| u2netp | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 250.14 | 4.0 | 1.11x |
| u2netp | 1 | pytorch-eager | - | n/a | 276.65 | 3.6 | 1.00x |

## Checkpoint provenance

| Model | Weights | SHA-256 |
|---|---|---|
| cutoutnet | `cutoutnet-small.pt` | `7877d96d498a0631...` |
| cutoutnet-base | `cutoutnet-base.pt` | `8c7acbb0b825d81c...` |
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
| trivial-ones | 1.39 | 0.18 | 0.14 | 0.07 | 0.000 |
| trivial-center | 1.34 | 0.62 | 0.14 | 0.07 | 0.000 |
| classical-saliency | 0.39 | 0.96 | 0.13 | 0.07 | 0.000 |
| classical | 0.40 | 272.52 | 0.16 | 0.08 | 0.000 |
| classical-saliency-grabcut | 0.39 | 367.02 | 0.15 | 0.08 | 0.000 |
| cutoutnet-tiny | 0.79 | 10.19 | 0.06 | 0.09 | 0.026 |
| cutoutnet | 0.76 | 31.77 | 0.06 | 0.09 | 0.042 |
| cutoutnet-base | 0.76 | 40.42 | 0.06 | 0.09 | 0.069 |
| cutoutnet-onnx ONNX/CPU | 0.77 | 16.80 | 0.08 | 0.09 | 0.043 |
| u2netp | 1.53 | 288.48 | 0.19 | 0.11 | 0.068 |
| u2netp-onnx ONNX/CPU | 1.54 | 249.87 | 0.13 | 0.11 | 0.035 |
| u2net | 1.63 | 610.89 | 0.18 | 0.11 | 0.583 |
| u2net-onnx ONNX/CPU | 1.56 | 594.56 | 0.19 | 0.12 | 0.264 |
| birefnet random-init | 3.97 | 224.96 | 0.46 | 0.14 | 0.027 |
| cutoutnet compiled | 0.80 | 19.08 | 0.06 | 0.17 | 0.037 |
| cutoutnet compiled | 0.77 | 17.85 | 0.04 | 0.07 | 0.038 |
| cutoutnet | 0.81 | 19.32 | 0.04 | 0.09 | 0.039 |
| cutoutnet | 0.83 | 28.65 | 0.05 | 0.08 | 0.038 |
| cutoutnet-onnx ONNX/CPU | 0.76 | 16.83 | 0.03 | 0.07 | 0.020 |

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
| cutoutnet-base | 0.8615 | 0.9103 | 0.0508 | 0.9183 | 0.9423 | 0.8987 | 0.8247 | 0.0575 | 0.9382 | 0.9115 |
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
close to useless, for six reasons that all apply on the machine these numbers came
from:

1. **The first call is not representative.** PyTorch and oneDNN choose convolution
   algorithms lazily and cache them; onnxruntime builds an execution plan; CUDA creates
   a context and autotunes. Across this run's cases the first inference cost 1.0-2.9x
   the steady-state median. On a GPU the multiple is larger, because context creation
   and autotuning happen there too, but this machine has no GPU and that figure is not
   measured here. The harness runs warmup iterations and *discards* them, reporting the
   first iteration separately as `first_inference_ms` and model load as
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
   weights on the same machine can differ by more than an order of magnitude depending
   on how many intra-op threads the runtime was given, and which direction they move in
   is a property of the machine rather than of the model - see
   [Thread scaling](#thread-scaling) for the curve this run measured and
   `benchmarks/results/README.md` for an archived run of the same sweep in which more
   threads were dramatically worse. Every row records the thread count the runtime
   actually ran with, taken from the runtime rather than from the request, because ONNX
   Runtime silently resolves a request of 0 to one thread per core.

6. **Position in the run changes the number, and it is not noise.** This suite measures
   `cutoutnet` eager at batch 1 on 1 thread twice - once in the main table, once as the
   matching rung of the thread sweep - and the two land about 1.5x apart, each with a
   standard deviation inside 4% of its own median. That is not contention - the harness
   sampled the load before both timing loops and both were idle - and it is not warmup,
   which is discarded from both. It is where each sat in the run: latency here depends
   on what ran earlier in the same process, and `benchmarks/order_effect.py` times the
   identical configuration after each of several preludes and finds that one particular
   earlier model reproduces the faster figure while others, including a much more
   expensive one, change nothing.

   The mechanism is not established here, and two plausible ones were tested and
   rejected: pre-faulting up to 1 GiB of heap before the timed loop changes nothing, and
   running a compiled case first changes nothing. What follows for a reader is concrete
   regardless of the cause - **compare rows that ran near each other**, which is why the
   thread sweep is a self-contained block rather than figures scattered through the main
   table, and treat a 1.5x agreement between two distant rows as the floor on this
   harness's cross-row precision. The experiment's own output, with the per-arm load
   samples that rule out contention, is archived in `benchmarks/results/experiments/`.

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

