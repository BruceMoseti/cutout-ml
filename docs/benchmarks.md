<!-- GENERATED FILE - do not edit by hand.
     Produced by `python -m cutoutml.benchmarks.render_report`
     from benchmarks/results/20260810T155155Z-7fc50b03.json. -->

# Benchmarks

## Environment

- **Hardware**: Intel(R) Xeon(R) Processor, 8 vCPU (8 physical cores), 47 GB RAM, no GPU (CPU-only)
- **GPU**: none  <-- all numbers below are CPU-only; no GPU was available on this machine
- **OS / Python**: Linux 6.12.94+ (x86_64) / Python 3.12.3
- **Intra-op threads**: 1 per runtime, pinned by the harness - see [Thread scaling](#thread-scaling)
- **Git commit**: `ba1bbda1c71c` on `cursor/cutoutml-platform-3514`
- **Libraries**: celery 5.6.3, fastapi 0.141.1, numpy 2.5.2, onnx 1.22.0, onnxruntime 1.28.0, opencv-python-headless 5.0.0.93, pillow 12.3.0, scipy 1.18.0, sqlalchemy 2.0.51, torch 2.13.0+cpu
- **Run id**: `20260810T155155Z-7fc50b03` (2026-08-10T15:51:55Z, 295.53 s wall clock)

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
| trivial-ones | numpy | fp32 | 1 | 1 | 0.3590 | 0.6410 | 0.4142 | 0.0000 | 0.12 | 0.13 | 8358.1 | 332.5 MiB | n/a |
| trivial-center | numpy | fp32 | 1 | 1 | 0.4382 | 0.2644 | 0.6167 | 0.0959 | 0.64 | 0.69 | 1535.4 | 333.0 MiB | n/a |
| classical-saliency | opencv+numpy | fp32 | 1 | 1 | 0.1508 | 0.3772 | 0.3130 | 0.1325 | 0.91 | 0.95 | 1103.6 | 334.3 MiB | n/a |
| classical | opencv+numpy | fp32 | 1 | 1 | 0.6503 | 0.1522 | 0.7677 | 0.5782 | 261.52 | 281.58 | 3.8 | 334.8 MiB | n/a |
| classical-saliency-grabcut | opencv+numpy | fp32 | 1 | 1 | 0.1574 | 0.3695 | 0.3137 | 0.2906 | 380.49 | 408.82 | 2.6 | 334.8 MiB | n/a |
| cutoutnet-tiny | pytorch-eager | fp32 | 1 | 1 | 0.8241 | 0.0693 | 0.8936 | 0.7564 | 10.03 | 10.76 | 98.6 | 357.3 MiB | 0.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 31.33 | 31.50 | 31.9 | 346.6 MiB | 4.5 MiB |
| cutoutnet-base | pytorch-eager | fp32 | 1 | 1 | 0.8615 | 0.0508 | 0.9183 | 0.8247 | 40.17 | 40.37 | 24.9 | 375.8 MiB | 16.8 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 16.65 | 17.42 | 59.6 | 428.8 MiB | 4.4 MiB |
| u2netp | pytorch-eager | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 263.03 | 273.14 | 3.8 | 480.7 MiB | 4.6 MiB |
| u2netp-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6380 | 0.1388 | 0.7205 | 0.6306 | 249.51 | 251.26 | 4.0 | 905.7 MiB | 4.4 MiB |
| u2net | pytorch-eager | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 623.39 | 627.02 | 1.6 | 748.1 MiB | 168.2 MiB |
| u2net-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 1 | 1 | 0.6974 | 0.1221 | 0.7758 | 0.7111 | 594.83 | 596.49 | 1.7 | 1350.6 MiB | 167.8 MiB |
| birefnet random-init | pytorch-eager | fp32 | 1 | 1 | n/a * | n/a * | n/a * | n/a * | 204.77 | 205.50 | 4.9 | 730.8 MiB | 11.8 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 1 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 18.58 | 18.96 | 53.7 | 831.3 MiB | 4.5 MiB |
| cutoutnet compiled | pytorch-compile:inductor:default | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 17.15 | 18.90 | 57.3 | 844.0 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 4 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 18.01 | 18.08 | 55.5 | 844.1 MiB | 4.5 MiB |
| cutoutnet | pytorch-eager | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 26.39 | 26.99 | 37.8 | 844.1 MiB | 4.5 MiB |
| cutoutnet-onnx ONNX/CPU | onnxruntime:CPUExecutionProvider | fp32 | 8 | 1 | 0.8544 | 0.0573 | 0.9114 | 0.7965 | 17.01 | 17.40 | 58.6 | 1084.1 MiB | 4.4 MiB |

`n/a *` = accuracy not measurable for this row: the network ran with **random weights** so that latency could still be benchmarked without a loadable checkpoint. Latency in those rows is real; accuracy is meaningless.

## Machine contention

Every case was measured on a quiet machine: external demand never exceeded 0.05 of 8 cores. The latency figures are this hardware's.

## Thread scaling

| Runtime | Threads | p50 ms | p95 ms | stddev ms | img/s | Speedup vs 1 thread |
|---|---|---|---|---|---|---|
| onnxruntime:CPUExecutionProvider | 1 | 16.3 | 16.4 | 0.1 | 61.4 | 1.00x |
| onnxruntime:CPUExecutionProvider | 2 | 9.2 | 10.1 | 0.4 | 107.0 | 1.77x |
| onnxruntime:CPUExecutionProvider | 4 | 5.6 | 5.7 | 0.1 | 177.5 | 2.90x |
| onnxruntime:CPUExecutionProvider | 8 | 5.5 | 6.1 | 0.4 | 179.6 | 2.95x |
| pytorch-eager | 1 | 19.8 | 20.7 | 0.4 | 50.1 | 1.00x |
| pytorch-eager | 2 | 13.1 | 13.2 | 0.1 | 76.3 | 1.51x |
| pytorch-eager | 4 | 9.1 | 9.4 | 0.1 | 109.9 | 2.18x |
| pytorch-eager | 8 | 7.3 | 7.7 | 0.2 | 136.1 | 2.70x |

Within each runtime the weights, the batch size and the image are identical; the only variable is how many intra-op threads the runtime was given. Compare down a runtime's rows, not across runtimes - the two runtimes execute different code.

- **onnxruntime:CPUExecutionProvider**: 3x between its own extremes - 5.5 ms at 8 thread(s) against 16.3 ms at 1 (`threadscale-onnx-t1`). That is, threads bought what they should have.
- **pytorch-eager**: 3x between its own extremes - 7.3 ms at 8 thread(s) against 19.8 ms at 1 (`threadscale-eager-t1`). That is, threads bought what they should have.

- **Repeatability**: `cutoutnet` at 1 thread(s) measured 19.8 ms here and 31.3 ms in the table above - 1.6x apart for the same configuration. Both rows sampled an idle machine, so contention does not account for it. What differs is where each sat in the run: latency here depends measurably on what ran earlier in the same process. `benchmarks/order_effect.py` isolates that effect - timing this configuration after a larger model, on a quiet machine, reproduces the faster figure with a standard deviation under 0.2 ms - and its result is archived under `benchmarks/results/experiments/`. Compare rows within this sweep, which ran back to back, rather than against the table above.

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
| cutoutnet | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 16.65 | 59.6 | 1.88x |
| cutoutnet | 1 | pytorch-compile:inductor:default | yes | 4.2 | 18.58 | 53.7 | 1.69x |
| cutoutnet | 1 | pytorch-eager | - | n/a | 31.33 | 31.9 | 1.00x |
| cutoutnet | 8 | onnxruntime:CPUExecutionProvider | - | n/a | 17.01 | 58.6 | 1.55x |
| cutoutnet | 8 | pytorch-compile:inductor:default | yes | 2.3 | 17.15 | 57.3 | 1.54x |
| cutoutnet | 8 | pytorch-eager | - | n/a | 26.39 | 37.8 | 1.00x |
| u2net | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 594.83 | 1.7 | 1.05x |
| u2net | 1 | pytorch-eager | - | n/a | 623.39 | 1.6 | 1.00x |
| u2netp | 1 | onnxruntime:CPUExecutionProvider | - | n/a | 249.51 | 4.0 | 1.05x |
| u2netp | 1 | pytorch-eager | - | n/a | 263.03 | 3.8 | 1.00x |

## Checkpoint provenance

A path is not provenance - the next training run overwrites the file - so each accuracy row is tied to the digest of the weights it was measured against.

One caveat applies to a checkpoint that was *converted* rather than trained here, which is how both U^2-Net files are produced from the published ONNX releases. A digest here that no longer matches your local file does not necessarily mean the row is stale: an earlier converter embedded the conversion timestamp in the checkpoint, so it identified a conversion rather than a set of weights. The timestamp now lives in `models/conversions/<model>.json` and the converter is reproducible - two conversions of one graph to the same destination name produce byte-identical checkpoints, which `test_two_conversions_of_one_graph_produce_byte_identical_checkpoints` asserts. Only rows measured before that change are affected, and re-measuring settles it: both models against re-derived files reproduced their published IoU to all sixteen digits. `benchmarks/results/README.md` names the two digests concerned.

Comparing digests has one trap worth knowing, because it is easy to conclude the converter is non-deterministic when it is not. `torch.save` writes a zip whose internal archive name comes from the destination path, so two checkpoints are only digest-comparable when written to the same file name. Converted to `a.pt` and `b.pt` the same weights hash differently; converted to `a/w.pt` and `b/w.pt` they hash identically. The identity that avoids the question altogether is `source_sha256`, the digest of the ONNX the checkpoint came from, which is pinned in `download_weights.py`, listed in `NOTICE`, recorded in the conversion record, and published in the table below for runs measured after it was surfaced there.

| Model | Weights | SHA-256 | Converted from |
|---|---|---|---|
| cutoutnet | `cutoutnet-small.pt` | `7877d96d498a0631...` | trained in-repo |
| cutoutnet-base | `cutoutnet-base.pt` | `8c7acbb0b825d81c...` | trained in-repo |
| cutoutnet-onnx | `cutoutnet-small.onnx` | `45540e5ef2f1e94d...` | trained in-repo |
| cutoutnet-tiny | `cutoutnet-tiny.pt` | `3fe10d23bcf4a0b3...` | trained in-repo |
| u2net | `u2net.pt` | `26a059bb7fb26a94...` | `8d10d2f3bb75ae3b...` |
| u2net-onnx | `u2net.onnx` | `8d10d2f3bb75ae3b...` | trained in-repo |
| u2netp | `u2netp.pt` | `def963cd69515e11...` | `309c8469258dda74...` |
| u2netp-onnx | `u2netp.onnx` | `309c8469258dda74...` | trained in-repo |

## Per-stage timing breakdown

Where the wall clock actually goes for one image. Useful because the model
is frequently not the bottleneck - preprocessing and alpha refinement are
resolution-dependent while inference is fixed at the letterboxed size.

| Model | Preprocess ms | Inference ms | Postprocess ms | Refine ms | Cold start s |
|---|---|---|---|---|---|
| trivial-ones | 1.41 | 0.18 | 0.14 | 0.08 | 0.000 |
| trivial-center | 1.40 | 0.62 | 0.14 | 0.08 | 0.000 |
| classical-saliency | 0.39 | 0.97 | 0.13 | 0.07 | 0.000 |
| classical | 0.40 | 262.11 | 0.14 | 0.07 | 0.000 |
| classical-saliency-grabcut | 0.40 | 384.55 | 0.14 | 0.08 | 0.000 |
| cutoutnet-tiny | 0.78 | 10.14 | 0.06 | 0.09 | 0.023 |
| cutoutnet | 0.77 | 31.79 | 0.06 | 0.09 | 0.036 |
| cutoutnet-base | 0.75 | 40.06 | 0.06 | 0.09 | 0.085 |
| cutoutnet-onnx ONNX/CPU | 0.76 | 16.74 | 0.08 | 0.09 | 0.031 |
| u2netp | 1.52 | 263.66 | 0.17 | 0.10 | 0.069 |
| u2netp-onnx ONNX/CPU | 1.46 | 249.21 | 0.13 | 0.10 | 0.035 |
| u2net | 1.68 | 623.85 | 0.21 | 0.12 | 0.452 |
| u2net-onnx ONNX/CPU | 1.58 | 595.32 | 0.18 | 0.12 | 0.261 |
| birefnet random-init | 3.74 | 201.55 | 0.41 | 0.12 | 0.027 |
| cutoutnet compiled | 0.79 | 18.66 | 0.06 | 0.09 | 0.037 |
| cutoutnet compiled | 0.78 | 17.34 | 0.03 | 0.06 | 0.037 |
| cutoutnet | 0.78 | 18.11 | 0.04 | 0.07 | 0.038 |
| cutoutnet | 0.83 | 26.56 | 0.04 | 0.07 | 0.038 |
| cutoutnet-onnx ONNX/CPU | 0.76 | 16.73 | 0.04 | 0.06 | 0.020 |

## Full accuracy metrics

| Model | IoU | Dice | MAE | F-beta | max F-beta | S-measure | Boundary F1 | BER | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial-ones | 0.3590 | 0.5087 | 0.6410 | 0.4142 | 0.4142 | 0.1856 | 0.0000 | 0.5000 | 0.3590 | 1.0000 |
| trivial-center | 0.4382 | 0.5989 | 0.2644 | 0.6167 | 0.6180 | 0.5698 | 0.0959 | 0.2729 | 0.6472 | 0.6155 |
| classical-saliency | 0.1508 | 0.2537 | 0.3772 | 0.3130 | 0.3373 | 0.3351 | 0.1325 | 0.4684 | 0.4352 | 0.2003 |
| classical | 0.6503 | 0.7533 | 0.1522 | 0.7677 | 0.7760 | 0.7415 | 0.5782 | 0.1518 | 0.8424 | 0.7692 |
| classical-saliency-grabcut | 0.1574 | 0.2530 | 0.3695 | 0.3137 | 0.3403 | 0.3436 | 0.2906 | 0.4611 | 0.4463 | 0.2001 |
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
   a context and autotunes. Across this run's cases the first inference cost 1.0-2.7x
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
   matching rung of the thread sweep - and the two land about 1.6x apart, each with a
   standard deviation inside 3% of its own median. That is not contention - the harness
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
   table, and treat a 1.6x agreement between two distant rows as the floor on this
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

**Latency and accuracy are measured under different rules, deliberately.** A latency figure
is worth having only if it reflects normal execution, so the timed loop runs the model
exactly as production would. An accuracy figure is worth having only if it is reproducible,
so the scoring pass runs from a controlled evaluation state.

For learned models the distinction costs nothing: accuracy is fixed by the checkpoint, the
eval set, the preprocessing and the metric implementation, so `cutoutnet-fp32` and
`u2net-pretrained` return the IoU published here to ten decimal places however they are
invoked. Classical methods with stochastic initialisation are the reason the rule has to be
stated. OpenCV's GrabCut seeds its colour model from a process-global RNG, so six
consecutive calls on one unchanged image return six slightly different masks - and because
each case is timed before it is scored, `--repetitions` used to decide how many draws
preceded the scoring pass. A latency knob moved an accuracy number, in stable increments
that looked nothing like noise. The harness now resets that RNG immediately before every
accuracy pass, so a score depends on the model and the eval set and not on what ran before
it. Nothing is reset inside the timed section, where it would measure work production does
not do.

### Calibration references

The table includes deliberately content-blind rows (`trivial-ones`, `trivial-center`).
They exist because IoU is only interpretable relative to what predicting *nothing*
achieves. On this set the foreground covers 35.9% of the frame, so "predict everything"
already scores 0.3590 IoU. Any row that does not clearly beat those has learned nothing.
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

