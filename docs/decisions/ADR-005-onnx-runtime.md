# ADR-005: ONNX Runtime as the second serving runtime

Status: Accepted

## Context

Training needs PyTorch. Serving does not necessarily, and there are three costs to
serving from PyTorch that are worth trying to avoid:

- **Image size and cold start.** `torch` with CUDA is roughly 2.5 GB of wheels. A container
  built on it takes minutes to pull on a cold node. `onnxruntime` is about 50 MB, and its
  GPU build is far smaller than the full CUDA PyTorch stack. On anything that scales from
  zero — a serverless GPU, a spot worker replacing a reclaimed one — that difference is the
  dominant term in time-to-first-inference.
- **Graph-level optimisation left on the table.** Eager PyTorch executes operator by
  operator. An ahead-of-time graph compiler can fuse Conv+BN+ReLU, fold constants, and
  choose memory layouts across the whole network. On CPU this is usually a real speedup; on
  GPU the gap narrows because cuDNN already fuses much of it.
- **Portability.** An ONNX graph is the input to TensorRT, OpenVINO, CoreML and several
  hardware vendors' stacks. Exporting once opens all of them; not exporting closes all of
  them.

Against that: an export is a second artefact that can silently disagree with the model it
was derived from, which is the failure mode that matters most.

## Decision

**Export to ONNX and serve it through ONNX Runtime as a peer runtime in the registry**,
not as a replacement for PyTorch. `cutoutnet-onnx` is a separate `ModelSpec` pointing at
the same weights via a `.onnx` artefact, so both runtimes appear as rows in the benchmark
table and the comparison is measured rather than asserted.

Three implementation decisions carry most of the value:

1. **Export is verified numerically or it is not a successful export.**
   `cutoutml.models.export_onnx` always runs a parity check and refuses to report success
   if the maximum absolute difference exceeds tolerance. Bit-exactness is not the goal and
   is not achievable — onnxruntime fuses operators, may reassociate additions, and picks
   its own GEMM kernels. What matters is that the *alpha map* is indistinguishable, so the
   tolerance is 1e-3 on probabilities **after** sigmoid: less than one 8-bit alpha level,
   i.e. a difference that cannot survive quantisation into a PNG. A separate, looser 2e-3
   tolerance applies to raw logits, where a large-magnitude logit difference is harmless
   because sigmoid has saturated.

2. **The execution provider that was actually selected is recorded, not the one that was
   requested.** Provider priority is TensorRT → CUDA → ROCm → CoreML → CPU, but listing
   `CUDAExecutionProvider` does not make it available. onnxruntime will silently fall back,
   and a benchmark row labelled "GPU" that ran on CPU is worse than no row at all. The
   adapter reports the provider in its metadata, the benchmark renderer prints it in the
   runtime column, and `device="cpu"` forces CPU-only even on a CUDA machine so that a
   comparable CPU row can be produced deliberately.

3. **onnxruntime is an optional extra.** `pip install '.[onnx]'`. The registry's
   `runtime_available()` reports it as unusable when the import fails, so a deployment
   without it degrades to a missing model rather than an import error at first request.

## Alternatives considered

**Serve from PyTorch only.** Simplest, one artefact, no parity risk. Rejected because the
image-size and cold-start argument is real, and because the ONNX export is a prerequisite
for TensorRT — closing that door for simplicity is expensive later. It remains the default:
`cutoutnet` is the PyTorch spec and it is what `CUTOUTML_DEFAULT_MODEL` points at.

**`torch.compile` instead of ONNX.** Also implemented, also benchmarked, and it addresses
a different problem. `torch.compile` gives graph-level optimisation *inside* PyTorch — so
it keeps the 2.5 GB dependency and the multi-second warm compile, and it does not help
cold start or portability. It is the right tool when PyTorch must be in the image anyway;
it is not a substitute for an exportable graph. Having both as benchmark rows is more
useful than choosing between them on principle, which is why the suite runs
`cutoutnet-fp32`, `cutoutnet-fp32-compiled` and `cutoutnet-onnx-cpu` at identical batch
size and resolution so the only difference is the backend.

**TorchScript.** Rejected: effectively deprecated in favour of `torch.compile` and
`torch.export`, still requires PyTorch at serve time, and gives none of the portability.

**TensorRT as the only accelerated path.** Rejected: NVIDIA-only, version-coupled to the
CUDA and driver stack, and engines are not portable between GPU generations, so it cannot
be the baseline. The adapter exists and degrades to a clear `RuntimeError` without CUDA +
TensorRT, and its row is absent from the benchmark table rather than estimated.

**OpenVINO.** A reasonable CPU-serving choice, and reachable from the same ONNX export.
Not implemented: no measurement to justify a third runtime, and the export already makes
it a small change if the numbers ever warrant it.

## Consequences

Good:

- A serving image can be built without PyTorch.
- The PyTorch-vs-ONNX question is answered by a measured row in `docs/benchmarks.md` for
  this workload on this hardware, rather than by folklore.
- The ONNX graph is the entry point to TensorRT and other vendor runtimes.
- A broken export is caught by the parity check at export time, not by a user reporting
  that masks look different.

Bad, and accepted:

- Two artefacts per model. Retraining without re-exporting leaves a stale `.onnx`, and
  nothing enforces the pairing — the parity check only runs at export. `make weights`
  re-exports as its final step for exactly this reason.
- Dynamic shapes are constrained. The export uses a dynamic batch dimension but a fixed
  spatial size; changing the input resolution requires a new export, whereas the PyTorch
  adapter simply accepts it.
- The ONNX adapter duplicates preprocessing (normalisation constants live in its spec
  `options`) rather than sharing the PyTorch adapter's, because there is no torch module to
  ask. A mismatch there would produce subtly wrong masks; the parity check covers it at
  export time only.
- `onnxruntime` version skew against the export opset is a real operational hazard. The
  opset is pinned at 17 in the exporter.
