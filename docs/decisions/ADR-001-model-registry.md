# ADR-001: Models are declared in a registry and loaded by name

Status: Accepted

## Context

The system has to serve more than one segmentation model, and the reasons are not
hypothetical:

- Different accuracy/latency points. A thumbnail preview wants the cheapest model that
  is good enough; a paying export wants the best one.
- Different runtimes for the same weights. The same CutoutNet checkpoint runs under
  PyTorch eager, `torch.compile`, ONNX Runtime and TensorRT, and which one is fastest
  depends on the hardware it lands on.
- Models that cannot run everywhere. TensorRT needs CUDA. ONNX Runtime is an optional
  dependency. Published U^2-Net weights are not redistributable here. A model that
  cannot run on this machine must still be *describable*.
- Baselines that are not neural networks at all. GrabCut and a fixed centred ellipse
  exist to calibrate the accuracy column, and they have to flow through the same
  pipeline as a network or the comparison is not apples to apples.

Four consumers need to know about models: the API (`GET /v1/models`, and validating the
`model` field of a job), the Celery worker, the benchmark harness, and the CLI. The
question is what those four know about a model, and when.

The naive shape is a chain of conditionals at each call site:

```python
if name == "cutoutnet":
    model = CutoutNetAdapter(weights="models/cutoutnet/cutoutnet-small.pt", size=(256, 256))
elif name == "u2net":
    ...
```

That has four copies of the same knowledge which drift independently, and it means
importing every adapter — including TensorRT — to answer "what models exist?".

## Decision

A model is a **declarative `ModelSpec`** in `src/cutoutml/models/registry.py`: name,
dotted path to its adapter class, architecture label, input size, licence, source URL,
default weights path, runtime, whether it needs weights, whether random initialisation
is permitted, a description, tags, and an options dict passed to the constructor.

Three rules make it useful:

1. **Adapters are imported lazily**, by dotted path, at instantiation time. Listing the
   catalogue never imports torch, onnxruntime or tensorrt. `GET /v1/models` is cheap and
   cannot fail because of a broken optional dependency.

2. **Availability is computed, not declared.** `weights_available()` checks the spec's
   candidate artefact paths on disk; `runtime_available()` checks that the runtime is
   importable and, for TensorRT, that a CUDA device exists. Both are evaluated per call
   rather than cached, so a checkpoint that appears in `models/` — a finished training
   run, a mounted volume — shows up without a restart. This is what lets the API tell a
   caller *before* they submit a job that `u2net` has no weights here, instead of
   failing the job asynchronously twenty seconds later.

3. **`random_init` is opt-in per spec.** The benchmark harness needs to build a network
   with random weights so it can measure the latency of an architecture whose
   checkpoint is unavailable. That capability must never be reachable from an API
   request, because it returns confident-looking noise. Specs declare
   `supports_random_init`, `get_model()` refuses otherwise, and the harness marks any
   random-weight row `accuracy_valid=False` so its accuracy columns render as `n/a`.

Adding a model is one `register(ModelSpec(...))` call plus an adapter class implementing
`preprocess / predict / postprocess`. Nothing in the API, the pipelines, the worker or
the harness changes.

## Alternatives considered

**Conditionals at each call site.** Rejected: four drifting copies of the same mapping,
and no way to enumerate models without importing all of them.

**Entry-point plugins (`importlib.metadata`).** The registry becomes extensible by
third-party packages without touching this repository. Rejected for now: it buys
extensibility nobody has asked for and costs discoverability — you can no longer answer
"what models are there?" by reading one file. The registry is a plain dict, so this
remains a small change if a plugin ecosystem ever exists.

**A YAML/TOML model catalogue.** Editable without a Python change and appealing for
operators. Rejected: the options are typed and adapter-specific (`normalization` is a
pair of RGB triples, `input_size` a tuple, `max_batch` an int), and a config file turns
every typo into a runtime error at first inference rather than a type error at import.
The specs are already data; they just happen to be data written in Python where mypy can
see them.

**A base class per model with class attributes, discovered by subclass walking.**
Rejected: discovery requires importing every subclass, which defeats the lazy-import
property that makes the catalogue cheap.

## Consequences

Good:

- `GET /v1/models` returns the catalogue with per-model availability, costs no model
  imports, and is the same data the CLI's `cutoutml models` prints.
- The benchmark harness takes a list of `(model, precision, batch, resolution)` cases and
  needs no knowledge of any specific model. New model, new row, no harness change.
- Licence and upstream source are attached to the model rather than living in a README
  that drifts. `docs/models.md` is generated from the same specs.
- A missing checkpoint degrades to a skipped benchmark case with a reason, not a crash.

Bad, and accepted:

- Adapter constructor arguments are passed as an untyped `dict`, so a bad option in a
  spec is a `TypeError` at instantiation rather than a type error at import. Mitigated by
  the registry test that instantiates every spec whose artefacts exist.
- Two sources of truth for a model's input size: the spec and whatever the checkpoint
  was trained at. Nothing enforces agreement; a mismatch shows up as bad accuracy. The
  training script writes its resolution into the run JSON so the two can be compared by
  hand.
- The registry is process-global mutable state with a lock around it. Tests that register
  temporary specs must unregister them, and `unregister()` exists only for that.
