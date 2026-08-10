# Conversion records

One JSON per converted checkpoint, written by `cutoutml.models.u2net.from_onnx` and
committed. The checkpoints themselves are not
([ADR-008](../../docs/decisions/ADR-008-no-committed-weights.md)), which is the reason these
files exist: the U²-Net weights here are recovered from a BatchNorm-folded ONNX graph and the
claim that the recovery is exact rests on a parity figure. Without a record, that figure
would appear in the docs with nothing in the repository behind it.

Each record carries the parity measured against onnxruntime and the tolerance it was
required to meet, the SHA-256 of the source graph and of the checkpoint produced, the
convolution count that had to line up, the upstream licence, and the onnxruntime and torch
versions that produced the comparison — because parity is a claim about two implementations
agreeing, and which implementations matters.

`converted_at` is here and deliberately *not* inside the checkpoint. The benchmark suite
records the digest of the weights behind every accuracy row and the archive index reads a
changed digest as changed weights, so a clock reading in the checkpoint would make two
conversions of one graph look like two different sets of weights. Converting the same graph
twice to the same path produces the same digest; `tests/test_u2net_weights.py` pins that.

Regenerate with `make weights-pretrained`.
