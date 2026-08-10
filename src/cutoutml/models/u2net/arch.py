"""U^2-Net architecture, implemented from the published design.

Reference
---------
Qin et al., "U^2-Net: Going Deeper with Nested U-Structure for Salient Object
Detection", Pattern Recognition 106 (2020). Upstream reference implementation is
Apache-2.0 licensed.

This is an independent implementation written from the paper's description of the
architecture; it is not a copy of the reference repository. It is however
*shape-compatible* with the widely distributed ``u2net.pth`` / ``u2netp.pth``
checkpoints: every tensor exists with the same dimensions, only the parameter
*names* differ because this version expresses the repeated stages as
``nn.ModuleList``s instead of numbered attributes.
:func:`cutoutml.models.u2net.adapter.remap_official_state_dict` performs the
name translation, so official weights load without modification.

Design in one paragraph
-----------------------
The outer network is a 6-stage U-Net. Every stage, instead of being a couple of
plain convolutions, is itself a small U-Net: a **ReSidual U-block** (RSU). An RSU
of depth *L* downsamples *L-1* times inside the stage, so the receptive field
grows enormously without an ImageNet backbone and without ever operating on a
tiny feature map. The deepest two stages use the *dilated* variant (``RSU4F``),
which keeps resolution constant and grows the receptive field with dilation
instead of pooling, because by that point the feature map is already small. Each
decoder stage emits a side output; all six are upsampled to input resolution and
fused with a 1x1 convolution, giving deep supervision at training time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class REBNConv(nn.Module):
    """Conv -> BatchNorm -> ReLU with a configurable dilation.

    Padding is set to ``dilation`` so the spatial size is always preserved, which
    is what lets the dilated RSU variants keep a constant resolution.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Bilinearly resize ``src`` to ``ref``'s spatial size.

    Sizes are taken from the tensor rather than hardcoded so odd input
    resolutions (which produce off-by-one feature maps after pooling) still
    concatenate cleanly.
    """
    return F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)


class RSU(nn.Module):
    """ReSidual U-block, the single building block of U^2-Net.

    Parameters
    ----------
    depth:
        Number of encoder convolutions in the inner U (``7`` for RSU-7 etc.).
        With ``dilated=False`` the block pools ``depth - 2`` times.
    dilated:
        The ``*F`` variant. Instead of pooling, inner convolutions use dilations
        ``1, 2, 4, 8, ...``; spatial size is constant throughout the block. Used
        for the two deepest stages where further pooling would destroy detail.

    The forward pass ends with ``+ hxin``, the residual connection that gives the
    block its name: the inner U learns a *correction* to a plain 3x3 transform of
    the input rather than the full mapping.
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        *,
        depth: int = 7,
        dilated: bool = False,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"RSU depth must be >= 2, got {depth}")
        self.depth = depth
        self.dilated = dilated

        # Input transform; its output is what the residual is added back to.
        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        # Encoder path. `depth - 1` convolutions, the last of which is the
        # bottleneck operating at the coarsest scale.
        enc: list[nn.Module] = []
        for i in range(depth - 1):
            src = out_ch if i == 0 else mid_ch
            dirate = 2**i if dilated else 1
            enc.append(REBNConv(src, mid_ch, dirate=dirate))
        self.encoder = nn.ModuleList(enc)

        # The "bottom" convolution with doubled dilation, bridging encoder and
        # decoder at the coarsest scale.
        bottom_dirate = 2 ** (depth - 1) if dilated else 2
        self.rebnconv_bottom = REBNConv(mid_ch, mid_ch, dirate=bottom_dirate)

        # Decoder path, consuming skip connections; input channels are doubled by
        # the concatenation.
        dec: list[nn.Module] = []
        for i in range(depth - 2, -1, -1):
            dirate = 2**i if dilated else 1
            out_c = out_ch if i == 0 else mid_ch
            dec.append(REBNConv(mid_ch * 2, out_c, dirate=dirate))
        self.decoder = nn.ModuleList(dec)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)

        skips: list[torch.Tensor] = []
        h = hxin
        for i, conv in enumerate(self.encoder):
            h = conv(h)
            skips.append(h)
            # Pool between encoder stages only; never after the last one, and
            # never at all in the dilated variant.
            if not self.dilated and i < len(self.encoder) - 1:
                h = self.pool(h)

        h = self.rebnconv_bottom(h)

        for i, conv in enumerate(self.decoder):
            skip = skips[len(skips) - 1 - i]
            h = conv(torch.cat((h, skip), dim=1))
            if not self.dilated and i < len(self.decoder) - 1:
                h = _upsample_like(h, skips[len(skips) - 2 - i])

        return h + hxin


class U2Net(nn.Module):
    """The full nested U-structure.

    ``stages`` is a table of ``(in_ch, mid_ch, out_ch, depth, dilated)`` for the
    encoder. Six side outputs plus the fused output are returned as logits, fusion
    first, so callers that only want the final mask can take ``out[0]`` while the
    training loop supervises all seven.

    Why the decoder widths are a table and not a rule
    ------------------------------------------------
    ``decoders`` gives ``(mid_ch, out_ch)`` for the five decoder stages, shallowest
    first (``stage1d`` .. ``stage5d``). They are transcribed from the published
    architecture rather than derived from ``stages``, because the published widths do
    not follow the encoder: with encoder widths ``64, 128, 256, 512, 512, 512`` the
    decoders emit ``64, 64, 128, 256, 512`` and use bottleneck widths
    ``16, 32, 64, 128, 256`` - so neither the output nor the bottleneck of a decoder
    matches its paired encoder stage, and the shallowest decoder does not follow the
    same relation as the other four.

    Only ``in_ch`` is derived, because it is genuinely structural: a decoder consumes
    the concatenation of the decoder above it and its encoder skip connection. Inner
    depth and dilation are taken from the paired encoder stage, which the nested
    U-structure does fix.

    Getting these widths wrong is silent in three separate ways, which is why they are
    written out explicitly here: every stage in the ``lite`` variant is 64 wide, so the
    error does not appear there; a from-scratch training run simply learns whatever
    shapes it is given; and :meth:`load_state_dict` is called with ``strict=False`` to
    tolerate the upstream key renaming, so mismatched tensors are skipped and the model
    runs on its random initialisation with nothing worse than a log warning. The
    property that "this is shape-compatible with ``u2net.pth``" is enforced instead by
    :mod:`cutoutml.models.u2net.from_onnx`, which fails if any tensor does not fit.
    """

    def __init__(
        self,
        stages: list[tuple[int, int, int, int, bool]],
        decoders: list[tuple[int, int]],
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        if len(stages) != 6:
            raise ValueError(f"U2Net expects 6 encoder stages, got {len(stages)}")
        if len(decoders) != 5:
            raise ValueError(f"U2Net expects 5 decoder stages, got {len(decoders)}")
        self.out_ch = out_ch

        self.encoders = nn.ModuleList(
            [RSU(i, m, o, depth=d, dilated=f) for (i, m, o, d, f) in stages]
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        encoder_out = [s[2] for s in stages]
        decoder_out = [out_c for (_, out_c) in decoders]

        # Decoder stage k consumes concat(output of the decoder above it, encoder k
        # output). The deepest encoder stage feeds the first decoder directly, since
        # stage 6 has no decoder of its own.
        blocks: list[nn.Module] = []
        for k in range(4, -1, -1):
            above = encoder_out[5] if k == 4 else decoder_out[k + 1]
            mid, out_c = decoders[k]
            _, _, _, depth, dilated = stages[k]
            blocks.append(RSU(encoder_out[k] + above, mid, out_c, depth=depth, dilated=dilated))
        self.decoders = nn.ModuleList(blocks)  # order: stage5d, 4d, 3d, 2d, 1d

        # Side output heads: one per decoder stage plus one from the deepest encoder.
        side_channels = [*decoder_out, encoder_out[5]]
        self.side = nn.ModuleList([nn.Conv2d(c, out_ch, 3, padding=1) for c in side_channels])
        self.outconv = nn.Conv2d(out_ch * 6, out_ch, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for convs, unit init for BN.

        Matters for the ``--random-init`` latency mode: a badly initialised net can
        produce NaNs, and NaN propagation actually changes CPU timings on some
        BLAS builds, so we keep the random path numerically sane.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # ---- encoder
        feats: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            feats.append(h)
            if i < len(self.encoders) - 1:
                h = self.pool(h)

        # ---- decoder, deepest first
        h = feats[5]
        dec_feats: list[torch.Tensor] = []
        for i, dec in enumerate(self.decoders):
            k = 4 - i  # encoder stage this decoder pairs with
            up = _upsample_like(h, feats[k])
            h = dec(torch.cat((up, feats[k]), dim=1))
            dec_feats.append(h)

        # ---- side outputs, all resampled to input resolution
        # dec_feats is [stage5d, 4d, 3d, 2d, 1d]; side1 comes from stage1d.
        sources = [dec_feats[4], dec_feats[3], dec_feats[2], dec_feats[1], dec_feats[0], feats[5]]
        sides = [
            _upsample_like(head(src), x) if idx > 0 else head(src)
            for idx, (head, src) in enumerate(zip(self.side, sources, strict=True))
        ]
        fused = self.outconv(torch.cat(sides, dim=1))
        return (fused, *sides)


def u2net_full(out_ch: int = 1) -> U2Net:
    """The 44M-parameter U^2-Net from the paper (``u2net.pth``).

    Widths transcribed from the reference implementation. The decoder table is the part
    worth checking against upstream: ``stage1d`` uses a 16-channel bottleneck where its
    paired encoder stage uses 32, and every decoder emits its shallower neighbour's
    width rather than its own stage's.
    """
    stages: list[tuple[int, int, int, int, bool]] = [
        (3, 32, 64, 7, False),
        (64, 32, 128, 6, False),
        (128, 64, 256, 5, False),
        (256, 128, 512, 4, False),
        (512, 256, 512, 4, True),
        (512, 256, 512, 4, True),
    ]
    decoders = [(16, 64), (32, 64), (64, 128), (128, 256), (256, 512)]
    return U2Net(stages, decoders, out_ch=out_ch)


def u2net_lite(out_ch: int = 1) -> U2Net:
    """U^2-Net-P, the 1.1M-parameter variant (``u2netp.pth``).

    Every stage, encoder and decoder alike, uses 16 mid / 64 out channels, which is why
    it is ~40x smaller while keeping the same nested topology and receptive field. That
    uniformity is also why this variant cannot detect a wrong decoder width table.
    """
    stages: list[tuple[int, int, int, int, bool]] = [
        (3, 16, 64, 7, False),
        (64, 16, 64, 6, False),
        (64, 16, 64, 5, False),
        (64, 16, 64, 4, False),
        (64, 16, 64, 4, True),
        (64, 16, 64, 4, True),
    ]
    decoders = [(16, 64)] * 5
    return U2Net(stages, decoders, out_ch=out_ch)
