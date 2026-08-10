"""A compact bilateral-reference segmentation network.

Relationship to BiRefNet
------------------------
This is an **architecture-inspired reimplementation**, not a port. It reproduces
the two structural ideas that give BiRefNet (Zheng et al., "Bilateral Reference
for High-Resolution Dichotomous Image Segmentation", CAAI AIR 2024) its
high-resolution edge quality, in a form small enough to run on a CPU:

1. **Localization module (LM).** A strided encoder plus a global-context head
   produces a coarse but semantically reliable map of *where* the subject is. Its
   job is not detail; it is to stop the decoder hallucinating subjects in
   background clutter.

2. **Reconstruction module (RM) with bilateral reference.** Each decoder stage is
   given two extra sources of evidence besides the encoder skip:

   * an **inner reference** - the *source image* resampled to that stage's
     resolution, so the decoder can always consult original pixels instead of
     features that have already been downsampled and blurred;
   * an **outer reference** - an explicit gradient/edge map of the source, which
     concentrates capacity where alpha actually varies.

   The real BiRefNet additionally tiles the inner reference into adaptive patches
   at native resolution and uses deformable convolutions; both are omitted here
   because they dominate cost on CPU. That omission is the main reason this is
   *inspired-by* rather than equivalent.

Deep supervision follows the paper's spirit: the network returns the fused
logits, per-stage side logits, the coarse LM logits, and a **gradient logit** map
supervised against the Sobel magnitude of the ground-truth alpha. See
``docs/models.md`` for the licensing situation around official weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNAct(nn.Sequential):
    """Conv -> BN -> activation, the repeated unit of this network."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        padding = dilation * (kernel - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, out_ch, kernel, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with an optional 1x1 projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride=stride)
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, act=False)
        self.act = nn.ReLU(inplace=True)
        self.proj: nn.Module = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else ConvBNAct(in_ch, out_ch, 1, stride=stride, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.proj(x))


class GlobalContext(nn.Module):
    """Pyramid-pooling context head for the localization module.

    Branches: three adaptive-average-pooled scales plus a dilated 3x3, all fused
    by a 1x1. Adaptive pooling (rather than fixed kernels) makes the module
    resolution-agnostic, which matters because this network is meant to be run at
    several input sizes.
    """

    def __init__(self, in_ch: int, out_ch: int, scales: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.scales = scales
        branch_ch = max(16, out_ch // 4)
        self.branches = nn.ModuleList([ConvBNAct(in_ch, branch_ch, 1) for _ in scales])
        self.dilated = ConvBNAct(in_ch, branch_ch, 3, dilation=3)
        self.fuse = ConvBNAct(in_ch + branch_ch * (len(scales) + 1), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        feats = [x]
        for scale, branch in zip(self.scales, self.branches, strict=True):
            pooled = F.adaptive_avg_pool2d(x, scale)
            feats.append(F.interpolate(branch(pooled), size=size, mode="bilinear", align_corners=False))
        feats.append(self.dilated(x))
        return self.fuse(torch.cat(feats, dim=1))  # type: ignore[no-any-return]


def sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    """Per-image Sobel gradient magnitude, normalised to roughly ``[0, 1]``.

    Computed on the luminance of the input rather than on features, so it is the
    same signal whether the network is training or serving. Implemented as a
    grouped conv with fixed kernels; it costs two 3x3 convolutions on one channel.
    """
    if x.shape[1] == 3:
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    else:
        gray = x[:, :1]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray.dtype, device=gray.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # Per-sample max-normalisation keeps the reference scale-invariant across
    # very flat and very textured images.
    flat_max = mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return mag / flat_max


class BilateralReferenceBlock(nn.Module):
    """One reconstruction-module stage.

    Inputs, all resampled to this stage's resolution:

    * ``deeper`` - features from the stage below (or the LM output)
    * ``skip``   - the matching encoder feature map
    * ``inner``  - the source image (3 channels)
    * ``outer``  - the source gradient magnitude (1 channel)

    The two reference channels are projected before fusion so a raw-pixel
    reference cannot dominate the learned features by sheer magnitude.
    """

    def __init__(self, deeper_ch: int, skip_ch: int, out_ch: int, ref_ch: int = 16) -> None:
        super().__init__()
        self.inner_proj = ConvBNAct(3, ref_ch, 3)
        self.outer_proj = ConvBNAct(1, ref_ch, 3)
        self.fuse = ConvBNAct(deeper_ch + skip_ch + 2 * ref_ch, out_ch, 1)
        self.refine = ResidualBlock(out_ch, out_ch)
        self.attn = nn.Sequential(
            nn.Conv2d(out_ch, max(8, out_ch // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, out_ch // 4), out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        deeper: torch.Tensor,
        skip: torch.Tensor,
        image: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        size = skip.shape[2:]
        deeper_up = F.interpolate(deeper, size=size, mode="bilinear", align_corners=False)
        inner = self.inner_proj(F.interpolate(image, size=size, mode="area"))
        outer = self.outer_proj(F.interpolate(gradient, size=size, mode="area"))
        fused = self.fuse(torch.cat([deeper_up, skip, inner, outer], dim=1))
        refined = self.refine(fused)
        # Channel gating: lets the block suppress the reference channels where
        # the encoder feature is already confident.
        return refined * self.attn(F.adaptive_avg_pool2d(refined, 1))  # type: ignore[no-any-return]


class BiRefNetCompact(nn.Module):
    """Localization + bilateral-reference reconstruction, CPU-sized.

    Parameters
    ----------
    width:
        Base channel count. ``(w, 2w, 4w, 8w)`` across the four encoder stages.
    depths:
        Residual blocks per encoder stage.
    """

    def __init__(
        self,
        width: int = 32,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        decoder_width: int = 48,
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        chs = (width, width * 2, width * 4, width * 8)
        self.channels = chs

        self.stem = nn.Sequential(
            ConvBNAct(3, width, 3, stride=2),
            ConvBNAct(width, width, 3),
        )

        # ---- encoder: four strided stages -> 1/4, 1/8, 1/16, 1/32
        self.stages = nn.ModuleList(
            [
                self._make_stage(width if i == 0 else chs[i - 1], chs[i], depths[i])
                for i in range(4)
            ]
        )

        # ---- localization module
        self.lm_context = GlobalContext(chs[3], decoder_width * 2)
        self.lm_head = nn.Conv2d(decoder_width * 2, out_ch, 1)

        # ---- reconstruction module: 1/32 -> 1/16 -> 1/8 -> 1/4
        self.rm = nn.ModuleList(
            [
                BilateralReferenceBlock(decoder_width * 2, chs[2], decoder_width),
                BilateralReferenceBlock(decoder_width, chs[1], decoder_width),
                BilateralReferenceBlock(decoder_width, chs[0], decoder_width),
            ]
        )
        self.side_heads = nn.ModuleList([nn.Conv2d(decoder_width, out_ch, 1) for _ in range(3)])

        # ---- full-resolution fusion, which also sees the raw image + gradient
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_width + 4, decoder_width // 2, 3),
            ResidualBlock(decoder_width // 2, decoder_width // 2),
        )
        self.out_head = nn.Conv2d(decoder_width // 2, out_ch, 3, padding=1)
        self.grad_head = nn.Conv2d(decoder_width // 2, out_ch, 3, padding=1)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, depth: int) -> nn.Sequential:
        blocks: list[nn.Module] = [ResidualBlock(in_ch, out_ch, stride=2)]
        blocks += [ResidualBlock(out_ch, out_ch) for _ in range(max(0, depth - 1))]
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns ``(fused, side_1_4, side_1_8, side_1_16, coarse_lm, gradient)``.

        All outputs are logits at input resolution so the loss can supervise them
        against the same target without extra bookkeeping.
        """
        full_size = x.shape[2:]
        gradient = sobel_gradient(x)

        h = self.stem(x)
        feats: list[torch.Tensor] = []
        for stage in self.stages:
            h = stage(h)
            feats.append(h)

        ctx = self.lm_context(feats[3])
        coarse = self.lm_head(ctx)

        d = ctx
        sides: list[torch.Tensor] = []
        for i, block in enumerate(self.rm):
            d = block(d, feats[2 - i], x, gradient)
            sides.append(self.side_heads[i](d))

        d_full = F.interpolate(d, size=full_size, mode="bilinear", align_corners=False)
        fused_in = torch.cat([d_full, x, gradient], dim=1)
        fused_feat = self.fuse(fused_in)

        logits = self.out_head(fused_feat)
        grad_logits = self.grad_head(fused_feat)

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=full_size, mode="bilinear", align_corners=False)

        return (
            logits,
            up(sides[2]),
            up(sides[1]),
            up(sides[0]),
            up(coarse),
            grad_logits,
        )


def birefnet_compact(out_ch: int = 1) -> BiRefNetCompact:
    """The default configuration (~5M params) used by the registry."""
    return BiRefNetCompact(width=32, depths=(2, 2, 2, 2), decoder_width=48, out_ch=out_ch)


def birefnet_tiny(out_ch: int = 1) -> BiRefNetCompact:
    """A smaller variant for latency experiments on very constrained hardware."""
    return BiRefNetCompact(width=16, depths=(1, 2, 2, 2), decoder_width=32, out_ch=out_ch)
