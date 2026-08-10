"""CutoutNet: a small, fully original encoder-decoder matting network.

Why it exists
-------------
Every other architecture in this repository needs pretrained weights that cannot
be downloaded in a sandboxed build, which means their accuracy numbers would be
either fabricated or absent. CutoutNet is sized so it can be **trained from
scratch on 8 CPU cores in a few minutes**, so the accuracy column of the
benchmark table contains at least one number that was genuinely measured on a
genuinely trained model.

Architecture
------------
::

    input 3xHxW
      |
    stem  (3x3 s2, 3 -> 16)                            1/2
      |
    MBConv stages  c=(24, 40, 80, 128)  s=(2,2,2,2)    1/4 1/8 1/16 1/32
      |                    |    |    |
      |            lateral 1x1 -> fpn_width (top-down FPN-lite)
      |                    v
      |            P2 (1/4) --- side heads at P3, P4
      |                    |
    coarse logits (1/4) ---+
      |
    upsample to HxW
      |
    refinement head:  cat(coarse_logits, image, sobel(image))
      -> depthwise-separable stack -> residual delta on the logits
      |
    final logits HxW

Three deliberate choices:

* **Inverted residuals with depthwise separable convolutions** (MobileNetV2
  style): expand 1x1 -> depthwise 3x3 -> project 1x1, residual when shape
  matches. This gives the receptive field of a much larger network for a fraction
  of the multiply-accumulates, which is what makes CPU training viable.
* **FPN-lite decoder** rather than a full U-Net: a single 1x1 lateral per stage
  and additive top-down fusion. Fewer parameters in the decoder means more of the
  budget goes to the encoder, where it buys more accuracy.
* **A residual refinement head that sees the original pixels.** The decoder works
  at 1/4 resolution; the head is the only part that sees full-resolution detail,
  and it predicts a *correction* to the upsampled logits rather than a fresh mask.
  Predicting a delta is markedly easier to optimise - at initialisation the head
  is near-identity, so training starts from "correct but soft" rather than noise.

Outputs are logits: ``(final, coarse, side_1_8, side_1_16)``. The training loss
supervises all four; inference uses ``[0]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBN(nn.Sequential):
    """Conv -> BN -> (ReLU6), the standard mobile unit.

    ReLU6 rather than ReLU: it bounds activations, which keeps the network
    friendlier to int8/fp16 quantisation later without costing anything now.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    """Lightweight channel attention (squeeze-and-excitation).

    Costs ~``2 * C^2 / r`` parameters and negligible FLOPs, and measurably helps a
    thin network decide between "subject" and "similarly coloured background".
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w), inplace=True)))
        return x * w


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual bottleneck with optional SE."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand: int = 4,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(8, int(round(in_ch * expand)))
        self.use_residual = stride == 1 and in_ch == out_ch

        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(ConvBN(in_ch, hidden, 1))
        layers.append(ConvBN(hidden, hidden, 3, stride=stride, groups=hidden))
        if use_se:
            layers.append(SqueezeExcite(hidden))
        # Linear bottleneck: no activation on the projection, per MobileNetV2.
        layers.append(ConvBN(hidden, out_ch, 1, act=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out  # type: ignore[no-any-return]


class SeparableConv(nn.Sequential):
    """Depthwise 3x3 followed by pointwise 1x1."""

    def __init__(self, in_ch: int, out_ch: int, act: bool = True) -> None:
        super().__init__(
            ConvBN(in_ch, in_ch, 3, groups=in_ch),
            ConvBN(in_ch, out_ch, 1, act=act),
        )


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    """Sobel magnitude of the input luminance, per-sample max-normalised.

    Handed to the refinement head as an explicit "here is where edges are" channel.
    Learning this from three convolutions would be trivially possible but wasteful
    in a model this small.
    """
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray.dtype, device=gray.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(2, 3), padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return mag / mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


class RefinementHead(nn.Module):
    """Full-resolution residual correction of the upsampled coarse logits.

    Input is ``cat(coarse_logits, rgb, gradient)`` = ``1 + 3 + 1`` channels. The
    final convolution is zero-initialised so the head starts as an exact identity
    and training begins from the decoder's already-reasonable output.
    """

    def __init__(self, width: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.entry = ConvBN(5, width, 3)
        self.body = nn.Sequential(*[SeparableConv(width, width) for _ in range(depth)])
        self.delta = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, coarse_logits: torch.Tensor, image: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        h = self.entry(torch.cat([coarse_logits, image, gradient], dim=1))
        h = self.body(h)
        return coarse_logits + self.delta(h)  # type: ignore[no-any-return]


class CutoutNet(nn.Module):
    """The full network.

    Parameters
    ----------
    channels:
        Output channels of the four encoder stages (strides 4, 8, 16, 32).
    depths:
        Inverted-residual blocks per stage.
    fpn_width:
        Channel width of every FPN lateral / output, shared across levels.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (24, 32, 64, 96),
        depths: tuple[int, int, int, int] = (2, 3, 4, 3),
        stem_width: int = 16,
        fpn_width: int = 48,
        head_width: int = 16,
        expand: int = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.fpn_width = fpn_width

        self.stem = ConvBN(3, stem_width, 3, stride=2)

        stages: list[nn.Module] = []
        in_ch = stem_width
        for out_ch, depth in zip(channels, depths, strict=True):
            blocks: list[nn.Module] = [
                InvertedResidual(in_ch, out_ch, stride=2, expand=expand)
            ]
            blocks += [
                InvertedResidual(out_ch, out_ch, stride=1, expand=expand)
                for _ in range(max(0, depth - 1))
            ]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        self.laterals = nn.ModuleList([ConvBN(c, fpn_width, 1) for c in channels])
        # One smoothing conv per fusion step (3 fusions for 4 levels).
        self.smooth = nn.ModuleList([SeparableConv(fpn_width, fpn_width) for _ in range(3)])

        self.coarse_head = nn.Conv2d(fpn_width, 1, 3, padding=1)
        self.side_heads = nn.ModuleList([nn.Conv2d(fpn_width, 1, 1) for _ in range(2)])
        self.refine = RefinementHead(width=head_width)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Re-zero the identity-init delta, which the loop above overwrote.
        nn.init.zeros_(self.refine.delta.weight)
        nn.init.zeros_(self.refine.delta.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        size = x.shape[2:]

        h = self.stem(x)
        feats: list[torch.Tensor] = []
        for stage in self.stages:
            h = stage(h)
            feats.append(h)

        # ---- FPN-lite top-down
        laterals = [lat(f) for lat, f in zip(self.laterals, feats, strict=True)]
        p = laterals[3]
        pyramid = [p]
        for i in range(2, -1, -1):
            up = F.interpolate(p, size=laterals[i].shape[2:], mode="nearest")
            p = self.smooth[2 - i](up + laterals[i])
            pyramid.append(p)
        # pyramid = [P5(1/32), P4(1/16), P3(1/8), P2(1/4)]

        coarse_small = self.coarse_head(pyramid[3])
        coarse = F.interpolate(coarse_small, size=size, mode="bilinear", align_corners=False)

        gradient = image_gradient(x)
        final = self.refine(coarse, x, gradient)

        side_1_8 = F.interpolate(
            self.side_heads[0](pyramid[2]), size=size, mode="bilinear", align_corners=False
        )
        side_1_16 = F.interpolate(
            self.side_heads[1](pyramid[1]), size=size, mode="bilinear", align_corners=False
        )
        return (final, coarse, side_1_8, side_1_16)


def cutoutnet_small() -> CutoutNet:
    """1.14M parameters / 4.6 MB fp32. The configuration trained and benchmarked."""
    return CutoutNet(
        channels=(24, 40, 80, 128),
        depths=(2, 3, 4, 3),
        stem_width=16,
        fpn_width=64,
        head_width=24,
    )


def cutoutnet_tiny() -> CutoutNet:
    """0.12M parameters, for latency-floor experiments."""
    return CutoutNet(
        channels=(16, 24, 40, 64),
        depths=(1, 2, 2, 2),
        stem_width=12,
        fpn_width=32,
        head_width=12,
        expand=3,
    )


def cutoutnet_base() -> CutoutNet:
    """4.34M parameters / 17.4 MB fp32, inside the 25 MB checkpoint budget."""
    return CutoutNet(
        channels=(32, 56, 112, 176),
        depths=(2, 4, 6, 4),
        stem_width=24,
        fpn_width=72,
        head_width=24,
        expand=5,
    )


ARCHITECTURES = {
    "tiny": cutoutnet_tiny,
    "small": cutoutnet_small,
    "base": cutoutnet_base,
}
