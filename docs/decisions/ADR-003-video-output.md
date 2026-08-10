# ADR-003: Three video output modes, and alpha capability is probed rather than assumed

Status: Accepted

## Context

Background removal produces an alpha channel. For a still image that is trivial: PNG and
WebP both carry alpha, every viewer renders it, done. For video it is the hardest part of
the feature, for reasons that are all about containers and codecs rather than about
segmentation.

- **MP4/H.264 cannot carry alpha.** Not "does not by default" — the pixel formats H.264
  uses in practice have no alpha plane. Encode RGBA to MP4 and ffmpeg will happily accept
  it, drop the alpha, and hand you an opaque video. Nothing errors.
- **WebM/VP9 can** carry alpha as `yuva420p`, and Chrome and Firefox play it. But whether
  a given ffmpeg/libvpx build actually *writes* the alpha plane varies by build, and when
  it does not, the failure is silent: you get a valid WebM whose alpha is 255 everywhere.
- **Decoding is a second trap.** ffmpeg's default VP9 decoder path does not reconstruct
  the alpha plane; reading a genuinely transparent WebM back without `-c:v libvpx-vp9`
  yields alpha = 255 for every pixel. A naive round-trip test therefore reports "alpha is
  broken" on a file that is perfectly fine, and — worse — a naive test written the other
  way round can report success on a file that is opaque.
- **ProRes 4444 and QuickTime RLE** carry alpha reliably and are what a video editor
  expects, at several times the file size.
- **Frame sequences** sidestep codecs entirely and are the only thing guaranteed to
  round-trip alpha losslessly, at the cost of being thousands of files.

The failure mode being designed against is specific: a user removes a background, gets a
file back, and only discovers days later that it is opaque.

## Decision

**Four output modes**, chosen per request, with `composite` as the default:

| Mode | Container | Carries alpha | Why it exists |
|---|---|---|---|
| `composite` | MP4/H.264 | No | Universally playable. A background is burned in — green, a colour, an image, or a blurred copy of the source. This is what most users actually want. |
| `transparent` | WebM/VP9 (`yuva420p`), MOV/ProRes 4444, QuickTime RLE | Yes | Real transparency for compositing downstream. |
| `frames` | RGBA PNG per frame, optionally zipped | Yes, losslessly | Largest output, and the only option a video editor will always accept. |
| `mask` | MP4, greyscale | N/A | The alpha channel as a video, for users who composite in their own tool. |

**MP4 plus alpha is refused, not silently downgraded.** `encoder_args()` raises rather
than producing an opaque file. A hard error at submission is enormously cheaper than a
wrong file discovered later.

**Alpha capability is probed at runtime, per container.** `alpha_roundtrip_works()`
encodes a synthetic clip with known transparent and opaque regions, decodes it back with
an explicitly alpha-aware decoder, and checks that both extremes survive. It answers "can
*this* ffmpeg binary do it" rather than "does this codec support it in principle", which
is the only question that matters in a container image somebody else built.
`working_alpha_containers()` runs the probe across all three and is what the readiness
check and `cutoutml doctor` report. On the machine these docs were written on, all three
of `webm`, `mov` and `qtrle` pass.

**Temporal smoothing is optional and measured.** Per-frame segmentation flickers: a pixel
near the decision boundary flips between frames even when nothing moved, which reads as a
crawling edge. Two smoothers exist (EMA, temporal median), and `estimate_flicker()`
reports mean absolute frame-to-frame alpha difference with and without them, so the
trade-off — smoothing always costs responsiveness on fast motion — is a number rather
than an opinion. The default is EMA at weight 0.65.

## Alternatives considered

**Only MP4 composite.** Simplest and never surprises anyone. Rejected: transparency is
the actual product for anyone doing further compositing, and "we cannot do transparent
video" is a poor answer when WebM and ProRes both can.

**Only WebM transparent.** Rejected: no Safari VP9-alpha support to rely on, no
hardware decode in many players, and it forces a codec choice on users who just wanted a
green screen.

**Assume alpha support from a static table of containers.** This is what the first
implementation did and it is exactly how the silent-opaque-output bug appears. A static
table describes the format; the probe describes the binary that is installed. Keeping the
table (`ALPHA_CONTAINERS`) as the list of *candidates* to probe is useful; trusting it as
an answer is not.

**PyAV instead of subprocess pipes.** Nicer API, in-process frame objects, no pipe
plumbing. Rejected: it adds a heavyweight build dependency to get less control over the
handful of encoder flags that actually matter for alpha (`-pix_fmt yuva420p`,
`-auto-alt-ref 0`), and ffmpeg is already present in any image that can process video.
The subprocess approach also makes streaming discipline explicit — frames enter in
batches and leave immediately, so peak memory is `O(batch_size)` rather than
`O(frame_count)`. A 60-second 4K clip is roughly 1.5 GB of raw RGB; holding it would be a
bug, not an inefficiency.

**Write alpha as a second video track.** Rejected: nothing consumes it without a manual
ffmpeg invocation, which is a worse experience than a zip of PNGs.

## Consequences

Good:

- A transparent request either produces genuine transparency or fails loudly.
- The readiness endpoint can report "transparent video unavailable in this image" before
  a user submits a job that cannot succeed.
- `composite` covers the common case with a format that plays everywhere.
- Memory is bounded by batch size, not clip length, so a long clip is slow rather than
  fatal.

Bad, and accepted:

- The alpha probe spawns two ffmpeg processes per container. It is cached per process, but
  the first readiness check after a cold start pays roughly 300 ms for all three.
- Four output modes is four code paths and four sets of tests.
- ProRes and QuickTime RLE outputs are very large, and the API does not warn about size
  before encoding — the caller finds out from the result manifest.
- Temporal smoothing is a global per-job setting, not adaptive. Fast motion with smoothing
  on shows lag; the mitigation is to document `--smoothing none` rather than to guess.
