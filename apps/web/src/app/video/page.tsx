'use client';

import { useCallback, useState } from 'react';
import { AuthPanel } from '@/components/AuthPanel';
import { Dropzone } from '@/components/Dropzone';
import { ProgressBar } from '@/components/ProgressBar';
import { StatusBadge } from '@/components/StatsPanel';
import { ApiError, api, type Asset } from '@/lib/api';
import { formatBytes, formatDuration } from '@/lib/format';
import { useJobPolling, useSession } from '@/lib/hooks';

type Mode = 'composite' | 'transparent' | 'frames' | 'mask';

const MAX_VIDEO_BYTES = 256 * 1024 * 1024;

/**
 * Output-mode descriptions.
 *
 * The MP4 alpha limitation is stated in the UI rather than only in the docs, because it is
 * the single most common surprise in this domain: MP4/H.264 has no alpha channel, so
 * "transparent MP4" is not a thing a player can show. The API rejects the combination
 * instead of silently returning an opaque file, and the picker explains why before the user
 * queues a five-minute job.
 */
const MODES: { key: Mode; label: string; blurb: string; container: string }[] = [
  {
    key: 'composite',
    label: 'Composite',
    container: 'mp4',
    blurb: 'Background burned in, H.264/MP4. Plays everywhere. No transparency.',
  },
  {
    key: 'transparent',
    label: 'Transparent',
    container: 'mov',
    blurb:
      'Real alpha channel. MP4 cannot carry one, so this writes QuickTime (ProRes 4444 / RLE). Large files.',
  },
  {
    key: 'frames',
    label: 'PNG sequence',
    container: 'zip',
    blurb: 'One RGBA PNG per frame, zipped. Lossless alpha, biggest output, always importable.',
  },
  {
    key: 'mask',
    label: 'Mask only',
    container: 'mp4',
    blurb: 'Greyscale alpha as video. Useful as a matte in an editor.',
  },
];

export default function VideoPage() {
  const { session, refresh } = useSession();
  const [asset, setAsset] = useState<Asset | null>(null);
  const [mode, setMode] = useState<Mode>('composite');
  const [smoothing, setSmoothing] = useState<'none' | 'ema' | 'median'>('ema');
  const [maxFrames, setMaxFrames] = useState(240);
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { job, error: pollError } = useJobPolling(jobId);

  const selected = MODES.find((m) => m.key === mode)!;

  const onFiles = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setAsset(await api.uploadAsset(file, 'video'));
      setJobId(null);
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  const start = useCallback(async () => {
    if (!asset) return;
    setBusy(true);
    setError(null);
    try {
      const created = await api.process(asset.id, {
        video: {
          mode,
          container: selected.container === 'zip' ? 'mp4' : selected.container,
          smoothing,
          max_frames: maxFrames,
          batch_size: 4,
          measure_flicker: smoothing !== 'none',
        },
      });
      setJobId(created.id);
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
    } finally {
      setBusy(false);
    }
  }, [asset, maxFrames, mode, selected.container, smoothing]);

  if (!session.ready) {
    return <div className="h-64 animate-pulse rounded-xl border border-ink-800 bg-ink-900/40" />;
  }
  if (!session.authenticated) {
    return <AuthPanel onAuthenticated={() => void refresh()} />;
  }

  const running = job !== null && !['succeeded', 'failed', 'cancelled'].includes(job.status);
  const manifest = (job?.result ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Video</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          Frames are decoded from an ffmpeg pipe in bounded batches, so peak memory is a few
          frames rather than the whole clip. On this CPU-only host a clip takes roughly one
          second of wall clock per frame — keep the frame cap low.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="space-y-4">
          {!asset ? (
            <Dropzone
              onFiles={(files) => void onFiles(files)}
              accept="video/*"
              maxBytes={MAX_VIDEO_BYTES}
              disabled={busy}
              hint={`MP4, MOV, WebM, MKV or AVI up to ${formatBytes(MAX_VIDEO_BYTES)}.`}
            />
          ) : (
            <div className="panel p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-mono text-sm text-ink-100">{asset.original_filename}</p>
                  <p className="mt-1 text-xs text-ink-400">
                    {formatBytes(asset.size_bytes)} · {asset.content_type}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setAsset(null);
                    setJobId(null);
                  }}
                  className="btn-ghost !py-1 !text-xs"
                >
                  Replace
                </button>
              </div>

              {running && (
                <div className="mt-4">
                  <ProgressBar
                    fraction={job?.progress ?? 0}
                    label={job?.progress_message ?? 'Queued'}
                    indeterminate={(job?.progress ?? 0) === 0}
                  />
                </div>
              )}

              {job?.status === 'succeeded' && (
                <div className="mt-4 space-y-2">
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
                    {[
                      ['Frames', String(manifest.frames_processed ?? '—')],
                      ['Source fps', String(manifest.fps ?? '—')],
                      ['Container', String(manifest.container ?? '—')],
                      ['Has alpha', manifest.has_alpha ? 'yes' : 'no'],
                      ['Smoothing', String(manifest.smoothing ?? '—')],
                      ['Batch size', String(manifest.batch_size ?? '—')],
                      ['OOM retries', String(manifest.oom_retries ?? 0)],
                      [
                        'Flicker (raw → smoothed)',
                        manifest.flicker_raw
                          ? `${Number(manifest.flicker_raw).toFixed(4)} → ${Number(manifest.flicker_smoothed ?? 0).toFixed(4)}`
                          : 'not measured',
                      ],
                    ].map(([label, value]) => (
                      <div key={label}>
                        <dt className="label">{label}</dt>
                        <dd className="mt-0.5 font-mono text-sm">{value}</dd>
                      </div>
                    ))}
                  </dl>
                  <a
                    href={`/api/v1/jobs/${job.id}/outputs/${String((manifest.outputs as { kind: string }[] | undefined)?.[0]?.kind ?? 'output')}`}
                    className="btn-primary mt-2 inline-flex"
                  >
                    Download result
                  </a>
                  <p className="text-[11px] text-ink-500">
                    The download link goes through the API so ownership is checked on the read.
                  </p>
                </div>
              )}

              {job?.status === 'failed' && (
                <p className="mt-4 rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
                  {job.error_code}: {job.error_message}
                </p>
              )}
            </div>
          )}

          {(error || pollError) && (
            <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
              {error ?? pollError}
            </p>
          )}
        </section>

        <aside className="panel space-y-4 p-4">
          <div>
            <span className="label">Output mode</span>
            <div className="mt-1.5 space-y-1.5">
              {MODES.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => setMode(option.key)}
                  className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                    mode === option.key
                      ? 'border-accent/60 bg-accent/10'
                      : 'border-ink-700 hover:border-ink-500'
                  }`}
                >
                  <span className="flex items-center justify-between text-sm font-medium">
                    {option.label}
                    <span className="chip !text-[10px]">{option.container}</span>
                  </span>
                  <span className="mt-1 block text-[11px] leading-relaxed text-ink-400">
                    {option.blurb}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label htmlFor="smoothing" className="label">
              Temporal smoothing
            </label>
            <select
              id="smoothing"
              value={smoothing}
              onChange={(event) => setSmoothing(event.target.value as typeof smoothing)}
              className="field mt-1"
            >
              <option value="none">None</option>
              <option value="ema">EMA (no latency, attenuates flicker)</option>
              <option value="median">Median (removes dropouts, adds latency)</option>
            </select>
            <p className="mt-1 text-[11px] text-ink-500">
              Flicker is measured with and without smoothing and reported on the finished job, so
              the trade-off is a number rather than an opinion.
            </p>
          </div>

          <div>
            <label htmlFor="max-frames" className="label">
              Frame cap — {maxFrames}
            </label>
            <input
              id="max-frames"
              type="range"
              min={30}
              max={900}
              step={30}
              value={maxFrames}
              onChange={(event) => setMaxFrames(Number(event.target.value))}
              className="mt-2 w-full accent-accent"
            />
            <p className="mt-1 text-[11px] text-ink-500">
              ≈ {formatDuration(maxFrames)} of wall clock on CPU at one frame per second.
            </p>
          </div>

          <button
            type="button"
            onClick={() => void start()}
            disabled={!asset || busy || running}
            className="btn-primary w-full"
          >
            {running ? 'Processing…' : 'Start job'}
          </button>
          {job && (
            <div className="flex items-center justify-between text-xs text-ink-400">
              <span className="font-mono">{job.queue}</span>
              <StatusBadge status={job.status} />
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
