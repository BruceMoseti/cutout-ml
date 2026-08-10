'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  api,
  fetchOutputObjectUrl,
  type Asset,
  type ImageOutput,
  type JobResult,
  type ModelInfo,
} from '@/lib/api';
import { formatBytes } from '@/lib/format';
import { useAsync, useJobPolling } from '@/lib/hooks';
import { BeforeAfterSlider } from './BeforeAfterSlider';
import { Dropzone } from './Dropzone';
import { ModelSelector } from './ModelSelector';
import { ProgressBar } from './ProgressBar';
import { StatsPanel } from './StatsPanel';

type BackgroundMode = 'transparent' | 'color' | 'blur';
type ViewMode = 'compare' | 'result' | 'mask';

const MAX_IMAGE_BYTES = 32 * 1024 * 1024;

/** The API output kinds needed for each background mode, plus the mask for visualisation. */
function outputsFor(mode: BackgroundMode): ImageOutput[] {
  const base: ImageOutput[] = ['transparent_png', 'mask_png'];
  if (mode === 'color') base.push('color_composite');
  if (mode === 'blur') base.push('blurred_background');
  return base;
}

const PRESET_COLORS: { label: string; rgb: [number, number, number] }[] = [
  { label: 'White', rgb: [255, 255, 255] },
  { label: 'Black', rgb: [16, 20, 27] },
  { label: 'Chroma green', rgb: [0, 177, 64] },
  { label: 'Studio grey', rgb: [113, 121, 133] },
  { label: 'Cyan', rgb: [34, 211, 238] },
];

interface QueuedItem {
  file: File;
  asset: Asset | null;
  jobId: string | null;
  error: string | null;
}

export function Studio() {
  const models = useAsync<ModelInfo[]>(async () => (await api.models()).items, []);
  const [model, setModel] = useState('cutoutnet');
  const [backgroundMode, setBackgroundMode] = useState<BackgroundMode>('transparent');
  const [color, setColor] = useState<[number, number, number]>([255, 255, 255]);
  const [view, setView] = useState<ViewMode>('compare');
  const [batch, setBatch] = useState<QueuedItem[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const active = batch[activeIndex] ?? null;
  const { job, error: pollError } = useJobPolling(active?.jobId ?? null);
  const [result, setResult] = useState<JobResult | null>(null);

  // Blob URLs for the original preview and each fetched output. Revoked on replacement
  // and on unmount: a studio session that processes twenty images would otherwise pin
  // every decoded bitmap for the lifetime of the tab.
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const outputUrls = useRef<Map<string, string>>(new Map());
  const [urlVersion, setUrlVersion] = useState(0);

  const revokeOutputs = useCallback(() => {
    outputUrls.current.forEach((url) => URL.revokeObjectURL(url));
    outputUrls.current.clear();
  }, []);

  useEffect(() => () => revokeOutputs(), [revokeOutputs]);

  useEffect(() => {
    if (!active) {
      setOriginalUrl(null);
      return;
    }
    const url = URL.createObjectURL(active.file);
    setOriginalUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [active]);

  useEffect(() => {
    if (job?.status !== 'succeeded' || !job.id) {
      setResult(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const fetched = await api.jobResult(job.id);
        if (cancelled) return;
        setResult(fetched);
        revokeOutputs();
        await Promise.all(
          fetched.outputs.map(async (output) => {
            const url = await fetchOutputObjectUrl(`/v1/jobs/${job.id}/outputs/${output.kind}`);
            outputUrls.current.set(output.kind, url);
          }),
        );
        if (!cancelled) setUrlVersion((v) => v + 1);
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [job?.status, job?.id, revokeOutputs]);

  const resultUrl = useMemo(() => {
    void urlVersion;
    const preferred =
      backgroundMode === 'color'
        ? 'color_composite'
        : backgroundMode === 'blur'
          ? 'blurred_background'
          : 'transparent_png';
    return outputUrls.current.get(preferred) ?? outputUrls.current.get('transparent_png') ?? null;
  }, [backgroundMode, urlVersion]);

  const maskUrl = useMemo(() => {
    void urlVersion;
    return outputUrls.current.get('mask_png') ?? null;
  }, [urlVersion]);

  const submit = useCallback(
    async (items: QueuedItem[]) => {
      setBusy(true);
      setError(null);
      const updated = [...items];
      try {
        for (let i = 0; i < updated.length; i += 1) {
          const item = updated[i]!;
          if (item.jobId) continue;
          try {
            const asset = item.asset ?? (await api.uploadAsset(item.file, 'image'));
            const created = await api.process(asset.id, {
              model,
              image: {
                outputs: outputsFor(backgroundMode),
                background_color: color,
                blur_sigma: 14,
              },
            });
            updated[i] = { ...item, asset, jobId: created.id, error: null };
          } catch (err) {
            updated[i] = {
              ...item,
              error: err instanceof ApiError ? `${err.message} (${err.code})` : String(err),
            };
          }
          setBatch([...updated]);
        }
      } finally {
        setBusy(false);
      }
    },
    [backgroundMode, color, model],
  );

  const onFiles = useCallback(
    (files: File[]) => {
      const items: QueuedItem[] = files.map((file) => ({
        file,
        asset: null,
        jobId: null,
        error: null,
      }));
      setBatch(items);
      setActiveIndex(0);
      void submit(items);
    },
    [submit],
  );

  const reprocess = useCallback(() => {
    if (!active?.asset) return;
    void (async () => {
      setBusy(true);
      try {
        const created = await api.process(active.asset!.id, {
          model,
          image: {
            outputs: outputsFor(backgroundMode),
            background_color: color,
            blur_sigma: 14,
          },
        });
        setBatch((prev) =>
          prev.map((item, index) => (index === activeIndex ? { ...item, jobId: created.id } : item)),
        );
      } catch (err) {
        setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
      } finally {
        setBusy(false);
      }
    })();
  }, [active, activeIndex, backgroundMode, color, model]);

  const running = job !== null && !['succeeded', 'failed', 'cancelled'].includes(job.status);

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section className="space-y-5">
        {batch.length === 0 ? (
          <Dropzone
            onFiles={onFiles}
            multiple
            maxBytes={MAX_IMAGE_BYTES}
            hint={`PNG, JPEG, WebP, GIF, BMP or TIFF up to ${formatBytes(MAX_IMAGE_BYTES)}. Drop several for a batch.`}
          />
        ) : (
          <div className="panel overflow-hidden">
            <div className="panel-header">
              <div className="flex items-center gap-1">
                {(['compare', 'result', 'mask'] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setView(mode)}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      view === mode ? 'bg-accent/15 text-accent' : 'text-ink-400 hover:text-ink-100'
                    }`}
                  >
                    {mode === 'compare' ? 'Compare' : mode === 'result' ? 'Result' : 'Mask'}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                {resultUrl && (
                  <a href={resultUrl} download className="btn-ghost !py-1 !text-xs">
                    Download
                  </a>
                )}
                <button
                  type="button"
                  onClick={() => {
                    revokeOutputs();
                    setBatch([]);
                    setResult(null);
                  }}
                  className="btn-ghost !py-1 !text-xs"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="p-4">
              {running && (
                <div className="mb-4">
                  <ProgressBar
                    fraction={job?.progress ?? 0}
                    label={job?.progress_message ?? `Job ${job?.status}`}
                    indeterminate={(job?.progress ?? 0) === 0}
                  />
                </div>
              )}

              {originalUrl && view === 'compare' && resultUrl && (
                <BeforeAfterSlider beforeSrc={originalUrl} afterSrc={resultUrl} />
              )}
              {originalUrl && view === 'compare' && !resultUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={originalUrl} alt="Original" className="w-full rounded-lg border border-ink-700" />
              )}
              {view === 'result' && resultUrl && (
                <div className="checkerboard rounded-lg border border-ink-700 p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={resultUrl} alt="Cut out" className="mx-auto max-h-[70vh]" />
                </div>
              )}
              {view === 'mask' && maskUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={maskUrl}
                  alt="Predicted alpha mask"
                  className="mx-auto max-h-[70vh] rounded-lg border border-ink-700"
                />
              )}
              {view !== 'compare' && !resultUrl && !running && (
                <p className="py-10 text-center text-sm text-ink-400">No output yet.</p>
              )}
            </div>
          </div>
        )}

        {batch.length > 1 && (
          <div className="panel p-3">
            <span className="label">Batch — {batch.length} images</span>
            <div className="mt-2 flex flex-wrap gap-2">
              {batch.map((item, index) => (
                <button
                  key={`${item.file.name}-${index}`}
                  type="button"
                  onClick={() => setActiveIndex(index)}
                  className={`max-w-[200px] truncate rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                    index === activeIndex
                      ? 'border-accent/60 bg-accent/10 text-accent'
                      : 'border-ink-700 text-ink-300 hover:border-ink-500'
                  }`}
                  title={item.error ?? item.file.name}
                >
                  {item.error ? '⚠ ' : item.jobId ? '' : '… '}
                  {item.file.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {(error || pollError || active?.error) && (
          <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
            {error ?? pollError ?? active?.error}
          </p>
        )}
      </section>

      <aside className="space-y-5">
        <div className="panel space-y-4 p-4">
          {models.loading && <p className="text-sm text-ink-400">Loading models…</p>}
          {models.error && <p className="text-sm text-bad">{models.error}</p>}
          {models.data && (
            <ModelSelector models={models.data} value={model} onChange={setModel} disabled={busy} />
          )}

          <div>
            <span className="label">Background</span>
            <div className="mt-1.5 grid grid-cols-3 gap-1.5">
              {(['transparent', 'color', 'blur'] as BackgroundMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setBackgroundMode(mode)}
                  className={`rounded-md border px-2 py-1.5 text-xs font-medium transition-colors ${
                    backgroundMode === mode
                      ? 'border-accent/60 bg-accent/10 text-accent'
                      : 'border-ink-700 text-ink-300 hover:border-ink-500'
                  }`}
                >
                  {mode === 'transparent' ? 'Alpha' : mode === 'color' ? 'Colour' : 'Blur'}
                </button>
              ))}
            </div>
          </div>

          {backgroundMode === 'color' && (
            <div>
              <span className="label">Colour</span>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {PRESET_COLORS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    title={preset.label}
                    aria-label={preset.label}
                    onClick={() => setColor(preset.rgb)}
                    style={{ backgroundColor: `rgb(${preset.rgb.join(',')})` }}
                    className={`h-7 w-7 rounded-md border-2 transition-transform hover:scale-110 ${
                      color.join(',') === preset.rgb.join(',')
                        ? 'border-accent'
                        : 'border-ink-700'
                    }`}
                  />
                ))}
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={reprocess}
            disabled={busy || !active?.asset || running}
            className="btn-primary w-full"
          >
            {running ? 'Processing…' : 'Re-run with these settings'}
          </button>
          <p className="text-[11px] leading-relaxed text-ink-500">
            Re-running with identical settings returns the original job rather than doing the
            work twice — the request carries an idempotency key derived from the content hash
            plus the parameters.
          </p>
        </div>

        <StatsPanel job={job} result={result} />
      </aside>
    </div>
  );
}
