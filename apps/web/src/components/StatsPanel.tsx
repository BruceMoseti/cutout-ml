'use client';

import type { Job, JobResult } from '@/lib/api';
import { formatBytes, formatMs, formatPercent } from '@/lib/format';

interface Props {
  job: Job | null;
  result: JobResult | null;
}

interface Row {
  label: string;
  value: string;
  hint?: string;
}

/**
 * Live inference statistics for the current job.
 *
 * The stage breakdown is the part worth staring at. For a small model on a large image the
 * network is frequently *not* the bottleneck - decode, letterbox and alpha refinement can
 * together exceed inference - and showing the split is what turns "it feels slow" into a
 * specific thing to fix. The numbers come from the pipeline's own timings, recorded on the
 * job result, not from anything measured in the browser.
 */
export function StatsPanel({ job, result }: Props) {
  if (!job) {
    return (
      <div className="panel p-4 text-sm text-ink-400">
        Statistics appear here once a job has run.
      </div>
    );
  }

  const manifest = (job.result ?? {}) as Record<string, unknown>;
  const timings = (manifest.timings_ms ?? {}) as Record<string, number>;
  const run = job.runs?.[job.runs.length - 1];

  const rows: Row[] = [
    { label: 'Model', value: job.model_name },
    { label: 'Precision', value: job.precision },
    { label: 'Queue', value: job.queue },
    { label: 'Device', value: run?.device_name ?? run?.device ?? '—' },
    {
      label: 'Resolution',
      value:
        manifest.width && manifest.height ? `${manifest.width}×${manifest.height}` : '—',
    },
    {
      label: 'Attempts',
      value: String(job.attempts),
      hint: run?.oom_retry ? 'included an out-of-memory retry at a smaller batch size' : undefined,
    },
    {
      label: 'Wall clock',
      value: run?.duration_seconds ? formatMs(run.duration_seconds * 1000) : '—',
    },
    {
      label: 'Alpha coverage',
      value: formatPercent(manifest.alpha_coverage as number | undefined),
      hint: 'share of pixels the model considers foreground',
    },
    { label: 'Peak RSS', value: formatBytes(run?.peak_rss_bytes) },
  ];

  const stageOrder = ['decode', 'preprocess', 'inference', 'postprocess', 'refine', 'encode'];
  const stages = stageOrder
    .filter((name) => typeof timings[name] === 'number')
    .map((name) => ({ name, ms: timings[name] as number }));
  const stageTotal = stages.reduce((sum, stage) => sum + stage.ms, 0);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="text-sm font-semibold">Inference stats</h2>
        <StatusBadge status={job.status} />
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 p-4">
        {rows.map((row) => (
          <div key={row.label} title={row.hint}>
            <dt className="label">{row.label}</dt>
            <dd className="mt-0.5 truncate font-mono text-sm text-ink-100">{row.value}</dd>
          </div>
        ))}
      </dl>

      {stages.length > 0 && (
        <div className="border-t border-ink-700/70 p-4">
          <div className="flex items-baseline justify-between">
            <span className="label">Stage breakdown</span>
            <span className="font-mono text-xs text-ink-300">{formatMs(stageTotal)} total</span>
          </div>

          <div className="mt-2.5 flex h-2 overflow-hidden rounded-full bg-ink-800">
            {stages.map((stage, index) => (
              <div
                key={stage.name}
                className={STAGE_COLORS[index % STAGE_COLORS.length]}
                style={{ width: `${(stage.ms / stageTotal) * 100}%` }}
                title={`${stage.name}: ${formatMs(stage.ms)}`}
              />
            ))}
          </div>

          <ul className="mt-3 space-y-1">
            {stages.map((stage, index) => (
              <li key={stage.name} className="flex items-center gap-2 text-xs">
                <span
                  aria-hidden
                  className={`h-2 w-2 shrink-0 rounded-sm ${STAGE_COLORS[index % STAGE_COLORS.length]}`}
                />
                <span className="flex-1 text-ink-300">{stage.name}</span>
                <span className="font-mono text-ink-200">{formatMs(stage.ms)}</span>
                <span className="w-12 text-right font-mono text-ink-500">
                  {((stage.ms / stageTotal) * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result && result.outputs.length > 0 && (
        <div className="border-t border-ink-700/70 p-4">
          <span className="label">Outputs</span>
          <ul className="mt-2 space-y-1">
            {result.outputs.map((output) => (
              <li key={output.kind} className="flex items-center gap-2 text-xs">
                <span className="flex-1 text-ink-300">{output.kind}</span>
                <span className="font-mono text-ink-200">{formatBytes(output.size_bytes)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {job.error_message && (
        <div className="border-t border-ink-700/70 p-4">
          <span className="label text-bad">Error — {job.error_code}</span>
          <p className="mt-1 break-words text-xs text-bad/90">{job.error_message}</p>
        </div>
      )}
    </div>
  );
}

const STAGE_COLORS = [
  'bg-accent-deep',
  'bg-accent',
  'bg-accent-soft',
  'bg-good',
  'bg-warn',
  'bg-ink-500',
];

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'succeeded'
      ? 'border-good/40 bg-good/10 text-good'
      : status === 'failed'
        ? 'border-bad/40 bg-bad/10 text-bad'
        : status === 'cancelled'
          ? 'border-ink-600 bg-ink-800 text-ink-300'
          : 'border-accent/40 bg-accent/10 text-accent';
  return (
    <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${tone}`}>
      {status}
    </span>
  );
}
