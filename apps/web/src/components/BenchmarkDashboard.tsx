'use client';

import { useMemo, useState } from 'react';
import { api, type BenchmarkCaseResult, type BenchmarkReport } from '@/lib/api';
import {
  formatBytes,
  formatDuration,
  formatMetric,
  formatMs,
  formatThroughput,
  formatTimestamp,
} from '@/lib/format';
import { useAsync } from '@/lib/hooks';

type SortKey = 'name' | 'iou' | 'mae' | 'p50' | 'throughput' | 'size';

/**
 * Benchmark dashboard.
 *
 * Reads the JSON that `benchmarks/run.py` wrote, via `GET /v1/benchmarks`. It never
 * computes anything: a latency measured in a browser tab while the API is serving the
 * request would be a measurement of the browser, and it would carry no provenance.
 *
 * The provenance block is shown above the table rather than tucked into a tooltip, because
 * a benchmark table without the CPU, the commit and the sample count attached is an
 * anecdote. Rows whose weights were random are rendered with their accuracy columns
 * explicitly blanked and labelled, so a latency-only row cannot be misread as an accuracy
 * claim.
 */
export function BenchmarkDashboard() {
  const reports = useAsync<BenchmarkReport[]>(async () => (await api.benchmarks(5)).items, []);
  const [sortKey, setSortKey] = useState<SortKey>('iou');
  const [descending, setDescending] = useState(true);

  const report = reports.data?.[0] ?? null;

  const rows = useMemo(() => {
    if (!report) return [];
    const cases = report.cases.filter((c) => c.status !== 'failed');
    const value = (c: BenchmarkCaseResult): number | string => {
      switch (sortKey) {
        case 'name':
          return c.case.name;
        case 'iou':
          return c.accuracy_valid ? (c.accuracy?.iou ?? -1) : -1;
        case 'mae':
          return c.accuracy_valid ? (c.accuracy?.mae ?? 2) : 2;
        case 'p50':
          return c.latency?.per_image_p50_ms ?? Number.POSITIVE_INFINITY;
        case 'throughput':
          return c.latency?.throughput_images_per_second ?? -1;
        case 'size':
          return c.model_size_bytes ?? -1;
      }
    };
    return [...cases].sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      const cmp = typeof va === 'string' ? String(va).localeCompare(String(vb)) : (va as number) - (vb as number);
      return descending ? -cmp : cmp;
    });
  }, [report, sortKey, descending]);

  if (reports.loading) {
    return <div className="h-64 animate-pulse rounded-xl border border-ink-800 bg-ink-900/40" />;
  }
  if (reports.error) {
    return <p className="rounded-lg border border-bad/40 bg-bad/10 p-3 text-sm text-bad">{reports.error}</p>;
  }
  if (!report) {
    return (
      <div className="panel p-6 text-sm text-ink-300">
        No benchmark results yet. Run <code className="font-mono text-accent">python benchmarks/run.py</code>{' '}
        and reload.
      </div>
    );
  }

  const env = report.environment as Record<string, unknown>;
  const dataset = report.dataset as Record<string, unknown>;
  const config = report.config as Record<string, unknown>;

  const provenance: [string, string][] = [
    ['CPU', String(env.cpu_model ?? env.hardware ?? 'unknown')],
    ['Cores', `${env.cpu_count_logical ?? env.cpu_count ?? '?'} logical`],
    ['GPU', String(env.gpu ?? 'none')],
    ['torch', String((env.libraries as Record<string, string> | undefined)?.torch ?? '?')],
    ['Commit', String(env.git_commit ?? '?').slice(0, 10) + (env.git_dirty ? ' (dirty)' : '')],
    ['Measured', formatTimestamp(report.created_at)],
    ['Dataset', String(dataset.dataset_id ?? dataset.name ?? 'synthetic')],
    ['Eval samples', String(config.accuracy_samples ?? '?')],
    ['Reps / warmup', `${config.repetitions ?? '?'} / ${config.warmup ?? '?'}`],
    ['Suite duration', formatDuration(report.duration_seconds)],
  ];

  const header: { key: SortKey; label: string; hint?: string }[] = [
    { key: 'name', label: 'Case' },
    { key: 'iou', label: 'IoU', hint: 'higher is better' },
    { key: 'mae', label: 'MAE', hint: 'lower is better' },
    { key: 'p50', label: 'p50 / img', hint: 'median per-image latency' },
    { key: 'throughput', label: 'Throughput' },
    { key: 'size', label: 'Model size' },
  ];

  return (
    <div className="space-y-5">
      <div className="panel">
        <div className="panel-header">
          <h2 className="text-sm font-semibold">Provenance</h2>
          <span className="font-mono text-[11px] text-ink-400">{report.run_id}</span>
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2.5 p-4 sm:grid-cols-3 lg:grid-cols-5">
          {provenance.map(([label, value]) => (
            <div key={label}>
              <dt className="label">{label}</dt>
              <dd className="mt-0.5 truncate font-mono text-xs text-ink-100" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="panel overflow-hidden">
        <div className="panel-header">
          <h2 className="text-sm font-semibold">
            Results — {rows.length} cases
          </h2>
          <span className="text-[11px] text-ink-400">
            CPU only. A CUDA GPU changes the latency columns entirely and the accuracy columns
            not at all.
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-ink-700/70 bg-ink-850/50">
                {header.map((column) => (
                  <th
                    key={column.key}
                    scope="col"
                    className="px-3 py-2 text-left text-[11px] font-medium uppercase tracking-wider text-ink-400"
                  >
                    <button
                      type="button"
                      title={column.hint}
                      onClick={() => {
                        if (sortKey === column.key) setDescending((d) => !d);
                        else {
                          setSortKey(column.key);
                          setDescending(column.key !== 'p50' && column.key !== 'mae' && column.key !== 'name');
                        }
                      }}
                      className="inline-flex items-center gap-1 hover:text-ink-100"
                    >
                      {column.label}
                      {sortKey === column.key && <span aria-hidden>{descending ? '↓' : '↑'}</span>}
                    </button>
                  </th>
                ))}
                <th scope="col" className="px-3 py-2 text-left text-[11px] font-medium uppercase tracking-wider text-ink-400">
                  Runtime
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.case.name} className="border-b border-ink-800/70 last:border-0 hover:bg-ink-850/40">
                  <td className="table-cell font-medium text-ink-100">
                    {row.case.name}
                    {row.status === 'skipped' && (
                      <span className="ml-2 chip !text-[10px]">skipped</span>
                    )}
                    {row.notes && (
                      <span className="ml-2 text-[11px] text-warn" title={row.notes}>
                        ⚠
                      </span>
                    )}
                  </td>
                  <td className="table-cell font-mono">
                    {row.accuracy_valid ? (
                      formatMetric(row.accuracy?.iou)
                    ) : (
                      <span className="text-ink-500" title="random weights: accuracy is not measurable">
                        n/a
                      </span>
                    )}
                  </td>
                  <td className="table-cell font-mono">
                    {row.accuracy_valid ? formatMetric(row.accuracy?.mae) : <span className="text-ink-500">n/a</span>}
                  </td>
                  <td className="table-cell font-mono">{formatMs(row.latency?.per_image_p50_ms)}</td>
                  <td className="table-cell font-mono">
                    {formatThroughput(row.latency?.throughput_images_per_second)}
                  </td>
                  <td className="table-cell font-mono text-ink-300">{formatBytes(row.model_size_bytes)}</td>
                  <td className="table-cell text-xs text-ink-300">{row.runtime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {reports.data && reports.data.length > 1 && (
        <div className="panel p-4">
          <span className="label">Earlier runs</span>
          <ul className="mt-2 space-y-1 text-xs">
            {reports.data.slice(1).map((earlier) => (
              <li key={earlier.run_id} className="flex flex-wrap items-center gap-3 text-ink-300">
                <span className="font-mono">{earlier.run_id}</span>
                <span>{formatTimestamp(earlier.created_at)}</span>
                <span>{String((earlier.summary as Record<string, unknown>).cases_ok ?? '?')} cases ok</span>
                <span className="text-ink-500">
                  best IoU {formatMetric((earlier.summary as Record<string, number>).best_iou)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
