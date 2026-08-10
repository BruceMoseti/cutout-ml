/**
 * Display formatting helpers.
 *
 * Kept in one module because the benchmark dashboard and the inference panel must agree:
 * a latency shown as `12.4 ms` in one place and `0.0124 s` in the other makes two real
 * numbers look like a contradiction.
 */

/** Bytes as a human-readable size. Binary units, because that is what disk tools report. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const exponent = Math.min(Math.floor(Math.log2(Math.abs(bytes)) / 10), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(exponent === 0 ? 0 : value < 10 ? 2 : 1)} ${units[exponent]}`;
}

/**
 * Milliseconds with a precision that scales to the magnitude.
 *
 * Sub-millisecond values need two decimals to be distinguishable at all, while a
 * 4-second video frame does not benefit from `4321.87 ms`.
 */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—';
  if (ms < 1) return `${ms.toFixed(3)} ms`;
  if (ms < 100) return `${ms.toFixed(2)} ms`;
  if (ms < 10_000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatThroughput(imagesPerSecond: number | null | undefined): string {
  if (!imagesPerSecond) return '—';
  if (imagesPerSecond < 10) return `${imagesPerSecond.toFixed(2)} img/s`;
  return `${imagesPerSecond.toFixed(1)} img/s`;
}

/** A 0..1 metric as a percentage, or an em dash when the metric is not valid. */
export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function formatPercent(fraction: number | null | undefined): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return '—';
  return `${(fraction * 100).toFixed(1)}%`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** A short, readable label for a model name in a dense table. */
export function shortModelName(name: string): string {
  return name.replace(/^cutoutnet/, 'CutoutNet').replace(/^u2net/, 'U²-Net').replace(/^birefnet/, 'BiRefNet');
}
