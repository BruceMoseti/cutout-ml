'use client';

interface Props {
  fraction: number;
  label?: string | null;
  /** Show an animated indeterminate bar when the backend has not reported progress yet. */
  indeterminate?: boolean;
}

export function ProgressBar({ fraction, label, indeterminate = false }: Props) {
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-ink-300">{label ?? 'Working…'}</span>
        {!indeterminate && <span className="font-mono text-ink-200">{pct}%</span>}
      </div>
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : pct}
        className="relative mt-1.5 h-1.5 overflow-hidden rounded-full bg-ink-800"
      >
        {indeterminate ? (
          <div className="absolute inset-y-0 w-1/3 animate-shimmer rounded-full bg-accent/70" />
        ) : (
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  );
}
