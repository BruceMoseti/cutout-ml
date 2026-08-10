'use client';

import { BenchmarkDashboard } from '@/components/BenchmarkDashboard';
import { AuthPanel } from '@/components/AuthPanel';
import { useSession } from '@/lib/hooks';

export default function BenchmarksPage() {
  const { session, refresh } = useSession();

  if (!session.ready) {
    return <div className="h-64 animate-pulse rounded-xl border border-ink-800 bg-ink-900/40" />;
  }
  if (!session.authenticated) {
    return <AuthPanel onAuthenticated={() => void refresh()} />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Benchmarks</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          Every row comes from a JSON file under <code className="font-mono text-ink-200">benchmarks/results/</code>,
          written by <code className="font-mono text-ink-200">benchmarks/run.py</code>. Latency is measured after
          discarding warmup iterations and reported as percentiles rather than a mean, because a
          mean hides the tail that users actually notice. Rows built with random weights show
          latency only; their accuracy is marked <span className="font-mono">n/a</span> rather than
          filled in with a number that would be meaningless.
        </p>
      </div>
      <BenchmarkDashboard />
    </div>
  );
}
