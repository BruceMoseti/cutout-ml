'use client';

import { AuthPanel } from '@/components/AuthPanel';
import { Studio } from '@/components/Studio';
import { useSession } from '@/lib/hooks';

export default function StudioPage() {
  const { session, refresh, signOut } = useSession();

  if (!session.ready) {
    return <div className="h-64 animate-pulse rounded-xl border border-ink-800 bg-ink-900/40" />;
  }

  if (!session.authenticated) {
    return <AuthPanel onAuthenticated={() => void refresh()} />;
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Segmentation studio</h1>
          <p className="mt-1 text-sm text-ink-400">
            Drop an image, pick a model, and compare the cut-out against the original.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-ink-400">
          <span className="font-mono">{session.email}</span>
          <button type="button" onClick={signOut} className="btn-ghost !py-1 !text-xs">
            Sign out
          </button>
        </div>
      </div>
      <Studio />
    </div>
  );
}
