'use client';

import { AuthPanel } from '@/components/AuthPanel';
import { api, type ModelInfo } from '@/lib/api';
import { useAsync, useSession } from '@/lib/hooks';

export default function ModelsPage() {
  const { session, refresh } = useSession();
  const models = useAsync<{ items: ModelInfo[]; default_model: string }>(() => api.models(), []);

  if (!session.ready) {
    return <div className="h-64 animate-pulse rounded-xl border border-ink-800 bg-ink-900/40" />;
  }
  if (!session.authenticated) {
    return <AuthPanel onAuthenticated={() => void refresh()} />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Model registry</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          Adding a model is one registry entry plus an adapter class — the API, the pipelines,
          the worker and the benchmark harness are written against a single interface and never
          change. Entries marked <span className="text-warn">no weights on this host</span> are
          architectures that exist in the repository but whose checkpoints are not present;
          they are listed rather than hidden.
        </p>
      </div>

      {models.error && (
        <p className="rounded-lg border border-bad/40 bg-bad/10 p-3 text-sm text-bad">{models.error}</p>
      )}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {models.data?.items.map((model) => (
          <article key={model.name} className="panel flex flex-col p-4">
            <div className="flex items-start justify-between gap-2">
              <h2 className="font-mono text-sm font-semibold text-ink-100">{model.name}</h2>
              {model.name === models.data?.default_model && <span className="chip">default</span>}
            </div>
            <p className="mt-0.5 text-xs text-ink-400">{model.architecture}</p>
            <p className="mt-2.5 flex-1 text-xs leading-relaxed text-ink-300">{model.description}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              <span className="chip">{model.runtime}</span>
              <span className="chip">
                {model.input_size[0]}×{model.input_size[1]}
              </span>
              {model.tags.map((tag) => (
                <span key={tag} className="chip">
                  {tag}
                </span>
              ))}
            </div>
            <p
              className={`mt-3 text-[11px] font-medium ${
                model.weights_available ? 'text-good' : 'text-warn'
              }`}
            >
              {model.weights_available ? 'weights present' : 'no weights on this host'}
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-500">{model.license}</p>
          </article>
        ))}
      </div>
    </div>
  );
}
