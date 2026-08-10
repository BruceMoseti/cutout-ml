'use client';

import type { ModelInfo } from '@/lib/api';

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (name: string) => void;
  disabled?: boolean;
}

/**
 * Model picker.
 *
 * Models whose checkpoint is not on disk are rendered but disabled, with the reason
 * visible. Hiding them would be friendlier and less honest: the registry is the point of
 * the project, and "this architecture exists but its weights are not downloadable here"
 * is information the reader should see rather than a gap they have to infer.
 */
export function ModelSelector({ models, value, onChange, disabled }: Props) {
  const selected = models.find((m) => m.name === value);

  return (
    <div className="space-y-2">
      <label htmlFor="model" className="label">
        Model
      </label>
      <select
        id="model"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="field"
      >
        {models.map((model) => (
          <option key={model.name} value={model.name} disabled={!model.weights_available}>
            {model.name} — {model.architecture}
            {model.weights_available ? '' : ' (no weights on this host)'}
          </option>
        ))}
      </select>

      {selected && (
        <div className="space-y-1.5 rounded-lg border border-ink-700/60 bg-ink-850/60 p-3">
          <p className="text-xs leading-relaxed text-ink-300">{selected.description}</p>
          <div className="flex flex-wrap gap-1.5">
            <span className="chip">{selected.runtime}</span>
            <span className="chip">
              {selected.input_size[0]}×{selected.input_size[1]}
            </span>
            {selected.tags.map((tag) => (
              <span key={tag} className="chip">
                {tag}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-ink-500">{selected.license}</p>
        </div>
      )}
    </div>
  );
}
