'use client';

import { useCallback, useRef, useState } from 'react';
import { formatBytes } from '@/lib/format';

interface Props {
  onFiles: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxBytes?: number;
  hint?: string;
  disabled?: boolean;
}

/**
 * Drag-and-drop file input.
 *
 * Two details that most drop zones get wrong:
 *
 * - `dragenter`/`dragleave` fire for every descendant element, so a naive
 *   `onDragLeave={() => setActive(false)}` makes the highlight strobe as the pointer
 *   crosses child boundaries. This keeps a depth counter instead.
 * - The visible drop target is a `<label>` wired to a real `<input type="file">`, so
 *   click-to-browse and keyboard activation work for free rather than being simulated
 *   with a click handler that screen readers cannot find.
 */
export function Dropzone({
  onFiles,
  accept = 'image/*',
  multiple = false,
  maxBytes,
  hint,
  disabled = false,
}: Props) {
  const [active, setActive] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);
  const depth = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept_ = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const list = Array.from(files);
      const tooBig = maxBytes ? list.filter((f) => f.size > maxBytes) : [];
      if (tooBig.length > 0) {
        setRejected(
          `${tooBig.map((f) => f.name).join(', ')} exceeds the ${formatBytes(maxBytes)} limit`,
        );
        return;
      }
      setRejected(null);
      onFiles(multiple ? list : list.slice(0, 1));
    },
    [maxBytes, multiple, onFiles],
  );

  return (
    <div>
      <label
        onDragEnter={(event) => {
          event.preventDefault();
          depth.current += 1;
          setActive(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          depth.current -= 1;
          if (depth.current <= 0) {
            depth.current = 0;
            setActive(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          depth.current = 0;
          setActive(false);
          if (!disabled) accept_(event.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border
          border-dashed px-6 py-10 text-center transition-colors
          ${active ? 'border-accent bg-accent/5' : 'border-ink-600 hover:border-ink-500 hover:bg-ink-900/50'}
          ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          disabled={disabled}
          className="sr-only"
          onChange={(event) => {
            accept_(event.target.files);
            // Reset so re-selecting the same file fires `change` again.
            event.target.value = '';
          }}
        />
        <svg
          viewBox="0 0 24 24"
          aria-hidden
          className={`h-8 w-8 ${active ? 'text-accent' : 'text-ink-400'}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
        >
          <path d="M12 16V4m0 0L8 8m4-4l4 4" />
          <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
        </svg>
        <span className="text-sm font-medium text-ink-100">
          {active ? 'Drop to upload' : 'Drag files here, or click to browse'}
        </span>
        {hint && <span className="text-xs text-ink-400">{hint}</span>}
      </label>
      {rejected && <p className="mt-2 text-xs text-bad">{rejected}</p>}
    </div>
  );
}
