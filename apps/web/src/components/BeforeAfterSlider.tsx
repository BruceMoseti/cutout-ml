'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

interface Props {
  beforeSrc: string;
  afterSrc: string;
  /** Rendered behind the "after" image, so transparency is visible rather than black. */
  afterBackground?: 'checker' | 'none';
  alt?: string;
}

/**
 * Draggable before/after comparison.
 *
 * Implementation notes that matter for it to feel right:
 *
 * - The two images are stacked and the *top* one is revealed with `clip-path: inset()`
 *   rather than by resizing a container. Resizing re-lays-out the image on every
 *   pointer move, which visibly reflows and lets the two halves drift out of alignment;
 *   clipping is compositor-only and stays pixel-aligned.
 * - Pointer events, not mouse events, so a touch drag works without a second code path.
 * - The handle is a real `role="slider"` with arrow-key support. A comparison slider that
 *   only responds to dragging is unusable with a keyboard, and this is the primary control
 *   on the page.
 */
export function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  afterBackground = 'checker',
  alt = 'Segmentation result',
}: Props) {
  const [position, setPosition] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updateFromClientX = useCallback((clientX: number) => {
    const node = containerRef.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    setPosition(Math.min(100, Math.max(0, ratio * 100)));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (event: PointerEvent) => updateFromClientX(event.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [dragging, updateFromClientX]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 10 : 2;
    if (event.key === 'ArrowLeft') setPosition((p) => Math.max(0, p - step));
    else if (event.key === 'ArrowRight') setPosition((p) => Math.min(100, p + step));
    else if (event.key === 'Home') setPosition(0);
    else if (event.key === 'End') setPosition(100);
    else return;
    event.preventDefault();
  };

  return (
    <div
      ref={containerRef}
      className="relative select-none overflow-hidden rounded-lg border border-ink-700 bg-ink-900"
      onPointerDown={(event) => {
        setDragging(true);
        updateFromClientX(event.clientX);
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={beforeSrc} alt={`${alt} (original)`} className="block w-full" draggable={false} />

      <div
        className={`absolute inset-0 ${afterBackground === 'checker' ? 'checkerboard' : ''}`}
        style={{ clipPath: `inset(0 0 0 ${position}%)` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={afterSrc} alt={alt} className="block h-full w-full object-contain" draggable={false} />
      </div>

      <div
        className="pointer-events-none absolute inset-y-0 w-px bg-accent/80 shadow-[0_0_12px_rgba(34,211,238,0.6)]"
        style={{ left: `${position}%` }}
      />

      <div
        role="slider"
        tabIndex={0}
        aria-label="Comparison position"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(position)}
        onKeyDown={onKeyDown}
        className="absolute top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize
          place-items-center rounded-full border border-accent/60 bg-ink-900/90 text-accent
          transition-transform hover:scale-105"
        style={{ left: `${position}%` }}
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 6l-4 6 4 6M15 6l4 6-4 6" />
        </svg>
      </div>

      <span className="pointer-events-none absolute left-3 top-3 chip">Original</span>
      <span className="pointer-events-none absolute right-3 top-3 chip">Cut out</span>
    </div>
  );
}
