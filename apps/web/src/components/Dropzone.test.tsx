/**
 * The drop zone's two non-obvious behaviours, asserted rather than assumed: the
 * dragenter/dragleave depth counter (without it the highlight strobes as the pointer
 * crosses child elements) and the client-side size guard.
 */

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Dropzone } from './Dropzone';

function file(name: string, bytes: number): File {
  const f = new File(['x'], name, { type: 'image/png' });
  // File size is read-only, and the guard exists to be tested.
  Object.defineProperty(f, 'size', { value: bytes });
  return f;
}

function dropzone(): HTMLElement {
  return screen.getByText(/drag files here/i).closest('label') as HTMLElement;
}

describe('Dropzone', () => {
  it('hands dropped files to the caller', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} />);

    const dropped = file('photo.png', 1024);
    fireEvent.drop(dropzone(), { dataTransfer: { files: [dropped] } });

    expect(onFiles).toHaveBeenCalledWith([dropped]);
  });

  it('keeps only the first file unless multiple is set', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} />);

    const a = file('a.png', 10);
    const b = file('b.png', 10);
    fireEvent.drop(dropzone(), { dataTransfer: { files: [a, b] } });

    expect(onFiles).toHaveBeenCalledWith([a]);
  });

  it('passes every file through when multiple is set', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} multiple />);

    const a = file('a.png', 10);
    const b = file('b.png', 10);
    fireEvent.drop(dropzone(), { dataTransfer: { files: [a, b] } });

    expect(onFiles).toHaveBeenCalledWith([a, b]);
  });

  it('refuses a file over the limit locally and names it', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} maxBytes={1024} />);

    fireEvent.drop(dropzone(), { dataTransfer: { files: [file('huge.png', 4096)] } });

    expect(onFiles).not.toHaveBeenCalled();
    expect(screen.getByText(/huge\.png exceeds/i)).toBeInTheDocument();
  });

  it('stays highlighted while the pointer crosses child elements', () => {
    render(<Dropzone onFiles={vi.fn()} />);
    const label = dropzone();

    fireEvent.dragEnter(label);
    expect(screen.getByText(/drop to upload/i)).toBeInTheDocument();

    // Entering a child fires dragenter again and then dragleave for the parent; a naive
    // handler would clear the highlight here.
    fireEvent.dragEnter(label);
    fireEvent.dragLeave(label);
    expect(screen.getByText(/drop to upload/i)).toBeInTheDocument();

    fireEvent.dragLeave(label);
    expect(screen.getByText(/drag files here/i)).toBeInTheDocument();
  });

  it('ignores a drop while disabled', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} disabled />);

    fireEvent.drop(dropzone(), { dataTransfer: { files: [file('photo.png', 10)] } });

    expect(onFiles).not.toHaveBeenCalled();
  });

  it('exposes a real file input, so click-to-browse and the keyboard work', () => {
    const onFiles = vi.fn();
    render(<Dropzone onFiles={onFiles} accept="video/*" />);

    const input = dropzone().querySelector('input[type=file]') as HTMLInputElement;
    expect(input.accept).toBe('video/*');

    const chosen = file('clip.mp4', 2048);
    fireEvent.change(input, { target: { files: [chosen] } });
    expect(onFiles).toHaveBeenCalledWith([chosen]);
  });
});
