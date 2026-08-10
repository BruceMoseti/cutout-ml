import type { Metadata, Viewport } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'CutoutML — segmentation studio',
  description:
    'Upload an image or video, cut out the subject with a model trained in this repository, and read the real CPU-measured benchmark numbers.',
};

export const viewport: Viewport = {
  themeColor: '#07090c',
  width: 'device-width',
  initialScale: 1,
};

const NAV = [
  { href: '/', label: 'Studio' },
  { href: '/video', label: 'Video' },
  { href: '/benchmarks', label: 'Benchmarks' },
  { href: '/models', label: 'Models' },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="sticky top-0 z-40 border-b border-ink-800/80 bg-ink-950/80 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-5">
            <Link href="/" className="flex items-center gap-2.5">
              <span
                aria-hidden
                className="grid h-7 w-7 place-items-center rounded-md bg-accent/15 text-accent"
              >
                {/* A scissors-through-frame glyph: the product in one mark. */}
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <rect x="3" y="3" width="18" height="18" rx="3" />
                  <path d="M8 8l8 8M16 8l-8 8" />
                </svg>
              </span>
              <span className="text-sm font-semibold tracking-tight">CutoutML</span>
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-2.5 py-1.5 text-ink-300 transition-colors hover:bg-ink-850 hover:text-white"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="ml-auto flex items-center gap-3">
              <a
                href="/api/docs"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-ink-400 transition-colors hover:text-ink-100"
              >
                API docs ↗
              </a>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-5 py-7">{children}</main>
        <footer className="mx-auto max-w-[1400px] px-5 pb-10 pt-4 text-xs text-ink-500">
          Every number shown in this UI comes from a JSON artifact produced by a script in
          this repository. Benchmarks are CPU-measured; there is no GPU in the machine that
          produced them.
        </footer>
      </body>
    </html>
  );
}
