import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/**
 * Vitest rather than Jest: it reuses the Vite transform pipeline, so TS + JSX need no
 * separate Babel configuration, and a watch run starts in well under a second.
 *
 * `environment: 'jsdom'` is required because the tests exercise the comparison slider's
 * pointer and keyboard behaviour, which is the component most likely to break silently.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
});
