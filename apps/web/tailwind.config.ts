import type { Config } from 'tailwindcss';

/**
 * The palette is a single dark scale plus one accent.
 *
 * Two constraints drove it. First, this is an app for looking at cut-out images, so the
 * chrome has to recede: near-black surfaces with low-chroma borders keep attention on the
 * canvas, and a light theme would make every transparent PNG look washed out against it.
 * Second, the accent is cyan rather than the usual indigo because the checkerboard used
 * for transparency is neutral grey and indigo reads as "part of the image" against it.
 */
const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#07090c',
          900: '#0b0e13',
          850: '#10141b',
          800: '#151a23',
          700: '#1d2430',
          600: '#2a3341',
          500: '#3d4859',
          400: '#5b6779',
          300: '#8b96a6',
          200: '#bcc4d0',
          100: '#e6eaf0',
        },
        accent: {
          DEFAULT: '#22d3ee',
          soft: '#67e8f9',
          deep: '#0e7490',
        },
        good: '#34d399',
        warn: '#fbbf24',
        bad: '#f87171',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      backgroundImage: {
        // The standard transparency checkerboard, as two offset conic gradients so it
        // needs no image asset and scales with `bg-size`.
        checker:
          'conic-gradient(from 90deg at 1px 1px, #0000 25%, #ffffff0f 0) 0 0/16px 16px, #1a1f28',
        'grid-fade':
          'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(34,211,238,0.10), transparent)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 240ms cubic-bezier(0.16, 1, 0.3, 1) both',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
};

export default config;
