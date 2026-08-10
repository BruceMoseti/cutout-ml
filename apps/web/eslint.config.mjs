import { FlatCompat } from '@eslint/eslintrc';

/**
 * ESLint flat config.
 *
 * `eslint-config-next` still ships as an eslintrc-style config, so it is bridged through
 * FlatCompat rather than rewritten - reimplementing it would mean re-deriving Next's
 * rules for the App Router and drifting from them on every upgrade.
 */
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

export default [
  { ignores: ['.next/**', 'node_modules/**', 'coverage/**'] },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/no-explicit-any': 'error',
      eqeqeq: ['error', 'smart'],
    },
  },
];
