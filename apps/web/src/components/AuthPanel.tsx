'use client';

import { useState } from 'react';
import { ApiError, api, tokenStore } from '@/lib/api';

interface Props {
  onAuthenticated: () => void;
}

/**
 * Sign in or register.
 *
 * Both actions land on the same form because the API issues a token from either, and a
 * separate "create account" page for a two-field form is friction with no benefit. The
 * 10-character minimum mirrors the server's rule so the failure is caught before a round
 * trip; the server remains the authority.
 */
export function AuthPanel({ onAuthenticated }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = mode === 'login' ? await api.login(email, password) : await api.register(email, password);
      tokenStore.set(result.access_token);
      onAuthenticated();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `${err.message}${err.requestId ? ` — request ${err.requestId.slice(0, 8)}` : ''}`
          : String(err),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-sm panel animate-fade-up p-6">
      <h1 className="text-lg font-semibold tracking-tight">
        {mode === 'login' ? 'Sign in' : 'Create an account'}
      </h1>
      <p className="mt-1 text-sm text-ink-400">
        Assets and jobs are scoped to your account; every read is checked against the owner.
      </p>

      <form onSubmit={submit} className="mt-5 space-y-3">
        <div>
          <label htmlFor="email" className="label">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="field mt-1"
            placeholder="you@example.com"
          />
        </div>
        <div>
          <label htmlFor="password" className="label">
            Password
          </label>
          <input
            id="password"
            type="password"
            required
            minLength={10}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="field mt-1"
            placeholder="at least 10 characters"
          />
        </div>

        {error && (
          <p role="alert" className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
            {error}
          </p>
        )}

        <button type="submit" disabled={busy} className="btn-primary w-full">
          {busy ? 'Working…' : mode === 'login' ? 'Sign in' : 'Create account'}
        </button>
      </form>

      <button
        type="button"
        onClick={() => {
          setMode(mode === 'login' ? 'register' : 'login');
          setError(null);
        }}
        className="mt-4 text-xs text-ink-400 underline-offset-4 hover:text-accent hover:underline"
      >
        {mode === 'login' ? 'Need an account? Register' : 'Already registered? Sign in'}
      </button>
    </div>
  );
}
