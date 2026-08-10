'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, api, tokenStore, type Job } from './api';

export interface Session {
  email: string | null;
  ready: boolean;
  authenticated: boolean;
}

/**
 * Resolve the stored token to a session on mount.
 *
 * `ready` exists separately from `authenticated` so the UI can render a neutral skeleton
 * instead of flashing the login form for one frame on every reload - the token is in
 * `localStorage`, which is unavailable during server rendering, so the first client render
 * always starts unauthenticated.
 */
export function useSession() {
  const [session, setSession] = useState<Session>({
    email: null,
    ready: false,
    authenticated: false,
  });

  const refresh = useCallback(async () => {
    if (!tokenStore.get()) {
      setSession({ email: null, ready: true, authenticated: false });
      return;
    }
    try {
      const me = await api.me();
      setSession({ email: me.email, ready: true, authenticated: true });
    } catch {
      tokenStore.clear();
      setSession({ email: null, ready: true, authenticated: false });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signOut = useCallback(() => {
    tokenStore.clear();
    setSession({ email: null, ready: true, authenticated: false });
  }, []);

  return { session, refresh, signOut };
}

const TERMINAL: ReadonlySet<string> = new Set(['succeeded', 'failed', 'cancelled']);

/**
 * Poll a job until it reaches a terminal state.
 *
 * Backs off from 400 ms to 3 s. A fixed short interval is wasteful for a five-minute
 * video job and a fixed long one makes a 40 ms image job feel slow, and neither is worth
 * a WebSocket for a poll that a single indexed SELECT answers.
 */
export function useJobPolling(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    let cancelled = false;
    let delay = 400;

    const tick = async () => {
      try {
        const next = await api.job(jobId);
        if (cancelled) return;
        setJob(next);
        if (TERMINAL.has(next.status)) return;
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
        return;
      }
      delay = Math.min(3000, Math.round(delay * 1.45));
      timer.current = setTimeout(() => void tick(), delay);
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [jobId]);

  return { job, error };
}

/** Load a value once on mount, exposing loading and error states. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loader()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading };
}
