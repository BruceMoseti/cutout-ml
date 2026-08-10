/**
 * The API client's job is the three things no call site should repeat: attach the bearer
 * token, unwrap the error envelope, and keep the `request_id` reachable so a UI message
 * can quote the same id that appears in the server logs. Each is asserted here against a
 * stubbed `fetch`, because a mistake in any of them is invisible until a user hits it.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, api, fetchOutputObjectUrl, tokenStore } from './api';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function lastRequest(): { url: string; init: RequestInit; headers: Headers } {
  const [url, init] = fetchMock.mock.calls.at(-1) as [string, RequestInit];
  return { url, init, headers: new Headers(init.headers) };
}

describe('authentication', () => {
  it('attaches the stored token to every request', async () => {
    tokenStore.set('a-stored-token');
    fetchMock.mockResolvedValue(jsonResponse({ items: [], default_model: 'cutoutnet' }));

    await api.models();

    expect(lastRequest().headers.get('Authorization')).toBe('Bearer a-stored-token');
  });

  it('sends no Authorization header when there is no token', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ items: [], default_model: 'cutoutnet' }));

    await api.models();

    expect(lastRequest().headers.has('Authorization')).toBe(false);
  });
});

describe('request bodies', () => {
  it('declares JSON for a serialised body', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ access_token: 't', user_id: 'u', expires_in: 1 }));

    await api.login('someone@example.com', 'a password');

    expect(lastRequest().headers.get('Content-Type')).toBe('application/json');
  });

  it('leaves FormData alone so the browser can set the multipart boundary', async () => {
    // Setting Content-Type by hand here would omit the boundary and the upload would
    // arrive unparseable, which is the classic multipart bug.
    fetchMock.mockResolvedValue(jsonResponse({ id: 'asset-1' }, 201));

    await api.uploadAsset(new File(['bytes'], 'photo.png', { type: 'image/png' }), 'image');

    const { init, headers } = lastRequest();
    expect(headers.has('Content-Type')).toBe(false);
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('kind')).toBe('image');
  });
});

describe('error handling', () => {
  it('unwraps the envelope into a typed error that keeps the request id', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: 'asset_not_ready',
            message: 'asset is awaiting_upload; upload its content before processing',
            request_id: 'abc123',
            details: { status: 'awaiting_upload' },
          },
        },
        409,
      ),
    );

    const error = await api.job('job-1').catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(409);
    expect(apiError.code).toBe('asset_not_ready');
    expect(apiError.requestId).toBe('abc123');
    expect(apiError.details).toEqual({ status: 'awaiting_upload' });
    expect(apiError.message).toContain('upload its content');
  });

  it('still throws a usable error when the body is not an envelope', async () => {
    // A proxy or load balancer can return HTML or an empty body, and the client must not
    // fail with a JSON parse error that hides the real status.
    fetchMock.mockResolvedValue(new Response('', { status: 502, statusText: 'Bad Gateway' }));

    const error = (await api.models().catch((e: unknown) => e)) as ApiError;

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(502);
    expect(error.code).toBe('unknown');
  });

  it('treats 204 as an empty success rather than parsing the body', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await expect(api.cancelJob('job-1')).resolves.toBeUndefined();
  });
});

describe('authenticated output downloads', () => {
  it('fetches through the API with the token, because an img src cannot', async () => {
    tokenStore.set('download-token');
    fetchMock.mockResolvedValue(new Response(new Blob(['png']), { status: 200 }));

    const url = await fetchOutputObjectUrl('/v1/jobs/job-1/outputs/transparent_png');

    expect(url.startsWith('blob:')).toBe(true);
    expect(lastRequest().url).toBe('/api/v1/jobs/job-1/outputs/transparent_png');
    expect(lastRequest().headers.get('Authorization')).toBe('Bearer download-token');
  });

  it('reports a failed download as an ApiError rather than an opaque rejection', async () => {
    fetchMock.mockResolvedValue(
      new Response('', { status: 403, headers: { 'X-Request-ID': 'req-9' } }),
    );

    const error = (await fetchOutputObjectUrl('/v1/jobs/j/outputs/mask_png').catch(
      (e: unknown) => e,
    )) as ApiError;

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(403);
    expect(error.requestId).toBe('req-9');
  });
});

describe('token storage', () => {
  it('round-trips and clears', () => {
    expect(tokenStore.get()).toBeNull();
    tokenStore.set('t');
    expect(tokenStore.get()).toBe('t');
    tokenStore.clear();
    expect(tokenStore.get()).toBeNull();
  });
});
