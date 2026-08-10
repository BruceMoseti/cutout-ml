/**
 * Typed client for the CutoutML API.
 *
 * Everything goes through {@link request}, which does three things no call site should
 * repeat: attach the bearer token, unwrap the error envelope into a typed
 * {@link ApiError}, and surface the `request_id` so a UI error message can quote the
 * same id that appears in the server logs.
 *
 * The base URL is `/api` and Next rewrites it to the FastAPI service, so requests are
 * same-origin and need no CORS preflight.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '/api';

export type Precision = 'fp32' | 'fp16' | 'bf16';

export type ImageOutput =
  | 'transparent_png'
  | 'transparent_webp'
  | 'mask_png'
  | 'color_composite'
  | 'background_composite'
  | 'blurred_background';

export type JobStatus = 'pending' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface ErrorEnvelope {
  code: string;
  message: string;
  request_id: string | null;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly details?: Record<string, unknown>;

  constructor(status: number, envelope: ErrorEnvelope) {
    super(envelope.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = envelope.code;
    this.requestId = envelope.request_id;
    this.details = envelope.details;
  }
}

export interface Asset {
  id: string;
  kind: 'image' | 'video';
  status: string;
  original_filename: string | null;
  content_type: string | null;
  size_bytes: number;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  frame_count: number | null;
  fps: number | null;
  created_at: string;
  storage_backend: string;
}

export interface JobRun {
  attempt: number;
  status: string;
  device: string | null;
  device_name: string | null;
  batch_size: number | null;
  oom_retry: boolean;
  retryable_error: boolean | null;
  error_code: string | null;
  duration_seconds: number | null;
  frames_processed: number | null;
  peak_rss_bytes: number | null;
  peak_vram_bytes: number | null;
}

export interface Job {
  id: string;
  asset_id: string;
  status: JobStatus;
  kind: 'image' | 'video';
  model_name: string;
  precision: string;
  queue: string;
  progress: number;
  progress_message: string | null;
  attempts: number;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  runs?: JobRun[];
}

export interface ResultOutput {
  kind: string;
  storage_key: string;
  url: string;
  size_bytes: number;
  content_type: string;
}

export interface JobResult {
  job_id: string;
  asset_id: string;
  status: string;
  outputs: ResultOutput[];
  metrics: Record<string, unknown> | null;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  runtime: string;
  input_size: number[];
  license: string;
  source: string;
  description: string;
  tags: string[];
  weights_available: boolean;
  supports_random_init: boolean;
  default_weights: string | null;
}

export interface ModelList {
  items: ModelInfo[];
  default_model: string;
}

export interface BenchmarkCaseResult {
  case: {
    name: string;
    model: string;
    precision: string;
    batch_size: number;
    resolution: number[] | null;
    random_init: boolean;
    compile: boolean;
  };
  status: 'ok' | 'skipped' | 'failed';
  runtime: string;
  accuracy: Record<string, number> | null;
  accuracy_valid: boolean;
  model_size_bytes: number | null;
  stage_timings_ms: Record<string, number> | null;
  notes: string;
  error: string | null;
  latency: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    mean_ms: number;
    stddev_ms: number;
    per_image_p50_ms: number;
    throughput_images_per_second: number;
    cold_start_seconds: number | null;
    first_inference_ms: number | null;
    peak_rss_bytes: number;
    repetitions: number;
    warmup: number;
    batch_size: number;
  } | null;
}

export interface BenchmarkReport {
  schema_version: number;
  run_id: string;
  created_at: string;
  duration_seconds: number;
  environment: Record<string, unknown>;
  config: Record<string, unknown>;
  dataset: Record<string, unknown>;
  cases: BenchmarkCaseResult[];
  summary: Record<string, unknown>;
}

const TOKEN_KEY = 'cutoutml.token';

/**
 * Token storage.
 *
 * `localStorage` is used because this is a demo console with no server-rendered session,
 * and it is the honest choice to document rather than hide: a token in `localStorage` is
 * readable by any script on the origin, so an XSS is an account takeover. A production
 * deployment should move to an httpOnly, `SameSite=Strict` refresh cookie. See
 * `docs/security.md`.
 */
export const tokenStore = {
  get(): string | null {
    if (typeof window === 'undefined') return null;
    return window.localStorage.getItem(TOKEN_KEY);
  },
  set(token: string): void {
    window.localStorage.setItem(TOKEN_KEY, token);
  },
  clear(): void {
    window.localStorage.removeItem(TOKEN_KEY);
  },
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = tokenStore.get();
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const payload: unknown = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const envelope =
      payload && typeof payload === 'object' && 'error' in payload
        ? ((payload as { error: ErrorEnvelope }).error)
        : { code: 'unknown', message: response.statusText, request_id: null };
    throw new ApiError(response.status, envelope);
  }
  return payload as T;
}

export const api = {
  register: (email: string, password: string) =>
    request<{ access_token: string; user_id: string; expires_in: number }>('/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<{ access_token: string; user_id: string; expires_in: number }>('/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  me: () => request<{ id: string; email: string; is_admin: boolean }>('/v1/auth/me'),

  models: () => request<ModelList>('/v1/models'),

  uploadAsset: (file: File, kind: 'image' | 'video' = 'image') => {
    const form = new FormData();
    form.append('file', file);
    form.append('kind', kind);
    return request<Asset>('/v1/assets', { method: 'POST', body: form });
  },

  listAssets: (limit = 50) => request<{ items: Asset[]; total: number }>(`/v1/assets?limit=${limit}`),

  process: (assetId: string, body: Record<string, unknown>) =>
    request<Job>(`/v1/assets/${assetId}/process`, { method: 'POST', body: JSON.stringify(body) }),

  job: (jobId: string) => request<Job>(`/v1/jobs/${jobId}`),

  jobResult: (jobId: string) => request<JobResult>(`/v1/jobs/${jobId}/result`),

  cancelJob: (jobId: string) => request<Job>(`/v1/jobs/${jobId}/cancel`, { method: 'POST' }),

  benchmarks: (limit = 3) =>
    request<{ items: BenchmarkReport[]; total_files: number }>(`/v1/benchmarks?limit=${limit}`),

  health: () => request<{ status: string; checks: { name: string; ok: boolean; detail: string }[] }>(
    '/health/ready',
  ),
};

/**
 * Fetch an authenticated output as a blob URL.
 *
 * Outputs are served through the API so ownership is enforced on reads, which means an
 * `<img src>` cannot fetch them directly - the browser will not attach the bearer token.
 * Fetching to a blob URL is the workaround; callers must revoke the URL when done or the
 * blob leaks for the lifetime of the document.
 */
export async function fetchOutputObjectUrl(url: string): Promise<string> {
  const token = tokenStore.get();
  const target = url.startsWith('http') ? url : `${API_BASE}${url}`;
  const response = await fetch(target, {
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
  if (!response.ok) {
    throw new ApiError(response.status, {
      code: 'output_fetch_failed',
      message: `could not fetch output (${response.status})`,
      request_id: response.headers.get('X-Request-ID'),
    });
  }
  return URL.createObjectURL(await response.blob());
}
