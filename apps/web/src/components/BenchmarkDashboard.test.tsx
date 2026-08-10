import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { BenchmarkCaseResult, BenchmarkReport } from '@/lib/api';

const benchmarks = vi.fn();
vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return { ...actual, api: { ...actual.api, benchmarks } };
});

const { BenchmarkDashboard } = await import('./BenchmarkDashboard');

/**
 * The dashboard's job is not to draw a table, it is to refuse to publish a latency
 * figure without the caveats attached to it. These tests pin the caveats: a row timed
 * on a busy machine must be marked in place, a thread count must be visible, and a row
 * with random weights must not show an accuracy. Each of those can regress silently
 * while the table still looks correct, which is exactly why they are asserted here and
 * not left to review.
 */

function caseResult(overrides: Partial<BenchmarkCaseResult> = {}): BenchmarkCaseResult {
  return {
    case: {
      name: 'cutoutnet',
      model: 'cutoutnet',
      precision: 'fp32',
      batch_size: 1,
      resolution: null,
      random_init: false,
      compile: false,
    },
    status: 'ok',
    runtime: 'pytorch-eager',
    accuracy: { iou: 0.8544, mae: 0.0573 },
    accuracy_valid: true,
    model_size_bytes: 4_718_592,
    stage_timings_ms: null,
    notes: '',
    error: null,
    latency_trustworthy: true,
    load: null,
    latency: {
      threads: 1,
      p50_ms: 46.7,
      p95_ms: 50.0,
      p99_ms: 52.0,
      mean_ms: 47.0,
      stddev_ms: 1.0,
      per_image_p50_ms: 46.7,
      throughput_images_per_second: 21.4,
      cold_start_seconds: 0.05,
      first_inference_ms: 60.0,
      peak_rss_bytes: 400_000_000,
      repetitions: 20,
      warmup: 3,
      batch_size: 1,
    },
    ...overrides,
  };
}

function report(cases: BenchmarkCaseResult[], config: Record<string, unknown> = {}): BenchmarkReport {
  return {
    schema_version: 2,
    run_id: '20260101T000000Z-abcd1234',
    created_at: '2026-01-01T00:00:00Z',
    duration_seconds: 120,
    environment: {
      cpu_model: 'Intel(R) Xeon(R) Processor',
      cpu_count_logical: 8,
      gpu: 'none',
      libraries: { torch: '2.13.0+cpu' },
      git_commit: 'abc1234567',
      git_dirty: false,
    },
    config: { accuracy_samples: 64, repetitions: 20, warmup: 3, threads: 1, ...config },
    dataset: { dataset_id: 'synthetic-v1.0.0' },
    cases,
    summary: { cases_ok: cases.length },
  };
}

describe('BenchmarkDashboard', () => {
  beforeEach(() => {
    benchmarks.mockReset();
  });

  it('shows the measured row with its thread count', async () => {
    benchmarks.mockResolvedValue({ items: [report([caseResult()])] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('cutoutnet')).toBeInTheDocument());
    expect(screen.getByText('0.854')).toBeInTheDocument();
    expect(screen.getByText('46.70 ms')).toBeInTheDocument();
    expect(screen.getByText('/1t')).toBeInTheDocument();
    expect(screen.getByText('1 per runtime')).toBeInTheDocument();
  });

  it('marks a row measured under contention and explains the marker', async () => {
    benchmarks.mockResolvedValue({
      items: [
        report([
          caseResult(),
          caseResult({
            case: { ...caseResult().case, name: 'u2net' },
            latency_trustworthy: false,
            load: {
              external_busy_cores: 7.5,
              logical_cpus: 8,
              quiet: false,
              summary: 'CONTENDED: 7.5 of 8 cores busy with external work',
            },
          }),
        ]),
      ],
    });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('u2net')).toBeInTheDocument());
    const marker = screen.getByTitle('CONTENDED: 7.5 of 8 cores busy with external work');
    expect(marker).toHaveTextContent('†');
    expect(screen.getByText(/1 of 2 rows were timed/)).toBeInTheDocument();
    expect(screen.getByText(/upper bounds/)).toBeInTheDocument();
    expect(screen.getByText(/accuracy columns are unaffected/)).toBeInTheDocument();
  });

  it('carries no contention marker when every row was measured on a quiet machine', async () => {
    benchmarks.mockResolvedValue({ items: [report([caseResult()])] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('cutoutnet')).toBeInTheDocument());
    expect(screen.queryByText('†')).not.toBeInTheDocument();
    expect(screen.queryByText(/upper bounds/)).not.toBeInTheDocument();
  });

  it('states that single-threaded latency is pessimistic rather than letting it read as best-case', async () => {
    benchmarks.mockResolvedValue({ items: [report([caseResult()])] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('cutoutnet')).toBeInTheDocument());
    expect(screen.getByText(/a per-core cost/)).toBeInTheDocument();
    expect(screen.getByText(/dedicated machine would beat/)).toBeInTheDocument();
  });

  it('omits the thread caveat when the run was not single-threaded', async () => {
    benchmarks.mockResolvedValue({ items: [report([caseResult()], { threads: 0 })] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('cutoutnet')).toBeInTheDocument());
    expect(screen.getByText('one per core')).toBeInTheDocument();
    expect(screen.queryByText(/dedicated machine would beat/)).not.toBeInTheDocument();
  });

  it('blanks accuracy for a random-weights row instead of printing a number', async () => {
    benchmarks.mockResolvedValue({
      items: [
        report([
          caseResult({
            case: { ...caseResult().case, name: 'birefnet', random_init: true },
            accuracy: null,
            accuracy_valid: false,
            notes: 'accuracy: n/a - random weights (latency only)',
          }),
        ]),
      ],
    });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('birefnet')).toBeInTheDocument());
    expect(screen.getAllByText('n/a')).toHaveLength(2);
    expect(screen.getByText('46.70 ms')).toBeInTheDocument();
  });

  it('reports a run recorded before threads were pinned as not pinned', async () => {
    const older = report([caseResult({ latency_trustworthy: null })]);
    delete (older.config as Record<string, unknown>).threads;
    benchmarks.mockResolvedValue({ items: [older] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText('not pinned')).toBeInTheDocument());
    expect(screen.queryByText('†')).not.toBeInTheDocument();
  });

  it('prompts for a run rather than rendering an empty table', async () => {
    benchmarks.mockResolvedValue({ items: [] });
    render(<BenchmarkDashboard />);

    await waitFor(() => expect(screen.getByText(/No benchmark results yet/)).toBeInTheDocument());
  });
});
