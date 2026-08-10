import { describe, expect, it } from 'vitest';
import {
  formatBytes,
  formatDuration,
  formatMetric,
  formatMs,
  formatPercent,
  formatThroughput,
} from './format';

describe('formatBytes', () => {
  it('uses binary units', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1024)).toBe('1.00 KiB');
    expect(formatBytes(4_743_375)).toBe('4.52 MiB');
  });

  it('renders an em dash for absent values rather than 0 B', () => {
    // A missing measurement and a zero-byte file are different facts; showing "0 B" for
    // an unmeasured model size would be a fabricated number.
    expect(formatBytes(null)).toBe('—');
    expect(formatBytes(undefined)).toBe('—');
  });
});

describe('formatMs', () => {
  it('scales precision to magnitude', () => {
    expect(formatMs(0.4213)).toBe('0.421 ms');
    expect(formatMs(12.345)).toBe('12.35 ms');
    expect(formatMs(1234.5)).toBe('1234.5 ms');
    expect(formatMs(45_000)).toBe('45.00 s');
  });

  it('does not invent a value for NaN', () => {
    expect(formatMs(Number.NaN)).toBe('—');
    expect(formatMs(null)).toBe('—');
  });
});

describe('metric formatting', () => {
  it('keeps three decimals so IoU differences stay visible', () => {
    expect(formatMetric(0.826476)).toBe('0.826');
    expect(formatMetric(0.0615, 4)).toBe('0.0615');
    expect(formatMetric(null)).toBe('—');
  });

  it('formats coverage as a percentage', () => {
    expect(formatPercent(0.40888)).toBe('40.9%');
    expect(formatPercent(undefined)).toBe('—');
  });

  it('formats throughput and duration', () => {
    expect(formatThroughput(3.2178)).toBe('3.22 img/s');
    expect(formatThroughput(42.5)).toBe('42.5 img/s');
    expect(formatThroughput(0)).toBe('—');
    expect(formatDuration(12.34)).toBe('12.3s');
    expect(formatDuration(2292.7)).toBe('38m 13s');
  });
});
