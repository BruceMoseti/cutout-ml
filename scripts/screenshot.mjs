#!/usr/bin/env node
/**
 * Capture the README screenshots from a running stack.
 *
 * Not a test and not run in CI: it writes tracked files, and a job that commits its own output
 * is a job nobody reviews. Run it by hand after a change to what the app looks like.
 *
 *   make api                       # in one terminal
 *   make worker                    # in another
 *   cd apps/web && NEXT_PUBLIC_API_URL=http://localhost:8000/v1 npm run dev
 *   node scripts/screenshot.mjs
 *
 * It signs in through the real form and uploads a real image, so the cutout in the picture was
 * produced by the worker rather than pasted in. Playwright is not a dependency of this
 * repository; point PLAYWRIGHT_MODULE at an installation that has it.
 *
 *   PLAYWRIGHT_MODULE=/path/to/node_modules/playwright/index.js node scripts/screenshot.mjs
 */

import zlib from 'node:zlib';

import { mkdir, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';

// Playwright is not a dependency of this repository -- adding a browser automation stack to a
// Python project for two screenshots would be a poor trade. Point PLAYWRIGHT_MODULE at an
// installation that has one; ESM ignores NODE_PATH, so the path has to be a real specifier.
const playwright = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
// Playwright ships CommonJS, so a dynamic import may expose its exports under `default`.
const chromium = playwright.chromium ?? playwright.default?.chromium;

const API = process.env.SHOT_API_URL ?? 'http://localhost:8000';
const WEB = process.env.SHOT_BASE_URL ?? 'http://localhost:3000';
const OUT = path.resolve('docs/media');
const VIEWPORT = { width: 1440, height: 960 };
const PASSWORD = 'correct horse battery staple';

/**
 * A subject with soft edges and a busy background, because a hard-edged rectangle on white
 * makes every segmenter look perfect and proves nothing about the alpha.
 */
function subjectPng() {
  const width = 512;
  const height = 512;
  const pixels = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 3;
      // Background: a diagonal gradient with stripes, so a lazy threshold cannot win.
      const stripe = (x + y) % 64 < 32 ? 24 : 0;
      pixels[i] = 40 + stripe + Math.floor((x / width) * 60);
      pixels[i + 1] = 60 + Math.floor((y / height) * 70);
      pixels[i + 2] = 120 + stripe;
      // Foreground: a filled circle with a feathered rim.
      const dx = x - width / 2;
      const dy = y - height / 2;
      const r = Math.sqrt(dx * dx + dy * dy);
      const edge = Math.min(1, Math.max(0, (150 - r) / 12));
      if (edge > 0) {
        pixels[i] = Math.round(pixels[i] * (1 - edge) + 235 * edge);
        pixels[i + 1] = Math.round(pixels[i + 1] * (1 - edge) + 80 * edge);
        pixels[i + 2] = Math.round(pixels[i + 2] * (1 - edge) + 60 * edge);
      }
    }
  }
  return encodePng(width, height, pixels);
}

/** Minimal PNG encoder: one uncompressed-deflate IDAT, so nothing has to be installed. */
function encodePng(width, height, rgb) {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    raw[y * (width * 3 + 1)] = 0; // filter: none
    rgb.copy(raw, y * (width * 3 + 1) + 1, y * width * 3, (y + 1) * width * 3);
  }
  const chunk = (type, data) => {
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([length, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolour
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

let crcTable;
function crc32(buffer) {
  if (!crcTable) {
    crcTable = new Int32Array(256);
    for (let n = 0; n < 256; n += 1) {
      let c = n;
      for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c;
    }
  }
  let crc = -1;
  for (const byte of buffer) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff];
  return (crc ^ -1) >>> 0;
}

async function register(email) {
  const response = await fetch(`${API}/v1/auth/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email, password: PASSWORD }),
  });
  if (!response.ok) throw new Error(`register -> ${response.status} ${await response.text()}`);
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const image = path.join(OUT, '.subject.png');
  await writeFile(image, subjectPng());

  const email = `shot-${randomUUID().slice(0, 8)}@example.com`;
  await register(email);

  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: VIEWPORT });
    await page.goto(WEB, { waitUntil: 'networkidle' });
    await page.locator('input[type="email"]').fill(email);
    await page.locator('input[type="password"]').fill(PASSWORD);
    await page.getByRole('button', { name: /sign in/i }).click();

    // The studio replaces the sign-in form once the session lands.
    await page.locator('input[type="file"]').waitFor({ timeout: 30_000 });
    await page.locator('input[type="file"]').setInputFiles(image);

    // The worker has to finish before there is anything worth photographing, and how long
    // that takes depends on the model, so wait for the result rather than for a duration.
    await page
      .getByText(/succeeded|done|complete/i)
      .first()
      .waitFor({ timeout: 180_000 });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(OUT, 'studio.png') });
    console.log('wrote studio.png');

    await page.goto(`${WEB}/benchmarks`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, 'benchmarks.png'), fullPage: false });
    console.log('wrote benchmarks.png');
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
