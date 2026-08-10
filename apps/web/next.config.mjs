/**
 * Next.js configuration.
 *
 * `rewrites` proxies `/api/*` to the FastAPI service so that the browser only ever
 * talks to one origin. That is not cosmetic: a same-origin fetch needs no CORS
 * preflight on every request, and the access token can be sent without a wildcard
 * CORS policy on the API. `NEXT_PUBLIC_API_URL` overrides the target for a split
 * deployment.
 *
 * The security headers are the small set that is unambiguously correct for this app.
 * A full CSP is deliberately not asserted here: Next injects inline bootstrap scripts,
 * so a strict `script-src` needs per-request nonces wired through the document, and a
 * CSP that has to include `unsafe-inline` provides confidence rather than protection.
 */

/** @type {import('next').NextConfig} */
const apiUrl = process.env.CUTOUTML_API_URL ?? 'http://127.0.0.1:8000';

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Emits .next/standalone: the server plus only the node_modules Next traced as
  // reachable. docker/web.Dockerfile copies that instead of a full install, which is the
  // difference between a ~60 MB runtime image and a ~500 MB one.
  output: 'standalone',
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${apiUrl}/:path*` }];
  },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
        ],
      },
    ];
  },
};

export default nextConfig;
