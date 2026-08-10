# CutoutML web console image.
#
# Three stages. The third copies only Next's `standalone` output, which bundles the
# server plus exactly the node_modules it traced as reachable -- a few tens of MB rather
# than the ~500 MB a full `node_modules` install would carry into production.
#
# NOTE ON VERIFICATION: written and reviewed, never built (no Docker daemon on the
# authoring machine). `npm ci`, `npm run build`, `tsc --noEmit` and `vitest run` were all
# run directly on the host, so the build steps themselves are known to work; what is
# unverified is this file's staging and the standalone copy paths.

# ------------------------------------------------------------------------- deps
FROM node:22-bookworm-slim AS deps

WORKDIR /app
# Only the lockfile and manifest, so a source edit does not reinstall the world.
COPY apps/web/package.json apps/web/package-lock.json ./
# `npm ci` rather than `npm install`: it fails on a lockfile that disagrees with
# package.json instead of quietly resolving something else, which is what makes an
# image build reproducible.
RUN npm ci

# ------------------------------------------------------------------------ build
FROM node:22-bookworm-slim AS build

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/ ./

# Baked in at build time because Next inlines NEXT_PUBLIC_* into the client bundle.
# Leaving it unset is the better default: the browser then calls the frontend's own
# origin and next.config.mjs proxies /api to the API service, so no CORS is involved.
ARG NEXT_PUBLIC_API_URL=""
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ---------------------------------------------------------------------- runtime
FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# node:* images already ship an unprivileged `node` user (uid 1000); reusing it avoids
# a redundant useradd and matches what most Node base images expect.
USER node

# `standalone` requires `output: 'standalone'` in next.config.mjs. static/ and public/
# are copied separately because the traced output deliberately excludes them.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
