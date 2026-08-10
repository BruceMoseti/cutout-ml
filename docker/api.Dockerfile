# CutoutML API image.
#
# Two stages so the runtime layer carries no compiler and no pip cache. The wheels are
# built once into a virtualenv and that directory is copied across, which is simpler and
# more reliable than `pip install --user` plus PATH juggling.
#
# NOTE ON VERIFICATION: this file was written and reviewed but never built. The machine
# it was authored on has no Docker daemon, so `docker build` was not run. Reviewed for:
# correct stage boundaries, no dev toolchain in the runtime layer, a non-root runtime
# user, ffmpeg present (the video pipeline shells out to it), and a healthcheck that
# hits the readiness probe rather than a liveness stub. See docs/benchmarks.md for the
# full list of what could and could not be verified in this environment.

# ---------------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by any dependency without a manylinux wheel for this
# platform; it stays in the builder and never reaches the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The CPU wheel index is explicit. The default PyPI `torch` wheel bundles CUDA and is
# roughly 2.5 GB; the CPU build is a tenth of that, and an API container never runs a
# kernel on a GPU it does not have. GPU workers override this (see worker.Dockerfile).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Dependency metadata is copied on its own so that editing application code does not
# invalidate the (slow) dependency layer.
COPY pyproject.toml README.md ./
COPY src/cutoutml/__init__.py src/cutoutml/__init__.py

RUN pip install --upgrade pip \
 && pip install --extra-index-url "${TORCH_INDEX_URL}" '.[api,onnx]'

# --------------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CUTOUTML_LOG_FORMAT=json

# ffmpeg is a hard runtime dependency, not an optional extra: the video pipeline
# shells out to it and `GET /health/ready` reports it missing. curl is here only for
# the healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 cutoutml

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=cutoutml:cutoutml pyproject.toml alembic.ini ./
COPY --chown=cutoutml:cutoutml src/ src/
COPY --chown=cutoutml:cutoutml services/ services/
COPY --chown=cutoutml:cutoutml infra/ infra/

# Installed non-editable in the runtime layer so `import cutoutml` does not depend on
# the source tree layout or on PYTHONPATH being set correctly by the entrypoint.
RUN pip install --no-deps --no-cache-dir . \
 && mkdir -p /data/storage /app/models \
 && chown -R cutoutml:cutoutml /data /app/models

USER cutoutml
ENV CUTOUTML_STORAGE_ROOT=/data/storage \
    CUTOUTML_MODEL_WEIGHTS_DIR=/app/models

EXPOSE 8000

# Readiness, not liveness: a container that is up but cannot reach Postgres should not
# be sent traffic. --start-period covers migration and first model load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready || exit 1

# No --reload and no shell form: the process must be PID 1 so that SIGTERM from the
# orchestrator reaches uvicorn and in-flight requests drain.
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--no-server-header"]
