# CutoutML Celery worker image.
#
# Deliberately a separate Dockerfile from the API rather than the same image with a
# different command, for one reason that matters and one that follows from it:
#
#  * A GPU worker needs a CUDA-enabled torch wheel and (in a real deployment) a
#    CUDA base image. The API needs neither and should not pay 2.5 GB for them.
#  * The worker's healthcheck has to be `celery inspect ping`, not an HTTP probe,
#    because a worker exposes no port.
#
# Build the CPU worker with the defaults. For a GPU worker, override both build args:
#
#     docker build -f docker/worker.Dockerfile \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
#
# NOTE ON VERIFICATION: written and reviewed, never built -- no Docker daemon was
# available on the authoring machine, and no GPU either, so the CUDA variant above is
# reasoned from the published wheel layout rather than tested. Treated as untested
# throughout the docs rather than presented as working.

ARG BASE_IMAGE=python:3.12-slim-bookworm

# ---------------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src/cutoutml/__init__.py src/cutoutml/__init__.py

RUN pip install --upgrade pip \
 && pip install --extra-index-url "${TORCH_INDEX_URL}" '.[api,onnx]'

# --------------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CUTOUTML_LOG_FORMAT=json

# python3 is installed explicitly because the CUDA base images do not ship it, while
# the python:slim base already has it and the install is a no-op there. One RUN that
# works for both beats two Dockerfiles that differ by a line.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 cutoutml

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=cutoutml:cutoutml pyproject.toml alembic.ini ./
COPY --chown=cutoutml:cutoutml src/ src/
COPY --chown=cutoutml:cutoutml services/ services/
COPY --chown=cutoutml:cutoutml infra/ infra/

RUN pip install --no-deps --no-cache-dir . \
 && mkdir -p /data/storage /app/models \
 && chown -R cutoutml:cutoutml /data /app/models

USER cutoutml
ENV CUTOUTML_STORAGE_ROOT=/data/storage \
    CUTOUTML_MODEL_WEIGHTS_DIR=/app/models

# One thread per worker child. Celery's prefork pool gives every child the host's full
# core count, so the default on an 8-core box with -c 4 is 32 threads contending for 8
# cores -- measurably slower than 4x2. Compose raises this for the single-slot video
# worker, which benefits from intra-op parallelism instead.
ENV CUTOUTML_TORCH_NUM_THREADS=2

# `celery inspect ping` actually round-trips through the broker, so it fails when Redis
# is unreachable or the worker has wedged -- unlike a `pgrep celery` style check, which
# passes for a hung process.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD celery -A services.inference.app.celery_app:celery inspect ping -d "celery@$(hostname)" \
        || exit 1

# Queue is overridden per replica in docker-compose.yml; `cpu` is the safe default
# because it is the only queue that works without a GPU.
CMD ["celery", "-A", "services.inference.app.celery_app:celery", "worker", \
     "-Q", "cpu", "-c", "2", "-n", "cpu@%h", "--loglevel=info"]
