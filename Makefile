# CutoutML developer entry points.
#
# Every target assumes the virtualenv at ./.venv (created by `make venv`). PYTHON can be
# overridden to point somewhere else, which is what the Docker images and CI do:
#
#     make test PYTHON=python3

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
NPM    ?= npm
WEB    ?= apps/web

.DEFAULT_GOAL := help
.PHONY: help venv install install-web fmt lint typecheck test test-integration check \
        train train-suite weights weights-pretrained export-onnx eval-data bench bench-quick \
        render-bench \
        migrate migrate-down api worker worker-gpu web doctor docker-build compose-up \
        compose-down compose-config clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- setup

venv: ## Create the virtualenv
	python3 -m venv .venv

install: ## Install the package plus every optional extra needed for development
	$(PIP) install -U pip
	$(PIP) install -e '.[api,onnx,dev]'

install-web: ## Install frontend dependencies
	cd $(WEB) && $(NPM) ci

# ---------------------------------------------------------------- quality gates

fmt: ## Format Python and check the frontend formatting
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint: ## Ruff lint + format check (no writes)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck: ## mypy
	$(PYTHON) -m mypy

test: ## Unit tests (no external services required)
	$(PYTHON) -m pytest -q -m 'not integration'

test-integration: ## Tests that need Postgres, Redis and ffmpeg
	$(PYTHON) -m pytest -q -m integration

check-generated: ## Verify the generated artefacts are in step with their sources
	$(PYTHON) scripts/eval_data.py --verify
	$(PYTHON) -m cutoutml.benchmarks.render_report
	git diff --exit-code -- docs/benchmarks.md README.md

check: lint typecheck test check-generated ## Every CI check that needs no services

# --------------------------------------------------------------------- models

train: ## Train the default model (cutoutnet-small)
	$(PYTHON) -m cutoutml.training.train --arch cutoutnet-small

train-suite: ## Train every CPU-feasible architecture on one identical budget
	scripts/train_suite.sh

weights: ## Produce every checkpoint the committed benchmark suite needs
	scripts/train_suite.sh cutoutnet-tiny cutoutnet-small cutoutnet-base u2net-lite
	$(MAKE) export-onnx

weights-pretrained: ## Fetch the published U^2-Net weights and convert them for PyTorch
	$(PYTHON) -m cutoutml.models.download_weights --model u2netp-onnx
	$(PYTHON) -m cutoutml.models.download_weights --model u2net-onnx
	$(PYTHON) -m cutoutml.models.u2net.from_onnx \
		--onnx models/u2net/u2netp.onnx --output models/u2net/u2netp.pt --variant lite
	$(PYTHON) -m cutoutml.models.u2net.from_onnx \
		--onnx models/u2net/u2net.onnx --output models/u2net/u2net.pt --variant full

export-onnx: ## Export the default checkpoint to ONNX
	$(PYTHON) -m cutoutml.cli export-onnx cutoutnet -o models/cutoutnet/cutoutnet-small.onnx

# ------------------------------------------------------------------ benchmarks

eval-data: ## Replay the committed eval-set manifest and verify its fingerprint
	$(PYTHON) scripts/eval_data.py --verify

bench: ## Full benchmark suite, then re-render the markdown tables
	$(PYTHON) benchmarks/run.py

bench-quick: ## Smoke-test the harness (few repetitions, no markdown rewrite)
	$(PYTHON) benchmarks/run.py --quick --no-render

render-bench: ## Re-render docs/benchmarks.md and the README table from the latest JSON
	$(PYTHON) -m cutoutml.benchmarks.render_report

# ------------------------------------------------------------------- database

migrate: ## Apply migrations
	$(PYTHON) -m alembic upgrade head

migrate-down: ## Roll back one migration
	$(PYTHON) -m alembic downgrade -1

# -------------------------------------------------------------------- services

api: ## Run the API with reload
	$(PYTHON) -m uvicorn services.api.app.main:app --reload --host 127.0.0.1 --port 8000

worker: ## Run the Celery worker on the CPU queue
	$(PYTHON) -m celery -A services.inference.app.celery_app:celery worker \
		-Q cpu -c 2 -n cpu@%h --loglevel=info

worker-gpu: ## Run the GPU workers (separate concurrencies: see ADR-002)
	$(PYTHON) -m celery -A services.inference.app.celery_app:celery worker \
		-Q image-gpu -c 2 -n image@%h --loglevel=info &
	$(PYTHON) -m celery -A services.inference.app.celery_app:celery worker \
		-Q video-gpu -c 1 -n video@%h --loglevel=info

web: ## Run the Next.js dev server
	cd $(WEB) && $(NPM) run dev

doctor: ## Report what this machine can actually do
	$(PYTHON) -m cutoutml.cli doctor

# ---------------------------------------------------------------------- docker

compose-config: ## Validate docker-compose.yml
	docker compose config -q

docker-build: ## Build both service images
	docker compose build api worker

compose-up: ## Bring up the full stack
	docker compose up -d --build

compose-down: ## Tear the stack down, keeping volumes
	docker compose down

clean: ## Remove caches and build artefacts (keeps checkpoints and results)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
	rm -rf $(WEB)/.next $(WEB)/tsconfig.tsbuildinfo
