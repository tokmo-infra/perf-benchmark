ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm
ARG RUST_IMAGE=rust:1.87.0-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.15
ARG NODE_IMAGE=node:24.15.0-bookworm-slim
FROM ${NODE_IMAGE} AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=peer
COPY frontend ./
RUN npm run build
FROM ${RUST_IMAGE} AS rust
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS builder
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:/usr/local/cargo/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 \
    CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
WORKDIR /opt/perf-benchmark
COPY pyproject.toml uv.lock README.md ./
COPY perf_benchmark ./perf_benchmark
COPY --from=frontend /build/perf_benchmark/app/workbench ./perf_benchmark/app/workbench
RUN uv sync --locked --no-dev --extra observation --extra production-traces \
    --group gpu-deploy --no-editable

FROM ${PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY examples /opt/perf-benchmark/examples
COPY repository /opt/perf-benchmark/repository
COPY deploy/gpu /opt/perf-benchmark/deploy/gpu
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/work XDG_CACHE_HOME=/work/cache HF_HOME=/work/cache/huggingface
RUN python -c 'from perf_benchmark import _native; import paramiko, scipy, PIL; assert _native.now() > 0' \
    && mkdir -p /work/cache /work/targets /work/results /work/logs /work/config /work/data /work/tokenizer \
    && mkdir -p /work/cache/perf-benchmark/result-sealing \
    && mkdir -p /opt/perf-benchmark/deploy/observation \
    && ln -s /work/targets /opt/perf-benchmark/deploy/observation/targets \
    && mkdir -p /opt/perf-benchmark/examples/debug/results \
    && chmod -R a+rwX /work /opt/perf-benchmark/examples/debug/results
WORKDIR /work
USER 10001:10001
CMD ["sleep", "infinity"]
