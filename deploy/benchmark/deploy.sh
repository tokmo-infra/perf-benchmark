#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
environment_file=${DEPLOY_ENV_FILE:-$base/../.env}
if [[ -f "$environment_file" ]]; then
  set -a; source "$environment_file"; set +a
fi
export OBSERVATION_NETWORK=${OBSERVATION_NETWORK:-perf-observation}
export WORKBENCH_PUBLIC_URL=${WORKBENCH_PUBLIC_URL:-http://127.0.0.1:${WORKBENCH_PORT:-8088}}
export WORKBENCH_PUBLIC_URL=${WORKBENCH_PUBLIC_URL%/}
export BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$base/config}
export BENCHMARK_DATA=${BENCHMARK_DATA:-$base/../data}
export BENCHMARK_TOKENIZER=${BENCHMARK_TOKENIZER:-$base/../tokenizer}
export BENCHMARK_UID=${BENCHMARK_UID:-$(id -u)} BENCHMARK_GID=${BENCHMARK_GID:-$(id -g)}
compose=(docker compose -f "$base/compose.yaml")
case "${1:-up}" in
  up)
    for path in "$BENCHMARK_CONFIG" "$BENCHMARK_DATA" "$BENCHMARK_TOKENIZER"; do
      [[ "$path" = /* ]] || { echo 'Use absolute host paths' >&2; exit 2; }
      mkdir -p "$path"
    done
    mkdir -p "$base/../observation/targets" "$base/../observation/logs"
    [[ -f "$BENCHMARK_CONFIG/run.yaml" ]] || cp "$base/run.yaml" "$BENCHMARK_CONFIG/run.yaml"
    docker network inspect "$OBSERVATION_NETWORK" >/dev/null 2>&1 \
      || docker network create "$OBSERVATION_NETWORK" >/dev/null
    "${compose[@]}" up -d --build --wait --wait-timeout 180
    "${compose[@]}" exec -T benchmark perf-benchmark --help >/dev/null
    echo "Workbench ready: $WORKBENCH_PUBLIC_URL/#/runs (Grafana diagnostics in the same page)"
    echo 'Benchmark toolbox: docker exec -it perf-benchmark bash'
    ;;
  build) "${compose[@]}" build ;;
  status) "${compose[@]}" ps ;;
  stop) "${compose[@]}" stop ;;
  *) echo 'Usage: bash deploy/benchmark/deploy.sh up|build|status|stop' >&2; exit 2 ;;
esac
