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
compose=(docker compose -f "$base/compose.yaml")
case "${1:-up}" in
  up)
    : "${GRAFANA_ADMIN_PASSWORD:?Set GRAFANA_ADMIN_PASSWORD in deploy/.env}"
    mkdir -p "$base/targets" "$base/logs"
    docker network inspect "$OBSERVATION_NETWORK" >/dev/null 2>&1 \
      || docker network create "$OBSERVATION_NETWORK" >/dev/null
    "${compose[@]}" up -d --build --wait --wait-timeout 180
    echo "Observation ready; Grafana diagnostics use $WORKBENCH_PUBLIC_URL/grafana/ after starting Benchmark"
    ;;
  build) "${compose[@]}" build ;;
  status) "${compose[@]}" ps; "${compose[@]}" exec -T observation supervisorctl status ;;
  logs) "${compose[@]}" logs --tail=100 observation ;;
  stop) "${compose[@]}" stop ;;
  *) echo 'Usage: bash deploy/observation/deploy.sh up|build|status|logs|stop' >&2; exit 2 ;;
esac
