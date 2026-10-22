#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in
  docker) engine=(docker) ;;
  nerdctl) engine=(nerdctl --namespace "${NERDCTL_NAMESPACE:-default}") ;;
  *) echo 'Usage: bash deploy/gpu/dcgm.sh docker|nerdctl [up|status|logs|stop]' >&2; exit 2 ;;
esac
name=perf-dcgm
case "${2:-up}" in
  up)
    : "${DCGM_EXPORTER_IMAGE:?Set a driver-compatible pinned image}"
    : "${GPU_BIND_IP:?Set the GPU server private bind IP}"
    "${engine[@]}" run -d --name "$name" --restart unless-stopped \
      --gpus all --cap-add SYS_ADMIN -p "${GPU_BIND_IP}:9400:9400" \
      --mount "type=bind,src=$base/dcgm-counters.csv,dst=/etc/dcgm-exporter/perf.csv,readonly" \
      "$DCGM_EXPORTER_IMAGE" --collectors /etc/dcgm-exporter/perf.csv --collect-interval 5000
    for ((i=0; i<30; i++)); do
      if metrics=$(curl --fail --silent --max-time 2 "http://${GPU_BIND_IP}:9400/metrics") \
          && [[ "$metrics" == *DCGM_FI_DEV_GPU_UTIL\{* ]]; then
        echo "DCGM ready: ${GPU_BIND_IP}:9400"
        exit 0
      fi
      sleep 2
    done
    echo 'DCGM not ready; inspect the perf-dcgm container logs' >&2
    exit 1
    ;;
  status) "${engine[@]}" inspect "$name" ;;
  logs) "${engine[@]}" logs --tail 100 "$name" ;;
  stop) "${engine[@]}" stop "$name" ;;
  *) echo 'Actions: up|status|logs|stop' >&2; exit 2 ;;
esac
