#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in
  docker) engine=(docker) ;;
  nerdctl) engine=(nerdctl --namespace "${NERDCTL_NAMESPACE:-default}") ;;
  *) echo 'Usage: bash deploy/gpu/nsight.sh docker|nerdctl build|run [inference command...]' >&2; exit 2 ;;
esac
action=${2:-}
if (($# >= 2)); then shift 2; fi
case "$action" in
  build)
    : "${INFERENCE_IMAGE:?Set the pinned inference image}"
    : "${NSIGHT_IMAGE:?Set the diagnostic image tag}"
    test -f "$base/nsight-systems-cli.deb"
    "${engine[@]}" build -f "$base/nsight.Dockerfile" --build-arg "INFERENCE_IMAGE=$INFERENCE_IMAGE" \
      -t "$NSIGHT_IMAGE" "$base"
    ;;
  run)
    : "${NSIGHT_IMAGE:?Set the diagnostic image}" "${BENCHMARK_RUN_ID:?Set associated Run ID}"
    : "${MODEL_DIR:?Set absolute model directory}" "${REPORT_DIR:?Set absolute report directory}"
    : "${GPU_BIND_IP:?Set private server IP}" "${GPU_DEVICES:?Set GPU UUID or device selection}"
    (($#)) || { echo 'Pass inference executable and arguments' >&2; exit 2; }
    [[ "$MODEL_DIR" = /* && "$REPORT_DIR" = /* ]] || { echo 'Use absolute model/report directories' >&2; exit 2; }
    test -d "$MODEL_DIR"
    mkdir -p "$REPORT_DIR"
    "${engine[@]}" run --rm --init --gpus "\"device=$GPU_DEVICES\"" \
      -p "${GPU_BIND_IP}:${DIAGNOSTIC_PORT:-8001}:8000" \
      --mount "type=bind,src=$MODEL_DIR,dst=/models,readonly" \
      --mount "type=bind,src=$REPORT_DIR,dst=/reports" \
      -e "BENCHMARK_RUN_ID=$BENCHMARK_RUN_ID" -e "DIAGNOSTIC_HOST=$(hostname)" \
      -e "DIAGNOSTIC_RUN_ID=${DIAGNOSTIC_RUN_ID:-}" \
      -e "CAPTURE_SECONDS=${CAPTURE_SECONDS:-30}" \
      -e "CAPTURE_DELAY_SECONDS=${CAPTURE_DELAY_SECONDS:-60}" \
      "$NSIGHT_IMAGE" "$@"
    ;;
  *) echo 'Actions: build|run [inference command...]' >&2; exit 2 ;;
esac
