#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
(cd frontend && npm ci --omit=peer && npm run build)
test -f perf_benchmark/app/workbench/index.html
uv build "$@"
