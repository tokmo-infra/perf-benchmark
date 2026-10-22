#!/bin/sh
set -eu
# Initialize only this platform's data directories; preserve existing contents.
for name in alloy tempo prometheus loki pyroscope grafana; do
  mkdir -p "/data/$name"
  chown 10001:10001 "/data/$name"
done
python /opt/observation/workbench_links.py
export GF_SERVER_ROOT_URL="${WORKBENCH_PUBLIC_URL%/}/grafana/"
exec supervisord -c /etc/supervisord.conf
