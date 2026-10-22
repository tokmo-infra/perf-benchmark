# Reuse upstream release binaries and Grafana assets; do not build backends from source.
FROM grafana/alloy:v1.19.0 AS alloy
FROM grafana/tempo:2.7.2 AS tempo
FROM prom/prometheus:v3.2.1 AS prometheus
FROM grafana/loki:3.7.2 AS loki
FROM grafana/pyroscope:2.2.0 AS pyroscope
FROM grafana/grafana:11.6.11 AS grafana
USER root
RUN grafana cli --pluginsDir /opt/grafana-plugins plugins install yesoreyeram-infinity-datasource 4.0.0
RUN grafana cli --pluginsDir /opt/grafana-plugins plugins install volkovlabs-table-panel 3.6.1
RUN grafana cli --pluginsDir /opt/grafana-plugins plugins install volkovlabs-variable-panel 4.1.0
FROM python:3.11.15-slim-trixie
RUN groupadd --gid 10001 observation \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin observation \
    && python -m venv /opt/supervisor \
    && /opt/supervisor/bin/pip install --disable-pip-version-check --no-cache-dir supervisor==4.3.0
ENV PATH="/opt/supervisor/bin:$PATH"
COPY --from=alloy /bin/alloy /usr/local/bin/alloy
COPY --from=tempo /tempo /usr/local/bin/tempo
COPY --from=prometheus /bin/prometheus /usr/local/bin/prometheus
COPY --from=loki /usr/bin/loki /usr/local/bin/loki
COPY --from=pyroscope /usr/bin/pyroscope /usr/local/bin/pyroscope
COPY --from=grafana /usr/share/grafana /usr/share/grafana
COPY --from=grafana /etc/grafana /etc/grafana
COPY --from=grafana /opt/grafana-plugins /opt/grafana-plugins
COPY config.alloy tempo.yaml prometheus.yaml loki.yaml pyroscope.yaml /etc/observation/
COPY datasources.yaml /etc/grafana/provisioning/datasources/perf.yaml
COPY dashboards.yaml /etc/grafana/provisioning/dashboards/perf.yaml
COPY *.json /etc/grafana/dashboards/
COPY supervisord.conf /etc/supervisord.conf
COPY start.sh health.py workbench_links.py /opt/observation/
RUN mkdir -p /etc/alloy/targets /var/log/perf-benchmark /data \
    && alloy --version && tempo -version && prometheus --version \
    && loki -version && pyroscope -version && /usr/share/grafana/bin/grafana --version
ENV GF_PATHS_HOME=/usr/share/grafana GF_PATHS_CONFIG=/etc/grafana/grafana.ini \
    GF_PATHS_DATA=/data/grafana GF_PATHS_LOGS=/data/grafana/logs \
    GF_PATHS_PLUGINS=/opt/grafana-plugins GF_PATHS_PROVISIONING=/etc/grafana/provisioning \
    GF_USERS_ALLOW_SIGN_UP=false WORKBENCH_PUBLIC_URL=http://127.0.0.1:8088 PYTHONUNBUFFERED=1
EXPOSE 3000 12345 4317 4318 9999 9090 3200 3100 4040
HEALTHCHECK --interval=5s --timeout=15s --start-period=120s --retries=3 CMD ["python", "/opt/observation/health.py"]
ENTRYPOINT ["sh", "/opt/observation/start.sh"]
