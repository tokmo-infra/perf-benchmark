# Build on the exact inference image. Supply NVIDIA's matching standalone CLI .deb.
# This recipe supports Debian/Ubuntu based images with Python 3 and apt-get.
ARG INFERENCE_IMAGE
FROM ${INFERENCE_IMAGE}
USER root
COPY nsight-systems-cli.deb /tmp/nsight-systems-cli.deb
RUN apt-get update && apt-get install -y --no-install-recommends /tmp/nsight-systems-cli.deb \
    && rm /tmp/nsight-systems-cli.deb && rm -rf /var/lib/apt/lists/* \
    && nsys --version && python3 --version
COPY capture.py /opt/perf-diagnostics/capture.py
ENTRYPOINT ["python3", "/opt/perf-diagnostics/capture.py"]
CMD []
