"""Report readiness only when all six platform services are ready."""

from urllib.request import urlopen


def check():
    for port, path in [(12345, "/-/ready"), (9090, "/-/ready"), (3200, "/ready"),
                       (3100, "/ready"), (4040, "/ready"), (3000, "/grafana/api/health")]:
        try:
            with urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError(f"Service on port {port} is not ready")
        except OSError as exc:
            raise RuntimeError(f"Service on port {port}: {exc}") from exc


if __name__ == "__main__":
    try:
        check()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
