"""Observation values. No SDK, I/O or measurement formulas live here."""

from dataclasses import dataclass, fields
from hashlib import blake2b
from pathlib import Path
import re
from time import perf_counter_ns, time_ns
from urllib.parse import urlsplit

from perf_benchmark.core.validation import positive_number


@dataclass(frozen=True, slots=True)
class MonitoringTarget:
    """A remote scrape endpoint; labels associate a service with this Run."""

    address: str
    service_name: str
    scheme: str = "http"
    metrics_path: str = "/metrics"
    engine: str = ""
    model: str = ""
    version: str = ""

    @classmethod
    def parse(cls, value):
        target = cls(**value)
        if any(not isinstance(getattr(target, f.name), str) for f in fields(cls)):
            raise ValueError("observation target fields must be strings")
        address = re.fullmatch(r"(?:[^:/\s?#@]+|\[[0-9A-Fa-f:]+\]):([0-9]+)", target.address)
        if address is None or not 1 <= int(address[1]) <= 65535:
            raise ValueError("target address must be host:port without credentials")
        if target.scheme not in ("http", "https") or not target.service_name:
            raise ValueError("target requires service_name and HTTP scheme")
        if not re.fullmatch(r"/[^?#\s]*", target.metrics_path):
            raise ValueError("target metrics_path must be an absolute path without query")
        return target


@dataclass(frozen=True, slots=True)
class MonitoringTargets:
    engines: tuple[MonitoringTarget, ...] = ()
    gpu: tuple[MonitoringTarget, ...] = ()
    profiles: tuple[MonitoringTarget, ...] = ()



@dataclass(frozen=True, slots=True)
class ObservationSettings:
    enabled: bool = False
    trace_mode: str = "benchmark"
    trace_export: bool = True
    poll_seconds: float = 2.0
    page_size: int = 500
    sample_rate: float = 0.01
    slow_seconds: float = 10.0
    slow_spans_per_second: float = 20.0
    host: str = "127.0.0.1"
    port: int = 0
    discovery_dir: str | None = None
    advertise_host: str = "127.0.0.1"
    targets: MonitoringTargets = MonitoringTargets()
    otlp_endpoint: str = "http://127.0.0.1:4318/v1/traces"

    @classmethod
    def parse(cls, value: object, *, base_dir: Path | None = None) -> "ObservationSettings":
        if not isinstance(value, dict):
            raise TypeError("observation must be an object")
        if unknown := set(value) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown observation settings: {sorted(unknown)}")
        value = dict(value)
        targets = value.get("targets", {})
        if not isinstance(targets, dict) or set(targets) - {"engines", "gpu", "profiles"}:
            raise ValueError("invalid observation.targets")
        if any(not isinstance(group, (list, tuple)) for group in targets.values()):
            raise ValueError("observation targets must be lists")
        value["targets"] = MonitoringTargets(**{
            name: tuple(MonitoringTarget.parse(target) for target in group) for name, group in targets.items()
        })
        if any(target.metrics_path != "/metrics" for target in value["targets"].profiles):
            raise ValueError("Profile targets use standard pprof paths, not metrics_path")
        directory = value.get("discovery_dir")
        if directory is not None:
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError("observation.discovery_dir must be a nonempty path")
            value["discovery_dir"] = str(((base_dir or Path.cwd()) / Path(directory).expanduser()).resolve())
        if any(getattr(value["targets"], f.name) for f in fields(MonitoringTargets)) and directory is None:
            raise ValueError("observation.targets requires discovery_dir")
        result = cls(**value)
        if any(type(getattr(result, name)) is not bool for name in ("enabled", "trace_export")):
            raise TypeError("observation.enabled and trace_export must be boolean")
        if result.trace_mode not in ("benchmark", "observe", "debug"):
            raise ValueError("observation.trace_mode must be benchmark, observe or debug")
        for name in ("poll_seconds", "slow_seconds", "slow_spans_per_second"):
            positive_number(getattr(result, name), f"observation.{name}")
        if positive_number(result.sample_rate, "observation.sample_rate", allow_zero=True) > 1:
            raise ValueError("observation.sample_rate must be at most 1")
        for name, maximum in (("page_size", 5000), ("port", 65535)):
            number = getattr(result, name)
            if type(number) is not int or not (0 if name == "port" else 1) <= number <= maximum:
                raise ValueError(f"invalid observation.{name}")
        if not isinstance(result.host, str) or not result.host:
            raise ValueError("observation.host must be nonempty")
        advertised = result.advertise_host
        if (not isinstance(advertised, str) or not advertised
                or not re.fullmatch(r"[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\]", advertised)
                or advertised in ("0.0.0.0", "[::]")):
            raise ValueError("observation.advertise_host must be a reachable hostname or IP")
        if not isinstance(result.otlp_endpoint, str):
            raise ValueError("observation.otlp_endpoint must be an HTTP URL")
        endpoint = urlsplit(result.otlp_endpoint)
        if (endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.port == 0
                or endpoint.username is not None or endpoint.query or endpoint.fragment):
            raise ValueError("observation.otlp_endpoint must be an HTTP URL without credentials")
        return result


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    monotonic_ns: int
    unix_ns: int
    uncertainty_ns: int

    @classmethod
    def capture(cls) -> "ClockAnchor":
        before = perf_counter_ns()
        wall = time_ns()
        after = perf_counter_ns()
        return cls((before + after) // 2, wall, after - before)

    def unix(self, timestamp_ns: int) -> int:
        return self.unix_ns + timestamp_ns - self.monotonic_ns


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: str
    sampled: bool

    @classmethod
    def create(cls, run_id: str, turn_id: str, attempt: int, rate: float, *,
               seed: int, session_ordinal: int, turn_ordinal: int) -> "TraceContext":
        # Length-prefixed run identity prevents ambiguous concatenation; IDs are
        # reproducible across replay, but isolated between Runs.
        identity = f"{len(run_id)}:{run_id}{turn_id}".encode()
        trace = blake2b(identity, digest_size=16).hexdigest()
        parent = blake2b(identity + b":turn", digest_size=8).hexdigest()
        span = blake2b(identity + f":attempt:{attempt}".encode(), digest_size=8).hexdigest()
        sampling = blake2b(f"{seed}:trace:{session_ordinal}:{turn_ordinal}".encode(), digest_size=16)
        return cls(trace, span, parent, int.from_bytes(sampling.digest()) < int(rate * (1 << 128)))

    @property
    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"
