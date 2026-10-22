"""Strict configuration for the native-pipeline static benchmark application."""
from dataclasses import dataclass, field
import math
from pathlib import Path
from urllib.parse import urlsplit
import yaml
from perf_benchmark.app.config import case_description

DEFAULT_PREFILL_MATRIX = {
    128: [1, 2, 4, 8, 16, 32, 64], 512: [1, 2, 4, 8, 16, 32, 64],
    2048: [1, 2, 4, 8, 16, 32, 64], 8192: [1, 2, 4, 8, 16, 32, 64],
    32768: [1, 2, 4, 8, 16], 131072: [1, 2, 4], 524288: [1],
}
DEFAULT_DECODE_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class ServiceConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "default-model"
    api_key: str = field(default="EMPTY", repr=False)
    timeout_seconds: float = 600.0
    stream: bool = True
    tokenizer_path: str | None = None
    endpoint: str = "/chat/completions"
    extra_body: dict = field(default_factory=dict)
    extra_headers: dict = field(default_factory=dict, repr=False)
    cache_salt_enabled: bool = False
    capacity_snapshot: str | None = None

    def normalized_url(self):
        base, endpoint = self.base_url.rstrip("/"), self.endpoint.lstrip("/")
        if base.endswith("/v1") and endpoint.startswith("v1/"):
            endpoint = endpoint[3:]
        return base + "/" + endpoint


@dataclass
class PrefillConfig:
    enabled: bool = True
    rounds: int = 3
    max_tokens: int = 1
    seed: int = 42
    temperature: float = 0.1
    matrix: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_PREFILL_MATRIX.items()})
    requests_per_concurrency: int | None = None
    randomize_per_request: bool = True


@dataclass
class DecodeConfig:
    enabled: bool = True
    rounds: int = 3
    concurrencies: list = field(default_factory=lambda: list(DEFAULT_DECODE_CONCURRENCIES))
    max_tokens: int = 2560
    ignore_eos: bool = False
    temperature: float = 0.1
    reasoning_effort: str | None = "high"
    prompts: list | None = None
    requests_per_concurrency: int | None = None


@dataclass
class OutputConfig:
    excel_path: str = "results/static_benchmark_results.xlsx"
    json_path: str | None = None


@dataclass
class StaticBenchmarkConfig:
    name: str = "static-benchmark"
    case: str = ""
    service: ServiceConfig = field(default_factory=ServiceConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    config_dir: Path = field(default_factory=Path.cwd)


def _object(value, label, allowed):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(label + " unknown settings: " + ", ".join(sorted(unknown)))
    return dict(value)


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(label + " must be a positive integer")
    return value


def _concurrencies(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(label + " must be a nonempty list")
    parsed = [_positive(v, label) for v in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(label + " must not repeat values")
    return parsed


def load_static_benchmark_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        raw, base = path_or_dict, Path.cwd()
    else:
        path = Path(path_or_dict).expanduser().resolve()
        raw, base = yaml.safe_load(path.read_text()), path.parent
    raw = _object(raw, "config", ("case", "name", "service", "prefill", "decode", "output"))
    case = case_description(raw)
    sections = {}
    for key, cls in (("service", ServiceConfig), ("prefill", PrefillConfig),
                     ("decode", DecodeConfig), ("output", OutputConfig)):
        values = _object(raw.get(key, {}), key, cls.__dataclass_fields__)
        sections[key] = cls(**values)
    config = StaticBenchmarkConfig(name=raw.get("name", "static-benchmark"), case=case, config_dir=base, **sections)
    service, prefill, decode = config.service, config.prefill, config.decode
    if not isinstance(config.name, str) or not config.name.strip():
        raise ValueError("name must be nonempty")
    for label, value in (("service.stream", service.stream),
                         ("service.cache_salt_enabled", service.cache_salt_enabled),
                         ("prefill.enabled", prefill.enabled),
                         ("prefill.randomize_per_request", prefill.randomize_per_request),
                         ("decode.enabled", decode.enabled), ("decode.ignore_eos", decode.ignore_eos)):
        if type(value) is not bool:
            raise ValueError(label + " must be boolean")
    if not service.stream:
        raise ValueError("static TTFT/decode measurements require service.stream: true")
    if not prefill.enabled and not decode.enabled:
        raise ValueError("enable prefill or decode")
    parsed = urlsplit(service.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("service.base_url must be an http(s) URL without credentials/query/fragment")
    if service.endpoint not in ("/chat/completions", "/v1/chat/completions"):
        raise ValueError("static mode currently supports chat/completions endpoints")
    if not isinstance(service.model, str) or not service.model.strip():
        raise ValueError("service.model must be nonempty")
    if type(service.timeout_seconds) not in (int, float) or not math.isfinite(service.timeout_seconds) or service.timeout_seconds <= 0:
        raise ValueError("service.timeout_seconds must be positive")
    if not isinstance(service.extra_body, dict) or not isinstance(service.extra_headers, dict):
        raise ValueError("extra_body and extra_headers must be objects")
    reserved = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                "temperature", "reasoning_effort", "ignore_eos", "cache_salt"}
    if reserved & service.extra_body.keys():
        raise ValueError("extra_body conflicts with static workload settings")
    for point in (prefill, decode):
        _positive(point.rounds, "rounds")
        if point.rounds < 3:
            raise ValueError("static benchmarking requires at least three rounds")
        _positive(point.max_tokens, "max_tokens")
        if point.requests_per_concurrency is not None:
            _positive(point.requests_per_concurrency, "requests_per_concurrency")
        if type(point.temperature) not in (int, float) or not math.isfinite(point.temperature) or point.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
    if type(prefill.seed) is not int or prefill.seed < 0:
        raise ValueError("prefill.seed must be a nonnegative integer")
    if not isinstance(prefill.matrix, dict) or not prefill.matrix:
        raise ValueError("prefill.matrix must be a nonempty object")
    matrix = {}
    for key, values in prefill.matrix.items():
        if type(key) is int:
            length = key
        elif isinstance(key, str) and key.isdecimal():
            length = int(key)
        else:
            raise ValueError("prefill length must be a positive integer")
        _positive(length, "prefill length")
        if length in matrix:
            raise ValueError("duplicate prefill length")
        matrix[length] = _concurrencies(values, "prefill concurrency")
    prefill.matrix = matrix
    decode.concurrencies = _concurrencies(decode.concurrencies, "decode concurrency")
    for point, concurrencies in ((prefill, [c for values in matrix.values() for c in values]),
                                 (decode, decode.concurrencies)):
        if point.enabled and point.requests_per_concurrency is not None and point.requests_per_concurrency < max(concurrencies):
            raise ValueError("requests_per_concurrency must be at least every selected concurrency")
    if decode.reasoning_effort is not None and (not isinstance(decode.reasoning_effort, str) or not decode.reasoning_effort):
        raise ValueError("reasoning_effort must be a nonempty string or null")
    if decode.prompts is not None and (not isinstance(decode.prompts, list) or not decode.prompts or
                                      any(not isinstance(p, str) or not p.strip() for p in decode.prompts)):
        raise ValueError("decode.prompts must be a nonempty string list")
    for label in ("excel_path", "json_path"):
        value = getattr(config.output, label)
        if value is None and label == "json_path":
            continue
        if not isinstance(value, str) or not value:
            raise ValueError("output." + label + " must be a path")
        path = Path(value).expanduser()
        setattr(config.output, label, str((base / path).resolve()))
    if service.tokenizer_path:
        service.tokenizer_path = str((base / Path(service.tokenizer_path).expanduser()).resolve())
    if service.capacity_snapshot:
        service.capacity_snapshot = str((base / Path(service.capacity_snapshot).expanduser()).resolve())
    if config.output.excel_path == config.output.json_path:
        raise ValueError("Excel and JSON outputs must differ")
    return config
