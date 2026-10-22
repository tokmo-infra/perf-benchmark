"""Body-free experiment conditions and external EvalScope association."""
import json
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml


from perf_benchmark.core.profiles import GENERATION_KEYS


def _generation(settings):
    result = {key: value for key, value in settings.get("extra_body", {}).items()
              if key in GENERATION_KEYS}
    if "max_tokens" in settings:
        result["max_tokens"] = settings["max_tokens"]
    return result


def _location(value):
    if isinstance(value, list):
        return [_location(item) for item in value]
    if isinstance(value, dict):
        return {key: _location(item) for key, item in value.items()}
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        return urlunsplit((parsed.scheme, host + (":" + str(parsed.port) if parsed.port else ""), parsed.path, "", ""))
    return value


def conditions(config, declarations, base_dir: Path | None = None):
    if set(declarations) - {"deployment", "cache", "quantization", "template"}:
        raise ValueError("conditions supports deployment, cache, quantization and template identifiers")
    rows = {}
    def add(key, value, status="observed"):
        rows[key] = {"value": value, "status": "unknown" if value is None else status}
    pipeline = config["pipeline"]
    for stage, keys in {
        "loader": {"dataset", "data_files", "split", "revision", "name"},
        "normalization": {"source", "variant", "mode", "tokenizer_path", "max_context_tokens", "root_interval_s", "block_size"},
        "selector": {"limit", "max_graphs", "max_sessions", "min_input_tokens", "max_input_tokens", "min_output_tokens", "max_output_tokens", "sort_by"},
    }.items():
        binding = pipeline.get(stage, {})
        add(stage + ".plugin", binding.get("plugin"))
        settings = binding.get("settings", {})
        add(stage + ".settings", {k: _location(settings[k]) for k in keys if k in settings})
        if any(_location(settings[k]) != settings[k] for k in keys if k in settings):
            add(stage + ".source_parameters", None)
        # Unknown settings prevent an incomplete whitelist from claiming equivalence.
        add(stage + ".unverified_fields", sorted(set(settings) - keys), "declared")
    executor = pipeline.get("executor", {})
    settings = executor.get("settings", {})
    add("protocol", executor.get("plugin"))
    add("model", settings.get("model"))
    add("seed", config.get("seed", 0))
    add("generation", _generation(settings))
    add("stream", settings.get("stream"))
    scheduler = pipeline.get("scheduler", {})
    add("scheduler.plugin", scheduler.get("plugin"))
    scheduling = scheduler.get("settings", {})
    # Input-independent load controls are compared explicitly; arbitrary request bodies stay out.
    load_keys = {"max_concurrency", "max_active_sessions", "request_rate", "session_rate", "max_pending_requests",
                 "max_pending_sessions", "stop", "on_exhausted", "arrival_distribution", "arrival_gamma_shape",
                 "arrival_ramp", "repetitions", "validity", "time_scale"}
    add("load", {k: scheduling[k] for k in load_keys if k in scheduling})
    add("scheduler.unverified_fields", sorted(set(scheduling) - load_keys - {"sweep"}), "declared")
    sweep = scheduling.get("sweep", {})
    add("sweep", {k: sweep[k] for k in ("axis", "values", "search", "slo", "validity") if k in sweep})
    for key in ("deployment", "cache", "quantization", "template"):
        value = declarations.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a nonempty condition identifier")
        add(key, value, "declared")
    add("tokenizer_fingerprint", None)
    source = {"status": "unknown", "value": None}
    if base_dir is not None and pipeline.get("loader", {}).get("plugin") == "hf-datasets":
        from perf_benchmark.plugins.loader.hf_datasets import source_identity
        source = source_identity(pipeline["loader"].get("settings", {}), base_dir)
    rows["source_identity"] = source
    return {"conditions": rows, "source_content":
            "已核实固定来源身份" if source["status"] == "observed" else "原始数据内容一致性未验证"}


def quality(settings, config, base) -> dict:
    anonymous_trace = config["pipeline"]["normalization"]["plugin"] in {
        "agentx", "qwen-bailian", "mooncake", "burstgpt", "azure-2024", "servegen"}
    if not settings:
        return {"status": "not_applicable" if anonymous_trace else "unverified",
                "reason": "anonymous_trace" if anonymous_trace else "quality_not_imported"}
    if set(settings) - {"report", "config", "tool_version", "metric", "threshold", "direction", "template_id"}:
        raise ValueError("unknown EvalScope association setting")
    for key in ("report", "config", "tool_version", "metric", "threshold"):
        if key not in settings:
            raise ValueError(f"EvalScope {key} is required")
    report_path = (base / settings["report"]).resolve(strict=True)
    config_path = (base / settings["config"]).resolve(strict=True)
    report = json.loads(report_path.read_text())
    evaluated = yaml.safe_load(config_path.read_text())
    matches = [m for m in report.get("metrics", []) if m.get("name") == settings["metric"]]
    if len(matches) != 1:
        raise ValueError("EvalScope selected metric must match exactly once")
    score, threshold = matches[0].get("score"), settings["threshold"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (score, threshold)):
        raise ValueError("quality score and threshold must be finite numbers")
    direction = settings.get("direction", "higher")
    if direction not in {"higher", "lower"}:
        raise ValueError("quality direction must be higher or lower")
    executor = config["pipeline"]["executor"]["settings"]
    identity = config.get("application", {}).get("conditions", {})
    expected = {"model": executor.get("model"), "quantization": identity.get("quantization"),
                "template": identity.get("template"), "generation": _generation(executor)}
    actual = {"model": evaluated.get("model"), "quantization": evaluated.get("model_args", {}).get("quantization"),
              "template": settings.get("template_id"), "generation": {k: v for k, v in evaluated.get("generation_config", {}).items() if k in GENERATION_KEYS}}
    checks = {k: "unverified" if actual[k] is None or expected[k] is None else "match" if actual[k] == expected[k] else "mismatch" for k in expected}
    if report.get("model_name") != actual["model"]:
        checks["report_model"] = "mismatch" if report.get("model_name") and actual["model"] else "unverified"
    passed = score >= threshold if direction == "higher" else score <= threshold
    status = "fail" if "mismatch" in checks.values() else "unverified" if "unverified" in checks.values() else "pass" if passed else "fail"
    if anonymous_trace:
        status = "not_applicable"
    return {"status": status, "identity": expected, "identity_checks": checks, "score": score, "threshold": threshold, "direction": direction,
            "metric": settings["metric"], "score_passed": passed, "tool": "EvalScope", "tool_version": settings["tool_version"],
            "dataset": report.get("dataset_name"), "scoring_method": settings["metric"],
            "report": str(report_path), "evaluation_config": str(config_path)}
