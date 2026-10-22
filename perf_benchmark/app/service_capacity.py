"""Bounded, pre-request engine capacity observations; never infer tokens from VRAM."""
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

LIMIT = 4 * 1024 * 1024


def integer(value):
    try:
        number = float(value)
        return int(number) if not isinstance(value, bool) and math.isfinite(number) and number >= 0 and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def pool(tier, identity, tokens, source, *, status="observed", reason=None):
    return dict(tier=tier, pool=identity, capacity_tokens=tokens, unit="token",
                status=status if tokens is not None else "unknown", source=source, reason=reason)


def attention_dp(topology):
    """Resolved nightly separates model replicas from attention-DP schedulers."""
    if "attn_dp_size" in topology:
        replicas = integer(topology.get("dp_size", 1))
        if replicas != 1:
            return None, "replica_dp_topology_unsupported"
        dp = integer(topology["attn_dp_size"])
    else:
        # Older server-info used dp_size for DP attention.
        dp = integer(topology.get("dp_size", 1))
    if not dp or dp > 128:
        return None, "invalid_attention_dp_size"
    return dp, None


def sglang(info):
    """SGLang /get_server_info internal_states are scheduler readbacks per DP."""
    if not isinstance(info, dict) or "internal_states" not in info:
        return []
    states = info["internal_states"]
    dp, reason = attention_dp(info)
    if reason:
        return [pool("gpu", "service", None, "/get_server_info", reason=reason)]
    if not isinstance(states, list) or len(states) != dp:
        return [pool("gpu", "service", None, "/get_server_info", reason="incomplete_dp_states")]
    rows = []
    for i, state in enumerate(states):
        memory = state.get("memory_usage", {}) if isinstance(state, dict) else {}
        tokens = integer(memory.get("token_capacity"))
        if tokens is None and dp == 1:
            tokens = integer(info.get("max_total_num_tokens"))
        rows.append(pool("gpu", f"dp_state:{i}", tokens,
                         f"/get_server_info:internal_states[{i}].memory_usage.token_capacity"))
    return rows


def host_identity(labels, topology):
    """Resolve independent attention-DP pools; TP and PP only mirror a pool."""
    topology = topology if isinstance(topology, dict) else {}
    dp, reason = attention_dp(topology)
    tp = integer(topology.get("tp_size"))
    if reason:
        return "service", reason
    if "attn_dp_size" in topology:
        # Scheduler dp_rank is the attention-DP rank supplied by the controller.
        # Global TP locates the same pool; CP/TP and PP ranks mirror its budget.
        rank = integer(labels.get("tp_rank"))
        if not tp or tp % dp or rank is None or rank >= tp:
            return "service", "invalid_attention_tp_topology"
        attention_rank = rank // (tp // dp)
        if "dp_rank" in labels and integer(labels["dp_rank"]) != attention_rank:
            return "service", "inconsistent_attention_dp_rank"
        return "dp:" + str(attention_rank), None
    if not topology and "dp_rank" not in labels:
        return "service", "independent_pool_topology_required"
    if "dp_size" not in topology:
        dp = None
    rank = integer(labels.get("dp_rank"))
    if "dp_rank" in labels:
        if rank is None or rank >= (dp if dp else 128):
            return "service", "invalid_dp_rank"
        return "dp:" + str(rank), None
    if dp == 1:
        rank = integer(labels.get("tp_rank", 0))
        if rank is None or (tp is not None and rank >= tp):
            return "service", "invalid_tp_rank"
        return "dp:0", None
    if dp and tp and topology.get("enable_dp_attention") is True and tp % dp == 0:
        rank = integer(labels.get("tp_rank"))
        if rank is not None and rank < tp:
            return "dp:" + str(rank // (tp // dp)), None
    return "service", "independent_pool_topology_required"


def metrics(text, engine, model, topology=None):
    rows, seen = [], {}
    names = {"sglang:hicache_host_total_tokens", "vllm:cache_config_info"}
    # Parse only the two bounded capacity families, not request histograms.
    selected = "\n".join(line for line in text.splitlines() if line.split("{", 1)[0] in names)
    for family in text_string_to_metric_families(selected):
        for sample in family.samples:
            labels = sample.labels
            if labels.get("model_name") not in (None, model):
                continue
            if sample.name == "sglang:hicache_host_total_tokens" and engine == "sglang":
                # all-schedulers can emit every TP/PP replica. Membership is
                # established by topology rather than equal capacity values.
                identity, reason = host_identity(labels, topology)
                row = pool("host_cpu", identity, integer(sample.value) if reason is None else None,
                           "/metrics:" + sample.name, reason=reason)
            elif sample.name == "vllm:cache_config_info" and engine != "sglang":
                # num_gpu_blocks*block_size is a block-pool budget. Hybrid groups
                # can share that budget, so retain it as slots, not logical KV.
                blocks, size = integer(labels.get("num_gpu_blocks")), integer(labels.get("block_size"))
                row = pool("gpu_block_slots", "engine:" + labels.get("engine", "unknown"),
                           blocks * size if blocks and size else None, "/metrics:" + sample.name,
                           status="derived", reason="allocator_slots; full-model capacity requires cache-group geometry")
            else:
                continue
            key = (row["tier"], row["pool"])
            if len(row["pool"]) > 128 or len(rows) >= 128:
                raise ValueError("capacity pool labels exceed bounded metadata")
            if key in seen and seen[key]["capacity_tokens"] != row["capacity_tokens"]:
                seen[key].update(capacity_tokens=None, status="unknown", reason="conflicting_pool_reports")
            elif key not in seen:
                seen[key] = row
                rows.append(row)
    return rows


def snapshot(path, model):
    """Deployment adapters may supply verified geometry when HTTP omits it."""
    if path.stat().st_size > LIMIT:
        raise ValueError("capacity snapshot exceeds 4 MiB")
    value = json.loads(path.read_text())
    required = {"model", "observed_at", "pools"}
    if not isinstance(value, dict) or not required <= value.keys() or set(value) - required - {"engine"} or value["model"] != model:
        raise ValueError("capacity snapshot must match the served model")
    engine = value.get("engine", "unknown")
    if not isinstance(engine, str) or engine not in {"sglang", "vllm", "unknown"}:
        raise ValueError("capacity snapshot engine must be sglang, vllm or unknown")
    if datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("capacity snapshot timestamp requires a timezone")
    if not isinstance(value["pools"], list) or len(value["pools"]) > 256:
        raise ValueError("invalid capacity snapshot pools")
    rows, identities = [], set()
    for item in value["pools"]:
        if set(item) != {"tier", "pool", "capacity_tokens", "source"}:
            raise ValueError("invalid capacity snapshot pool")
        if item["tier"] not in {"gpu", "host_cpu"} or not all(
            isinstance(item[k], str) and 0 < len(item[k]) <= 512 for k in ("pool", "source")
        ) or integer(item["capacity_tokens"]) is None:
            raise ValueError("invalid capacity snapshot token capacity")
        key = item["tier"], item["pool"]
        if key in identities:
            raise ValueError("duplicate capacity snapshot pool")
        identities.add(key)
        rows.append(pool(*key, integer(item["capacity_tokens"]), item["source"], status="deployment_observed"))
    return dict(observed_at=value["observed_at"], engine=engine, pools=rows)


async def collect(config, settings, base_dir):
    """Best effort GETs finish before Coordinator starts benchmark measurement."""
    if not isinstance(settings, dict) or set(settings) - {"enabled", "timeout_seconds", "snapshot"}:
        raise ValueError("unknown service_capacity setting")
    if type(settings.get("enabled", True)) is not bool:
        raise ValueError("service_capacity.enabled must be boolean")
    seconds = settings.get("timeout_seconds", 2.0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError("service_capacity.timeout_seconds must be in (0, 10]")
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), engine="unknown", pools=[], probes={})
    if not settings.get("enabled", True):
        return dict(result, status="disabled")
    executor = config["pipeline"]["executor"]
    target = executor.get("settings", {})
    model = target.get("model")
    if settings.get("snapshot") is not None:
        imported = snapshot((Path(base_dir) / settings["snapshot"]).resolve(), model)
        result.update(imported, status="deployment_snapshot")
        return result
    from perf_benchmark.plugins import create_registry
    binding = create_registry().bind("executor", executor["plugin"], target)
    plugin = binding.plugin
    if not hasattr(plugin, "request_headers") or not target.get("base_url"):
        return dict(result, status="unavailable", reason="single_http_service_required")
    parsed = urlsplit(target["base_url"])
    # Preserve a reverse-proxy prefix; /v1 is the protocol namespace.
    path = parsed.path.rstrip("/")
    root = urlunsplit((parsed.scheme, parsed.netloc, path[:-3] if path.endswith("/v1") else path, "", ""))
    headers = plugin.request_headers(binding.settings)
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=seconds),
            connector=aiohttp.TCPConnector(ssl=target.get("verify_ssl", False))) as client:
        async def get(endpoint):
            try:
                async with client.get(root + endpoint, allow_redirects=False) as response:
                    result["probes"][endpoint] = "http_" + str(response.status)
                    if response.status != 200:
                        return None
                    data = bytearray()
                    async for part in response.content.iter_chunked(65536):
                        data.extend(part)
                        if len(data) > LIMIT:
                            result["probes"][endpoint] = "response_too_large"
                            return None
                    return bytes(data).decode("utf-8")
            except (aiohttp.ClientError, TimeoutError, UnicodeError):
                result["probes"][endpoint] = "unavailable"
                return None
        info, text = await asyncio.gather(get("/get_server_info"), get("/metrics"))
    try:
        state = json.loads(info) if info else {}
        result["pools"] = sglang(state)
        if result["pools"]:
            result["engine"] = "sglang"
        if text:
            result["pools"].extend(metrics(text, result["engine"], model, state))
            if any(p["tier"] == "gpu_block_slots" for p in result["pools"]):
                result["engine"] = "vllm"
    except (ValueError, TypeError, AttributeError):
        result["probes"]["parse"] = "unsupported_response"
    for tier in ("gpu", "host_cpu"):
        if not any(p["tier"] == tier for p in result["pools"]):
            result["pools"].append(pool(tier, "service", None, "engine", reason="token_capacity_not_exposed"))
    result["status"] = "observed" if any(p["capacity_tokens"] is not None for p in result["pools"]) else "unavailable"
    return result
