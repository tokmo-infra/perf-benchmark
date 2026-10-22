"""Capacity totals stay separate from usage, hardware bytes and block slots."""
import asyncio
import json
import time

from aiohttp import web
import pytest

from perf_benchmark.app.service_capacity import collect, metrics, sglang, snapshot


def test_sglang_counts_dp_pools_once_not_tp_shards():
    info = dict(dp_size=2, tp_size=4, max_total_num_tokens=900,
                internal_states=[dict(memory_usage=dict(token_capacity=n)) for n in (900, 800)])
    rows = sglang(info)
    assert [p["capacity_tokens"] for p in rows] == [900, 800]
    assert sum(p["capacity_tokens"] for p in rows) == 1700
    info["internal_states"].pop()
    assert sglang(info)[0]["reason"] == "incomplete_dp_states"
    assert sglang(dict(dp_size=1, tp_size=4, internal_states=[{}], max_total_num_tokens=900))[0]["capacity_tokens"] == 900


def test_nightly_attention_dp_states_and_global_tp_host_identity():
    info = dict(attn_dp_size=2, dp_size=1, tp_size=8, pp_size=2,
                enable_dp_attention=False,
                internal_states=[dict(memory_usage=dict(token_capacity=n)) for n in (900, 800)])
    assert [p["capacity_tokens"] for p in sglang(info)] == [900, 800]
    # CP ranks also mirror the attention-DP pool: group width includes CP.
    info["attn_cp_size"] = 2
    text = "\n".join(
        f'sglang:hicache_host_total_tokens{{model_name="m",dp_rank="{tp // 4}",tp_rank="{tp}",pp_rank="{pp}"}} {2700 if tp < 4 else 2400}'
        for pp in range(2) for tp in range(8))
    rows = metrics(text, "sglang", "m", info)
    assert [(r["pool"], r["capacity_tokens"]) for r in rows] == [("dp:0", 2700), ("dp:1", 2400)]
    info["internal_states"].pop()
    assert sglang(info)[0]["reason"] == "incomplete_dp_states"


def test_nightly_replica_dp_and_invalid_rank_are_explicitly_unknown():
    info = dict(attn_dp_size=2, dp_size=2, tp_size=8,
                internal_states=[dict(memory_usage=dict(token_capacity=900))] * 4)
    text = 'sglang:hicache_host_total_tokens{model_name="m",dp_rank="0",tp_rank="0"} 2700'
    assert sglang(info)[0]["reason"] == "replica_dp_topology_unsupported"
    assert metrics(text, "sglang", "m", info)[0]["reason"] == "replica_dp_topology_unsupported"
    info["dp_size"] = 1
    assert metrics(text.replace('dp_rank="0"', 'dp_rank="1"'), "sglang", "m", info)[0]["reason"] == "inconsistent_attention_dp_rank"
    info["attn_dp_size"] = 3
    assert metrics(text, "sglang", "m", info)[0]["reason"] == "invalid_attention_tp_topology"


def test_nightly_flat_attention_dp_host_metrics_match_real_deepseek83():
    # Fixed nightly controller passes computed attention-DP rank to each scheduler.
    info = dict(attn_dp_size=4, dp_size=1, tp_size=4, attn_cp_size=1, pp_size=1,
                enable_dp_attention=True,
                internal_states=[dict(memory_usage=dict(token_capacity=4136192))] * 4)
    text = "\n".join(
        f'sglang:hicache_host_total_tokens{{model_name="DeepSeek-V4-Flash-0731",dp_rank="{rank}",tp_rank="{rank}",pp_rank="0"}} 1.2408576e+07'
        for rank in (1, 0, 3, 2))
    rows = metrics(text, "sglang", "DeepSeek-V4-Flash-0731", info)
    assert {r["pool"]: r["capacity_tokens"] for r in rows} == {f"dp:{i}": 12408576 for i in range(4)}
    assert sum(r["capacity_tokens"] for r in rows) == 49634304
    assert sum(r["capacity_tokens"] for r in sglang(info)) == 16544768


@pytest.mark.parametrize("rank", ["-1", "4", "bad", "1.5"])
def test_nightly_attention_dp_label_must_match_global_tp(rank):
    text = f'sglang:hicache_host_total_tokens{{model_name="m",dp_rank="{rank}",tp_rank="0"}} 1000'
    row = metrics(text, "sglang", "m", dict(attn_dp_size=4, dp_size=1, tp_size=4))[0]
    assert row["capacity_tokens"] is None and row["reason"] == "inconsistent_attention_dp_rank"


def test_metrics_deduplicate_keep_total_not_free_and_separate_vllm_slots():
    text = '\n'.join([
        'sglang:hicache_host_total_tokens{model_name="m",tp_rank="0",pp_rank="0"} 3000',
        'sglang:hicache_host_total_tokens{model_name="m",tp_rank="0",pp_rank="1"} 3000',
        'sglang:hicache_host_total_tokens{model_name="m",tp_rank="2"} 2800',
        'sglang:hicache_host_total_tokens{model_name="other",tp_rank="3"} 9000',
        'sglang:kv_available_tokens{model_name="m",tp_rank="0"} 12',
        'sglang:hicache_host_used_tokens{model_name="m",tp_rank="0"} 15',
    ])
    topology = {"dp_size": 2, "tp_size": 4, "enable_dp_attention": True}
    assert [r["capacity_tokens"] for r in metrics(text, "sglang", "m", topology)] == [3000, 2800]
    assert metrics(text + '\nsglang:hicache_host_total_tokens{model_name="m",tp_rank="0"} 20', "sglang", "m", topology)[0]["status"] == "unknown"
    vllm = 'vllm:cache_config_info{model_name="m",engine="0",block_size="64",num_gpu_blocks="1000"} 1'
    row = metrics(vllm, "unknown", "m")[0]
    assert row["tier"] == "gpu_block_slots" and row["capacity_tokens"] == 64000
    assert "full-model" in row["reason"]


def test_hicache_all_tp_and_pp_replicas_are_one_logical_pool():
    text = "\n".join(f'sglang:hicache_host_total_tokens{{model_name="m",tp_rank="{tp}",pp_rank="{pp}"}} 947008'
                     for pp in range(2) for tp in range(8))
    rows = metrics(text, "sglang", "m", {"dp_size": 1, "tp_size": 8, "pp_size": 2})
    assert len(rows) == 1 and rows[0]["pool"] == "dp:0" and rows[0]["capacity_tokens"] == 947008


def test_equal_capacities_in_distinct_dp_pools_are_not_deduplicated():
    text = "\n".join(f'sglang:hicache_host_total_tokens{{model_name="m",tp_rank="{rank}"}} 1000' for rank in range(8))
    rows = metrics(text, "sglang", "m", {"dp_size": 2, "tp_size": 8, "enable_dp_attention": True})
    assert [p["pool"] for p in rows] == ["dp:0", "dp:1"]
    assert [p["capacity_tokens"] for p in rows] == [1000, 1000]
    explicit = "\n".join(f'sglang:hicache_host_total_tokens{{model_name="m",dp_rank="{dp}",tp_rank="{tp}"}} 1000'
                         for dp in range(2) for tp in range(4))
    assert len(metrics(explicit, "sglang", "m")) == 2


def test_ambiguous_or_conflicting_topology_keeps_capacity_unknown():
    text = 'sglang:hicache_host_total_tokens{model_name="m",tp_rank="0"} 1000\n'
    assert metrics(text, "sglang", "m")[0]["reason"] == "independent_pool_topology_required"
    text += 'sglang:hicache_host_total_tokens{model_name="m",tp_rank="1"} 999\n'
    row = metrics(text, "sglang", "m", {"dp_size": 1, "tp_size": 8})[0]
    assert row["capacity_tokens"] is None and row["reason"] == "conflicting_pool_reports"


def test_snapshot_records_shared_host_once_and_checks_identity(tmp_path):
    value = dict(model="m", observed_at="2026-10-16T12:00:00+08:00", pools=[
        dict(tier="gpu", pool="dp:0", capacity_tokens=64000, source="registered blocks"),
        dict(tier="host_cpu", pool="shared", capacity_tokens=120000, source="complete chunks")])
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(value))
    assert snapshot(path, "m")["engine"] == "unknown"
    assert snapshot(path, "m")["pools"][1]["capacity_tokens"] == 120000
    with pytest.raises(ValueError, match="match"):
        snapshot(path, "another-model")
    value["pools"].append(value["pools"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate"):
        snapshot(path, "m")


@pytest.fixture
def capacity_snapshot(tmp_path):
    value = dict(model="m", observed_at="2026-10-17T12:00:00+08:00", pools=[
        dict(tier="gpu", pool="dp:0", capacity_tokens=64000, source="registered blocks")])
    return tmp_path / "capacity.json", value


@pytest.mark.parametrize("engine", ["sglang", "vllm", "unknown"])
def test_snapshot_preserves_explicit_engine(capacity_snapshot, engine):
    path, value = capacity_snapshot
    value["engine"] = engine
    path.write_text(json.dumps(value))
    imported = snapshot(path, "m")
    assert imported["engine"] == engine and imported["observed_at"] == value["observed_at"]
    assert imported["pools"][0]["capacity_tokens"] == 64000


@pytest.mark.parametrize("engine", [None, "", "SGLang", "other", False, 1, [], {}])
def test_snapshot_rejects_invalid_engine(capacity_snapshot, engine):
    path, value = capacity_snapshot
    value["engine"] = engine
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="snapshot engine"):
        snapshot(path, "m")


@pytest.mark.parametrize("field", ["model", "observed_at", "pools"])
def test_snapshot_engine_retains_required_fields(capacity_snapshot, field):
    path, value = capacity_snapshot
    value["engine"] = "sglang"
    value.pop(field)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="snapshot must match"):
        snapshot(path, "m")


def test_snapshot_engine_retains_strict_field_allowlist(capacity_snapshot):
    path, value = capacity_snapshot
    value.update(engine="sglang", engine_version="nightly")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="snapshot must match"):
        snapshot(path, "m")


@pytest.mark.parametrize("engine", [None, "sglang", "vllm", "unknown"])
async def test_collect_imports_snapshot_engine_without_http_or_plugin_binding(capacity_snapshot, monkeypatch, engine):
    path, value = capacity_snapshot
    if engine is not None:
        value["engine"] = engine
    path.write_text(json.dumps(value))
    def unexpected_http(*args, **kwargs):
        pytest.fail("snapshot import attempted an HTTP probe")
    monkeypatch.setattr("perf_benchmark.app.service_capacity.aiohttp.ClientSession", unexpected_http)
    config = dict(pipeline=dict(executor=dict(plugin="unused-snapshot-executor", settings=dict(model="m"))))
    imported = await collect(config, {"snapshot": path.name}, path.parent)
    assert imported["engine"] == (engine if engine is not None else "unknown")
    assert imported["status"] == "deployment_snapshot" and imported["probes"] == {}
    assert imported["observed_at"] == value["observed_at"] and imported["pools"][0]["capacity_tokens"] == 64000


async def test_probe_is_bounded_authenticated_and_keeps_only_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPACITY_TEST_KEY", "SECRET")
    async def info(request):
        assert request.headers["Authorization"] == "Bearer SECRET"
        return web.json_response(dict(dp_size=1, internal_states=[dict(memory_usage=dict(token_capacity=123456))],
                                      secret="SECRET", environment="UNRELATED"))
    async def metric(request):
        return web.Response(text='sglang:hicache_host_total_tokens{model_name="m",tp_rank="0"} 370368\n')
    async def slow(request):
        await asyncio.sleep(.2)
        return web.Response(text="ignored")
    app = web.Application()
    app.router.add_get("/prefix/get_server_info", info)
    app.router.add_get("/prefix/metrics", metric)
    app.router.add_get("/slow/{path:.*}", slow)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    config = dict(pipeline=dict(executor=dict(plugin="openai-chat", settings=dict(
        base_url=base + "/prefix/v1", model="m", api_keys=[{"env": "CAPACITY_TEST_KEY"}]))))
    try:
        value = await collect(config, {}, tmp_path)
        assert value["engine"] == "sglang"
        assert [r["capacity_tokens"] for r in value["pools"]] == [123456, 370368]
        assert "SECRET" not in json.dumps(value) and "UNRELATED" not in json.dumps(value)
        config["pipeline"]["executor"]["settings"]["base_url"] = base + "/slow/v1"
        started = time.monotonic()
        value = await collect(config, {"timeout_seconds": .03}, tmp_path)
        assert time.monotonic() - started < .2
        assert value["status"] == "unavailable" and all(r["capacity_tokens"] is None for r in value["pools"])
    finally:
        await server.cleanup()


async def test_invalid_settings_and_disabled_probe(tmp_path):
    for settings in ({"timeout_seconds": True}, {"timeout_seconds": 20}, {"enabled": "yes"}, {"ratio": 3}):
        with pytest.raises(ValueError):
            await collect({}, settings, tmp_path)
    assert (await collect({}, {"enabled": False}, tmp_path))["status"] == "disabled"
