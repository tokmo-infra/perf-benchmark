"""Static workload CLI; measurement, cancellation and facts use the native pipeline."""
from pathlib import Path
from rich.console import Console
from perf_benchmark.static_benchmark.config import load_static_benchmark_config
from perf_benchmark.static_benchmark.runner import StaticBenchmarkRunner

console = Console()


async def run_static_cli(args):
    config = load_static_benchmark_config(args.config)
    if getattr(args, "only_prefill", False) and getattr(args, "only_decode", False):
        raise ValueError("choose only-prefill or only-decode")
    if getattr(args, "only_prefill", False):
        config.prefill.enabled, config.decode.enabled = True, False
    if getattr(args, "only_decode", False):
        config.prefill.enabled, config.decode.enabled = False, True
    if getattr(args, "output", None):
        config.output.excel_path = str(Path(args.output).resolve())
    if getattr(args, "base_url", None):
        config.service.base_url = args.base_url
    if getattr(args, "model", None):
        config.service.model = args.model
    # Validate overrides using the same loader; never print API keys or headers.
    from dataclasses import asdict
    values = asdict(config)
    values.pop("config_dir")
    config = load_static_benchmark_config(values)
    console.print(f"静态测试：{config.name}; 模型 {config.service.model}")
    console.print(f"Prefill矩阵: {config.prefill.matrix if config.prefill.enabled else 'disabled'}")
    console.print(f"Decode并发: {config.decode.concurrencies if config.decode.enabled else 'disabled'}")
    for name in ("prefill", "decode"):
        point = getattr(config, name)
        if point.enabled:
            console.print(f"{name}: {point.rounds}轮连续闭环；每轮请求数="
                          f"{point.requests_per_concurrency or '当前并发数'}；各轮请求统一统计")
    console.print(f"Excel: {config.output.excel_path}")
    if getattr(args, "dry_run", False):
        console.print("dry-run：仅校验配置；没有加载tokenizer或发送请求")
        return 0
    runner = StaticBenchmarkRunner(config)
    pf, dec = await runner.run_all()
    return 1 if any(r.status != "succeeded" or r.failed_requests or r.cancelled_requests
                    for r in (*pf, *dec)) else 0
