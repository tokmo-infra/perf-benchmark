"""Static workload application reusing the native perf-benchmark pipeline."""
from perf_benchmark.static_benchmark.config import StaticBenchmarkConfig, load_static_benchmark_config
from perf_benchmark.static_benchmark.runner import StaticBenchmarkRunner
from perf_benchmark.static_benchmark.metrics import AggregatedTestResult

__all__ = ["StaticBenchmarkConfig", "load_static_benchmark_config",
           "StaticBenchmarkRunner", "AggregatedTestResult"]
