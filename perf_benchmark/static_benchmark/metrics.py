"""Presentation of the native StandardMetrics results; no second metric implementation."""
from dataclasses import dataclass, field


@dataclass
class AggregatedTestResult:
    test_type: str
    concurrency: int
    rounds: int
    requests_per_round: int
    input_tokens: int | None
    total_requests: int
    succeeded_requests: int
    failed_requests: int
    cancelled_requests: int
    success_rate: float | None
    wall_time_s: float | None
    avg_actual_prompt_tokens: float | None
    avg_actual_completion_tokens: float | None
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    ttft_p99_ms: float | None
    prefill_speed_mean: float | None
    prefill_speed_p50: float | None
    prefill_speed_p90: float | None
    prefill_speed_p99: float | None
    total_prefill_speed: float | None
    decode_speed_mean: float | None
    decode_speed_p50: float | None
    decode_speed_p90: float | None
    decode_speed_p99: float | None
    total_decode_speed: float | None
    database: str
    run_id: str
    status: str
    native_metrics: dict = field(repr=False)
    service_capacity: dict = field(default_factory=dict)

    def eligible_samples(self, metric):
        statistics = self.native_metrics.get(metric, {}).get("statistics", {})
        value = statistics.get("eligible_n", statistics.get("n"))
        return int(value) if value is not None else None

    @property
    def tail_sample_note(self):
        speed = "prefill_throughput" if self.test_type == "prefill" else "decode_throughput"
        counts = (self.eligible_samples("ttft"), self.eligible_samples(speed))
        if any(n is None or n < 100 for n in counts):
            return "存在有效样本数<100或缺失的指标，P99仅供观察；可增加每轮请求数"
        return "P99为样本估计；100条仅提供约1%的经验分位分辨率，不保证尾延迟稳定"

    @classmethod
    def from_reader(cls, reader, run_id, *, test_type, concurrency, rounds, requests_per_round, input_tokens=None):
        trials = reader.trials(run_id)
        if len(trials) != 1:
            raise ValueError("a static matrix point must contain exactly one native Trial")
        metrics = reader.metric_points(run_id, scope="trial", scope_id=trials[0]["trial_id"], window="overall")
        def value(name, statistic="value", scale=1):
            result = metrics.get(name, {}).get("statistics", {}).get(statistic)
            return None if result is None else result * scale
        summary = reader.summary(run_id)
        fields = dict(test_type=test_type, concurrency=concurrency, rounds=rounds,
                      requests_per_round=requests_per_round, input_tokens=input_tokens,
                      total_requests=summary["turn_count"], succeeded_requests=int(value("succeeded") or 0),
                      failed_requests=int(value("failed") or 0), cancelled_requests=int(value("cancelled") or 0),
                      success_rate=value("success_rate", scale=100), wall_time_s=value("window_duration"),
                      avg_actual_prompt_tokens=value("mean_input_tokens"),
                      avg_actual_completion_tokens=value("mean_output_tokens"),
                      total_prefill_speed=value("workload_prompt"),
                      total_decode_speed=value("workload_completion"), database=reader.location,
                      run_id=run_id, status=summary["status"], native_metrics=metrics,
                      service_capacity=reader.measurements(run_id, scope="run", scope_id=run_id)
                          .get("evidence", {}).get("service_capacity", {}))
        for label, metric, scale in (("ttft", "ttft", 1000),
                                     ("prefill_speed", "prefill_throughput", 1),
                                     ("decode_speed", "decode_throughput", 1)):
            for display, statistic in (("mean", "am"), ("p50", "p50"), ("p90", "p90"), ("p99", "p99")):
                key = label + "_" + display + ("_ms" if label == "ttft" else "")
                fields[key] = value(metric, statistic, scale)
        return cls(**fields)
