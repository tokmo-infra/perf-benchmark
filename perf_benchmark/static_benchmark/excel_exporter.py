"""One workbook from native measurements, with explicit missing-value reasons."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from perf_benchmark.plugins.exporter.tabular.render import write_xlsx, display_value, without_availability
from perf_benchmark.plugins.exporter.tabular.append import publish_workbook


def export_benchmark_to_excel(excel_path, config, prefill_results, decode_results, *, include_config=True):
    path = Path(excel_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    def rows(results, prefill):
        for result in results:
            row = [
                result.input_tokens, result.concurrency, result.total_requests,
                result.succeeded_requests, result.failed_requests, result.cancelled_requests,
                result.success_rate, result.ttft_mean_ms, result.ttft_p50_ms,
                result.ttft_p90_ms, result.ttft_p99_ms,
                result.prefill_speed_mean if prefill else result.decode_speed_mean,
                result.prefill_speed_p50 if prefill else result.decode_speed_p50,
                result.prefill_speed_p90 if prefill else result.decode_speed_p90,
                result.prefill_speed_p99 if prefill else result.decode_speed_p99,
                result.total_prefill_speed if prefill else result.total_decode_speed,
                result.avg_actual_prompt_tokens, result.avg_actual_completion_tokens,
                result.wall_time_s, result.status, result.run_id, result.database,
            ]
            metrics = {6: ("success_rate", "value"), 15: ("workload_prompt" if prefill else "workload_completion", "value"),
                       16: ("mean_input_tokens", "value"), 17: ("mean_output_tokens", "value"),
                       18: ("window_duration", "value")}
            for offset, name in ((7, "ttft"), (11, "prefill_throughput" if prefill else "decode_throughput")):
                metrics.update({offset + i: (name, stat) for i, stat in enumerate(("am", "p50", "p90", "p99"))})
            for index, (name, stat) in metrics.items():
                row[index] = display_value(result.native_metrics.get(name, {}), row[index], stat)
            speed = "prefill_throughput" if prefill else "decode_throughput"
            row[2:2] = [result.rounds, result.requests_per_round]
            row.extend([result.eligible_samples("ttft"), result.eligible_samples(speed), result.tail_sample_note])
            label = (f"prefill-{result.input_tokens}-c{result.concurrency}" if prefill
                     else f"decode-c{result.concurrency}")
            yield [config.case+" / "+label, *(row if prefill else row[1:])]
    header = ["目标输入token", "并发", "轮数", "每轮请求数", "总请求数", "成功", "失败", "取消", "成功率(%)",
              "TTFT mean(ms)", "TTFT P50(ms)", "TTFT P90(ms)", "TTFT P99(ms)",
              "单请求速度 mean(tok/s)", "P50(tok/s)", "P90(tok/s)", "P99(tok/s)",
              "批次吞吐(tok/s，含TTFT)", "平均实际输入token", "平均实际输出token",
              "测量窗口(s)", "状态", "Run", "原生数据库", "TTFT有效样本n", "速度有效样本n", "P99样本说明"]
    def availability():
        for result in (*prefill_results, *decode_results):
            for name, metric in result.native_metrics.items():
                for statistic, value in metric.get("statistics", {}).items():
                    meta = metric.get("statistic_metadata", {}).get(statistic, metric)
                    yield [result.test_type, result.concurrency, result.input_tokens, result.run_id,
                           name, statistic, value, meta.get("unit", metric.get("unit")),
                           meta.get("availability"), meta.get("unavailable_reason")]
    metadata = [("case", config.case), ("任务", config.name), ("模型", config.service.model),
                ("服务", config.service.base_url), ("分词器", config.service.tokenizer_path),
                ("Decode思考档位(请求值)", config.decode.reasoning_effort),
                ("Decode输出上限", config.decode.max_tokens), ("自然EOS", not config.decode.ignore_eos),
                ("统计", "原生StandardMetrics；空白为缺失，详见指标可用性"),
                ("批次吞吐", "原生workload_prompt/workload_completion，包含排队和TTFT"),
                ("测量轮次", "每点至少三轮请求配额，连续闭环；全部请求由原生指标统一统计，分位数不按轮平均"),
                ("生成时间UTC", datetime.now(timezone.utc).isoformat())]
    specs = [
        ("Prefill性能测试", ["case",*header], lambda: rows(prefill_results, True)),
        ("Decode性能测试", ["case",*header[1:]], lambda: rows(decode_results, False)),
        ("指标可用性", ["阶段", "并发", "目标输入", "Run", "指标", "统计量", "值", "单位",
                       "可用性", "缺失原因"], availability),
        ("KV容量", ["Run", "采集时间", "引擎", "层级", "独立池", "容量(token)", "状态", "来源", "说明"],
         lambda: ([r.run_id, r.service_capacity.get("observed_at"), r.service_capacity.get("engine"),
                   p["tier"], p["pool"], p["capacity_tokens"], p["status"], p["source"], p.get("reason")]
                  for r in (*prefill_results, *decode_results) for p in r.service_capacity.get("pools", []))),
    ]
    if include_config:
        specs.insert(2, ("测试概览与配置", ["配置项", "值"], lambda: iter(metadata)))
    with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_xlsx(temporary, tuple(without_availability(spec) for spec in specs))
        publish_workbook(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
