import json
import re
from threading import current_thread, main_thread
from dataclasses import asdict
from typing import cast

from perf_benchmark.pipeline.contracts.results import RequestSnapshot

from perf_benchmark.core.models import RequestMeasurements

import pytest

from perf_benchmark.pipeline.contracts import (
    RunRecord,
)
from perf_benchmark.plugins.exporter.html import HtmlExporter, parse_settings


class PagedReader:
    location = "/tmp/results.db"

    def measurements(self, run_id, **kwargs):
        return {}

    def trials(self, run_id):
        return ()

    def summary(self, run_id):
        assert current_thread() is not main_thread()
        return {"name": "<script>alert(1)</script>", "status": "succeeded"}

    def read_updates(self, *args, **kwargs):
        raise AssertionError('summary export must not scan request changes')

    def iter_summary_records(self, run_id):
        from perf_benchmark.pipeline.contracts.results import MetricPoint
        assert current_thread() is not main_thread()
        yield RunRecord("r", "safe", "succeeded", "", None, "", "", "PRIVATE_ERROR", {})
        yield MetricPoint('trial', 't', 'total', 'tpot', 'am', None,
                          'ms', 'test-v1', 'unavailable', 'non_streaming_response', None)
        for index in range(1201):
            yield MetricPoint('trial', 't', f'timeline:{index}', 'e2e', 'am', .1,
                              's', 'test-v1', 'available', None, 'client')


async def test_html_streams_summaries_without_request_scan_and_escapes_scripts(tmp_path, stage_context):
    path = tmp_path / "result.html"
    result = await HtmlExporter(parse_settings({"output": str(path)})).export(
        "r", PagedReader(), stage_context("exporter")
    )
    assert result.outputs[0].location == str(path)
    output = path.read_text()
    pages = re.findall(
        r'<script type="application/json" class="result-page">(.*?)</script>', output
    )
    records = [row for page in pages for row in json.loads(page)]
    assert len(pages) == 3 and len(records) == 1203
    assert records[-1]["window"] == "timeline:1200"
    assert not any(r["kind"] in {"turn", "attempt"} for r in records)
    assert "perf-benchmark results serve" in output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in output
    assert all(
        secret not in output
        for secret in ["PRIVATE_ERROR", "PRIVATE_BODY", "PRIVATE_AUTH", "PRIVATE_USAGE"]
    )
    assert "<script src=" not in output and "fetch(" not in output
    unavailable = next(row for row in records if row.get("metric") == "tpot")
    assert unavailable["value"] is None
    assert unavailable["availability"] == "unavailable"
    assert unavailable["unavailable_reason"] == "non_streaming_response"
    assert 'id="measurement-semantics"' in output
    assert "ITL 是相邻文本输出事件的间隔" in output
    assert "TPOT 的 token 分母来自服务端 usage" in output
    assert "有效 / 预期样本" in output
    assert "成功 / 失败请求的统计资格" in output


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "https://user:password@example.com", "https://host/?token=secret"],
)
def test_report_rejects_unsafe_external_urls(url):
    with pytest.raises(ValueError):
        parse_settings({"grafana_url": url})


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
async def test_report_preserves_database_and_sidecars(tmp_path, stage_context, suffix):
    database = tmp_path / "results.db"
    target = tmp_path / ("results.db" + suffix)
    target.write_bytes(b"existing-result-file")
    reader = PagedReader()
    reader.location = str(database)
    exporter = HtmlExporter(parse_settings({"output": str(target)}))
    with pytest.raises(ValueError, match="must not replace"):
        await exporter.export("r", reader, stage_context("exporter"))
    assert target.read_bytes() == b"existing-result-file"


def test_summary_projection_excludes_attempt_details():
    from perf_benchmark.pipeline.contracts import AttemptResultRecord
    from perf_benchmark.plugins.exporter.html.render import projection

    started = 10**18 + 1
    record = AttemptResultRecord(
        turn_id="t",
        session_id="s",
        attempt=1,
        status="cancelled",
        error="PRIVATE_ERROR",
        error_origin="scenario",
        measurements=cast(RequestSnapshot, asdict(RequestMeasurements())),
        usage={},
        metrics={},
        scenario_passed=True,
        trigger_ns=started,
        decision_ns=started + 1,
        applied_ns=started + 2,
        plan={"private": "PRIVATE_PLAN"},
    )
    value = projection(record, "trial")
    assert value is None
