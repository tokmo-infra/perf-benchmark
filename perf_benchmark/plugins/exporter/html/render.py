"""Page committed results and stream a body-free, script-safe projection to disk."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import orjson
from jinja2 import Environment, FileSystemLoader, select_autoescape

from perf_benchmark.core.models import thaw_snapshot

from perf_benchmark.pipeline.contracts import (
    GroupedMetricRecord,
    MeasurementRecord,
    MetricPoint,
    MetricWindowResultRecord,
    RunRecord,
    SweepRecord,
    TrialResultRecord,
)


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def projection(record, trial_id):
    """Explicit allowlists: never embed metadata, request content, headers or errors."""
    common = {"trial_id": trial_id}
    if isinstance(record, SweepRecord):
        return {"kind": "sweep", **{name: plain(getattr(record, name)) for name in (
            "sequence", "axis", "phase", "status", "value", "trial_ids", "checks",
            "lower", "upper", "reason", "validity",
        )}}
    if isinstance(record, RunRecord):
        return {
            "kind": "run",
            **{
                name: getattr(record, name)
                for name in (
                    "id",
                    "name",
                    "status",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "counts",
                )
            },
        }
    if isinstance(record, TrialResultRecord):
        return {
            "kind": "trial",
            **{
                name: getattr(record, name)
                for name in (
                    "id",
                    "name",
                    "ordinal",
                    "status",
                    "started_at",
                    "finished_at",
                    "turn_count",
                    "succeeded_turns",
                    "failed_or_cancelled_turns",
                )
            },
        }
    if isinstance(record, (MetricPoint, MetricWindowResultRecord, GroupedMetricRecord)):
        kind = (
            "point"
            if isinstance(record, MetricPoint)
            else "group"
            if isinstance(record, GroupedMetricRecord)
            else "window"
        )
        return {"kind": kind, **common, **{f.name: plain(getattr(record, f.name)) for f in fields(record)}}
    if isinstance(record, MeasurementRecord) and record.name == "observation":
        assert isinstance(record.value, Mapping)
        observation = thaw_snapshot(record.value)
        settings = observation.get("settings", {})
        assert isinstance(settings, Mapping)
        targets = settings.get("targets", {})
        assert isinstance(targets, Mapping)
        clock = observation.get("clock", {})
        assert isinstance(clock, Mapping)
        return {
            "kind": "observation",
            "clock": {key: str(value) for key, value in clock.items()},
            "targets": {
                kind: [
                    {
                        key: value
                        for key, value in target.items()
                        if key
                        in {
                            "address",
                            "service_name",
                            "engine",
                            "model",
                            "version",
                            "scheme",
                            "metrics_path",
                        }
                    }
                    for target in targets.get(kind, ())
                ]
                for kind in ("engines", "gpu", "profiles")
            },
        }
    return None


def script_json(value):
    return (
        orjson.dumps(plain(value))
        .decode()
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_report(run_id, results, path, grafana_url=""):
    summary = results.summary(run_id)
    if summary["status"] not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("offline report requires a terminal Run; retry after execution finishes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"])
    )
    template = environment.get_template("report.html")
    # Keep any existing report intact until a complete reader replay succeeds.
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".report-",
            suffix=".html",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                template.render(title=summary["name"], run_id=run_id, grafana_url=grafana_url, database=results.location)
            )
            evidence = results.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {})
            trial_evidence = []
            for trial in results.trials(run_id):
                validity = trial.get("validity") or {}
                observations = {name: value for name, value, _unit in validity.get("observations", [])}
                profile = trial.get("load_profile") or {}
                target = next((profile[key] for key in ("target_concurrency", "max_active_sessions",
                    "target_request_rate", "target_session_rate") if profile.get(key) is not None), None)
                trial_evidence.append({"id": trial["trial_id"], "status": trial["status"],
                    "execution_status": trial.get("execution_status"), "validity": validity,
                    "target_pressure": target, "actual_pressure": observations.get("actual_pressure"),
                    "scheduling": trial.get("scheduling") or {}, "repetition": trial.get("repetition"),
                    "preparation": trial["metadata"].get("preparation"),
                    "graph_coverage": (trial.get("scheduling") or {}).get("graph_coverage", [])})
            evidence_template = environment.from_string(
                '<section><h2>实验条件、质量与图覆盖</h2><p>{{ source_content }}</p>'
                '<h3>实验条件</h3><pre style="white-space:pre-wrap">{{ evidence }}</pre>'
                '<h3>Trial 资格与压力</h3><table><tr><th>Trial</th><th>执行</th><th>负载资格</th>'
                '<th>目标</th><th>实际</th><th>offered / admitted / completed</th><th>原因</th></tr>'
                '{% for row in trial_rows %}<tr><td>{{ row.id }}</td><td>{{ row.execution_status }}</td>'
                '<td>{{ row.validity.status or "未核实" }}</td><td>{{ row.target_pressure }}</td>'
                '<td>{{ row.actual_pressure }}</td><td>{{ row.scheduling.offered_attempts }} / '
                '{{ row.scheduling.admitted_attempts }} / {{ row.scheduling.completed_attempts }}</td>'
                '<td>{{ row.validity.reasons }}</td></tr>{% endfor %}</table>'
                '<h3>完整 Trial 证据</h3><pre style="white-space:pre-wrap">{{ trials }}</pre></section>')
            stream.write(evidence_template.render(source_content=evidence.get("source_content", "原始数据内容一致性未验证"),
                                                 evidence=orjson.dumps(evidence, option=orjson.OPT_INDENT_2).decode(),
                                                 trial_rows=trial_evidence,
                                                 trials=orjson.dumps(trial_evidence, option=orjson.OPT_INDENT_2).decode()))
            batch = []
            for record in results.iter_summary_records(run_id):
                trial_id = getattr(record, 'trial_id', None)
                projected = projection(record, trial_id)
                if projected is not None:
                    batch.append(projected)
                if len(batch) == 500:
                    stream.write('<script type="application/json" class="result-page">' + script_json(batch) + '</script>\n')
                    batch.clear()
            if batch:
                stream.write('<script type="application/json" class="result-page">' + script_json(batch) + '</script>\n')
            stream.write("<script>initializeReport();</script></body></html>\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
