"""Stream body-free measurement tables, with explicit unknown values and units."""

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from shutil import copyfileobj
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
from unicodedata import east_asian_width, combining

from perf_benchmark.core.profiles import GENERATION_KEYS

DEFAULT_TABLES = ("summary", "minutes")
TABLE_NAMES = (*DEFAULT_TABLES, "requests", "kv_capacity")

MEASURED = (
    "maxtpm",
    "maxtpm_minutes",
    "maxtpm_input_tpm",
    "maxtpm_output_tpm",
    "maxtpm_cached_tpm",
    "maxtpm_uncached_tpm",
    "maxtpm_cache_hit_rate",
    "revenue_per_minute",
    "input_revenue_per_minute",
    "cached_input_revenue_per_minute",
    "output_revenue_per_minute",
)
SUMMARY_METRICS = ("maxtpm", "maxtpm_minutes", "maxtpm_input_tpm", "maxtpm_output_tpm",
                   "maxtpm_cache_hit_rate", "revenue_per_minute",
                   "cached_input_revenue_per_minute", "input_revenue_per_minute",
                   "output_revenue_per_minute")
REVENUE_METRICS = tuple(name for name in SUMMARY_METRICS if 'revenue_per_minute' in name)
OVERALL_DISTRIBUTIONS = ("ttft", "tpot", "itl", "e2e", "prefill_throughput",
                         "decode_throughput", "input_tokens", "output_tokens",
                         "session_duration", "session_first_turn_ttft",
                         "session_subsequent_turn_ttft")
OVERALL_SCALARS = ("window_duration", "succeeded", "workload_rps",
                   "total_prompt_tokens", "total_completion_tokens", "total_cached_tokens",
                   "workload_prompt", "workload_completion", "workload_total", "total_cost")
OVERALL_STATISTICS = (("mean", "am"), ("p50", "p50"), ("p90", "p90"), ("p99", "p99"))
MAXTPM_DISTRIBUTIONS = ("ttft", "tpot", "e2e", "itl", "prefill_throughput", "decode_throughput")
MAXTPM_STATISTICS = (*OVERALL_STATISTICS, ("n", "n"), ("eligible_n", "eligible_n"))
MAXTPM_SCALARS = ("workload_completion", "workload_rps", "succeeded")
SUMMARY_PREFIX = ("case", "model", "input_tokens_mean", "output_tokens_mean", "concurrency",
                  "active_users", "generation_parameters", "run_id", "name", "status")
PRICING_RATES = ("input_price", "cached_input_price", "output_price")
REASONLESS_METRICS = frozenset(("maxtpm_cache_hit_rate", "cached_input_revenue_per_minute",
                              "revenue_per_minute"))
SPEED_METRICS = frozenset(("prefill_throughput", "decode_throughput"))


def distribution_statistics(name, statistics=OVERALL_STATISTICS):
    return tuple(("time_weighted_mean", "time_wm") if name in SPEED_METRICS and label == "mean"
                 else (label, stat) for label, stat in statistics)


def value_columns(names):
    return tuple(column for name in names
                 for column in (name, name + "_unit", name + "_availability",
                                *(() if name in REASONLESS_METRICS else (name + "_reason",))))


VALUE_COLUMNS = value_columns(MEASURED)


class PartialValue(float):
    """Presentation marker; Excel retains the underlying numeric value."""


def display_value(metric, value, statistic="value"):
    status = metric.get("statistic_metadata", {}).get(statistic, metric)
    return PartialValue(value) if value is not None and status.get("availability") == "partial" else value


def values(metrics, names=MEASURED):
    return [
        value
        for name in names
        for m in (metrics.get(name, {}),)
        for value in (
            display_value(m, m.get("statistics", {}).get("value")),
            m.get("unit"),
            m.get("availability", "disabled"),
            *(() if name in REASONLESS_METRICS else (
                m.get("unavailable_reason", "metric_not_recorded" if not m and name in REVENUE_METRICS else None),
            )),
        )
    ]


def tables(run_id, reader):
    trials = reader.trials(run_id)
    capacity = reader.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {}).get("service_capacity")
    capacity_columns = ("gpu_kv_capacity_tokens (token)", "host_kv_capacity_tokens (token)") if capacity else ()

    def capacity_total(tier):
        pools = [p["capacity_tokens"] for p in capacity.get("pools", []) if p["tier"] == tier]
        return sum(pools) if pools and all(p is not None for p in pools) else None

    def summary():
        conditions = reader.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {}).get("conditions", {})
        for trial in trials:
            tid = trial["trial_id"]
            diagnostics = (
                reader.measurements(run_id, scope="trial", scope_id=tid)
                .get("diagnostics", {})
            )
            pricing = diagnostics.get("pricing", {})
            metrics = reader.metric_points(run_id, scope="trial", scope_id=tid, window="maxtpm")
            measured = reader.metric_points(run_id, scope="trial", scope_id=tid, window="overall")
            page = reader.request_page(run_id, scope="turn", trial_id=tid, offset=0, limit=500)
            profile = next((item.get("measurements", {}).get("request", {}).get("profile")
                            for item in page["items"] if item.get("measurements", {}).get("request", {}).get("profile")), {})
            generation = {key: value for source in (
                conditions.get("generation", {}).get("value") or {},
                trial.get("parameters", {}).get("request_parameters", {}),
            ) for key, value in source.items() if key in GENERATION_KEYS}
            generation.update({key: profile[key] for key in ("temperature", "top_p") if profile.get(key) is not None})
            load = trial.get("load_profile") or {}
            case = reader.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {}).get("case")
            row = [case, profile.get("model") or conditions.get("model", {}).get("value"),
                   measured.get("input_tokens", {}).get("statistics", {}).get("am"),
                   measured.get("output_tokens", {}).get("statistics", {}).get("am"),
                   load.get("target_concurrency") or load.get("max_concurrency"),
                   load.get("max_active_sessions"),
                   json.dumps(generation, ensure_ascii=False, sort_keys=True) if generation else None,
                   run_id, trial["name"], trial["status"]]
            if capacity:
                row.extend(capacity_total(tier) for tier in ("gpu", "host_cpu"))
            for name in OVERALL_DISTRIBUTIONS:
                metric = measured.get(name, {})
                row.extend(display_value(metric, metric.get("statistics", {}).get(stat), stat) for label, stat in distribution_statistics(name)
                           if label != "mean" or name not in {"input_tokens", "output_tokens"})
                row.extend((metric.get("unit"), metric.get("availability", "unavailable"),
                            metric.get("unavailable_reason", "metric_missing" if not metric else None)))
            row.extend(values(measured, OVERALL_SCALARS))
            rate_unit = f"{pricing['currency']}/{pricing['per_tokens']} tokens" if pricing else None
            for name in PRICING_RATES:
                row.extend((pricing.get(name), rate_unit))
            row.extend((pricing.get("per_tokens"), "tokens" if pricing else None,
                        pricing.get("source"), pricing.get("currency"), *values(metrics, SUMMARY_METRICS)))
            for name in MAXTPM_DISTRIBUTIONS:
                metric=metrics.get(name,{})
                row.extend(display_value(metric,metric.get('statistics',{}).get(stat),stat)
                           for _,stat in distribution_statistics(name, MAXTPM_STATISTICS))
                row.extend((metric.get('unit'),metric.get('availability','unavailable' if metrics else 'disabled'),
                            metric.get('unavailable_reason','metric_not_recorded' if metrics and not metric else None)))
            yield [*row,*values(metrics,MAXTPM_SCALARS)]

    def minutes():
        for trial in trials:
            tid = trial["trial_id"]
            windows = reader.metric_windows(run_id, scope="trial", scope_id=tid)
            for window in sorted(windows, key=lambda w: w["window"]):
                if not window["window"].startswith("maxtpm:minute:"):
                    continue
                metrics = reader.metric_points(
                    run_id, scope="trial", scope_id=tid, window=window["window"]
                )
                yield [
                    run_id,
                    int(window["window"].rsplit(":", 1)[1]),
                    str(window["start_ns"]),
                    str(window["end_ns"]),
                    window["duration_s"],
                    *values(metrics),
                ]

    def requests():
        for trial in trials:
            for scope in ("turn", "attempt"):
                offset = 0
                while True:
                    page = reader.request_page(
                        run_id, scope=scope, trial_id=trial["trial_id"], offset=offset, limit=500
                    )
                    for row in page["items"]:
                        usage = row.get("usage", {})
                        prompt = usage.get("prompt_tokens")
                        output = usage.get("completion_tokens")
                        cached = usage.get("cached_tokens")
                        valid = (
                            isinstance(prompt, int)
                            and not isinstance(prompt, bool)
                            and prompt >= 0
                            and isinstance(cached, int)
                            and not isinstance(cached, bool)
                            and 0 <= cached <= prompt
                        )
                        measurements = row.get("measurements", {})
                        request = (
                            measurements.get("request", {}) if scope == "turn" else measurements
                        )
                        yield [
                            run_id,
                            scope,
                            row.get("id", row.get("turn_id")),
                            row.get("attempt"),
                            row.get("status"),
                            str(request.get("response_completed_ns"))
                            if request.get("response_completed_ns") is not None
                            else None,
                            prompt,
                            output,
                            cached,
                            prompt - cached if valid else None,
                            cached / prompt if valid and prompt else None,
                            "available" if valid else "unavailable",
                            None if valid else "cache_usage_missing_or_invalid",
                        ]
                    offset += len(page["items"])
                    if offset >= page["total"] or not page["items"]:
                        break

    capacity_table = (("kv_capacity", ("run_id", "observed_at", "engine", "tier", "pool", "capacity_tokens (token)",
                                     "status", "source", "reason"),
                       lambda: ([run_id, capacity.get("observed_at"), capacity.get("engine"), p["tier"], p["pool"],
                                 p["capacity_tokens"], p["status"], p["source"], p.get("reason")]
                                for p in capacity.get("pools", []))),) if capacity else ()
    specs = (
        ("summary", (*SUMMARY_PREFIX, *capacity_columns, *(
            name + "_" + suffix for name in OVERALL_DISTRIBUTIONS
            for suffix in (*[label for label, _ in distribution_statistics(name)
                            if label != "mean" or name not in {"input_tokens", "output_tokens"}],
                           "unit", "availability", "reason")
        ), *value_columns(OVERALL_SCALARS),
            *(column for name in PRICING_RATES for column in (name, name + "_unit")),
            "per_tokens", "per_tokens_unit", "pricing_source", "currency",
            *value_columns(SUMMARY_METRICS), *(
            'maxtpm_'+name+'_'+suffix for name in MAXTPM_DISTRIBUTIONS
            for suffix in (*[label for label,_ in distribution_statistics(name, MAXTPM_STATISTICS)],'unit','availability','reason')
        ), *(
            'maxtpm_'+column for name in MAXTPM_SCALARS
            for column in (name,name+'_unit',name+'_availability',name+'_reason')
        )), summary),
        (
            "minutes",
            ("run_id", "minute", "start_ns", "end_ns", "duration_s", *VALUE_COLUMNS),
            minutes,
        ),
        (
            "requests",
            (
                "run_id",
                "scope",
                "request_id",
                "attempt",
                "status",
                "response_completed_ns",
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "uncached_input_tokens",
                "cache_hit_rate",
                "cache_availability",
                "cache_reason",
            ),
            requests,
        ),
    ) + capacity_table
    return tuple(without_availability(spec) for spec in specs)


def without_availability(spec):
    """Omit redundant status columns while retaining reasons and partial values."""
    name, header, factory = spec
    columns = tuple(i for i, label in enumerate(header)
                    if label not in {"availability", "可用性"} and not label.endswith("_availability"))
    return name, tuple(header[i] for i in columns), lambda: (
        tuple(row[i] for i in columns) for row in factory())


def csv_value(value):
    if isinstance(value, PartialValue):
        return str(value) + "*"
    # Spreadsheet programs treat formula-looking CSV text as code; export it as text.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def cell(value, *, header=False, reference=None):
    position = ' r="' + reference + '"' if reference is not None else ''
    if value is None:
        return '<c' + position + '/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            style = ' s="1"' if isinstance(value, PartialValue) else ""
            return f"<c{position}{style}><v>{value}</v></c>"
        return '<c' + position + '/>'
    value = str(value)
    # XML 1.0 permits only these control characters.
    value = "".join(
        ch
        for ch in value
        if ch in "\t\n\r"
        or 32 <= ord(ch) <= 0xD7FF
        or 0xE000 <= ord(ch) <= 0xFFFD
        or 0x10000 <= ord(ch) <= 0x10FFFF
    )
    style = ' s="2"' if header else ''
    return '<c'+position+style+' t="inlineStr"><is><t xml:space="preserve">' + escape(value) + "</t></is></c>"


def column_name(index):
    name = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def text_width(value):
    return max((sum(0 if combining(ch) else 2 if east_asian_width(ch) in 'WF' else 1
                    for ch in line) for line in str(value).splitlines()), default=0)


def _write_sheet(out, body, name, header, first_width):
    out.write(b'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
    out.write(b'<sheetViews><sheetView workbookViewId="0" showGridLines="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="20"/><cols>')
    for column, title in enumerate(header, 1):
        width = min(40, max(18, len(str(title)) * 0.85))
        if name in {"kv_capacity", "KV容量"}:
            width = (28, 34, 14, 20, 22, 20, 24, 56, 52)[column - 1]
        elif title == "P99样本说明":
            width = 44
        if column == 1:
            width = first_width
        out.write(f'<col min="{column}" max="{column}" width="{width}" customWidth="1"/>'.encode())
    out.write(b'</cols><sheetData>')
    copyfileobj(body, out, 1024 * 1024)
    out.write(b"</sheetData></worksheet>")


def write_xlsx(path, specs):
    # OOXML inline strings support streaming without a library or a shared-string table.
    sheets = []
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for name, header, factory in specs:
            rows = iter(factory())
            part = 1
            while True:
                sheet_name = name if part == 1 else f"{name}_{part}"
                index = len(sheets) + 1
                sheets.append(sheet_name)
                # Spool the current sheet body, not decoded records. This keeps
                # exact Unicode widths with one SQL/JSON traversal and bounded RAM.
                with TemporaryFile() as out:
                    first_width = min(255, max(18, text_width(header[0]) + 3))
                    row_number = 0
                    columns = [column_name(i) for i in range(1, len(header) + 1)]

                    def write_row(values, *, heading=False):
                        nonlocal row_number, first_width
                        row_number += 1
                        if values and values[0] is not None:
                            first_width = min(255, max(first_width, text_width(values[0]) + 3))
                        wrapped = not heading and (name in {"kv_capacity", "KV容量"} or "P99样本说明" in header)
                        opening = '<row r="' + str(row_number) + ('" ht="36" customHeight="1">' if heading else
                                   '" ht="48" customHeight="1">' if wrapped else '">')
                        cells = (cell(v, header=heading, reference=columns[i]+str(row_number)) for i, v in enumerate(values))
                        if wrapped:
                            cells = (c.replace(' t="inlineStr"', ' s="3" t="inlineStr"') if i else c
                                     for i,c in enumerate(cells))
                        out.write((opening + "".join(cells) + "</row>").encode())

                    write_row(header, heading=True)
                    for _ in range(1_048_575):
                        row = next(rows, None)
                        if row is None:
                            break
                        write_row(row)
                    out.seek(0)
                    with z.open(f"xl/worksheets/sheet{index}.xml", "w") as sheet:
                        _write_sheet(sheet, out, name, header, first_width)
                if row is None:
                    break
                part += 1
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView activeTab="0"/></bookViews><sheets>'
            + "".join(
                f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(sheets, 1)
            )
            + "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + '<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        z.writestr(
            "xl/styles.xml",
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="General&quot;*&quot;"/></numFmts>'
            '<fonts count="2"><font><sz val="10"/><name val="Arial"/></font><font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF243746"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>',
        )


def compact_summary(spec):
    name, header, factory = spec
    rows = tuple(factory())
    # Only the small per-Trial summary is materialized; detail tables stay streamed.
    columns = tuple(
        i for i in range(len(header))
        if header[i] == "case" or header[i] in REVENUE_METRICS
        or not rows or any(row[i] is not None and row[i] != "" for row in rows)
    )
    return (
        name,
        tuple(header[i] for i in columns),
        lambda: (tuple(row[i] for i in columns) for row in rows),
    )


def scale_maxtpm(spec):
    """Report maxTPM in millions of tokens/minute; native metrics stay unchanged."""
    name, header, factory = spec
    if 'maxtpm' in header and 'maxtpm_unit' in header:
        value_index, unit_index = header.index('maxtpm'), header.index('maxtpm_unit')
        labels = header
    elif 'maxtpm (tok/min)' in header:
        value_index, unit_index = header.index('maxtpm (tok/min)'), None
        labels = tuple('maxtpm (百万token/M)' if key == 'maxtpm (tok/min)' else key for key in header)
    else:
        return spec

    def rows():
        for source in factory():
            row = list(source)
            unit = row[unit_index] if unit_index is not None else 'tok/min'
            if unit == 'tok/min':
                value = row[value_index]
                if value is not None:
                    row[value_index] = (PartialValue(value / 1_000_000)
                                        if isinstance(value, PartialValue) else value / 1_000_000)
                if unit_index is not None:
                    row[unit_index] = '百万token/M'
            yield tuple(row)

    return name, labels, rows


def units_in_headers(spec):
    """Fold constant per-metric units into numeric headings without changing values.

    Factories stay repeatable and streamed. The heterogeneous long metrics table
    keeps its row-specific unit column; CSV retains its existing machine schema.
    """
    name, header, factory = spec
    unit_columns = [i for i, label in enumerate(header) if label.endswith('_unit')]
    fixed = {'duration_s': 's', 'start_ns': 'ns', 'end_ns': 'ns',
             'response_completed_ns': 'ns', 'input_tokens': 'tokens', 'output_tokens': 'tokens',
             'cached_tokens': 'tokens', 'uncached_input_tokens': 'tokens', 'cache_hit_rate': 'ratio'}
    fixed.update({'maxtpm_'+name+'_'+stat:'samples' for name in MAXTPM_DISTRIBUTIONS
                  for stat in ('n','eligible_n')})
    if not unit_columns:
        return name, tuple(label + ' (' + fixed[label] + ')' if label in fixed else label
                           for label in header), factory
    units = {i: set() for i in unit_columns}
    populated = [False] * len(header)
    for row in factory():
        for i, value in enumerate(row):
            populated[i] |= value is not None and value != ''
        for i in unit_columns:
            if row[i] is not None and row[i] != '':
                units[i].add(str(row[i]))
    labels, drop = list(header), set()
    for i in unit_columns:
        prefix = header[i][:-5]
        names = {prefix, *(prefix + '_' + label for label, _ in OVERALL_STATISTICS)}
        related = [j for j, label in enumerate(header) if label in names]
        numeric = [j for j in related if populated[j]]
        if not numeric:
            drop.add(i)
        elif len(units[i]) == 1:
            unit, = units[i]
            for j in related:
                labels[j] += ' (' + unit + ')'
            drop.add(i)
    for i, label in enumerate(header):
        if label in fixed and labels[i] == label:
            labels[i] += ' (' + fixed[label] + ')'
    columns = tuple(i for i in range(len(header)) if i not in drop)
    return name, tuple(labels[i] for i in columns), lambda: (
        tuple(row[i] for i in columns) for row in factory())


def render_tables(run_id, reader, directory, formats, *, excel_path=None, table_names=DEFAULT_TABLES):
    if reader.summary(run_id)["status"] not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("tabular export requires a terminal Run")
    directory = Path(directory)
    if directory == Path(reader.location).resolve():
        raise ValueError("tabular directory must not replace the result database")
    directory.mkdir(parents=True, exist_ok=True)
    specs = tuple(
        scale_maxtpm(compact_summary(spec) if spec[0] == "summary" else spec)
        for spec in tables(run_id, reader) if spec[0] in table_names
    )
    if not specs:
        raise ValueError("selected tables have no available data")
    filenames = ([name + ".csv" for name, _, _ in specs] if "csv" in formats else []) + (
        ["results.xlsx"] if "xlsx" in formats else []
    )
    targets = [Path(excel_path) if name=='results.xlsx' and excel_path else directory / name for name in filenames]
    db = str(Path(reader.location).resolve())
    if any(str(p.resolve()) in {db, db + "-wal", db + "-shm"} for p in targets):
        raise ValueError("tabular output must not replace the result database")
    with TemporaryDirectory(prefix=".tabular-", dir=directory) as tmp:
        staging = Path(tmp)
        if "csv" in formats:
            for name, header, factory in specs:
                with (staging / (name + ".csv")).open(
                    "w", encoding="utf-8-sig", newline=""
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    for row in factory():
                        writer.writerow([csv_value(v) for v in row])
        if "xlsx" in formats:
            write_xlsx(staging / "results.xlsx", tuple(units_in_headers(spec) for spec in specs))
        for name,target in zip(filenames,targets):
            target.parent.mkdir(parents=True,exist_ok=True)
            if name=='results.xlsx' and excel_path:
                from perf_benchmark.plugins.exporter.tabular.append import publish_workbook
                publish_workbook(staging/name, target)
            else:
                (staging/name).replace(target)
    return targets
