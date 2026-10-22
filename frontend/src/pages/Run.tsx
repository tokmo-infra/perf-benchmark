import { lazy, Suspense, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  grafanaLink,
  path,
  text,
  terminal,
  useRead,
  useRefresh,
  type Settings,
} from "../api";
import {
  ChartBoundary,
  ErrorMessage,
  Json,
  Panel,
  RunTitle,
  State,
  Table,
  Updated,
  Value,
} from "../components";
import DiagnosticFrame from "../DiagnosticFrame";
import {
  metricHelp,
  metricLabel,
  statisticLabel,
  stateLabel,
  windowLabel,
} from "../labels";
const Plot = lazy(() => import("../Plot"));

const metrics = [
  "ttft",
  "tpot",
  "itl",
  "e2e",
  "workload_rps",
  "actual_in_flight_concurrency",
  "input_tokens",
  "output_tokens",
  "prefill_throughput",
  "decode_throughput",
];

function Series({
  id,
  metric,
  params,
  onRange,
  scopeLabels,
}: {
  id: string;
  metric: string;
  params: URLSearchParams;
  onRange: (a: number, b: number) => void;
  scopeLabels: Record<string, string>;
}) {
  const query = Object.fromEntries(params);
  const [pointsOpen, setPointsOpen] = useState(false);
  const result = useRead(
    path(id, "series", {
      trial: query.trial,
      from: query.from,
      to: query.to,
      metric,
      statistics: query.statistics || (metric.endsWith("_throughput") ? "time_wm,p99,value" : "am,p99,value"),
    }),
  );
  return (
    <Panel
      title={metricLabel(metric)}
      aside={<State value={result.data?.state ?? "未取得"} />}
    >
      <ErrorMessage error={result.error} />
      <p className="muted">{metricHelp(metric)}</p>
      <ChartBoundary>
        <Suspense fallback={<div className="empty">加载图表…</div>}>
          <Plot
            rows={result.data?.items ?? []}
            title={metricLabel(metric) + " · " + stateLabel(result.data?.state)}
            identity={`${id}-${query.trial}-${query.from}-${query.to}`}
            onRange={onRange}
            scopeLabels={scopeLabels}
          />
        </Suspense>
      </ChartBoundary>
      <details onToggle={(e) => setPointsOpen(e.currentTarget.open)}>
        <summary>查看数据点与统计说明</summary>
        {pointsOpen && (
          <>
            <Table
              rows={result.data?.items ?? []}
              columns={[
                "series",
                "scope_id",
                "window",
                "value_text",
                "unit",
                "availability",
                "reason",
                "valid_sample_count",
                "expected_sample_count",
                "basis",
                "source",
                "formula_version",
                "start_ns",
                "end_ns",
              ]}
            />
            <Json value={result.data?.summary} />
          </>
        )}
      </details>
      <Updated time={result.dataUpdatedAt} stale={!!result.error} />
    </Panel>
  );
}

function Extra({
  id,
  operation,
  params,
}: {
  id: string;
  operation: string;
  params: URLSearchParams;
}) {
  const result = useRead(path(id, operation, Object.fromEntries(params)));
  return (
    <>
      <ErrorMessage error={result.error} />
      {operation === "stages" ? (
        <Table
          rows={result.data?.items ?? []}
          columns={["stage", "plugin", "status", "duration_s"]}
        />
      ) : (
        <Json value={result.data} />
      )}
      <Updated time={result.dataUpdatedAt} stale={!!result.error} />
    </>
  );
}

function Diagnostic({
  id,
  settings,
  params,
  relative,
  expired,
}: {
  id: string;
  settings: Settings | undefined;
  params: URLSearchParams;
  relative: unknown;
  expired: boolean;
}) {
  const [open, setOpen] = useState(false);
  const url = expired
    ? undefined
    : grafanaLink(
        settings?.grafana_url,
        relative,
        Intl.DateTimeFormat().resolvedOptions().timeZone,
        params,
      );
  return (
    <Panel title="监控与排查">
      <p className="muted">
        查看已采集的服务与 GPU 监控、请求链路、日志和性能分析。此视图由 Grafana
        提供；未配置采集的项目会显示无数据。登录后继续显示当前任务的监控。
      </p>
      <DiagnosticFrame
        url={url}
        title="测试任务监控与排查"
        unavailable={
          expired
            ? "监控数据已超过七天保留期，测试汇总仍可长期查看。"
            : settings?.grafana_url
              ? "此测试任务没有可用的监控链接。"
              : "尚未配置监控服务地址，暂时无法打开此视图。"
        }
      />
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>查看监控采集情况与 GPU 分析记录</summary>
        {open &&
          ["targets", "coverage", "diagnostics"].map((op) => (
            <Extra key={op} id={id} operation={op} params={params} />
          ))}
      </details>
    </Panel>
  );
}

export default function Run({ settings }: { settings?: Settings }) {
  const id = useParams().runId!;
  const [params, setParams] = useSearchParams();
  const [diagnostic, setDiagnostic] = useState(false);
  const run = useRead(`/api/runs/${encodeURIComponent(id)}`);
  const trials = useRead(path(id, "trials"));
  const row = run.data?.items?.[0];
  const scopeLabels = Object.fromEntries([
    [id, "任务汇总"],
    ...(trials.data?.items ?? []).map((trial) => [
      text(trial.trial_id),
      `测试轮次 · ${text(trial.name)}`,
    ]),
  ]);
  const query = Object.fromEntries(params);
  const values = useRead(
    path(id, "metrics", {
      trial: query.trial,
      window: query.window || "overall",
    }),
    !diagnostic,
  );
  const windows = useRead(
    path(id, "windows", { trial: query.trial }),
    !diagnostic,
  );
  const [chosen, setChosen] = useState([
    "ttft",
    "tpot",
    "workload_rps",
    "actual_in_flight_concurrency",
  ]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const refresh = useRefresh(!row || !terminal(row.status), id, !diagnostic);
  const change = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  };
  const range = (a: number, b: number) => {
    const next = new URLSearchParams(params);
    next.set("from", String(a));
    next.set("to", String(b));
    setParams(next);
  };
  const cards = (values.data?.items ?? []).filter(
    (r) =>
      ["ttft", "tpot", "e2e", "workload_rps", "workload_completion"].includes(
        text(r.metric),
      ) && ["p99", "value"].includes(text(r.statistic)),
  );
  const stats = useRead(
    path(id, "statistics", {
      trial: query.trial,
      window: query.window || "overall",
      view: query.view || "distribution",
    }),
    !diagnostic && !!expanded.statistics,
  );
  return (
    <>
      <RunTitle id={id} title={row ? text(row.name) : "测试任务"} />
      <ErrorMessage error={run.error} />
      <div className="view-tabs" role="tablist" aria-label="测试任务视图">
        <button
          role="tab"
          aria-selected={!diagnostic}
          onClick={() => setDiagnostic(false)}
        >
          测试结果
        </button>
        <button
          role="tab"
          aria-selected={diagnostic}
          onClick={() => setDiagnostic(true)}
        >
          监控与排查
        </button>
      </div>
      <div className="status-line">
        <State value={row?.status} />
        {row && (
          <span>
            Trace：{row.observation_enabled === false ? "未启用" : row.trace_mode === "debug" ? "Debug（全采样）" : row.trace_mode === "observe" ? "Observe（采样）" : "Benchmark"}
            {row.trace_mode !== "benchmark" && row.trace_export === false ? " · 导出关闭" : ""}
          </span>
        )}
        <State value={values.data?.state} />
        <Updated time={values.dataUpdatedAt} stale={!!values.error} />
        <button onClick={() => void refresh()}>刷新</button>
      </div>
      <div className="notice" title={text(values.data?.summary?.[0]?.message)}>
        {row?.status === "cancelled" || row?.status === "failed"
          ? "测试已取消或失败，仅保留已保存的结果，不能作为完整测试结论。"
          : row?.status === "succeeded"
            ? "测试已完成。下方显示保存的性能指标和请求结果。"
            : row?.status === "running"
              ? "测试进行中，新结果会随测试进度保存。"
              : "测试结果将在请求完成后显示。"}
      </div>
      <div className="toolbar">
        <label>
          测试轮次
          <select
            value={query.trial || ""}
            onChange={(e) => change("trial", e.target.value)}
          >
            <option value="">全部测试轮次</option>
            {trials.data?.items?.map((t) => (
              <option value={text(t.trial_id)} key={text(t.trial_id)}>
                {text(t.name)}
              </option>
            ))}
          </select>
        </label>
        {!diagnostic && (
          <label>
            统计范围
            <select
              value={query.window || "overall"}
              onChange={(e) => change("window", e.target.value)}
            >
              {[
                ...new Set([
                  "overall",
                  ...(windows.data?.items ?? []).map((w) => text(w.window)),
                ]),
              ].map((w) => (
                <option key={w} value={w}>
                  {windowLabel(w)}
                </option>
              ))}
            </select>
          </label>
        )}
        <Link
          className="button"
          to={`/runs/${encodeURIComponent(id)}/requests?${params}`}
        >
          查看请求列表 ↗
        </Link>
        <Link
          className="button"
          to={`/compare?${new URLSearchParams({ baseline: id, ...(query.trial ? { baseline_trial: query.trial } : {}), ...(query.from ? { from: query.from } : {}), ...(query.to ? { to: query.to } : {}) })}`}
        >
          与其他测试比较 ↗
        </Link>
      </div>
      <ErrorMessage error={values.error} />
      <p className="muted">
        一个测试任务可以包含多个轮次，每个轮次对应一组负载设置或一次重复测试。
      </p>
      <ErrorMessage error={trials.error} />
      <ErrorMessage error={windows.error} />
      {diagnostic ? (
        <>
          <div className="toolbar">
            <label>
              开始时间
              <input
                value={query.from || ""}
                onChange={(e) => change("from", e.target.value)}
                placeholder="留空使用测试开始时间"
              />
            </label>
            <label>
              结束时间
              <input
                value={query.to || ""}
                onChange={(e) => change("to", e.target.value)}
                placeholder="留空使用测试结束时间"
              />
            </label>
            <button
              onClick={() => {
                const next = new URLSearchParams(params);
                next.delete("from");
                next.delete("to");
                setParams(next);
              }}
            >
              查看整个测试过程
            </button>
          </div>
          <p className="muted">
            留空查看整个测试过程。手动填写时使用毫秒时间戳；结束时间也可填
            now，表示当前时间。
          </p>
          <Diagnostic
            id={id}
            settings={settings}
            params={params}
            relative={row?.run_url}
            expired={row?.trend_status === "超出 7 天诊断保留期"}
          />
        </>
      ) : (
        <>
          <p className="muted">
            P99 用于观察尾部耗时，帮助发现少量很慢的请求。
          </p>
          <div className="metric-strip">
            {cards.map((r, i) => (
              <div key={i}>
                <span className="eyebrow">
                  <span title={metricHelp(text(r.metric))}>
                    {metricLabel(text(r.metric))} /{" "}
                    {statisticLabel(text(r.statistic))}
                  </span>
                </span>
                <Value row={r} />
              </div>
            ))}
          </div>

          <Panel
            title="查看测试过程中的变化"
            aside={<span className="muted">每 10 秒汇总一个数据点</span>}
          >
            <div className="toolbar">
              <label>
                开始时间
                <input
                  value={query.from || ""}
                  onChange={(e) => change("from", e.target.value)}
                  placeholder="留空使用测试开始时间"
                />
              </label>
              <label>
                结束时间
                <input
                  value={query.to || ""}
                  onChange={(e) => change("to", e.target.value)}
                  placeholder="留空使用测试结束时间"
                />
              </label>
              <label>
                曲线显示的数值
                <select
                  value={query.statistics || ""}
                  onChange={(e) => change("statistics", e.target.value)}
                >
                  <option value="">主均值、P99、数值（速度采用时间加权）</option>
                  {[
                    "time_wm,p99,value",
                    "am,p99,value",
                    "am,value",
                    "p50,value",
                    "p90,value",
                    "p95,value",
                    "p99,value",
                  ].map((s) => (
                    <option key={s} value={s}>
                      {s.split(",").map(statisticLabel).join("、")}
                    </option>
                  ))}
                </select>
              </label>
              <button
                onClick={() => {
                  const next = new URLSearchParams(params);
                  next.delete("from");
                  next.delete("to");
                  setParams(next);
                }}
              >
                查看整个测试过程
              </button>
            </div>
            <p className="muted">
              可以在图表上拖选时间范围，只改变趋势图显示的数据，不重算下方统计。
              手动填写时间时使用毫秒时间戳；结束时间可填 now，表示当前时间。
            </p>
            <p className="muted">
              TTFT 是首个输出耗时，TPOT 是首个输出之后每个输出 token
              的平均耗时，E2E 是请求总耗时。每次最多查看 6 个指标。
            </p>
            <div className="checks">
              {metrics.map((m) => (
                <label key={m} title={metricHelp(m)}>
                  <input
                    type="checkbox"
                    checked={chosen.includes(m)}
                    disabled={!chosen.includes(m) && chosen.length >= 6}
                    onChange={(e) =>
                      setChosen((old) =>
                        e.target.checked
                          ? [...old, m]
                          : old.filter((x) => x !== m),
                      )
                    }
                  />
                  {metricLabel(m)}
                </label>
              ))}
            </div>
          </Panel>
          <div className="chart-grid">
            {chosen.map((m) => (
              <Series
                key={`${id}-${m}`}
                id={id}
                metric={m}
                params={params}
                onRange={range}
                scopeLabels={scopeLabels}
              />
            ))}
          </div>
          <Panel title="测试指标">
            <Table
              rows={values.data?.items ?? []}
              columns={[
                "metric",
                "statistic",
                "value",
                "unit",
                "availability",
                "reason",
              ]}
              label="测试指标"
            />
          </Panel>
          <Panel title="详细统计与测试信息">
            {["statistics", "evidence", "stages", "groups"].map((op) => (
              <details
                key={op}
                onToggle={(e) => {
                  const open = e.currentTarget.open;
                  setExpanded((old) => ({ ...old, [op]: open }));
                }}
              >
                <summary>
                  {
                    (
                      {
                        statistics: "查看完整统计数据（高级）",
                        evidence: "查看测试条件与结果可信度",
                        stages: "查看各执行阶段的状态与耗时",
                        groups: "按请求特征查看分组结果",
                      } as Record<string, string>
                    )[op]
                  }
                </summary>
                {expanded[op] &&
                  (op === "statistics" ? (
                    <>
                      <label>
                        统计内容
                        <select
                          value={query.view || "distribution"}
                          onChange={(e) => change("view", e.target.value)}
                        >
                          <option value="distribution">常用分布统计</option>
                          <option value="distribution_full">
                            完整分布统计
                          </option>
                          <option value="means">常用平均值</option>
                          <option value="means_all">全部平均值</option>
                          <option value="scalar">单项指标</option>
                          <option value="compare">测试轮次对比</option>
                          <option value="groups">请求特征分组</option>
                          <option value="repetitions">重复测试统计</option>
                        </select>
                      </label>
                      <ErrorMessage error={stats.error} />
                      <Json value={stats.data} />
                      <p className="muted">
                        原始指标记录保留数据来源、计算规则和统计范围，供核对结果。
                      </p>
                      <Json value={values.data?.items ?? []} />
                    </>
                  ) : (
                    <Extra id={id} operation={op} params={params} />
                  ))}
              </details>
            ))}
          </Panel>
        </>
      )}
    </>
  );
}
