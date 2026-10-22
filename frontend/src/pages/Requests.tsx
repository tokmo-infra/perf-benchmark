import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  grafanaLink,
  numeric,
  path,
  text,
  terminal,
  usePageCache,
  useRead,
  useRefresh,
  type Row,
  type Settings,
} from "../api";
import {
  ErrorMessage,
  Json,
  Pager,
  Panel,
  RunTitle,
  Table,
  Updated,
} from "../components";
import DiagnosticFrame from "../DiagnosticFrame";
import { fieldLabel, stateLabel } from "../labels";

type TraceSelection = {
  relative: string;
  label: string;
  trial?: string;
  from?: string;
  to?: string;
};

function Detail({
  id,
  turn,
  params,
  update,
  settings,
  trace,
  changeTrace,
}: {
  id: string;
  turn: string;
  params: URLSearchParams;
  update: (key: string, value: string) => void;
  settings?: Settings;
  trace: TraceSelection | null;
  changeTrace: (trace: TraceSelection | null) => void;
}) {
  const offset = Math.max(0, Number(params.get("attempt_offset")) || 0);
  const detail = useRead(path(id, "turn", { turn_id: turn }));
  const attemptsPath = path(id, "attempts", {
    turn_id: turn,
    offset,
    page_size: 50,
  });
  const attempts = useRead(attemptsPath, !trace);
  usePageCache(`/api/runs/${encodeURIComponent(id)}`, "attempts", attemptsPath);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const evidence = useRead(
    path(id, "server-evidence", { turn_id: turn }),
    evidenceOpen && !trace,
  );
  const row = detail.data?.items?.[0];
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const link = grafanaLink(
    settings?.grafana_url,
    row?.trace_url,
    timeZone,
    params,
  );
  const traceParams = new URLSearchParams(params);
  if (trace?.trial) traceParams.set("trial", trace.trial);
  if (trace?.from && !traceParams.has("from"))
    traceParams.set("from", trace.from);
  if (trace?.to && !traceParams.has("to")) traceParams.set("to", trace.to);
  const selectedUrl = grafanaLink(
    settings?.grafana_url,
    trace?.relative,
    timeZone,
    traceParams,
  );
  const selectedLink = selectedUrl ? new URL(selectedUrl) : undefined;
  selectedLink?.searchParams.set("var-run", id);
  const selectTrace = (item: Row, label: string) => {
    if (typeof item.trace_url === "string")
      changeTrace({
        relative: item.trace_url,
        label,
        trial: typeof item.trial_id === "string" ? item.trial_id : undefined,
        from:
          typeof item.start_ms === "number" ? String(item.start_ms) : undefined,
        to: typeof item.end_ms === "number" ? String(item.end_ms) : undefined,
      });
  };
  return (
    <Panel
      title="请求详情"
      aside={
        <button onClick={() => update("selected_turn", "")}>关闭详情</button>
      }
    >
      <ErrorMessage error={detail.error} />
      <ErrorMessage error={attempts.error} />
      {trace && row ? (
        <>
          <div className="toolbar">
            <button onClick={() => changeTrace(null)}>返回请求详情</button>
            <span>{trace.label} · 请求链路</span>
          </div>
          <DiagnosticFrame
            url={selectedLink?.href}
            title={`${trace.label} · 请求链路`}
            unavailable={
              settings?.grafana_url
                ? "没有可用的请求链路链接。"
                : "尚未配置监控服务地址，暂时无法查看请求链路。"
            }
          />
        </>
      ) : (
        <>
          {row && (
            <>
              <div className="toolbar">
                <span>数据来源记录 {text(row.source_record)}</span>
                {link ? (
                  <button onClick={() => selectTrace(row, "当前请求")}>
                    查看请求链路
                  </button>
                ) : (
                  <span className="muted">
                    {settings?.grafana_url
                      ? "此请求没有可用的链路数据"
                      : "尚未配置监控服务地址"}
                  </span>
                )}
              </div>
              <Table
                rows={detail.data?.metrics ?? []}
                columns={[
                  "metric",
                  "statistic",
                  "value",
                  "unit",
                  "availability",
                  "reason",
                ]}
              />
              <details>
                <summary>查看请求信息与数据来源（高级）</summary>
                <p className="muted">
                  这里保留请求标识、数据来源和指标计算规则，用于精确筛选与核对。
                  数据来源记录：
                  {text(row.source_record)}。
                </p>
                <button
                  onClick={() => void navigator.clipboard.writeText(turn)}
                >
                  复制请求 ID
                </button>
                <Json value={{ request: row, metrics: detail.data?.metrics }} />
              </details>
            </>
          )}
          <h3>全部执行记录</h3>
          <p className="muted">
            重试时，一个请求会有多条执行记录。重试等待时间可能依赖上一页的记录，
            缺少信息时会显示对应状态。首个输出耗时是客户端观测值，不等同于服务端处理输入的耗时。
          </p>
          <Table
            rows={attempts.data?.items ?? []}
            columns={[
              "attempt",
              "status",
              "e2e",
              "e2e_unit",
              "ttft",
              "ttft_unit",
              "tpot",
              "retry_wait_ms",
              "retry_wait_state",
              "context_page_state",
            ]}
            label="执行记录"
            action={(item) =>
              grafanaLink(
                settings?.grafana_url,
                item.trace_url,
                timeZone,
                params,
              ) ? (
                <button
                  onClick={() =>
                    selectTrace(item, `第 ${text(item.attempt)} 条执行记录`)
                  }
                >
                  查看本次执行链路
                </button>
              ) : (
                <span className="muted">
                  {settings?.grafana_url
                    ? "此执行记录没有可用的链路数据"
                    : "尚未配置监控服务地址"}
                </span>
              )
            }
          />
          <Pager
            offset={offset}
            size={50}
            total={attempts.data?.total ?? 0}
            onPage={(n) => update("attempt_offset", String(n))}
          />
          <details>
            <summary>查看精确计时与执行指标</summary>
            <Json
              value={{
                timings: attempts.data?.timings,
                metrics: attempts.data?.metrics,
              }}
            />
          </details>
          <details
            open={evidenceOpen}
            onToggle={(e) => setEvidenceOpen(e.currentTarget.open)}
          >
            <summary>查看服务端链路数据（高级）</summary>
            <p className="muted">
              这里核对是否采集到服务端的处理过程；客户端链路不能单独证明服务端耗时。
            </p>
            {evidenceOpen && (
              <>
                <ErrorMessage error={evidence.error} />
                <Json value={evidence.data} />
              </>
            )}
          </details>
          <Updated time={attempts.dataUpdatedAt} stale={!!attempts.error} />
        </>
      )}
    </Panel>
  );
}

export default function Requests({ settings }: { settings?: Settings }) {
  const id = useParams().runId!;
  const [params, setParams] = useSearchParams();
  const [traceSelection, setTraceSelection] = useState<{
    scope: string;
    trace: TraceSelection;
  } | null>(null);
  const trace =
    traceSelection?.scope === params.toString() ? traceSelection.trace : null;
  const query = Object.fromEntries(params);
  const offset = Math.max(0, Number(query.offset) || 0);
  const size = Number(query.page_size) || 50;
  const run = useRead(`/api/runs/${encodeURIComponent(id)}`);
  const trials = useRead(path(id, "trials"), !trace);
  const pagePath = path(id, "requests", {
    trial: query.trial,
    status: query.status,
    turn_id: query.turn_id,
    session_id: query.session_id,
    from: query.from,
    to: query.to,
    offset: query.offset ?? 0,
    page_size: query.page_size ?? 50,
  });
  const page = useRead(pagePath, !trace);
  usePageCache(`/api/runs/${encodeURIComponent(id)}`, "requests", pagePath);
  useRefresh(
    !run.data?.items?.[0] || !terminal(run.data.items[0].status),
    id,
    !trace,
  );
  const change = (key: string, value: string) => {
    setTraceSelection(null);
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (!["offset", "selected_turn", "attempt_offset"].includes(key))
      next.set("offset", "0");
    if (key === "selected_turn") next.set("attempt_offset", "0");
    setParams(next);
  };
  const total = numeric(page.data?.total) ?? 0;
  return (
    <>
      <RunTitle id={id} title="请求与执行记录" />
      <Link to={`/runs/${encodeURIComponent(id)}?${params}`}>
        ← 返回测试结果（保留筛选）
      </Link>
      <div className="notice">
        查看在所选时间段内执行过的请求，点击“详情”可查看每次执行和重试。
        列表与总数同时读取；测试运行中，翻页可能看到新写入的记录。
        请求明细从测试结束起保留七天。
      </div>
      <div className="toolbar">
        <label>
          测试轮次
          <select
            value={query.trial || ""}
            onChange={(e) => change("trial", e.target.value)}
          >
            <option value="">全部</option>
            {trials.data?.items?.map((r) => (
              <option key={text(r.trial_id)} value={text(r.trial_id)}>
                {text(r.name)}
              </option>
            ))}
          </select>
        </label>
        <label>
          状态
          <select
            value={query.status || ""}
            onChange={(e) => change("status", e.target.value)}
          >
            <option value="">全部</option>
            {["succeeded", "failed", "cancelled"].map((s) => (
              <option key={s} value={s}>
                {stateLabel(s)}
              </option>
            ))}
          </select>
        </label>
        {["from", "to"].map((key) => (
          <label key={key}>
            {key === "from" ? "开始时间" : "结束时间"}
            <input
              value={query[key] || ""}
              onChange={(e) => change(key, e.target.value)}
              placeholder={
                key === "to" ? "留空使用测试结束时间" : "留空使用测试开始时间"
              }
            />
          </label>
        ))}
        <label>
          每页请求数
          <select
            value={size}
            onChange={(e) => change("page_size", e.target.value)}
          >
            {[50, 100, 250, 1000].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <button onClick={() => void page.refetch()}>刷新本页</button>
      </div>
      <p className="muted">
        留空查看整个测试过程。手动填写时间时使用毫秒时间戳；结束时间可填
        now，表示当前时间。筛选时间不会截短请求本身的耗时。
      </p>
      <details>
        <summary>按标识或列表位置筛选（高级）</summary>
        <div className="toolbar">
          {["session_id", "turn_id"].map((key) => (
            <label key={key}>
              {fieldLabel(key)}
              <input
                value={query[key] || ""}
                onChange={(e) => change(key, e.target.value)}
                placeholder="粘贴完整标识"
              />
            </label>
          ))}
          <label>
            跳过前几条请求
            <input
              type="number"
              min="0"
              value={query.offset ?? "0"}
              onChange={(e) => change("offset", e.target.value)}
            />
          </label>
        </div>
      </details>
      <p className="muted">
        一个测试任务可以包含多个轮次。选择轮次后，可查看该组负载下的请求和重试记录。
      </p>
      <ErrorMessage error={run.error} />
      <ErrorMessage error={trials.error} />
      <ErrorMessage error={page.error} />
      {!trace && (
        <Panel
          title="已记录的请求"
          aside={<Updated time={page.dataUpdatedAt} stale={!!page.error} />}
        >
          <Table
            rows={(page.data?.items ?? []).map((item) => ({
              ...item,
              trial_name:
                trials.data?.items?.find(
                  (trial) => trial.trial_id === item.trial_id,
                )?.name ?? item.trial_id,
            }))}
            columns={[
              "trial_name",
              "status",
              "source_record",
              "input_tokens",
              "output_tokens",
              "e2e",
              "e2e_unit",
              "ttft",
              "tpot",
            ]}
            label="请求"
            action={(row) => (
              <button
                onClick={() => change("selected_turn", text(row.turn_id))}
              >
                详情
              </button>
            )}
          />
          <Pager
            offset={offset}
            size={size}
            total={total}
            onPage={(n) => change("offset", String(n))}
          />
        </Panel>
      )}
      {query.selected_turn && page.data && (
        <Detail
          key={`${id}-${query.selected_turn}`}
          id={id}
          turn={query.selected_turn}
          params={params}
          update={change}
          settings={settings}
          trace={trace}
          changeTrace={(next) =>
            setTraceSelection(
              next ? { scope: params.toString(), trace: next } : null,
            )
          }
        />
      )}
    </>
  );
}
