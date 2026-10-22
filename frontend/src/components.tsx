import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { Component } from "react";
import { ApiError, numeric, text, type Row } from "./api";
import {
  conditionLabel,
  fieldLabel,
  metricLabel,
  reasonLabel,
  stateLabel,
  statisticLabel,
  windowLabel,
} from "./labels";

export function ErrorMessage({ error }: { error: Error | null }) {
  if (!error) return null;
  const status = error instanceof ApiError ? error.status : undefined;
  const message =
    status === 404
      ? "未找到这份测试结果，请返回测试列表确认。"
      : status === 409
        ? "测试结果的身份有冲突，请检查结果目录中的重复记录。"
        : status === 410
          ? "请求明细已超过七天保留期，汇总结果仍可查看。"
          : status === 503
            ? "查询服务暂时繁忙，请稍后重试。"
            : status === 400
              ? "筛选条件无效，请检查输入。"
              : "暂时无法显示数据，请稍后刷新或检查结果服务。";
  return (
    <div role="alert" className="notice error">
      <strong>{message}</strong>
      <details>
        <summary>查看错误详情</summary>
        {status && <span>HTTP {status} · </span>}
        {error.message}
      </details>
    </div>
  );
}
export function State({ value }: { value: unknown }) {
  return (
    <span className={"state state-" + text(value)} title={text(value)}>
      {stateLabel(value)}
    </span>
  );
}
export function Value({ row }: { row: Row }) {
  const available = row.availability === "available" && row.value != null;
  const number = numeric(row.value);
  return (
    <span
      title={`${text(row.value_text ?? row.value)} ${text(row.unit)} · ${text(row.source)} · ${text(row.formula_version)}`}
    >
      {available ? (
        <>
          <strong>
            {number == null
              ? text(row.value_text ?? row.value)
              : number.toLocaleString(undefined, {
                  maximumSignificantDigits: 6,
                })}
          </strong>{" "}
          <small>{text(row.unit)}</small>
        </>
      ) : (
        <span className="muted">暂无数值 · {reasonLabel(row.reason)}</span>
      )}
    </span>
  );
}
export function Table({
  rows,
  columns,
  action,
  label = "数据",
}: {
  rows: Row[];
  columns: string[];
  action?: (row: Row) => ReactNode;
  label?: string;
}) {
  if (!rows.length)
    return <div className="empty">暂无记录，可调整筛选条件后重试。</div>;
  return (
    <div className="table-scroll">
      <table aria-label={label}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} title={c}>
                {fieldLabel(c)}
              </th>
            ))}
            {action && <th>操作</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={text(row.id ?? row.turn_id ?? row.trial_id ?? i) + "-" + i}
            >
              {columns.map((c) => (
                <td key={c} title={text(row[c])}>
                  {c === "status" || c === "state" || c === "process_state" ? (
                    <State value={row[c]} />
                  ) : c === "condition" ? (
                    conditionLabel(text(row[c]))
                  ) : c === "metric" ? (
                    metricLabel(text(row[c]))
                  ) : c === "statistic" ? (
                    statisticLabel(text(row[c]))
                  ) : c === "window" ? (
                    windowLabel(text(row[c]))
                  ) : c === "decision" ||
                    c === "availability" ||
                    c === "retry_wait_state" ||
                    c === "context_page_state" ? (
                    stateLabel(row[c])
                  ) : (c === "reason" || c === "unavailable_reason") &&
                    row[c] == null ? (
                    "—"
                  ) : c === "reason" || c === "unavailable_reason" ? (
                    reasonLabel(row[c])
                  ) : (c === "relative_delta" ||
                      c === "max_regression_ratio") &&
                    typeof row[c] === "number" ? (
                    numeric(row[c])!.toLocaleString(undefined, {
                      style: "percent",
                      maximumSignificantDigits: 6,
                    })
                  ) : typeof row[c] === "boolean" ? (
                    row[c] ? (
                      "是"
                    ) : (
                      "否"
                    )
                  ) : (
                    text(row[c])
                  )}
                </td>
              ))}
              {action && <td>{action(row)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Pager({
  offset,
  size,
  total,
  onPage,
}: {
  offset: number;
  size: number;
  total: number;
  onPage: (n: number) => void;
}) {
  const end = Math.min(offset + size, total);
  return (
    <div className="pager">
      <span>
        {total === 0 || offset >= total ? "0" : `${offset + 1}–${end}`} /{" "}
        <strong>{total.toLocaleString()}</strong> 条记录
      </span>
      <div>
        <button
          disabled={offset === 0}
          onClick={() => onPage(Math.max(0, offset - size))}
        >
          上一页
        </button>
        <button disabled={end >= total} onClick={() => onPage(offset + size)}>
          下一页
        </button>
        {offset >= total && total > 0 && (
          <button onClick={() => onPage(Math.floor((total - 1) / size) * size)}>
            返回最后一页
          </button>
        )}
      </div>
    </div>
  );
}
export function Panel({
  title,
  children,
  aside,
}: {
  title: string;
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}
export function Json({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(value, null, 2) ?? "未记录"}</pre>;
}
export function Updated({ time, stale }: { time: number; stale?: boolean }) {
  return (
    <span className="muted">
      {time ? `更新于 ${new Date(time).toLocaleTimeString()}` : "正在加载数据"}
      {stale ? " · 更新失败，当前显示上次数据" : ""}
    </span>
  );
}
export function RunTitle({ id, title }: { id: string; title: string }) {
  return (
    <header className="page-head">
      <div>
        <Link className="eyebrow" to="/runs">
          返回测试列表
        </Link>
        <h1>{title}</h1>
        <div className="identity">测试任务编号：{id}</div>
        <button onClick={() => void navigator.clipboard.writeText(id)}>
          复制任务编号
        </button>
      </div>
      <Link className="button" to={`/runs/${encodeURIComponent(id)}/requests`}>
        查看请求明细 ↗
      </Link>
    </header>
  );
}

export class ChartBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null } as { error: Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    return this.state.error ? (
      <ErrorMessage error={this.state.error} />
    ) : (
      this.props.children
    );
  }
}
