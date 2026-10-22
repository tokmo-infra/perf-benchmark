import { Link, useSearchParams } from "react-router-dom";
import { text, useRead, useRefresh } from "../api";
import { ErrorMessage, Pager, Panel, State, Updated } from "../components";
import { stateLabel } from "../labels";

export default function Runs() {
  const [params, setParams] = useSearchParams();
  const page = Math.max(1, Number(params.get("page")) || 1);
  const result = useRead("/api/runs?" + params);
  useRefresh(true, "runs");
  const filter = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    next.set(key, value);
    next.set("page", "1");
    setParams(next);
  };
  return (
    <>
      <header className="page-head">
        <div>
          <div className="eyebrow">测试记录</div>
          <h1>测试记录</h1>
          <p className="muted">
            选择一次测试，查看性能指标和请求明细。运行中的测试会持续更新已保存的结果。
          </p>
        </div>
        <Updated time={result.dataUpdatedAt} stale={!!result.error} />
      </header>
      <div className="toolbar">
        <label>
          搜索
          <input
            value={params.get("search") ?? ""}
            placeholder="测试名称或任务 ID"
            onChange={(e) => filter("search", e.target.value)}
          />
        </label>
        <label>
          状态
          <select
            value={params.get("status") ?? ""}
            onChange={(e) => filter("status", e.target.value)}
          >
            <option value="">全部</option>
            {[
              "created",
              "preparing",
              "running",
              "succeeded",
              "failed",
              "cancelled",
            ].map((s) => (
              <option key={s} value={s}>
                {stateLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <button onClick={() => void result.refetch()}>刷新</button>
        <Link className="button" to="/compare">
          对比两次测试 ↗
        </Link>
      </div>
      <ErrorMessage error={result.error} />
      {!!result.data?.errors?.length && (
        <div className="notice">部分结果未读取：{text(result.data.errors)}</div>
      )}
      {!!result.data?.duplicate_run_ids?.length && (
        <div role="alert" className="notice error">
          发现重复的测试任务 ID，无法区分结果：
          {result.data.duplicate_run_ids.join(", ")}
        </div>
      )}
      <Panel
        title="全部测试"
        aside={<span className="muted">汇总长期保留 · 请求明细保留七天</span>}
      >
        <div className="table-scroll">
          <table aria-label="测试记录">
            <thead>
              <tr>
                <th>测试名称</th>
                <th>状态</th>
                <th>开始时间</th>
                <th>请求数：成功 / 失败 / 取消</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {result.data?.items?.map((row) => (
                <tr key={text(row.id)}>
                  <td>
                    <Link
                      className="run-name"
                      to={`/runs/${encodeURIComponent(text(row.id))}`}
                    >
                      {text(row.name)}
                    </Link>
                    <div className="identity">{text(row.id)}</div>
                  </td>
                  <td>
                    <State value={row.status} />
                  </td>
                  <td>
                    {typeof row.start_ms === "number"
                      ? new Date(row.start_ms).toLocaleString()
                      : "未记录"}
                  </td>
                  <td>
                    {text(row.succeeded_turns)} / {text(row.failed_turns)} /{" "}
                    {text(row.cancelled_turns)}
                  </td>
                  <td>
                    <Link
                      to={`/runs/${encodeURIComponent(text(row.id))}/requests`}
                    >
                      请求明细 ↗
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!result.data?.items?.length && (
          <div className="empty">
            {result.isFetching
              ? "正在加载测试记录…"
              : "暂无符合条件的测试。可调整筛选条件；首次使用请在“启动测试”中查看启动方式。"}
          </div>
        )}
        <Pager
          offset={(page - 1) * 50}
          size={50}
          total={result.data?.total ?? 0}
          onPage={(n) => {
            const next = new URLSearchParams(params);
            next.set("page", String(n / 50 + 1));
            setParams(next);
          }}
        />
      </Panel>
    </>
  );
}
