import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, text, useRefresh, type Row, type Settings } from "../api";
import { ErrorMessage, Json, Panel, Table } from "../components";

export default function Control({ settings }: { settings?: Settings }) {
  const client = useQueryClient();
  const [config, setConfig] = useState("");
  const enabled = !!settings?.control_enabled;
  const configs = useQuery({
    queryKey: ["api", "/api/control/configs"],
    queryFn: ({ signal }) =>
      api<{ configs: string[] }>("/api/control/configs", signal),
    enabled,
  });
  const jobs = useQuery({
    queryKey: ["api", "/api/control/jobs"],
    queryFn: ({ signal }) => api<{ jobs: Row[] }>("/api/control/jobs", signal),
    enabled,
  });
  const action = useMutation({
    mutationFn: ({ endpoint, body }: { endpoint: string; body: unknown }) =>
      api<Record<string, unknown>>("/api/control/" + endpoint, undefined, body),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["api", "/api/control/jobs"] }),
  });
  useRefresh(enabled, "control");
  return (
    <>
      <header className="page-head">
        <div>
          <div className="eyebrow">开始一次性能测试</div>
          <h1>启动测试</h1>
        </div>
      </header>
      <div className="notice">
        测试启动后，关闭网页仍会继续运行。取消测试后，已经产生的结果仍可查看。
      </div>
      {!enabled ? (
        <Panel title="当前工作台用于查看结果">
          <p>
            当前部署没有开放网页启动或取消测试，仍可查看测试结果、请求明细和结果比较。
          </p>
          <p className="muted">
            如需从网页启动测试，请联系管理员准备测试配置并开启网页启动功能。
          </p>
          <Link to="/runs">浏览测试结果</Link>
        </Panel>
      ) : (
        <>
          <ErrorMessage error={configs.error} />
          <ErrorMessage error={jobs.error} />
          <ErrorMessage error={action.error} />
          {action.data && (
            <div role="status" className="notice">
              {action.variables?.endpoint === "cancel"
                ? "取消请求已处理，请查看下方测试状态。"
                : "启动请求已处理，请查看下方测试列表。"}
              <details>
                <summary>查看操作详情</summary>
                <Json value={action.data} />
              </details>
            </div>
          )}
          <Panel title="选择测试配置">
            <p className="muted">
              配置由管理员提前准备，例如模型地址、请求数量和并发设置。
            </p>
            {configs.data?.configs.length === 0 && (
              <p className="notice">暂无可用配置，请联系管理员添加测试配置。</p>
            )}
            <form
              className="toolbar"
              onSubmit={(e) => {
                e.preventDefault();
                action.mutate({ endpoint: "start", body: { config } });
              }}
            >
              <label>
                测试配置
                <select
                  required
                  value={config}
                  onChange={(e) => setConfig(e.target.value)}
                >
                  <option value="">请选择测试配置</option>
                  {configs.data?.configs.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <button className="primary" disabled={action.isPending}>
                启动测试
              </button>
            </form>
          </Panel>
          <Panel title="已启动的测试">
            <p className="muted">
              这里显示当前工作台启动的测试。结果开始保存后会出现“查看测试结果”。测试是否成功、是否已取消，请以结果页为准。
            </p>
            {jobs.data &&
              (jobs.data.jobs.length === 0 ? (
                <div className="empty">还没有从当前页面启动的测试。</div>
              ) : (
                <Table
                  rows={jobs.data.jobs}
                  columns={["config", "process_state"]}
                  action={(r) => (
                    <div className="toolbar">
                      {typeof r.run_id === "string" && (
                        <Link
                          to={`/runs/${encodeURIComponent(text(r.run_id))}`}
                        >
                          查看测试结果
                        </Link>
                      )}
                      {r.process_state === "running" && (
                        <button
                          disabled={action.isPending}
                          onClick={() => {
                            action.mutate({
                              endpoint: "cancel",
                              body: { id: r.id },
                            });
                          }}
                        >
                          取消测试
                        </button>
                      )}
                    </div>
                  )}
                />
              ))}
            <details>
              <summary>查看启动记录与运行详情</summary>
              <p className="muted">
                退出码用于排查启动程序，不能代替正式测试结果。关闭控制服务会取消它启动的测试并等待结果保存；服务重启后不会重新接管之前的进程。
              </p>
              <Json value={jobs.data?.jobs ?? []} />
            </details>
          </Panel>
        </>
      )}
    </>
  );
}
