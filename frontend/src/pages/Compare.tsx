import { lazy, Suspense } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { api, path, text, useRead, type Row } from "../api";
import { ChartBoundary, ErrorMessage, Json, Panel, Table } from "../components";
import {
  fieldLabel,
  metricHelp,
  metricLabel,
  statisticLabel,
  windowLabel,
} from "../labels";
const Plot = lazy(() => import("../Plot"));
interface Rule {
  metric: string;
  statistic: string;
  window: string;
  direction: string;
  threshold: string;
}
interface Comparison {
  conditions: Row[];
  metrics: Row[];
  load_qualified: boolean[];
  repetitions: Row[];
  quality: Row[];
  source_content: string;
  directly_comparable: boolean;
}

function unitGroups(rows: Row[]) {
  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    if (row.reason != null) continue;
    const key = text(row.unit);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(row);
  }
  return [...groups];
}

export default function Compare() {
  const [params, setParams] = useSearchParams();
  const identity = params.toString();
  const baseline = params.get("baseline") || "",
    candidate = params.get("candidate") || "";
  let rules: Rule[] = [
    {
      metric: "e2e",
      statistic: "p99",
      window: "overall",
      direction: "lower",
      threshold: "",
    },
  ];
  let rulesError: Error | null = null;
  if (params.has("rules")) {
    try {
      const decoded: unknown = JSON.parse(params.get("rules")!);
      if (
        !Array.isArray(decoded) ||
        !decoded.length ||
        decoded.length > 4 ||
        decoded.some(
          (r) =>
            !r ||
            typeof r !== "object" ||
            ["metric", "statistic", "window", "direction", "threshold"].some(
              (k) => typeof r[k] !== "string",
            ) ||
            Object.keys(r).some(
              (k) =>
                ![
                  "metric",
                  "statistic",
                  "window",
                  "direction",
                  "threshold",
                ].includes(k),
            ),
        )
      )
        throw new Error("比较指标设置无法读取");
      rules = decoded as Rule[];
    } catch {
      rulesError = new Error(
        "比较设置无法读取，请点击“添加比较指标”重新设置。",
      );
      rules = [];
    }
  }
  const a = useRead(baseline ? path(baseline, "trials") : null),
    b = useRead(candidate ? path(candidate, "trials") : null);
  const comparison = useMutation({
    mutationFn: async ({
      identity,
      payload,
    }: {
      identity: string;
      payload: unknown;
    }) => ({
      identity,
      value: await api<Comparison>("/api/compare", undefined, payload),
    }),
  });
  const data =
    comparison.data?.identity === identity ? comparison.data.value : undefined;
  const error =
    comparison.variables?.identity === identity ? comparison.error : null;
  const pending =
    comparison.variables?.identity === identity && comparison.isPending;
  const change = (key: string, value: string) => {
    comparison.reset();
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  };
  const edit = (i: number, key: keyof Rule, value: string) => {
    comparison.reset();
    const next = rules.map((r, n) => (n === i ? { ...r, [key]: value } : r));

    const q = new URLSearchParams(params);
    q.set("rules", JSON.stringify(next));
    setParams(q);
  };
  return (
    <>
      <header className="page-head">
        <div>
          <div className="eyebrow">查看性能变化</div>
          <h1>比较测试结果</h1>
          <p className="muted">
            选择两组已有结果，查看性能差异以及是否超过可接受的退化上限。
          </p>
          <p className="muted">
            在测试结果页点击“与其他测试比较”可自动选为基准；另一份结果的任务编号可在其结果页复制。
          </p>
          <p className="muted">
            测试任务包含一个或多个测试轮次，每个轮次对应一组负载设置（例如并发）或一次重复测试。
          </p>
        </div>
      </header>
      <ErrorMessage error={rulesError} />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (rulesError) return;
          comparison.mutate({
            identity,
            payload: {
              baseline,
              candidate,
              specification: {
                allowed_changes: (params.get("allowed_changes") || "")
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean),
                ...(params.get("baseline_trial")
                  ? { baseline_trial: params.get("baseline_trial") }
                  : {}),
                ...(params.get("candidate_trial")
                  ? { candidate_trial: params.get("candidate_trial") }
                  : {}),
                metrics: rules.map(({ threshold, ...r }) => ({
                  ...r,
                  ...(threshold === ""
                    ? {}
                    : { max_regression_ratio: Number(threshold) }),
                })),
              },
            },
          });
        }}
      >
        <div className="compare-sides">
          {[
            ["baseline", "基准结果", a],
            ["candidate", "待比较结果", b],
          ].map(([key, label, result]) => {
            const k = key as string,
              data = (result as typeof a).data;
            return (
              <Panel key={k} title={label as string}>
                <label>
                  测试任务编号
                  <input
                    required
                    value={params.get(k) || ""}
                    onChange={(e) => change(k, e.target.value)}
                  />
                </label>
                <label>
                  测试轮次
                  <select
                    value={params.get(k + "_trial") || ""}
                    onChange={(e) => change(k + "_trial", e.target.value)}
                  >
                    <option value="">整个测试任务的汇总</option>
                    {data?.items?.map((r) => (
                      <option key={text(r.trial_id)} value={text(r.trial_id)}>
                        {text(r.name)}
                      </option>
                    ))}
                  </select>
                </label>
                <ErrorMessage error={(result as typeof a).error} />
              </Panel>
            );
          })}
        </div>
        <Panel title="设置比较方式">
          <label>
            允许发生变化的测试条件
            <input
              value={params.get("allowed_changes") || ""}
              onChange={(e) => change("allowed_changes", e.target.value)}
              placeholder="条件标识，如 model；多个用逗号分隔"
            />
          </label>
          <p className="muted">
            留空时要求两组测试条件保持一致。只允许本次要验证的条件变化；条件标识可在“查看比较依据与原始判定”中查看。
          </p>
          {rules.map((r, i) => (
            <div className="toolbar rule" key={i}>
              {(["metric", "statistic", "window"] as const).map((k) => (
                <label key={k}>
                  {fieldLabel(k)}
                  <input
                    required
                    value={r[k]}
                    onChange={(e) => edit(i, k, e.target.value)}
                    title={k === "metric" ? metricHelp(r.metric) : undefined}
                  />
                  <span className="muted">
                    {k === "metric"
                      ? metricLabel(r.metric)
                      : k === "statistic"
                        ? statisticLabel(r.statistic)
                        : windowLabel(r.window)}
                  </span>
                </label>
              ))}
              <label>
                哪种变化更好
                <select
                  value={r.direction}
                  onChange={(e) => edit(i, "direction", e.target.value)}
                >
                  <option value="lower">越低越好</option>
                  <option value="higher">越高越好</option>
                </select>
              </label>
              <label>
                可接受的退化上限（选填）
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={r.threshold}
                  onChange={(e) => edit(i, "threshold", e.target.value)}
                  placeholder="例如 0.05 表示 5%；留空只看差异"
                />
              </label>
              {rules.length > 1 && (
                <button
                  type="button"
                  onClick={() => {
                    const next = rules.filter((_, n) => n !== i);

                    comparison.reset();
                    const q = new URLSearchParams(params);
                    q.set("rules", JSON.stringify(next));
                    setParams(q);
                  }}
                >
                  移除此指标
                </button>
              )}
            </div>
          ))}
          <p className="muted">
            P99 表示 99%
            的请求不超过该值，适合查看慢请求。退化上限填写比例，例如 0.05 表示
            5%；留空只展示差异。
          </p>
          <div className="toolbar">
            <button
              type="button"
              disabled={rules.length >= 4}
              onClick={() => {
                const next = [
                  ...rules,
                  {
                    metric: "ttft",
                    statistic: "p99",
                    window: "overall",
                    direction: "lower",
                    threshold: "",
                  },
                ];

                const q = new URLSearchParams(params);
                q.set("rules", JSON.stringify(next));
                setParams(q);
                comparison.reset();
              }}
            >
              添加比较指标
            </button>
            <button className="primary" disabled={pending}>
              比较结果
            </button>
          </div>
        </Panel>
      </form>
      <ErrorMessage error={error} />
      {data && (
        <>
          <div className="notice">
            {data.directly_comparable
              ? "测试条件满足比较要求，可以直接比较。"
              : "测试条件尚未满足直接比较要求，请先查看下方的条件差异。"}{" "}
            {data.source_content === "已核实固定来源身份"
              ? "两组测试的数据来源已确认"
              : data.source_content}
            。
          </div>
          <Panel title="测试条件与负载有效性">
            <Table
              rows={data.conditions}
              columns={["condition", "changed", "allowed_change", "verified"]}
            />
            <p>
              基准结果：
              {data.load_qualified[0]
                ? "负载有效性检查通过"
                : "尚未满足负载有效性要求"}
              ；待比较结果：
              {data.load_qualified[1]
                ? "负载有效性检查通过"
                : "尚未满足负载有效性要求"}
              。
            </p>
            <p className="muted">
              所选测试轮次需要执行成功，并具备有效的负载证据。未启用检查、证据不足或执行失败都可能尚未满足要求，此时不能据此确认服务容量或判定是否超过退化上限。
            </p>
            <details>
              <summary>查看比较依据与原始判定</summary>
              <Json value={data} />
            </details>
          </Panel>
          <Panel title="指标差异与结论">
            <p className="muted">
              差异按“待比较结果 −
              基准结果”计算。未设置上限时只报告差异，不判定是否退化。已设置上限时，“退化未超过上限”表示仍在允许范围内，不能据此认定性能已经改善；指标、条件或负载证据不足时，结论会显示“无法确定”。
            </p>
            <Table
              rows={data.metrics}
              columns={[
                "metric",
                "statistic",
                "window",
                "baseline",
                "candidate",
                "delta",
                "relative_delta",
                "unit",
                "decision",
                "reason",
                "max_regression_ratio",
              ]}
            />
            <ChartBoundary>
              <Suspense fallback={<p>加载图表…</p>}>
                {unitGroups(data.metrics).map(([unit, rows]) => (
                  <Plot
                    key={unit}
                    rows={rows}
                    title={"指标变化 · " + unit}
                    identity={baseline + "-" + candidate + "-" + unit}
                    comparison
                  />
                ))}
              </Suspense>
            </ChartBoundary>
          </Panel>
        </>
      )}
    </>
  );
}
