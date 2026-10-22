import createPlotlyComponent from "react-plotly.js/factory";
import PlotlyURL from "plotly.js-basic-dist-min?url";
import { lazy } from "react";
import type Plotly from "plotly.js";
import type { Data, Layout } from "plotly.js";
import { numeric, plotTime, text, type Row } from "./api";
import {
  metricLabel,
  reasonLabel,
  stateLabel,
  statisticLabel,
  windowLabel,
} from "./labels";

// A local, content-hashed vendor asset. Only chart pages load it; no runtime CDN.
const Graph = lazy(async () => {
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = PlotlyURL;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error("图表文件加载失败，请刷新页面重试。"));
    document.head.append(script);
  });
  return {
    default: createPlotlyComponent(
      (window as Window & { Plotly: typeof Plotly }).Plotly,
    ),
  };
});
export default function Plot({
  rows,
  title,
  identity,
  onRange,
  comparison = false,
  scopeLabels,
}: {
  rows: Row[];
  title: string;
  identity: string;
  comparison?: boolean;
  scopeLabels?: Record<string, string>;
  onRange?: (from: number, to: number) => void;
}) {
  const hasValues = comparison
    ? rows.some(
        (row) =>
          numeric(row.baseline) !== null || numeric(row.candidate) !== null,
      )
    : rows.some(
        (row) =>
          row.availability === "available" &&
          numeric(row.time) !== null &&
          numeric(row.value) !== null,
      );
  if (!hasValues) {
    const reasons = [
      ...new Set(
        rows
          .map((row) => row.reason ?? row.unavailable_reason)
          .filter((reason): reason is string => typeof reason === "string"),
      ),
    ];
    return (
      <div className="plot empty" aria-label={title} role="status">
        <p>
          {rows.length
            ? "此指标没有可用数值。"
            : "所选时段内还没有这项指标的数据。"}
        </p>
        {reasons.length > 0 && (
          <p title={reasons.join(" · ")}>
            {reasons.map(reasonLabel).join(" · ")}
          </p>
        )}
      </div>
    );
  }
  const traces: Data[] = [];
  if (comparison) {
    for (const side of ["baseline", "candidate"])
      traces.push({
        type: "bar",
        name: side === "baseline" ? "基准结果" : "待比较结果",
        x: rows.map(
          (r) =>
            `${metricLabel(text(r.metric))} / ${statisticLabel(text(r.statistic))} [${windowLabel(text(r.window))}]`,
        ),
        y: rows.map((r) => numeric(r[side])),
        customdata: rows.map((r) => [
          text(r.unit),
          windowLabel(text(r.window)),
          stateLabel(r.decision),
        ]),
        hovertemplate:
          "%{y} %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra>%{fullData.name}</extra>",
      });
  } else {
    const groups = new Map<string, Row[]>();
    for (const row of rows) {
      const key = text(row.series);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(row);
    }
    for (const [name, points] of groups)
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name:
          points[0]?.statistic != null
            ? `${statisticLabel(text(points[0].statistic))} · ${scopeLabels?.[text(points[0].scope_id)] ?? text(points[0].scope_id)}`
            : name,
        x: points.map((r) => numeric(r.time)),
        y: points.map((r) =>
          r.availability === "available" ? numeric(r.value) : null,
        ),
        connectgaps: false,
        customdata: points.map((r) => [
          text(r.unit),
          text(r.basis),
          text(r.valid_sample_count),
          text(r.expected_sample_count),
          reasonLabel(r.reason ?? r.unavailable_reason),
          stateLabel(r.state),
        ]),
        hovertemplate:
          "%{x|%Y-%m-%d %H:%M:%S.%L} UTC<br>%{y} %{customdata[0]}<br>%{customdata[1]}<br>有效/期望 %{customdata[2]}/%{customdata[3]} · %{customdata[5]}<extra>%{fullData.name}</extra>",
      });
  }
  const layout: Partial<Layout> = {
    autosize: true,
    height: 300,
    uirevision: identity,
    paper_bgcolor: "#fff",
    plot_bgcolor: "#fff",
    font: { family: "system-ui", size: 11, color: "#526158" },
    margin: { l: 60, r: 20, t: 22, b: 65 },
    colorway: ["#287a54", "#6e73ae", "#c78c42", "#4e91ae"],
    xaxis: comparison
      ? {}
      : { type: "date", gridcolor: "#eef1ed", title: { text: "UTC" } },
    yaxis: {
      gridcolor: "#eef1ed",
      zeroline: false,
      title: { text: text(rows[0]?.unit) },
    },
    legend: { orientation: "h", y: -0.23 },
    hovermode: "closest",
    barmode: "group",
  };
  return (
    <div className="plot" aria-label={title}>
      <Graph
        data={traces}
        layout={layout}
        config={{
          responsive: true,
          displaylogo: false,
          modeBarButtons: [
            ["toImage"],
            ["zoom2d", "pan2d", "select2d", "lasso2d"],
            ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
          ],
          toImageButtonOptions: {
            filename: title + "-" + identity,
            format: "png",
          },
        }}
        useResizeHandler
        style={{ width: "100%" }}
        onRelayout={(event) => {
          const bounds = event["xaxis.range"];
          const from = Array.isArray(bounds)
            ? bounds[0]
            : event["xaxis.range[0]"];
          const to = Array.isArray(bounds)
            ? bounds[1]
            : event["xaxis.range[1]"];
          if (onRange && from != null && to != null) {
            const a = plotTime(from);
            const b = plotTime(to);
            if (Number.isFinite(a) && Number.isFinite(b))
              onRange(Math.floor(a), Math.ceil(b));
          }
        }}
      />
    </div>
  );
}
