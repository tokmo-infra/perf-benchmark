import { Suspense } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import Plot from "../src/Plot";

vi.mock("react-plotly.js/factory", () => ({
  default:
    () =>
    ({ data }: { data: unknown }) => (
      <div data-testid="plotly-graph">{JSON.stringify(data)}</div>
    ),
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("truthful empty chart presentation", () => {
  it("does not mount Plotly axes for empty or entirely unavailable series and preserves the backend reason", () => {
    const view = render(
      <Plot rows={[]} title="workload_rps" identity="empty" />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "所选时段内还没有这项指标的数据。",
    );
    expect(document.head.querySelector("script[src]")).toBeNull();
    view.rerender(
      <Plot
        rows={[
          {
            availability: "unavailable",
            reason: "usage_not_reported",
            time: 100,
            value: 0,
          },
        ]}
        title="output_tokens"
        identity="unavailable"
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "此指标没有可用数值。",
    );
    expect(screen.getByRole("status").textContent).toContain(
      "服务未返回 token 用量",
    );
    expect(
      screen.getByText("服务未返回 token 用量").getAttribute("title"),
    ).toBe("usage_not_reported");
    expect(screen.queryByTestId("plotly-graph")).toBeNull();
    expect(document.head.querySelector("script[src]")).toBeNull();
    view.rerender(
      <Plot
        comparison
        rows={[
          { baseline: null, candidate: null, reason: "metric_unavailable" },
        ]}
        title="comparison"
        identity="comparison-empty"
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "所选范围没有此指标",
    );
    expect(screen.queryByTestId("plotly-graph")).toBeNull();
  });

  it("keeps an observed zero as a plotted value and keeps an unavailable neighbour as a gap", async () => {
    vi.stubGlobal("Plotly", {});
    render(
      <Suspense fallback={<div>加载图表</div>}>
        <Plot
          rows={[
            { series: "x", availability: "available", time: 100, value: 0 },
            { series: "x", availability: "unavailable", time: 200, value: 8 },
          ]}
          title="observed zero"
          identity="zero"
        />
      </Suspense>,
    );
    await waitFor(() =>
      expect(document.head.querySelector("script[src]")).not.toBeNull(),
    );
    fireEvent.load(document.head.querySelector("script[src]")!);
    const graph = await screen.findByTestId("plotly-graph");
    expect(JSON.parse(graph.textContent!)[0].y).toEqual([0, null]);
    expect(screen.queryByRole("status")).toBeNull();
  });
});
