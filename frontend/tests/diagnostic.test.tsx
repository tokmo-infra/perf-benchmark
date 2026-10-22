import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import DiagnosticFrame from "../src/DiagnosticFrame";
import Run from "../src/pages/Run";
import Requests from "../src/pages/Requests";

vi.mock("../src/Plot", () => ({ default: () => <div data-testid="plot" /> }));

beforeEach(() => vi.spyOn(document, "hidden", "get").mockReturnValue(false));
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function app(route: string, requests = false, expired = false) {
  const fetch = vi.fn(async (input: URL) => {
    const path = input.pathname;
    let data: object = { items: [] };
    if (path === "/api/runs/run")
      data = {
        items: [
          {
            name: "Run",
            status: "running",
            run_url: "/d/perf-run?var-run=run&var-trial=%24__all&from=1&to=now",
            trend_status: expired
              ? "超出 7 天诊断保留期"
              : "运行中，见采集覆盖",
          },
        ],
      };
    if (path.endsWith("/trials"))
      data = { items: [{ trial_id: "trial", name: "Trial" }] };
    if (path.endsWith("/windows")) data = { items: [{ window: "last30s" }] };
    if (path.endsWith("/metrics"))
      data = {
        items: [
          {
            name: "Trial",
            metric: "ttft",
            statistic: "p99",
            value: 1,
            unit: "ms",
            availability: "available",
          },
        ],
      };
    if (path.endsWith("/requests"))
      data = { items: [{ turn_id: "turn" }], total: 1 };
    if (path.endsWith("/turn"))
      data = {
        items: [
          {
            turn_id: "turn",
            trial_id: "trial",
            trace_url:
              "/explore?schemaVersion=1&" +
              new URLSearchParams({
                panes: JSON.stringify({
                  workbench: {
                    datasource: "tempo",
                    queries: [{ query: "trace" }],
                    range: { from: "1", to: "2" },
                  },
                }),
              }),
          },
        ],
      };
    if (path.endsWith("/attempts"))
      data = {
        items: [
          {
            attempt: 2,
            trial_id: "trial",
            trace_url:
              "/explore?schemaVersion=1&" +
              new URLSearchParams({
                panes: JSON.stringify({
                  workbench: {
                    datasource: "tempo",
                    queries: [{ query: "attempt-trace" }],
                    range: { from: "3", to: "4" },
                  },
                }),
              }),
          },
        ],
        total: 1,
      };
    return new Response(JSON.stringify(data));
  });
  vi.stubGlobal("fetch", fetch);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  const view = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route
            path={requests ? "/runs/:runId/requests" : "/runs/:runId"}
            element={
              requests ? (
                <Requests
                  settings={{
                    control_enabled: false,
                    grafana_url: "https://host/grafana",
                  }}
                />
              ) : (
                <Run
                  settings={{
                    control_enabled: false,
                    grafana_url: "https://host/grafana",
                  }}
                />
              )
            }
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...view, fetch, client };
}

describe("same-page diagnostics", () => {
  it("uses readable choices while sending the original metric and window keys", async () => {
    const view = app("/runs/run");
    await screen.findByRole("table", { name: "测试指标" });
    const option = await screen.findByRole("option", { name: "最后 30 秒" });
    expect(option.getAttribute("value")).toBe("last30s");
    expect(
      screen.getByRole("checkbox", { name: "首个输出耗时（TTFT）" }),
    ).toBeTruthy();
    fireEvent.change(screen.getByRole("combobox", { name: "统计范围" }), {
      target: { value: "last30s" },
    });
    await waitFor(() =>
      expect(
        view.fetch.mock.calls.some(
          ([url]) =>
            url.pathname.endsWith("/metrics") &&
            url.searchParams.get("window") === "last30s",
        ),
      ).toBe(true),
    );
    view.unmount();
    view.client.clear();
  });
  it("loads only the selected view, keeps scope, pauses result polling and releases the frame", async () => {
    const view = app("/runs/run?trial=trial&from=100&to=200");
    await waitFor(() => expect(screen.getAllByTestId("plot")).toHaveLength(4));
    expect(view.container.querySelector("iframe")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "监控与排查" }));
    await waitFor(() => expect(screen.queryByTestId("plot")).toBeNull());
    const frame = view.container.querySelector("iframe")!;
    const url = new URL(frame.src);
    expect(url.pathname).toBe("/grafana/d/perf-run");
    expect(Object.fromEntries(url.searchParams)).toMatchObject({
      "var-run": "run",
      "var-trial": "trial",
      from: "100",
      to: "200",
      kiosk: "1",
    });
    expect(url.searchParams.get("timezone")).toBe(
      Intl.DateTimeFormat().resolvedOptions().timeZone,
    );
    const heavyReads = () =>
      view.fetch.mock.calls.filter(([url]) =>
        /\/(series|metrics|windows|statistics)$/.test(url.pathname),
      ).length;
    const before = heavyReads();
    vi.useFakeTimers();
    await act(() => vi.advanceTimersByTimeAsync(6000));
    expect(heavyReads()).toBe(before);
    fireEvent.click(screen.getByRole("tab", { name: "测试结果" }));
    expect(view.container.querySelector("iframe")).toBeNull();
    await act(() => vi.advanceTimersByTimeAsync(6000));
    expect(heavyReads()).toBeGreaterThan(before);
    vi.useRealTimers();
    view.unmount();
    view.client.clear();
  });

  it("keeps expired diagnostic details unmounted while the permanent result remains readable", async () => {
    const view = app("/runs/run", false, true);
    await screen.findByRole("table", { name: "测试指标" });
    fireEvent.click(screen.getByRole("tab", { name: "监控与排查" }));
    expect(
      await screen.findByText(
        "监控数据已超过七天保留期，测试汇总仍可长期查看。",
      ),
    ).toBeTruthy();
    expect(view.container.querySelector("iframe")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "测试结果" }));
    expect(screen.getByRole("table", { name: "测试指标" })).toBeTruthy();
    view.unmount();
    view.client.clear();
  });

  it("opens Turn and Attempt Trace in the page and returns to the same detail", async () => {
    const view = app(
      "/runs/run/requests?selected_turn=turn&from=100&to=200",
      true,
    );
    await screen.findByRole("button", { name: "查看请求链路" });
    fireEvent.click(screen.getByRole("button", { name: "查看请求链路" }));
    let url = new URL(view.container.querySelector("iframe")!.src);
    expect(
      JSON.parse(url.searchParams.get("panes")!).workbench.queries[0].query,
    ).toBe("trace");
    expect(JSON.parse(url.searchParams.get("panes")!).workbench.range).toEqual({
      from: "100",
      to: "200",
    });
    expect(Object.fromEntries(url.searchParams)).toMatchObject({
      "var-run": "run",
      "var-trial": "trial",
      from: "100",
      to: "200",
    });
    expect(screen.queryByRole("table", { name: "执行记录" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "返回请求详情" }));
    expect(view.container.querySelector("iframe")).toBeNull();
    fireEvent.click(
      await screen.findByRole("button", { name: "查看本次执行链路" }),
    );
    url = new URL(view.container.querySelector("iframe")!.src);
    expect(
      JSON.parse(url.searchParams.get("panes")!).workbench.queries[0].query,
    ).toBe("attempt-trace");
    expect(view.container.querySelector("iframe")!.title).toContain(
      "第 2 条执行记录",
    );
    fireEvent.click(screen.getByRole("button", { name: "返回请求详情" }));
    expect(view.container.querySelector("iframe")).toBeNull();
    expect(screen.getByRole("heading", { name: "请求详情" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "查看请求链路" }));
    fireEvent.change(screen.getByRole("combobox", { name: "测试轮次" }), {
      target: { value: "trial" },
    });
    expect(view.container.querySelector("iframe")).toBeNull();
    await screen.findByRole("table", { name: "执行记录" });
    view.unmount();
    view.client.clear();
  });

  it("revokes an open Trace when the underlying request details expire", async () => {
    const view = app("/runs/run/requests?selected_turn=turn", true);
    fireEvent.click(
      await screen.findByRole("button", { name: "查看请求链路" }),
    );
    expect(view.container.querySelector("iframe")).not.toBeNull();
    view.fetch.mockImplementationOnce(
      async () => new Response('{"reason":"detail_expired"}', { status: 410 }),
    );
    fireEvent.click(screen.getByRole("button", { name: "刷新本页" }));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toContain("detail_expired"),
    );
    expect(view.container.querySelector("iframe")).toBeNull();
    view.unmount();
    view.client.clear();
  });

  it("does not mount a missing backend and unloads diagnostic work while the tab is hidden", () => {
    const view = render(
      <DiagnosticFrame title="诊断" unavailable="未配置 Grafana 地址。" />,
    );
    expect(view.container.querySelector("iframe")).toBeNull();
    view.rerender(
      <DiagnosticFrame
        title="诊断"
        url="https://host/grafana/d/perf-run?var-run=run"
        unavailable="不可用"
      />,
    );
    expect(view.container.querySelector("iframe")).not.toBeNull();
    vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    fireEvent(document, new Event("visibilitychange"));
    expect(view.container.querySelector("iframe")).toBeNull();
    expect(screen.getByText("页面暂未显示，已暂停加载监控。")).toBeTruthy();
    vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    fireEvent(document, new Event("visibilitychange"));
    expect(view.container.querySelector("iframe")).not.toBeNull();
  });

  it("reports a slow frame, treats load as document completion and allows a deliberate reload", async () => {
    vi.useFakeTimers();
    const view = render(
      <DiagnosticFrame
        title="诊断"
        url="https://host/grafana/d/perf-run"
        unavailable="不可用"
      />,
    );
    const frame = view.container.querySelector("iframe")!;
    await act(() => vi.advanceTimersByTimeAsync(15_000));
    expect(
      screen.getByText("监控页面仍在加载，可稍候或重新加载。"),
    ).toBeTruthy();
    fireEvent.load(frame);
    expect(screen.queryByRole("status")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "重新加载监控页面" }));
    expect(view.container.querySelector("iframe")).not.toBe(frame);
    expect(screen.getByText("正在加载运行监控…")).toBeTruthy();
  });
});
