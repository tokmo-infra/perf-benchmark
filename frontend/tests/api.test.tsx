import { afterEach, describe, expect, it, vi } from "vitest";
import {
  render,
  screen,
  cleanup,
  fireEvent,
  renderHook,
  act,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  api,
  ApiError,
  grafanaLink,
  path,
  plotTime,
  terminal,
  useRead,
  useRefresh,
} from "../src/api";
import { Pager } from "../src/components";
import { statisticLabel } from "../src/labels";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
describe("bounded HTTP and exact presentation", () => {
  it("labels token and time weighted means separately", () => {
    expect(statisticLabel("twm")).toBe("Token 加权平均值");
    expect(statisticLabel("time_wm")).toBe("时间加权平均值");
  });
  it("defers the final visible-data read when a Run becomes terminal while hidden", async () => {
    let value = 1;
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (url) =>
          new Response(
            JSON.stringify({
              items: String(url).includes("metrics")
                ? [{ value }]
                : [{ status: value === 1 ? "running" : "succeeded" }],
            }),
          ),
      ),
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: Infinity } },
    });
    const { result } = renderHook(
      () => {
        const run = useRead("/api/runs/deferred");
        const metrics = useRead("/api/runs/deferred/metrics?");
        useRefresh(!terminal(run.data?.items?.[0].status), "deferred");
        return metrics;
      },
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={client}>{children}</QueryClientProvider>
        ),
      },
    );
    await waitFor(() => expect(result.current.data?.items?.[0].value).toBe(1));
    hidden.mockReturnValue(true);
    value = 2;
    await act(async () => {
      client.setQueryData(["api", "/api/runs/deferred"], {
        items: [{ status: "succeeded" }],
      });
    });
    expect(result.current.data?.items?.[0].value).toBe(1);
    hidden.mockReturnValue(false);
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(result.current.data?.items?.[0].value).toBe(2));
  });
  it("bounds simultaneous requests and discards aborted waiting work", async () => {
    let running = 0,
      peak = 0;
    const releases: (() => void)[] = [];
    const fetch = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          running++;
          peak = Math.max(peak, running);
          releases.push(() => {
            running--;
            resolve(new Response("{}", { status: 200 }));
          });
        }),
    );
    vi.stubGlobal("fetch", fetch);
    const requests = Array.from({ length: 8 }, (_, i) => api("/api/" + i));
    const controller = new AbortController();
    const aborted = api("/never-sent", controller.signal).catch((e) => e);
    controller.abort();
    await Promise.resolve();
    expect(fetch).toHaveBeenCalledTimes(4);
    for (let i = 0; i < 8; i++) {
      releases.shift()!();
      await new Promise((r) => setTimeout(r, 0));
    }
    await Promise.all(requests);
    expect(await aborted).toHaveProperty("name", "AbortError");
    expect(peak).toBe(4);
    expect(fetch).toHaveBeenCalledTimes(8);
  });
  it("never retries control writes or unavailable detail", async () => {
    const fetch = vi.fn(
      async () => new Response('{"reason":"detail_expired"}', { status: 410 }),
    );
    vi.stubGlobal("fetch", fetch);
    await expect(api("/api/detail")).rejects.toMatchObject({
      status: 410,
      reason: "detail_expired",
    });
    await expect(
      api("/api/control/start", undefined, { config: "x" }),
    ).rejects.toBeInstanceOf(ApiError);
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("honors Retry-After on GET while limiting retry to once", async () => {
    const fetch = vi.fn(
      async () =>
        new Response("busy", { status: 503, headers: { "Retry-After": "0" } }),
    );
    vi.stubGlobal("fetch", fetch);
    await expect(api("/api/runs")).rejects.toMatchObject({ status: 503 });
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("removes expired details from view after a successful read", async () => {
    let expired = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        expired
          ? new Response('{"reason":"detail_expired"}', { status: 410 })
          : new Response('{"items":[{"turn_id":"old"}],"total":1}'),
      ),
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(
      () => useRead("/api/runs/r/requests?offset=0"),
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={client}>{children}</QueryClientProvider>
        ),
      },
    );
    await waitFor(() => expect(result.current.data?.total).toBe(1));
    expired = true;
    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() => expect(result.current.data).toBeUndefined());
    expect(result.current.error).toMatchObject({ reason: "detail_expired" });
  });
  it("preserves decimal nanoseconds as text and encodes IDs", async () => {
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response('{"timings":[{"monotonic_ns":"9223372036854775807"}]}'),
    );
    const result = await api<{ timings: { monotonic_ns: string }[] }>(
      "/api/time",
    );
    expect(result.timings[0].monotonic_ns).toBe("9223372036854775807");
    expect(
      path("中文/a", "requests", { session_id: "x?&y", offset: 500000 }),
    ).toContain("session_id=x%3F%26y");
  });
  it("revokes cached Turn and Attempt facts with an expired request page and resumes only explicitly", async () => {
    let expired = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url) =>
        expired && String(url).includes("/requests?")
          ? new Response('{"reason":"detail_expired"}', { status: 410 })
          : new Response('{"items":[{"turn_id":"old"}],"total":1}'),
      ),
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: Infinity } },
    });
    const { result } = renderHook(
      () => ({
        page: useRead("/api/runs/family/requests?offset=0"),
        turn: useRead("/api/runs/family/turn?turn_id=old"),
        attempts: useRead("/api/runs/family/attempts?turn_id=old"),
        summary: useRead("/api/runs/family/metrics?"),
      }),
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={client}>{children}</QueryClientProvider>
        ),
      },
    );
    await waitFor(() => expect(result.current.attempts.data?.total).toBe(1));
    expired = true;
    await act(async () => {
      await result.current.page.refetch();
    });
    await waitFor(() => {
      expect(result.current.page.data).toBeUndefined();
      expect(result.current.turn.data).toBeUndefined();
      expect(result.current.attempts.data).toBeUndefined();
    });
    expect(result.current.summary.data?.total).toBe(1);
    expect(
      client.getQueryData(["api", "/api/runs/family/turn?turn_id=old"]),
    ).toBeUndefined();
    expired = false;
    await act(async () => {
      await result.current.page.refetch();
    });
    await waitFor(() => expect(result.current.page.data?.total).toBe(1));
  });
  it("preserves Grafana subpaths and refuses arbitrary redirects", () => {
    expect(grafanaLink("https://host/grafana", "/d/perf-run?var-run=x")).toBe(
      "https://host/grafana/d/perf-run?var-run=x",
    );
    expect(
      grafanaLink("https://host", "//evil.test/d/perf-run"),
    ).toBeUndefined();
    expect(grafanaLink("https://host", "/not-a-dashboard")).toBeUndefined();
    expect(grafanaLink("https://host", "http://[invalid")).toBeUndefined();
    const trace = new URL(
      grafanaLink(
        "https://host/grafana",
        "/explore?schemaVersion=1&" +
          new URLSearchParams({
            panes: JSON.stringify({
              workbench: {
                datasource: "tempo",
                queries: [{ query: "abc" }],
                range: { from: "1", to: "2" },
              },
            }),
          }),
        "Asia/Shanghai",
        new URLSearchParams({
          trial: "t",
          from: "100",
          to: "200",
          redirect: "https://evil.test",
        }),
      )!,
    );
    expect(trace.pathname).toBe("/grafana/explore");
    expect(Object.fromEntries(trace.searchParams)).toMatchObject({
      schemaVersion: "1",
      timezone: "Asia/Shanghai",
      "var-trial": "t",
      from: "100",
      to: "200",
    });
    expect(JSON.parse(trace.searchParams.get("panes")!)).toEqual({
      workbench: {
        datasource: "tempo",
        queries: [{ query: "abc" }],
        range: { from: "100", to: "200" },
      },
    });
    expect(
      grafanaLink("https://host/grafana", "/explore?panes=invalid"),
    ).toBeUndefined();
  });
  it("maps Plotly date ranges to UTC independently of the browser timezone", () => {
    expect(plotTime("2026-09-29 07:59:59.125")).toBe(
      Date.parse("2026-09-29T07:59:59.125Z"),
    );
    expect(plotTime("2026-09-29")).toBe(Date.parse("2026-09-29T00:00:00Z"));
    expect(plotTime("2026-09-29T15:59:59.125+08:00")).toBe(
      plotTime("2026-09-29 07:59:59.125"),
    );
    expect(plotTime(1790668799125)).toBe(1790668799125);
  });
  it("reports exact empty/out-of-range pages without claiming rows", () => {
    const onPage = vi.fn();
    const view = render(
      <Pager offset={200} size={50} total={121} onPage={onPage} />,
    );
    expect(screen.getByText(/条记录/).textContent).toContain("0 / 121");
    fireEvent.click(screen.getByText("返回最后一页"));
    expect(onPage).toHaveBeenCalledWith(100);
    view.rerender(<Pager offset={0} size={50} total={0} onPage={onPage} />);
    expect(screen.getByText(/条记录/).textContent).toContain("0 / 0");
    expect((screen.getByText("下一页") as HTMLButtonElement).disabled).toBe(
      true,
    );
  });
});
