import {
  expect,
  test,
  type FrameLocator,
  type Locator,
  type Page,
} from "@playwright/test";

const defaultMetrics = [
  "ttft",
  "tpot",
  "workload_rps",
  "actual_in_flight_concurrency",
];
const allMetrics = [...defaultMetrics, "itl", "e2e"];
const metricNames: Record<string, string> = {
  ttft: "首个输出耗时（TTFT）",
  tpot: "每 token 输出耗时（TPOT）",
  workload_rps: "请求处理速率",
  actual_in_flight_concurrency: "同时执行的请求数",
  itl: "输出间隔（ITL）",
  e2e: "请求总耗时（E2E）",
};

async function assertSeriesCharts(
  page: Page,
  metrics = defaultMetrics,
  prefix = "",
) {
  const panels = page.locator(".chart-grid > .panel");
  await expect(panels).toHaveCount(metrics.length);
  let plots = 0;
  let points = 0;
  for (const metric of metrics) {
    const response = await page.request.get(
      `${prefix}/api/runs/edges-64/series?from=0&to=now&metric=${metric}&statistics=am,p99,value`,
    );
    expect(response.ok()).toBe(true);
    const result = await response.json();
    const rows: {
      availability: string;
      time: number | null;
      value: number | null;
      reason?: string;
      unavailable_reason?: string;
    }[] = result.items;
    const drawable = rows.some(
      (row) =>
        row.availability === "available" &&
        typeof row.time === "number" &&
        Number.isFinite(row.time) &&
        typeof row.value === "number" &&
        Number.isFinite(row.value),
    );
    const panel = panels.filter({
      has: page.getByRole("heading", {
        name: metricNames[metric],
        exact: true,
      }),
    });
    await expect(panel).toHaveCount(1);
    await expect(panel.locator(".js-plotly-plot")).toHaveCount(
      drawable ? 1 : 0,
    );
    if (drawable) {
      plots++;
      points += rows.length;
    } else {
      const empty = panel.getByRole("status");
      await expect(empty).toBeVisible();
      await expect(empty).toHaveAttribute(
        "aria-label",
        `${metricNames[metric]} · ${result.state === "final" ? "完整结果" : result.state === "partial" ? "部分结果" : result.state}`,
      );
      await expect(empty).toContainText(
        rows.length
          ? "此指标没有可用数值。"
          : "所选时段内还没有这项指标的数据。",
      );
      for (const reason of new Set(
        rows.map((row) => row.reason ?? row.unavailable_reason).filter(Boolean),
      ))
        expect(await empty.locator("p[title]").getAttribute("title")).toContain(
          reason!,
        );
    }
  }
  await expect(page.locator(".js-plotly-plot")).toHaveCount(plots);
  return { plots, points, panels: metrics.length };
}

async function diagnosticLogin(diagnostic: FrameLocator, ready: Locator) {
  const login = diagnostic.locator('input[name="user"]');
  await expect
    .poll(async () => (await login.count()) + (await ready.count()), {
      timeout: 30_000,
    })
    .toBeGreaterThan(0);
  if (await login.count()) {
    const user = process.env.GRAFANA_TEST_USER;
    const password = process.env.GRAFANA_TEST_PASSWORD;
    if (!user || !password)
      throw new Error(
        "Explicit GRAFANA_TEST_USER and GRAFANA_TEST_PASSWORD are required for the authenticated diagnostic stack",
      );
    await login.fill(user);
    await diagnostic.locator('input[name="password"]').fill(password);
    await diagnostic.locator('button[type="submit"]').click();
  }
  await expect(ready).toBeVisible({ timeout: 30_000 });
}

test("real frozen SQLite and Parquet: runs, million pages, Turn and all Attempts", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/#/runs");
  await expect(
    page.getByRole("table", { name: "测试记录", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "standard-1000000", exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "测试指标", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("tab", { name: "监控与排查" })).toBeVisible();
  await expect(page.locator("iframe")).toHaveCount(0);
  for (const offset of [0, 500000, 999950]) {
    await page.goto(
      `/#/runs/standard-1000000/requests?from=0&to=now&offset=${offset}`,
    );
    const response = await request.get(
      `/api/runs/standard-1000000/requests?from=0&to=now&offset=${offset}&page_size=50`,
    );
    const expected = await response.json();
    expect(expected.total).toBe(1000000);
    await expect(
      page
        .getByRole("table", { name: "请求", exact: true })
        .locator("tbody tr"),
    ).toHaveCount(50);
    await page
      .getByRole("button", { name: "详情", exact: true })
      .first()
      .click();
    await page
      .getByText("查看请求信息与数据来源（高级）", { exact: true })
      .click();
    await expect(
      page.locator("pre").filter({ hasText: '"request"' }).first(),
    ).toContainText(expected.items[0].turn_id);
    await expect(page.getByText("1,000,000", { exact: true })).toBeVisible();
  }
  await page.getByRole("button", { name: "详情", exact: true }).first().click();
  await expect(
    page.getByRole("heading", { name: "全部执行记录" }),
  ).toBeVisible();
  await expect(
    page.getByRole("table", { name: "执行记录", exact: true }),
  ).toBeVisible();
  await page.getByText("查看精确计时与执行指标").click();
  await expect(page.locator("pre").last()).toContainText("monotonic_ns");
  expect(errors).toEqual([]);
});

test("real series charts, fixed range URL, read-only mode and formal comparison", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/#/runs/edges-64?from=0&to=now");
  await assertSeriesCharts(page);
  await page
    .getByRole("checkbox", { name: "输出间隔（ITL）", exact: true })
    .check();
  await page
    .getByRole("checkbox", { name: "请求总耗时（E2E）", exact: true })
    .check();
  await assertSeriesCharts(page, allMetrics);
  await expect(
    page.getByRole("button", { name: "Share chart..." }),
  ).toHaveCount(0);
  const downloaded = page.waitForEvent("download");
  const points = page
    .locator("details")
    .filter({ has: page.getByText("查看数据点与统计说明", { exact: true }) })
    .first();
  await expect(points.getByRole("table")).toHaveCount(0);
  await points.getByText("查看数据点与统计说明", { exact: true }).click();
  await expect(
    points.getByRole("columnheader", { name: "开始时间（ns）", exact: true }),
  ).toBeVisible();
  await points.getByText("查看数据点与统计说明", { exact: true }).click();
  await expect(points.getByRole("table")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Download plot as a PNG" })
    .first()
    .click();
  expect((await downloaded).suggestedFilename()).toMatch(/\.png$/);
  await page
    .locator(".js-plotly-plot")
    .first()
    .evaluate(async (element) => {
      if (!(element instanceof HTMLElement))
        throw new Error("Plotly graph is missing");
      await window.Plotly.relayout(element, {
        "xaxis.range": ["2026-09-29 07:59:59", "2026-09-29 08:00:01"],
      });
    });
  await expect
    .poll(() => new URLSearchParams(page.url().split("?")[1]).get("from"))
    .toBe(String(Date.parse("2026-09-29T07:59:59Z")));
  await page.getByText("查看完整统计数据（高级）", { exact: true }).click();
  await expect(
    page.locator("pre").filter({ hasText: "distribution" }).first(),
  ).toBeVisible();
  await page.getByText("查看各执行阶段的状态与耗时", { exact: true }).click();
  await expect(
    page.getByRole("table").filter({ hasText: "loader" }).first(),
  ).toBeVisible();
  await page.getByLabel("开始时间").fill("1790639999000");
  await expect(page).toHaveURL(/from=1790639999000/);
  await page.getByRole("link", { name: "启动测试" }).click();
  await expect(
    page.getByRole("heading", { name: "当前工作台用于查看结果" }),
  ).toBeVisible();
  await page.goto("/#/compare?baseline=edges-64&candidate=varied-100000");
  await page.getByRole("button", { name: "比较结果" }).click();
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toBeVisible();
  await expect(page.getByRole("table").last()).toContainText(
    "仅展示差异（未设阈值）",
  );
  expect(errors).toEqual([]);
});

test("expiration and late responses do not display cached details or overwrite a new filter", async ({
  page,
}) => {
  await page.goto("/#/runs/edges-64/requests?from=0&to=now");
  await expect(
    page.getByRole("table", { name: "请求", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "详情", exact: true }).first().click();
  await expect(
    page.getByRole("table", { name: "执行记录", exact: true }),
  ).toBeVisible();
  // Explicit HTTP fault injection for UI handling; real data path covered above.
  await page.route("**/api/runs/edges-64/requests?**", (route) =>
    route.fulfill({
      status: 410,
      contentType: "application/json",
      body: '{"reason":"detail_expired"}',
    }),
  );
  await page.getByRole("button", { name: "刷新本页" }).click();
  await expect(page.getByRole("alert")).toContainText("detail_expired");
  await expect(
    page.getByRole("table", { name: "请求", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("table", { name: "执行记录", exact: true }),
  ).toHaveCount(0);
  await page.goto("/#/runs/edges-64");
  await expect(
    page.getByRole("table", { name: "测试指标", exact: true }),
  ).toBeVisible();
});

test("comparison history cannot display another identity, including a late response", async ({
  page,
}) => {
  await page.goto("/#/compare?baseline=edges-64&candidate=varied-100000");
  await page.getByRole("button", { name: "比较结果" }).click();
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toBeVisible();
  await page.evaluate(() => {
    window.location.hash =
      "/compare?baseline=edges-64&candidate=standard-1000000";
  });
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "比较结果" }).click();
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toBeVisible();
  await page.goBack();
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toHaveCount(0);
  const candidate = page.getByRole("textbox", { name: "测试任务编号" }).nth(1);
  await candidate.fill("varied-100000");
  await candidate.press("End");
  await expect(candidate).toBeFocused();
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let ready!: () => void, done!: () => void;
  const received = new Promise<void>((resolve) => {
    ready = resolve;
  });
  const completed = new Promise<void>((resolve) => {
    done = resolve;
  });
  await page.route("**/api/compare", async (route) => {
    const response = await route.fetch();
    ready();
    await held;
    await route.fulfill({ response });
    done();
  });
  await page.getByRole("button", { name: "比较结果" }).click();
  await received;
  await page.evaluate(() => {
    window.location.hash =
      "/compare?baseline=edges-64&candidate=standard-1000000";
  });
  release();
  await completed;
  await page.unrouteAll({ behavior: "wait" });
  await expect(
    page.getByRole("heading", { name: "指标差异与结论" }),
  ).toHaveCount(0);
});

test("terminal manual refresh covers visible queries and Compare keeps the selected Trial", async ({
  page,
  request,
}) => {
  await page.goto("/#/runs/edges-64?from=0&to=now");
  await assertSeriesCharts(page);
  await page.getByText("查看完整统计数据（高级）", { exact: true }).click();
  await expect(
    page.locator("pre").filter({ hasText: "distribution" }).first(),
  ).toBeVisible();
  const refreshed = new Set<string>();
  page.on("response", (r) => {
    const url = new URL(r.url());
    if (url.pathname.startsWith("/api/runs/edges-64"))
      refreshed.add(url.pathname);
  });
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect
    .poll(() =>
      [
        "/api/runs/edges-64",
        "/api/runs/edges-64/series",
        "/api/runs/edges-64/windows",
        "/api/runs/edges-64/statistics",
      ].every((p) => refreshed.has(p)),
    )
    .toBe(true);
  const response = await request.get("/api/runs/edges-64/trials");
  const trial = (await response.json()).items[0].trial_id;
  await page
    .getByRole("combobox", { name: "测试轮次", exact: true })
    .selectOption(trial);
  await page.getByRole("link", { name: "与其他测试比较 ↗" }).click();
  await expect(page).toHaveURL(
    new RegExp(`baseline_trial=${encodeURIComponent(trial)}`),
  );
  await expect(page).toHaveURL(/from=0.*to=now/);
});

test("subpath deployment, actual Plotly values and stable repeated navigation", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/tools/#/runs/edges-64?from=0&to=now");
  const expected = await assertSeriesCharts(page, defaultMetrics, "/tools");
  const response = await request.get(
    "/api/runs/edges-64/series?from=0&to=now&metric=ttft&statistics=am,p99,value",
  );
  const data = await response.json();
  const actual = () =>
    page
      .locator(".js-plotly-plot")
      .first()
      .evaluate((element) => {
        const graph = element as HTMLElement & {
          data: { y: (number | null)[] }[];
        };
        return graph.data
          .flatMap((trace) => trace.y)
          .filter((value) => value !== null)
          .sort((a, b) => a! - b!);
      });
  await expect.poll(actual).toEqual(
    data.items
      .filter((r: { availability: string }) => r.availability === "available")
      .map((r: { value: number }) => r.value)
      .sort((a: number, b: number) => a - b),
  );
  await page.screenshot({
    path: "/tmp/workbench-preview-2026-10-12.png",
    fullPage: false,
  });
  for (let i = 0; i < 10; i++) {
    await page.goto("/tools/#/runs/edges-64/requests?from=0&to=now&offset=50");
    await expect(
      page.getByRole("table", { name: "请求", exact: true }),
    ).toBeVisible();
    await page.goto("/tools/#/runs/edges-64?from=0&to=now");
    await assertSeriesCharts(page, defaultMetrics, "/tools");
  }
  const memory = await page.evaluate(() => {
    const perf = performance as Performance & {
      memory?: { usedJSHeapSize: number };
    };
    return {
      heap: perf.memory?.usedJSHeapSize,
      plots: document.querySelectorAll(".js-plotly-plot").length,
    };
  });
  expect(memory.plots).toBe(expected.plots);
  console.log("Repeated navigation memory", JSON.stringify(memory));
  expect(errors).toEqual([]);
});

test("fixed Grafana opens in the same page and is unloaded when returning to results", async ({
  page,
}) => {
  test.skip(
    !process.env.GRAFANA_TEST_URL,
    "explicit fixed-version diagnostic stack required",
  );
  await page.goto("/#/runs/edges-64?from=0&to=now");
  const original = page.url();
  const pages = page.context().pages().length;
  await expect(page.locator("iframe")).toHaveCount(0);
  await page.getByRole("tab", { name: "监控与排查" }).click();
  await expect(page.locator("iframe")).toHaveAttribute(
    "src",
    /\/d\/perf-run\?.*var-run=edges-64/,
  );
  const diagnostic = page.frameLocator("iframe");
  const runPanel = diagnostic.getByText("Run · 时间与状态", { exact: true });
  await diagnosticLogin(diagnostic, runPanel);
  expect(page.context().pages()).toHaveLength(pages);
  expect(page.url()).toBe(original);
  await page.getByRole("tab", { name: "测试结果" }).click();
  await expect(page.locator("iframe")).toHaveCount(0);
  await assertSeriesCharts(page);
});

test("real committed Trace opens its actual Attempt and output spans in the same page", async ({
  page,
}) => {
  const runId = process.env.WORKBENCH_TRACE_TEST_RUN_ID;
  test.skip(
    !runId ||
      !process.env.WORKBENCH_TEST_URL ||
      !process.env.GRAFANA_TEST_USER ||
      !process.env.GRAFANA_TEST_PASSWORD,
    "requires an explicit committed traced Run, WORKBENCH_TEST_URL and authenticated Grafana test credentials",
  );
  await page.goto(`/#/runs/${encodeURIComponent(runId!)}/requests`);
  await expect(
    page.getByRole("table", { name: "请求", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "详情", exact: true }).first().click();
  const original = page.url();
  const tabs = page.context().pages().length;
  expect(tabs).toBe(1);
  await expect(page.locator("iframe")).toHaveCount(0);
  await page.getByRole("button", { name: "查看请求链路", exact: true }).click();
  const diagnostic = page.frameLocator("iframe");
  await diagnosticLogin(
    diagnostic,
    diagnostic.getByText("benchmark.attempt", { exact: true }).first(),
  );
  await expect(
    diagnostic.getByText("benchmark.output_events", { exact: true }).first(),
  ).toBeVisible();
  expect(page.context().pages()).toHaveLength(tabs);
  expect(page.url()).toBe(original);
  await page.getByRole("button", { name: "返回请求详情" }).click();
  await expect(page.locator("iframe")).toHaveCount(0);
  await expect(
    page.getByRole("table", { name: "执行记录", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "查看本次执行链路" }).first().click();
  await diagnosticLogin(
    diagnostic,
    diagnostic.getByText("benchmark.attempt", { exact: true }).first(),
  );
  await expect(
    diagnostic.getByText("benchmark.output_events", { exact: true }).first(),
  ).toBeVisible();
  expect(page.context().pages()).toHaveLength(tabs);
  expect(page.url()).toBe(original);
  await page.getByRole("button", { name: "返回请求详情" }).click();
  await expect(page.locator("iframe")).toHaveCount(0);
});

test("browser starts an actual CLI child, links its formal 测试任务编号 and cancels it", async ({
  page,
  request,
}) => {
  await page.goto("/control-test/#/control");
  await page
    .getByRole("combobox", { name: "测试配置" })
    .selectOption("run.yaml");
  await page.getByRole("button", { name: "启动测试" }).click();
  await expect
    .poll(
      async () =>
        (await (await request.get("/control-test/fixture-state")).json())
          .received,
      { timeout: 30_000 },
    )
    .toBe(true);
  await expect(page.getByRole("link", { name: "查看测试结果" })).toBeVisible();
  const response = await request.get("/control-test/api/control/jobs");
  const job = (await response.json()).jobs[0];
  expect(job.run_id).toBeTruthy();
  expect(job.run_id).not.toBe(job.id);
  await page.getByRole("button", { name: "取消测试" }).click();
  await expect
    .poll(
      async () =>
        (await (await request.get("/control-test/api/control/jobs")).json())
          .jobs[0].process_state,
    )
    .toBe("exited");
  const run = await request.get(`/control-test/api/runs/${job.run_id}`);
  expect((await run.json()).items[0].status).toBe("cancelled");
  await page.getByRole("link", { name: "查看测试结果" }).click();
  await expect(page.locator(".status-line")).toContainText("已取消");
});

test("five-minute six-chart refresh remains bounded, pauses when hidden and stops at terminal", async ({
  page,
  request,
}) => {
  test.setTimeout(360_000);
  let terminal = false;
  // Only lifecycle state is injected to exercise polling over real frozen points.
  // This does not claim live serving/backfill correctness from synthetic status.
  await page.route("**/api/runs/edges-64", async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    data.items[0].status = terminal ? "succeeded" : "running";
    await route.fulfill({ response, json: data });
  });
  let active = 0,
    peak = 0,
    completed = 0,
    bytes = 0,
    lastActivity = Date.now();
  const pending = new Set<object>();
  page.on("request", (r) => {
    if (new URL(r.url()).pathname.startsWith("/api/")) {
      pending.add(r);
      lastActivity = Date.now();
      active++;
      peak = Math.max(peak, active);
    }
  });
  const finish = (r: object) => {
    if (pending.delete(r)) {
      active--;
      completed++;
      lastActivity = Date.now();
    }
  };
  page.on("requestfinished", finish);
  page.on("requestfailed", finish);
  page.on("response", async (r) => {
    if (r.url().includes("/series?")) bytes += (await r.body()).byteLength;
  });
  await page.goto("/#/runs/edges-64?from=0&to=now");
  await page
    .getByRole("checkbox", { name: "输出间隔（ITL）", exact: true })
    .check();
  await page
    .getByRole("checkbox", { name: "请求总耗时（E2E）", exact: true })
    .check();
  const expected = await assertSeriesCharts(page, allMetrics);
  const cdp = await page.context().newCDPSession(page);
  const memory: {
    elapsed_seconds: number;
    heap: number;
    nodes: number;
    plots: number;
    points: number;
  }[] = [];
  const sample = async (elapsed_seconds: number) => {
    await cdp.send("HeapProfiler.collectGarbage");
    const heap = (await cdp.send("Runtime.getHeapUsage")).usedSize;
    const dom = await cdp.send("Memory.getDOMCounters");
    const chart = await page.evaluate(() => ({
      plots: document.querySelectorAll(".js-plotly-plot").length,
      points: [...document.querySelectorAll(".js-plotly-plot")].reduce(
        (count, e) =>
          count +
          (e as HTMLElement & { data: { y: unknown[] }[] }).data.reduce(
            (n, t) => n + t.y.length,
            0,
          ),
        0,
      ),
    }));
    memory.push({ elapsed_seconds, heap, nodes: dom.nodes, ...chart });
  };
  await expect.poll(() => active).toBe(0);
  // Unavailable metrics legitimately return no points. Wait for the exact API
  // point count, rather than requiring every chart to manufacture a series.
  await page.waitForFunction(
    (expected) =>
      [...document.querySelectorAll(".js-plotly-plot")].reduce(
        (n, e) =>
          n +
          (e as HTMLElement & { data: { y: unknown[] }[] }).data.reduce(
            (count, t) => count + t.y.length,
            0,
          ),
        0,
      ) === expected,
    expected.points,
  );
  await sample(0);
  for (let i = 1; i <= 5; i++) {
    await page.waitForTimeout(60_000);
    await sample(i * 60);
  }
  console.log("Five-minute samples", JSON.stringify({ peak, memory }));
  expect(peak).toBeLessThanOrEqual(4);
  expect(
    memory.every(
      (s) => s.plots === expected.plots && s.points === expected.points,
    ),
  ).toBe(true);
  await expect.poll(() => active).toBe(0);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  const beforeHidden = completed;
  await page.waitForTimeout(6000);
  expect(completed).toBe(beforeHidden);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: false,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect.poll(() => completed).toBeGreaterThan(beforeHidden);
  terminal = true;
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.locator(".status-line")).toContainText("已完成", {
    timeout: 15_000,
  });
  await expect
    .poll(() => active === 0 && Date.now() - lastActivity > 1000)
    .toBe(true);
  const atTerminal = completed;
  await page.waitForTimeout(6000);
  expect(completed).toBe(atTerminal);
  console.log(
    "Five-minute browser evidence",
    JSON.stringify({ peak, completed, series_response_bytes: bytes, memory }),
  );
  await cdp.detach();
});
