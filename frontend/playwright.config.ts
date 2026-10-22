import { defineConfig } from "@playwright/test";
delete process.env.NO_COLOR;
const externalWorkbench = process.env.WORKBENCH_TEST_URL;
export default defineConfig({
  testDir: "./tests/browser",
  workers: 1,
  timeout: 60_000,
  use: {
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    },
    baseURL: externalWorkbench ?? "http://127.0.0.1:8089",
    viewport: { width: 1440, height: 1000 },
    timezoneId: "Asia/Shanghai",
    screenshot: "only-on-failure",
  },
  webServer: externalWorkbench
    ? undefined
    : {
        command:
          "../.venv/bin/python -m benchmarks.workbench_server --port 8089",
        cwd: ".",
        url: "http://127.0.0.1:8089/health",
        reuseExistingServer: false,
        timeout: 30_000,
      },
});
