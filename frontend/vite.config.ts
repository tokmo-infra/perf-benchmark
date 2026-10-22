import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "../perf_benchmark/app/workbench", emptyOutDir: true },
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8088", changeOrigin: false } },
  },
  test: { environment: "jsdom", include: ["tests/**/*.test.{ts,tsx}"] },
});
