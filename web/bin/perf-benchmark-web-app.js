#!/usr/bin/env node

const { spawn } = require("node:child_process");

const python = process.env.PERF_BENCHMARK_PYTHON || "python3";
const child = spawn(python, ["-m", "perf_benchmark", "web", ...process.argv.slice(2)], {
  stdio: "inherit",
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("error", (error) => {
  console.error(`Unable to start perf-benchmark: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  process.exitCode = signal ? 1 : (code ?? 1);
});
