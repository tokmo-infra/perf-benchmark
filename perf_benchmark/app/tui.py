"""Rich terminal client for the same bounded control and ResultReader HTTP API."""
import asyncio

import aiohttp
from rich.console import Console
from rich.table import Table


async def run(url):
    console = Console()
    base = url.rstrip("/")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
        async def call(path, data=None):
            async with client.request("GET" if data is None else "POST", base+path, json=data) as response:
                response.raise_for_status()
                return await response.json()
        console.print("perf-benchmark · SQLite 结果工作台")
        console.print("configs | start <YAML> | jobs | cancel <ID> | runs [页] | status <Run> | validity <Run> | metrics <Run> | coverage <Run> | evidence <Run> | requests <Run> [偏移] [Session] | turn <Run> <Turn> | compare <基线> <候选> [最大退化比例] [允许变化项,逗号分隔] | quit")
        while True:
            command = (await asyncio.to_thread(console.input, "[bold]> [/]")).split()
            if not command:
                continue
            action, *args = command
            try:
                if action == "quit":
                    return 0
                if action == "configs":
                    value = await call("/api/control/configs")
                elif action == "jobs":
                    value = await call("/api/control/jobs")
                elif action == "start":
                    value = await call("/api/control/start", {"config": " ".join(args)})
                elif action == "cancel":
                    value = await call("/api/control/cancel", {"id": args[0]})
                elif action == "runs":
                    value = await call("/api/runs?page="+(args[0] if args else "1"))
                elif action == "evidence":
                    value = await call("/api/runs/"+args[0]+"/evidence")
                elif action == "status":
                    run_info, trials = await asyncio.gather(call("/api/runs/"+args[0]),
                                                            call("/api/runs/"+args[0]+"/trials"))
                    console.print_json(data={"run": run_info, "trials": trials})
                    continue
                elif action == "validity":
                    value = await call("/api/runs/"+args[0]+"/evidence")
                    table = Table("Trial", "执行", "负载资格", "目标 / 实际", "原因")
                    for trial in value.get("trials", []):
                        validity = trial.get("validity") or {}
                        observed = {name: number for name, number, _unit in validity.get("observations", [])}
                        profile = trial.get("load_profile") or {}
                        target = next((profile[key] for key in ("target_concurrency", "max_active_sessions",
                            "target_request_rate", "target_session_rate") if profile.get(key) is not None), None)
                        table.add_row(str(trial["trial_id"]), str(trial.get("execution_status") or "未知"),
                                      validity.get("status", "未核实"),
                                      f"{target if target is not None else '—'} / {observed.get('actual_pressure', '—')}",
                                      ", ".join(validity.get("reasons", [])) or "—")
                    console.print(table)
                    continue
                elif action in {"metrics", "coverage"}:
                    value = await call("/api/runs/"+args[0]+"/"+action)
                elif action in {"requests", "turn"}:
                    from urllib.parse import urlencode
                    query = {"turn_id": args[1]} if action == "turn" else {"offset": int(args[1]) if len(args)>1 else 0,
                        "page_size": 50, "session_id": args[2] if len(args)>2 else ""}
                    value = await call("/api/runs/"+args[0]+"/"+action+"?"+urlencode(query))
                elif action == "compare":
                    value = await call("/api/compare", {"baseline": args[0], "candidate": args[1],
                        "specification": {"allowed_changes": args[3].split(",") if len(args)>3 else [],
                            "metrics": [{"metric": "e2e", "statistic": "p99",
                            **({"max_regression_ratio": float(args[2])} if len(args)>2 else {})}]}})
                else:
                    console.print("未知命令")
                    continue
                items = value.get("items")
                if action == "compare":
                    console.print(f"直接可比：{value['directly_comparable']} · 负载资格：{value['load_qualified']} · {value['source_content']}")
                    conditions = Table("条件", "基线", "候选", "已核实", "允许变化")
                    for row in value["conditions"]:
                        conditions.add_row(str(row["condition"]), str(row["baseline"]["value"])[:120],
                            str(row["candidate"]["value"])[:120], str(row["verified"]), str(row["allowed_change"]))
                    console.print(conditions)
                    metrics = Table("指标", "基线", "候选", "判定", "原因")
                    for row in value["metrics"]:
                        metrics.add_row(f"{row['metric']}.{row['statistic']}", str(row["baseline"]),
                            str(row["candidate"]), row["decision"], str(row["reason"] or "—"))
                    console.print(metrics)
                    console.print({"repetitions": value.get("repetitions", [])})
                elif items and action in {"runs", "requests"}:
                    keys = ("id", "name", "status") if action == "runs" else ("turn_id", "session_id", "status", "source_branch_id")
                    table = Table(*keys)
                    for item in items:
                        table.add_row(*(str(item.get(k, "不可用")) for k in keys))
                    console.print(table)
                    console.print({key: value[key] for key in ("total", "page", "pagination") if key in value})
                else:
                    console.print_json(data=value)
            except (IndexError, ValueError, aiohttp.ClientError) as exc:
                console.print(str(exc), style="red", markup=False)
