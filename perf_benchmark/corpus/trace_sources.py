"""Read the published source schemas; no request synthesis occurs here."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < (1 if positive else 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _time(value, scale=1, utc=False):
    if utc:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    return datetime.fromtimestamp(_number(value, "timestamp") / scale, timezone.utc).isoformat()


def raw_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()
    elif path.suffix in {".jsonl", ".json"}:
        with path.open() as stream:
            if path.suffix == ".json":
                value = json.load(stream)
                if isinstance(value, dict) and "rows" in value:  # official HF viewer download
                    yield from (entry["row"] for entry in value["rows"])
                elif isinstance(value, list):
                    yield from value
                else:
                    yield value
            else:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
    elif path.suffix == ".csv":
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream)
    else:
        raise ValueError("trace input must be official JSONL, JSON, CSV, or Parquet")


def source_requests(source, path, *, root_interval_s=None, mode="timeline"):
    if source == "servegen":
        raise ValueError("ServeGen chunk-*-trace.csv contains generation parameters, not requests")
    if source == "agentx":
        if root_interval_s is None or root_interval_s <= 0:
            raise ValueError("AgentX requires positive synthetic root_interval_s")
        identities = set()
        for root_index, row in enumerate(raw_rows(path)):
            if row.get("id") in identities:
                raise ValueError("duplicate AgentX root identity")
            identities.add(row.get("id"))
            yield from _agentx(row, root_index, root_interval_s, mode)
        return
    for index, row in enumerate(raw_rows(path)):
        identity = f"{source}:{index}"
        if source == "qwen-bailian":
            required = {"chat_id", "parent_chat_id", "timestamp", "input_length", "output_length", "type", "turn", "hash_ids"}
            if set(row) != required:
                raise ValueError("Bailian source fields do not match the official schema")
            meta = {"source_block_size": 16, "source_time_basis": "trace_relative", "source_type": row["type"],
                    "source_turn": row["turn"], "source_parent_id": str(row["parent_chat_id"])}
            yield dict(id=identity, session=identity, source_session=str(row["chat_id"]),
                       created=_time(row["timestamp"]), input=row["input_length"], output=row["output_length"],
                       hashes=row["hash_ids"], metadata=meta)
        elif source == "mooncake":
            if set(row) != {"timestamp", "input_length", "output_length", "hash_ids"}:
                raise ValueError("Mooncake source fields do not match the official schema")
            yield dict(id=identity, session=identity, source_session=None,
                       created=_time(row["timestamp"], 1000), input=row["input_length"],
                       output=row["output_length"], hashes=row["hash_ids"],
                       metadata={"source_block_size": 512, "source_time_basis": "trace_relative"})
        elif source == "burstgpt":
            old = {"Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"}
            new = old | {"Session ID", "Elapsed time"}
            schema = "v2" if set(row) == new else "legacy" if set(row) == old else None
            if schema is None:
                raise ValueError("BurstGPT CSV columns do not match legacy or v2 release")
            session = row.get("Session ID") if schema == "v2" and "conversation" in row["Log Type"].lower() else None
            yield dict(id=identity, session=identity, source_session=session,
                       created=_time(float(row["Timestamp"])), input=int(row["Request tokens"]),
                       output=int(row["Response tokens"]), hashes=None,
                       metadata={"source_schema": schema, "source_model": row["Model"], "source_log_type": row["Log Type"],
                                 "source_time_basis": "trace_relative", "source_elapsed_s": row.get("Elapsed time")})
        elif source == "azure-2024":
            if set(row) != {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}:
                raise ValueError("Azure 2024 CSV columns do not match the one-week release")
            yield dict(id=identity, session=identity, source_session=None,
                       created=_time(row["TIMESTAMP"], utc=True), input=int(row["ContextTokens"]),
                       output=int(row["GeneratedTokens"]), hashes=None,
                       metadata={"source_time_basis": "utc"})
        else:
            raise ValueError(f"unknown production trace source: {source}")


def _agentx(row, root_index, interval, mode):
    if set(row) != {"id", "models", "block_size", "hash_id_scope", "requests"}:
        raise ValueError("AgentX row does not match WEKA schema")
    if row["block_size"] != 64 or row["hash_id_scope"] != "local":
        raise ValueError("AgentX requires local 64-token hashes")
    root = str(row["id"])
    origin = root_index * interval
    root_ns = round(origin * 1e9)

    def walk(requests, branch, incoming=()):
        previous, last, pending = None, None, list(incoming)
        rows = []
        for index, request in enumerate(requests):
            at = _number(request["t"], "AgentX request t")
            if previous is not None and at < previous:
                raise ValueError("AgentX branch timestamps are not ordered")
            previous = at
            identity = f"agentx:{root}:{branch}:{index}"
            if request["type"] == "subagent":
                launch = list(pending)
                if last is not None:
                    parent, source = last
                    end = source["t"] + source["api_time"] if source.get("api_time") is not None else None
                    event = "completed" if end is not None and end <= at else "started"
                    anchor = end if event == "completed" else source["t"]
                    launch.append({"node_id": parent, "event": event, "delay_ns": round((at - anchor) * 1e9)})
                nested, terminals = walk(request["requests"], branch + "/" + str(request["agent_id"]), tuple(launch))
                rows.extend(nested)
                if request.get("status") != "async_launched":
                    pending.extend({"node_id": node, "event": "completed", "delay_ns": None} for node in terminals)
                continue
            if request["type"] not in {"n", "s"}:
                raise ValueError("unsupported AgentX request type")
            edges = list(pending)
            if last is not None:
                parent, source = last
                gap = request.get("think_time")
                if gap is None and source.get("api_time") is not None:
                    gap = at - source["t"] - source["api_time"]
                    if gap < 0:
                        gap = None
                edges.append({"node_id": parent, "event": "completed", "delay_ns": round(gap * 1e9) if gap is not None else None})
            causal = {"graph_id": root, "node_id": identity, "branch_id": branch, "root_ns": root_ns,
                      "not_before_ns": root_ns + round(at * 1e9), "dependencies": edges} if mode == "causal" else None
            metadata = {"source_block_size": 64, "hash_id_scope": "local", "source_time_basis": "synthetic_root_arrivals",
                        "source_relative_s": at, "source_model": request.get("model"), "source_branch_id": branch,
                        "trace_mode": mode, "dependency_source": "reconstructed_weka" if mode == "causal" else None}
            rows.append(dict(id=identity, session=f"agentx:{root}:{branch}" if mode == "causal" else identity,
                             source_session=root, created=_time(origin + at), input=request["in"],
                             output=request["out"], hashes=request["hash_ids"], metadata=metadata,
                             causal=causal, turn_order_key=index, hash_scope=root))
            last, pending = (identity, request), []
        terminals = [edge["node_id"] for edge in pending]
        if last is not None:
            terminals.append(last[0])
        return rows, terminals

    rows, _ = walk(row["requests"], "main")
    yield from rows
