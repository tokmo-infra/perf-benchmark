"""Invoke pinned ServeGen generation with safe local distribution parsing."""

import ast
import base64
import csv
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys

from .common import local_path, positive
from .trace_text import plain_text


UPSTREAM_COMMIT = "d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b"
UPSTREAM_FILES = {
    "construct.py": "68fc2d2424cc53cb1feb03fba3347ef58263c9fe3a771874b7fe165506f6ab61",
    "clientpool.py": "4329eb239d459328e50e440fb82647382c6dac04f5de863bcda73610aaed7f1f",
    "workload_types.py": "ba27f4649a1c47b94e7896d3c864ae866cfe280bf93f4359191dd73c219f26ad",
}


def _official(repo):
    for name, expected in UPSTREAM_FILES.items():
        path = repo / "servegen" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ServeGen source {name} differs from pinned {UPSTREAM_COMMIT}")
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise ValueError("ServeGen requires the production-traces optional dependencies") from None
    sys.path.insert(0, str(repo))
    try:
        clientpool = importlib.import_module("servegen.clientpool")
        construct = importlib.import_module("servegen.construct")
        types = importlib.import_module("servegen.workload_types")
    finally:
        sys.path.pop(0)
    if any(Path(module.__file__).resolve().parent != (repo / "servegen").resolve()
           for module in (clientpool, construct, types)):
        raise ValueError("loaded ServeGen module does not come from the pinned local source")
    return clientpool, construct, types


def _distribution(value):
    # The official loader eval()s this string. Its data is a literal dict; reject code.
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("ServeGen distribution must be a literal dict")
    if not parsed:
        return []
    if not all(type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in parsed.values()):
        raise ValueError("ServeGen distribution probabilities must be finite and nonnegative")
    if abs(sum(parsed.values()) - 1) > 1e-6:
        raise ValueError("ServeGen distribution probabilities must sum to one")
    if all(type(k) is int and k >= 0 for k in parsed):
        if max(parsed) > 1_000_000:
            raise ValueError("ServeGen distribution has too many bins")
        return [parsed.get(index, 0.0) for index in range(max(parsed) + 1)]
    if all(type(k) is float and math.isfinite(k) and 0 <= k <= 1 for k in parsed):
        # Upstream loader uses values() for the ratio PDF; retain that bin order.
        if list(parsed) != sorted(parsed):
            raise ValueError("ServeGen ratio bins must be sorted")
        return list(parsed.values())
    raise ValueError("ServeGen distribution bins must be nonnegative integers or sorted ratios")


def _clients(directory, Client):
    clients, inputs = [], []
    for dataset_path in sorted(directory.glob("chunk-*-dataset.json")):
        identifier = int(dataset_path.name.split("-")[1])
        trace_path = directory / f"chunk-{identifier}-trace.csv"
        if not trace_path.is_file():
            raise ValueError(f"ServeGen lacks {trace_path.name}")
        datasets = json.loads(dataset_path.read_text())
        distribution = {}
        for ts, fields in datasets.items():
            values = {name: _distribution(value) for name, value in fields.items()}
            if values and all(values.values()):
                distribution[int(ts)] = values
        trace = {}
        with trace_path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 6:
                    raise ValueError("ServeGen trace must contain six generation parameters")
                ts, rate, cv, family, shape, scale = row
                if family and family not in {"Gamma", "Weibull"}:
                    raise ValueError("unsupported ServeGen arrival distribution")
                trace[int(ts)] = ({"rate": float(rate), "cv": float(cv), "pat": (family, (float(shape), float(scale)))}
                                  if family else None)
        clients.append(Client(identifier, trace, distribution))
        inputs.extend((dataset_path, trace_path))
    if not clients:
        raise ValueError("ServeGen directory has no paired official chunk files")
    return clients, inputs


def _image_payload(target, processor):
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("mm-image requires Pillow from the production-traces extra") from None
    # Qwen-VL may expand an image to a model-dependent token count. Calibrate the
    # input with the local processor; keep the trace target and local count separate.
    side = max(28, min(2048, round((max(1, target) * 28 * 28) ** 0.5)))
    image = Image.new("RGB", (side, side), (127, 127, 127))
    processed = processor.image_processor(images=image, return_tensors="np")
    grid = processed.get("image_grid_thw")
    local_count = None
    if grid is not None:
        merge = getattr(processor.image_processor, "merge_size", 2)
        local_count = int(grid[0][0] * grid[0][1] * grid[0][2] // (merge * merge))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), local_count


def _generated(config, base, tokenizer):
    options = config.get("servegen")
    if not isinstance(options, dict):
        raise ValueError("ServeGen generator settings are required")
    if set(options) - {"repo", "category", "model", "duration_s", "rate_scale", "endpoint", "processor_path"}:
        raise ValueError("unknown ServeGen generator setting")
    repo = local_path(base, options.get("repo"))
    clientpool, construct, types = _official(repo)
    category = options.get("category")
    if category not in {"language", "reason", "multimodal"}:
        raise ValueError("ServeGen category must be language, reason or multimodal")
    model = options.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("ServeGen model is required")
    data = local_path(base, config.get("input"))
    clients, inputs = _clients(data, clientpool.Client)
    pool = clientpool.ClientPool.from_clients(types.Category(category), model, clients)
    duration = positive(options.get("duration_s"), "duration_s")
    scale = options.get("rate_scale", 1.0)
    if type(scale) not in {int, float} or not math.isfinite(scale) or scale <= 0:
        raise ValueError("rate_scale must be finite and positive")
    view = pool.span(0, duration)
    windows = view.get()
    rate_fn = {ts: sum(window.rate or 0 for window in windows if window.timestamp == ts) * scale
               for ts in {window.timestamp for window in windows}}
    requests = construct.generate_workload(pool, rate_fn, duration, seed=config.get("seed", 0))
    endpoint = options.get("endpoint", "chat")
    if endpoint not in {"chat", "responses"}:
        raise ValueError("ServeGen endpoint must be chat or responses")
    processor = None
    if category == "multimodal":
        if model != "mm-image" or not options.get("processor_path"):
            raise ValueError("only mm-image with a local Qwen-VL processor is supported")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(local_path(base, options["processor_path"]), local_files_only=True,
                                                  trust_remote_code=False)
    upstream = {"repository": "https://github.com/alibaba/ServeGen", "commit": UPSTREAM_COMMIT,
                "algorithm": "servegen.construct.generate_workload", "category": category, "model": model}

    def entries():
        import random
        rng = random.Random(config.get("seed", 0))
        for item in requests:
            data = item.data
            if category == "multimodal":
                if data.get("audio_tokens") or data.get("video_tokens"):
                    raise ValueError("mm-image source yielded unsupported audio/video items")
                text_count = int(data["text_tokens"])
                targets = [int(value) for value in data["image_tokens"]]
                text = plain_text(tokenizer, text_count, rng) if text_count else ""
                images = [_image_payload(value, processor) for value in targets]
                content = ([{"type": "text" if endpoint == "chat" else "input_text", "text": text}]
                           + [{"type": "image_url", "image_url": {"url": uri}} if endpoint == "chat"
                              else {"type": "input_image", "image_url": uri} for uri, _ in images])
                request = ({"messages": [{"role": "user", "content": content}]} if endpoint == "chat"
                           else {"input": [{"type": "message", "role": "user", "content": content}]})
                kind = "chat" if endpoint == "chat" else "responses"
                length = text_count + sum(targets)
                metadata = {"content_kind": "statistically_synthetic", "target_image_tokens": targets,
                            "processor_image_tokens": [count for _, count in images], "source_time_basis": "generated_relative"}
            else:
                length = int(data["input_tokens"])
                text = plain_text(tokenizer, length, rng)
                request, kind = {"prompt": text}, "completion"
                metadata = {"content_kind": "statistically_synthetic", "source_time_basis": "generated_relative"}
                if "reason_ratio" in data:
                    metadata["source_reason_ratio"] = float(data["reason_ratio"])
            yield {"id": f"servegen:{item.request_id}", "session": f"servegen:{item.request_id}",
                   "source_session": None, "created": __import__("datetime").datetime.fromtimestamp(float(item.timestamp), __import__("datetime").timezone.utc).isoformat(),
                   "input": length, "output": int(data["output_tokens"]), "request": request,
                   "kind": kind, "metadata": metadata}
    return entries(), inputs, upstream


def _conversations(config, base, tokenizer):
    path = local_path(base, config.get("input"))
    sessions = json.loads(path.read_text())
    if not isinstance(sessions, list):
        raise ValueError("ServeGen hashed conversations must be a session array")
    def entries():
        import random
        from datetime import datetime, timezone
        rng = random.Random(config.get("seed", 0))
        for session_index, turns in enumerate(sessions):
            if not isinstance(turns, list):
                raise ValueError("ServeGen conversation must contain turns")
            if not turns:
                continue
            graph = f"servegen:conversation:{session_index}"
            root_ns = round(float(turns[0]["timestamp"]) * 1e9)
            for index, turn in enumerate(turns):
                if set(turn) != {"turn", "timestamp", "input_token_count", "output_token_count", "input_tokens", "output_tokens"}:
                    raise ValueError("ServeGen hashed conversation fields differ from official data")
                length = int(turn["input_token_count"])
                identity = f"{graph}:{index}"
                yield {"id": identity,
                       "session": graph, "source_session": str(session_index),
                       "turn_order_key": index, "created": datetime.fromtimestamp(turn["timestamp"], timezone.utc).isoformat(),
                       "input": length, "output": int(turn["output_token_count"]),
                       "request": {"prompt": plain_text(tokenizer, length, rng)}, "kind": "completion",
                       "causal": {"graph_id": graph, "node_id": identity, "branch_id": graph,
                                  "root_ns": root_ns, "not_before_ns": round(float(turn["timestamp"]) * 1e9),
                                  "dependencies": ([{"node_id": f"{graph}:{index - 1}", "event": "completed",
                                                    "delay_ns": None}] if index else [])},
                       "metadata": {"content_kind": "hashed_conversation", "source_time_basis": "utc",
                                    "source_hash_count": len(turn["input_tokens"]), "prefix_blocks_known": False,
                                    "trace_mode": "causal", "dependency_source": "session_turn_order"}}
    return entries(), [path], {"repository": "https://github.com/alibaba/ServeGen", "commit": UPSTREAM_COMMIT,
                                "dataset": "conversations_hashed.json", "hash_replay": "structure_only_unknown_block_size"}


def servegen_requests(config, base, tokenizer):
    return _conversations(config, base, tokenizer) if config.get("mode") == "hashed-conversations" else _generated(config, base, tokenizer)
