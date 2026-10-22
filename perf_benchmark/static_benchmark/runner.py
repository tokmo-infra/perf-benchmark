"""Static workload orchestration over the existing eight-stage native pipeline."""
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from urllib.parse import urlsplit
from rich.console import Console
from perf_benchmark.app.main import execute_config
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver
from perf_benchmark.static_benchmark.metrics import AggregatedTestResult
from perf_benchmark.static_benchmark.prompts import get_decode_prompt
from perf_benchmark.static_benchmark.token_sampler import VocabularyTokenSampler

console = Console()


class StaticBenchmarkRunner:
    def __init__(self, config):
        self.config = config
        self.sampler = None
        self.prefill_results = []
        self.decode_results = []
        self.result_dir = Path(config.output.excel_path).with_suffix("").with_name(
            Path(config.output.excel_path).stem + "-facts")
        self._started = False
        self._exported = (0, 0)

    def _start(self):
        if not self._started:
            if self.config.output.json_path and Path(self.config.output.json_path).exists():
                raise FileExistsError("choose a new JSON output path")
            self.result_dir.mkdir(parents=True, exist_ok=True)
            self.result_dir = Path(mkdtemp(prefix="run-", dir=self.result_dir))
            self._started = True

    def _checkpoint(self):
        from perf_benchmark.static_benchmark.excel_exporter import export_benchmark_to_excel
        counts = (len(self.prefill_results), len(self.decode_results))
        if counts != self._exported:
            export_benchmark_to_excel(
                self.config.output.excel_path, self.config,
                self.prefill_results[self._exported[0]:], self.decode_results[self._exported[1]:],
                include_config=self._exported == (0, 0))
            self._exported = counts
        if self.config.output.json_path:
            path = Path(self.config.output.json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Measurements and availability only; no response bodies or credentials.
            path.write_text(json.dumps([asdict(r) for r in (*self.prefill_results, *self.decode_results)],
                                       ensure_ascii=False, indent=2)+"\n")

    def _pipeline(self, folder, source, concurrency, count, *, prefill):
        service = self.config.service
        point = self.config.prefill if prefill else self.config.decode
        url = urlsplit(service.normalized_url())
        extra = dict(service.extra_body, temperature=point.temperature)
        if not prefill:
            extra["ignore_eos"] = point.ignore_eos
            if point.reasoning_effort is not None:
                extra["reasoning_effort"] = point.reasoning_effort
        executor = dict(base_url=f"{url.scheme}://{url.netloc}", endpoint=url.path,
                        model=service.model, stream=True, timeout_seconds=service.timeout_seconds,
                        max_tokens=point.max_tokens, extra_body=extra, extra_headers=service.extra_headers,
                        cache_salt_enabled=service.cache_salt_enabled, connection_scope="trial")
        if service.api_key and service.api_key != "EMPTY":
            if any(k.lower() == "authorization" for k in service.extra_headers):
                raise ValueError("api_key conflicts with Authorization extra header")
            executor["extra_headers"] = dict(service.extra_headers, Authorization="Bearer " + service.api_key)
        return dict(case=self.config.case+" / "+folder.name,
                    name=self.config.name+"-"+folder.name, seed=self.config.prefill.seed,
                    application={"service_capacity": {"snapshot": service.capacity_snapshot}
                                 if service.capacity_snapshot else {}}, pipeline={
            "loader": {"plugin": "hf-datasets", "settings": {"dataset": "json", "data_files": str(source),
                                                               "split": "train"}},
            "normalization": {"plugin": "openai-messages", "settings": {
                "tokenizer_path": service.tokenizer_path}},
            "selector": {"plugin": "all", "settings": {}},
            "executor": {"plugin": "openai-chat", "settings": executor},
            "scheduler": {"plugin": "single-turn-closed-loop", "settings": {
                "max_concurrency": concurrency, "stop": {"session_count": count}, "on_exhausted": "error"}},
            "metrics": {"plugin": "standard", "settings": {}},
            "saver": {"plugin": "sqlite", "settings": {"path": str(folder / "run.db")}},
            "exporter": {},
        })

    async def _point(self, kind, concurrency, input_tokens=None):
        self._start()
        prefill = kind == "prefill"
        point = self.config.prefill if prefill else self.config.decode
        per_round = point.requests_per_concurrency or concurrency
        if per_round < concurrency:
            raise ValueError("requests_per_concurrency must be at least concurrency")
        count = point.rounds * per_round
        name = f"prefill-{input_tokens}-c{concurrency}" if prefill else f"decode-c{concurrency}"
        folder = self.result_dir / name
        folder.mkdir()
        rows = []
        point_seed = int.from_bytes(hashlib.sha256(
            f"{self.config.prefill.seed}:{name}".encode()).digest()[:8], "little")
        recipe = dict(stage=kind, concurrency=concurrency, rounds=point.rounds,
                      requests_per_round=per_round, count=count, input_tokens=input_tokens,
                      seed=point_seed, request_body_hashes=[])
        for index in range(count):
            if prefill:
                seed = point_seed + index * 17 if point.randomize_per_request else self.config.prefill.seed
                text = await asyncio.to_thread(self.sampler.sample_prompt, input_tokens, seed,
                                                adjust_for_chat_template=True)
                messages = [{"role": "user", "content": text}]
            else:
                text, system = get_decode_prompt(index, point.prompts)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
            row = dict(id=f"{name}-{index}", token_name=f"{name}-{index}", messages=messages,
                       created=index)
            rows.append(row)
            recipe["request_body_hashes"].append(hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
        recipe["tokenizer_fingerprint"] = self.sampler.fingerprint
        (folder / "workload.json").write_text(json.dumps(recipe, indent=2)+"\n")
        console.print(f"{name}: {point.rounds} rounds × {per_round} requests = {count}, "
                      f"concurrency {concurrency}, continuous native closed loop")
        with TemporaryDirectory(prefix=".static-input-", dir=self.result_dir) as temporary:
            source = Path(temporary) / "input.jsonl"
            source.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
            rows.clear()
            config = self._pipeline(folder, source, concurrency, count, prefill=prefill)
            run = await execute_config(config, folder)
        reader = await asyncio.to_thread(SQLiteSaver.open_reader, str(folder / "run.db"))
        result = await asyncio.to_thread(AggregatedTestResult.from_reader, reader, run.id,
                                         test_type=kind, concurrency=concurrency, input_tokens=input_tokens,
                                         rounds=point.rounds, requests_per_round=per_round)
        result.database = str(folder / "run.db")
        (self.prefill_results if prefill else self.decode_results).append(result)
        await asyncio.to_thread(self._checkpoint)
        # The combined workbook now contains the presentation; retain native facts,
        # not a second derived Excel for every matrix point.
        derived = folder / "run-tables/results.xlsx"
        derived.unlink()
        derived.parent.rmdir()
        return result

    async def run_prefill_benchmark(self, on_progress=None):
        if not self.config.prefill.enabled:
            return []
        if self.sampler is None:
            self.sampler = await asyncio.to_thread(
                VocabularyTokenSampler, self.config.service.tokenizer_path,
                base_dir=self.config.config_dir,
                template_options=self.config.service.extra_body.get("chat_template_kwargs"))
        output = []
        for length, concurrencies in sorted(self.config.prefill.matrix.items()):
            for concurrency in concurrencies:
                result = await self._point("prefill", concurrency, length)
                output.append(result)
                if on_progress:
                    on_progress(result)
                if result.status == "cancelled":
                    return output
        return output

    async def run_decode_benchmark(self, on_progress=None):
        if not self.config.decode.enabled:
            return []
        if self.sampler is None:
            self.sampler = await asyncio.to_thread(
                VocabularyTokenSampler, self.config.service.tokenizer_path,
                base_dir=self.config.config_dir,
                template_options=self.config.service.extra_body.get("chat_template_kwargs"))
        output = []
        for concurrency in self.config.decode.concurrencies:
            result = await self._point("decode", concurrency)
            output.append(result)
            if on_progress:
                on_progress(result)
            if result.status == "cancelled":
                return output
        return output

    async def run_all(self):
        await self.run_prefill_benchmark()
        if not any(r.status == "cancelled" for r in self.prefill_results):
            await self.run_decode_benchmark()
        return self.prefill_results, self.decode_results
