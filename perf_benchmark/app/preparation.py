"""Application-owned, result-free service preparation before each Trial."""
import asyncio
from dataclasses import replace
from itertools import chain

import aiohttp

from perf_benchmark.core.models import Trial, freeze_snapshot, thaw_snapshot
from perf_benchmark.core.profiles import Continuation, GENERATION_KEYS
from perf_benchmark.core.validation import positive_int, positive_number
from perf_benchmark.pipeline.context import ExecutorContext
from perf_benchmark.pipeline.contracts.executor import AttemptInput, AttemptPlan
from perf_benchmark.pipeline.contracts.preparation import PreparationFact
from perf_benchmark.pipeline.preparation.streaming import DatasetExhausted
from perf_benchmark.runtime.attempt import AttemptSignal


class PreparationFailed(RuntimeError):
    def __init__(self, fact: PreparationFact):
        self.fact = fact
        super().__init__(fact.failure_reason)


class PreparationCancelled(asyncio.CancelledError):
    def __init__(self, fact: PreparationFact):
        self.fact = fact
        super().__init__(fact.failure_reason)


class ServicePreparation:
    def __init__(self, settings, executor):
        if set(settings) - {"adapter", "exclusive", "warmup_requests", "timeout_seconds"}:
            raise ValueError("unknown service preparation setting")
        if settings.get("adapter") not in {"vllm", "sglang"} or settings.get("exclusive") is not True:
            raise ValueError("service preparation requires adapter vllm/sglang and exclusive=true")
        positive_int(settings.get("warmup_requests", 1), "warmup_requests")
        positive_number(settings.get("timeout_seconds", 60), "preparation timeout_seconds")
        if executor["settings"].get("cache_salt_enabled") is not True:
            raise ValueError("service preparation requires executor.settings.cache_salt_enabled=true")
        self.settings, self.executor = freeze_snapshot(settings), freeze_snapshot(executor)

    async def prepare(self, trial, cancellation, corpus, executor):
        work = asyncio.create_task(self._prepare(trial, cancellation, corpus, executor))
        stopped = asyncio.create_task(cancellation.wait())
        try:
            await asyncio.wait((work, stopped), return_when=asyncio.FIRST_COMPLETED)
            if cancellation.is_cancelled():
                work.cancel()
                try:
                    await work
                except PreparationCancelled:
                    raise
                raise asyncio.CancelledError(cancellation.reason)
            return await work
        finally:
            work.cancel()
            stopped.cancel()
            await asyncio.gather(work, stopped, return_exceptions=True)

    async def _prepare(self, trial, cancellation, corpus, executor):
        settings, target = thaw_snapshot(self.settings), thaw_snapshot(self.executor)["settings"]
        count = settings.get("warmup_requests", 1)
        completed = 0
        attempted = 0
        runtime = None
        try:
            async with asyncio.timeout(settings.get("timeout_seconds", 60)):
                sessions = corpus.warmup_sessions(Trial(id=trial.id), count)
                first = next(sessions)  # Validate capacity before contacting the service.
                base = target["base_url"].rstrip("/")
                headers = executor.request_headers(target)
                timeout = aiohttp.ClientTimeout(total=settings.get("timeout_seconds", 60))
                async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                    connector=aiohttp.TCPConnector(ssl=target.get("verify_ssl", False))) as client:
                    async with client.get(base + "/v1/models") as response:
                        response.raise_for_status()
                        models = await response.json()
                        if target["model"] not in {model.get("id") for model in models.get("data", [])}:
                            raise ValueError("configured model was not reported by service")
                runtime = await executor.create_runtime(trial.executor_trial(), ExecutorContext(cancellation))
                await runtime.prepare()
                for session in chain((first,), sessions):
                    history = {}
                    try:
                        for turn in session.turns:
                            if cancellation.is_cancelled():
                                raise asyncio.CancelledError(cancellation.reason)
                            turn.start()
                            prepared = await runtime.prepare_attempt(AttemptInput.capture(
                                session, turn, history, attempt=1, payloads=corpus.payloads,
                            ))
                            try:
                                attempted += 1
                                result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
                            finally:
                                settlement = await runtime.release(prepared)
                                if settlement is not None:
                                    await settlement
                            if result.status != "succeeded" or not result.continuation_valid:
                                raise ValueError(
                                    f"warmup request {completed + 1} failed: HTTP {result.request.http_status}"
                                    if result.request.http_status is not None else
                                    f"warmup request {completed + 1} failed: {result.status}"
                                )
                            turn.succeed()
                            completed += 1
                            if result.continuation_payload is not None:
                                history[turn.id] = Continuation(result.continuation_payload, result.continuation_profile)
                    finally:
                        history.clear()
                        await runtime.close_session(session.id)
        except asyncio.CancelledError as exc:
            fact = PreparationFact(settings["adapter"], count, completed,
                                   "salt_attempted_backend_unverified" if attempted else "not_sent",
                                   failure_reason=cancellation.reason or "cancelled")
            raise PreparationCancelled(fact) from exc
        except Exception as exc:
            reason = str(exc) if isinstance(exc, DatasetExhausted) or (
                isinstance(exc, ValueError) and str(exc).startswith("warmup request ")
            ) else f"{type(exc).__name__}: preparation failed after {completed} requests"
            fact = PreparationFact(settings["adapter"], count, completed,
                                   "salt_attempted_backend_unverified" if attempted else "not_sent",
                                   failure_reason=reason)
            raise PreparationFailed(fact) from exc
        finally:
            if runtime is not None:
                await runtime.close()
        return PreparationFact(settings["adapter"], count, completed, "salt_sent_backend_unverified")


class ApplicationPreparation:
    """Associate imported quality with effective generation settings, then prepare service."""
    def __init__(self, service, quality):
        self.service, self.quality = service, freeze_snapshot(quality)

    async def prepare(self, trial, cancellation, corpus, executor):
        quality = thaw_snapshot(self.quality)
        assert isinstance(quality, dict)
        status = quality["status"]
        if status in {"pass", "fail"}:
            expected = quality.get("identity", {}).get("generation", {})
            if any(key not in GENERATION_KEYS or expected.get(key) != value for key, value in trial.request_parameters.items()):
                status = "unverified"
        try:
            fact = (await self.service.prepare(trial, cancellation, corpus, executor) if self.service is not None
                    else PreparationFact("none", 0, 0, "unknown"))
        except PreparationFailed as exc:
            exc.fact = replace(exc.fact, quality_status=status)
            raise
        except PreparationCancelled as exc:
            exc.fact = replace(exc.fact, quality_status=status)
            raise
        return replace(fact, quality_status=status)
