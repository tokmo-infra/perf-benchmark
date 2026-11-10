"""Python lifecycle bridge to the in-process Rust HTTP executor."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from time import perf_counter_ns

from perf_benchmark.pipeline.contracts.executor import AttemptResult
from perf_benchmark.pipeline.results.measurement import ResponseTrace
from perf_benchmark.pipeline.resources import finish_task
from perf_benchmark.runtime.http.request import RequestRuntime


class RustHttpExecutorRuntime(RequestRuntime):
    def __init__(self, plugin, settings, cancellation, protocol: str):
        super().__init__(plugin, settings, cancellation)
        self.protocol = protocol
        self.cancellation = cancellation
        self.native = None
        self._background = set()
        self._background_errors = []
        self._opened = False
        self.safety_observer = None

    async def prepare(self):
        if self._opened or self._close_task is not None:
            raise RuntimeError("Rust runtime must be prepared exactly once")
        self._opened = True
        from perf_benchmark import _native
        for _ in range(3):
            before = perf_counter_ns()
            at = _native.now()
            if not before <= at <= perf_counter_ns():
                raise RuntimeError("Rust and Python do not share the same monotonic clock")
        self.native = _native.create_trial(self.protocol)

    async def _prepare(self, value, spec):
        from perf_benchmark import _native
        prepared = await super()._prepare(value, spec)
        try:
            prepared.native_body = _native.PreparedBody(prepared.body.segments, prepared.body.content_length)
            return prepared
        except BaseException:
            await self.release(prepared)
            raise

    async def execute_attempt(self, prepared, signal):
        task = asyncio.current_task()
        assert task is not None
        signal.bind(task.cancel)
        join_settlement = False
        try:
            if self.native is None or self._close_task is not None:
                raise RuntimeError("Rust runtime is not open")
            from perf_benchmark import _native
            spec = prepared.spec
            headers = {"Content-Type": prepared.body.content_type, **spec.headers}
            request_id = next((value for name, value in headers.items()
                               if name.lower() == "x-request-id"), None)
            if request_id is None:
                request_id = f"{prepared.input.turn.id}:a{prepared.input.attempt}"
                headers["X-Request-ID"] = request_id
            timeout = self.settings.get("timeout_seconds", 300)
            try:
                handle = self.native.submit(prepared.native_body, {
                    "url": spec.url, "method": spec.method,
                    "notify_start": signal.started_callback is not None,
                    "expectations": spec.response_expectations,
                    "headers": headers,
                    "body_length": prepared.body.content_length, "stream": spec.streaming,
                    "retain": spec.retain_output, "timeout": 0.0 if timeout is None else timeout,
                    "plan": asdict(signal.plan),
                    "session_id": prepared.input.session.id,
                    "connection_scope": self.settings.get("connection_scope", "session"),
                    "verify_ssl": self.settings.get("verify_ssl", False),
                }, cancelled=signal.reason is not None or task.cancelling() > 0)
            except _native.SafetyAdmissionRejected:
                # The native safety latch is authoritative even before Python
                # receives the overdue request. This Attempt was never submitted;
                # no transport timestamps or TTFT proof exist for it.
                self.cancellation.cancel("ttft_watchdog_safety_abort")
                return AttemptResult(
                    prepared.input.turn.id, prepared.input.attempt, status="cancelled",
                    error="native_safety_admission_rejected", error_origin="client",
                    request_facts=prepared.request_facts, continuation_valid=False,
                    transport_timing_contract="not-dispatched-v1",
                )
            prepared.native_attempt = handle
            operation = asyncio.ensure_future(handle.result())
            started = (
                asyncio.create_task(self._notify_start(handle, signal))
                if signal.started_callback is not None else None
            )
            try:
                value = await asyncio.shield(operation)
                if started is not None:
                    await finish_task(started)
            except asyncio.CancelledError:
                join_settlement = True
                handle.cancel()
                try:
                    value = await finish_task(operation)
                except asyncio.CancelledError:
                    value = operation.result()
                if started is not None:
                    try:
                        await finish_task(started)
                    except asyncio.CancelledError:
                        started.result()
            if (value.error_origin == 'safety' and value.error == 'TTFTWatchdogExceeded'
                    and value.http_status != 500):
                observer = getattr(self, 'safety_observer', None)
                if observer is None:
                    self.cancellation.cancel('ttft_watchdog_safety_abort')
                else:
                    first = value.samples[0][1] if value.samples else None
                    observer(dict(request_id=request_id,
                        turn_id=prepared.input.turn.id, attempt=prepared.input.attempt,
                        sent_ns=value.transport_sent_ns, first_token_ns=first,
                        first_token_pending=first is None, detected_ns=value.completed_ns,
                        clock='shared_native_monotonic',
                        basis='transport_dispatch_after_pool_acquire; excludes_tool_queue',
                        native_client_start_ns=value.started_ns, reason='TTFTWatchdogExceeded'))
            result = AttemptResult(prepared.input.turn.id, prepared.input.attempt,
                                   request_facts=prepared.request_facts)
            result.status = value.status
            result.transport_sent_ns = value.transport_sent_ns
            result.transport_timing_contract = 'sent-dispatch-first-output-v1'
            result.error, result.error_origin = value.error, value.error_origin
            result.retryable = value.retryable
            result.usage = value.usage
            result.continuation_valid = value.continuation_valid
            continuation = value.continuation
            result.continuation_payload = continuation
            result.trace = ResponseTrace(result.turn_id, value.samples, result.attempt)
            for key in ("trigger_ns", "decision_ns", "applied_ns"):
                setattr(result, key, getattr(value, key))
            request = result.request
            request.server_metrics = value.server_metrics
            request.attempts, request.response_mode = result.attempt, "stream" if spec.streaming else "non_stream"
            request.started_ns, request.response_completed_ns = value.started_ns, value.completed_ns
            request.http_status, request.output_chars = value.http_status, value.output_chars
            request.finish_reason = value.finish_reason
            request.output_kind = value.output_kind
            if result.trace.samples:
                request.first_output_ns, request.last_output_ns = result.trace.samples[0][1], result.trace.samples[-1][1]
                request.output_event_count = len(result.trace.samples)
            if result.error_origin == "client" and signal.reason is not None:
                result.error, result.error_origin = signal.reason, signal.origin
                result.trigger_ns, result.decision_ns = signal.trigger_ns, signal.decision_ns
                result.status = "failed" if signal.action == "fail" else "cancelled"
                result.retryable = signal.action == "fail" and result.recoverable
            return result
        except Exception:
            join_settlement = True
            self.cancellation.cancel("native execution failure")
            raise
        finally:
            signal.close()
            # Retain an already returned real HTTP/safety outcome even when the
            # whole-trial watcher cancels us during cleanup. Join, never replace
            # the true Attempt with an artificial successful/cancelled result.
            release = asyncio.create_task(self.release(prepared))
            try:
                settlement = await finish_task(release)
            except asyncio.CancelledError:
                settlement = release.result()
            if join_settlement and settlement is not None:
                try:
                    await finish_task(settlement)
                except asyncio.CancelledError:
                    settlement.result()

    async def _notify_start(self, handle, signal):
        try:
            at = await handle.started()
            if at is not None:
                signal.started_callback(at)
        except BaseException:
            handle.cancel()
            raise

    def _track(self, task):
        self._background.add(task)
        def finished(value):
            self._background.discard(value)
            if not value.cancelled() and (error := value.exception()) is not None:
                self._background_errors.append(error)
                self.cancellation.cancel("native cleanup failure")
        task.add_done_callback(finished)
        return task

    async def release(self, prepared):
        identity = id(prepared)
        if prepared.release_done:
            return None
        if prepared.release_task is not None:
            return prepared.release_task
        if prepared.native_attempt is None:
            await super().release(prepared)
            prepared.release_done = True
            return None
        self._prepared.pop(identity, None)
        if prepared.native_attempt.settled_done:
            await prepared.close()
            prepared.release_done = True
            return None
        async def finish():
            try:
                await prepared.native_attempt.settled()
            finally:
                await prepared.close()
                prepared.release_done = True
        task = self._track(asyncio.create_task(finish()))
        prepared.release_task = task
        return task

    async def close_session(self, session_id: str):
        if self.native is not None:
            self._track(asyncio.ensure_future(self.native.close_session(session_id)))

    async def _close(self):
        try:
            if self.native is not None:
                await finish_task(asyncio.ensure_future(self.native.close()))
        finally:
            try:
                await super()._close()
            finally:
                if self._background:
                    await finish_task(asyncio.gather(*tuple(self._background), return_exceptions=True))
        if self._background_errors:
            raise self._background_errors[0]
