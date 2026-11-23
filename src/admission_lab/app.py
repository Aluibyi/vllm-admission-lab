from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import math
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import yaml

from .scheduler import AdmissionError, Scheduler
from .sse import SSEParser, content_event
from .telemetry import Recorder


DEFAULT = {
    "backend_url": "http://127.0.0.1:8000", "model": "admission-7b",
    "policy": "cost_aging", "max_inflight": 16,
    "work_budget_ms": 200000.0, "queue_limit": 256,
    "tau_ms": 500.0, "aging_threshold_ms": 5000.0,
    "queue_timeout_ms": 30000.0, "backend_timeout_seconds": 120.0,
    "max_prompt_tokens": 8192, "max_output_tokens": 512,
    "cost_intercept_ms": 10.0, "prompt_ms_per_token": 0.1,
    "output_ms_per_token": 10.0,
    "cancel_path_verified": False,
    "event_log": "artifacts/gateway_events.jsonl",
}


def estimate(config, prompt_tokens, max_output_tokens):
    return (config["cost_intercept_ms"] +
            config["prompt_ms_per_token"] * prompt_tokens +
            config["output_ms_per_token"] * max_output_tokens)


class ManagedStream(StreamingResponse):
    def __init__(self, iterator, cleanup, **kwargs):
        super().__init__(iterator, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also covers disconnect before the generator's first iteration.
            task = asyncio.create_task(self.cleanup())
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)


def create_app(config=None, transport=None):
    settings = dict(DEFAULT)
    config_path = os.environ.get("ADMISSION_CONFIG")
    if config_path:
        with open(config_path, encoding="utf-8") as handle:
            settings.update(yaml.safe_load(handle) or {})
    settings.update(config or {})
    worst = estimate(settings, settings["max_prompt_tokens"], settings["max_output_tokens"])
    if not 0 < worst <= settings["work_budget_ms"]:
        raise ValueError("Work budget cannot accommodate maximum accepted request")

    @asynccontextmanager
    async def lifespan(app):
        app.state.scheduler = Scheduler(**{k: settings[k] for k in (
            "policy", "max_inflight", "work_budget_ms", "queue_limit",
            "tau_ms", "aging_threshold_ms", "queue_timeout_ms")})
        app.state.recorder = Recorder(settings["event_log"])
        app.state.paused = False
        app.state.cleanup_tasks = set()
        app.state.client = httpx.AsyncClient(
            base_url=settings["backend_url"], transport=transport,
            timeout=httpx.Timeout(settings["backend_timeout_seconds"], connect=10, pool=10),
            limits=httpx.Limits(max_connections=max(128, settings["max_inflight"] * 2),
                               max_keepalive_connections=max(32, settings["max_inflight"])))
        await app.state.recorder.start()
        await app.state.scheduler.start()
        async def sample():
            while True:
                app.state.recorder.event("snapshot", **await app.state.scheduler.snapshot())
                await asyncio.sleep(0.5)
        sampler = asyncio.create_task(sample())
        try:
            yield
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            await app.state.scheduler.close()
            if app.state.cleanup_tasks:
                await asyncio.gather(*list(app.state.cleanup_tasks), return_exceptions=True)
            await app.state.client.aclose()
            await app.state.recorder.close()

    app = FastAPI(title="vLLM Admission Lab", lifespan=lifespan)
    app.state.config = settings

    @app.get("/health")
    async def health():
        status = 503 if app.state.paused else 200
        return JSONResponse({"ok": not app.state.paused,
                             "log_dropped": app.state.recorder.dropped}, status_code=status)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": settings["model"], "object": "model"}]}

    @app.get("/debug/state")
    async def debug_state():
        return {**await app.state.scheduler.snapshot(), "paused": app.state.paused,
                "log_dropped": app.state.recorder.dropped,
                "cancel_path_verified": settings["cancel_path_verified"]}

    @app.delete("/requests/{request_id}")
    async def cancel_request(request_id: str):
        return {"request_id": request_id,
                "cancel_requested": await app.state.scheduler.cancel(request_id)}

    @app.post("/v1/completions")
    async def completions(request: Request):
        received = time.perf_counter()
        if app.state.paused:
            return JSONResponse({"error": "paused_after_unverified_cancel"}, status_code=503)
        try:
            body = await request.json()
            prompt = body.get("prompt")
            if not isinstance(prompt, list) or not prompt or any(
                    not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in prompt):
                raise ValueError("P0 API requires one nonempty token-ID prompt")
            output = body.get("max_tokens", 128)
            if not isinstance(output, int) or isinstance(output, bool) or not 1 <= output <= settings["max_output_tokens"]:
                raise ValueError("max_tokens outside configured range")
            if len(prompt) > settings["max_prompt_tokens"]:
                raise ValueError("prompt exceeds configured range")
            if body.get("model") != settings["model"]:
                raise ValueError("unknown model")
            if body.get("n", 1) != 1 or body.get("stream") is not True:
                raise ValueError("P0 requires n=1, stream=true")
            request_id = body.get("request_id") or request.headers.get("x-request-id") or str(uuid.uuid4())
            if not isinstance(request_id, str) or len(request_id) > 160:
                raise ValueError("invalid request_id")
            cost = estimate(settings, len(prompt), output)
            if not math.isfinite(cost):
                raise ValueError("nonfinite request cost")
            body["request_id"] = request_id
            body["priority"] = 0
        except (ValueError, TypeError, AttributeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

        scheduler = app.state.scheduler
        recorder = app.state.recorder
        try:
            entry = await scheduler.submit(request_id, cost)
        except AdmissionError as exc:
            recorder.event("rejected", request_id=request_id, reason=exc.reason)
            return JSONResponse({"error": exc.reason}, status_code=exc.status)
        recorder.event("enqueued", request_id=request_id, prompt_tokens=len(prompt),
                       max_output_tokens=output, cost_ms=cost,
                       validation_ms=(time.perf_counter() - received) * 1000)
        upstream = None
        open_task = None
        monitor = None
        next_task = None
        cancel_task = None
        state = {"terminal": "failed", "reason": "incomplete_stream", "done": False,
                 "first": None, "cleanup_done": False, "forwarded": False,
                 "upstream_terminal": False}
        cleanup_lock = asyncio.Lock()

        async def cleanup():
            async with cleanup_lock:
                if state["cleanup_done"]:
                    return
                state["cleanup_done"] = True
                for task in (monitor, next_task, cancel_task, open_task):
                    if task and not task.done():
                        task.cancel()
                pending = [t for t in (monitor, next_task, cancel_task, open_task) if t]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if (state["forwarded"] and not state["done"] and
                        not state["upstream_terminal"] and not settings["cancel_path_verified"]):
                    app.state.paused = True
                    await scheduler.pause()
                try:
                    response_to_close = upstream
                    if response_to_close is None and open_task and not open_task.cancelled():
                        if open_task.done() and open_task.exception() is None:
                            response_to_close = open_task.result()
                    if response_to_close is not None:
                        await response_to_close.aclose()
                finally:
                    await scheduler.release(entry, state["terminal"])
                recorder.event("terminal", request_id=request_id,
                               status=state["terminal"], reason=state["reason"],
                               duration_ms=(time.perf_counter() - received) * 1000,
                               queue_ms=((entry.admitted or time.perf_counter()) - entry.enqueued) * 1000,
                               cancel_unverified=app.state.paused)

        async def safe_cleanup():
            task = asyncio.create_task(cleanup())
            app.state.cleanup_tasks.add(task)
            task.add_done_callback(app.state.cleanup_tasks.discard)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)

        async def watch_disconnect():
            # Only receive consumer before StreamingResponse takes ownership.
            while True:
                if await request.is_disconnected():
                    await scheduler.cancel(request_id)
                    return
                await asyncio.sleep(0.025)

        try:
            monitor = asyncio.create_task(watch_disconnect())
            await asyncio.shield(entry.future)
            if entry.cancelled.is_set():
                raise AdmissionError("cancelled", 499)
            recorder.event("admitted", request_id=request_id,
                           queue_ms=(entry.admitted - entry.enqueued) * 1000)
            upstream_request = app.state.client.build_request("POST", "/v1/completions",
                                                             json=body, headers={"x-request-id": request_id})
            state["forwarded"] = True
            open_task = asyncio.create_task(app.state.client.send(upstream_request, stream=True))
            cancel_task = asyncio.create_task(entry.cancelled.wait())
            done, _ = await asyncio.wait([open_task, cancel_task], return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done:
                open_task.cancel()
                result = await asyncio.gather(open_task, return_exceptions=True)
                if result and isinstance(result[0], httpx.Response):
                    upstream = result[0]
                raise AdmissionError("cancelled", 499)
            upstream = await open_task
            state["forwarded"] = True
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            if upstream.status_code >= 400:
                state["upstream_terminal"] = True
                error_status = upstream.status_code
                state["reason"] = f"upstream_http_{error_status}"
                await safe_cleanup()
                return JSONResponse({"error": state["reason"]}, status_code=error_status)
        except AdmissionError as exc:
            state.update(terminal="cancelled" if exc.reason == "cancelled" else "failed", reason=exc.reason)
            await safe_cleanup()
            return JSONResponse({"error": exc.reason}, status_code=exc.status)
        except asyncio.CancelledError:
            state.update(terminal="cancelled", reason="client_disconnected_before_stream")
            await safe_cleanup()
            raise
        except Exception as exc:
            state["reason"] = f"upstream_connect_{type(exc).__name__}"
            await safe_cleanup()
            return JSONResponse({"error": state["reason"]}, status_code=502)

        async def stream():
            nonlocal next_task
            parser = SSEParser()
            iterator = upstream.aiter_raw().__aiter__()
            try:
                while True:
                    next_task = asyncio.create_task(iterator.__anext__())
                    done, _ = await asyncio.wait([next_task, cancel_task], return_when=asyncio.FIRST_COMPLETED)
                    if cancel_task in done:
                        state.update(terminal="cancelled", reason="explicit_cancel")
                        break
                    try:
                        chunk = next_task.result()
                    except StopAsyncIteration:
                        break
                    for event in parser.feed(chunk):
                        if event == "[DONE]":
                            state["done"] = True
                        else:
                            _, content = content_event(event)
                            if content and state["first"] is None:
                                state["first"] = time.perf_counter()
                                recorder.event("first_content", request_id=request_id,
                                               from_receive_ms=(state["first"] - received) * 1000)
                    yield chunk
                if state["done"]:
                    state.update(terminal="succeeded", reason="done")
            except asyncio.CancelledError:
                state.update(terminal="cancelled", reason="client_disconnected_streaming")
                raise
            except Exception as exc:
                state.update(terminal="failed", reason=f"stream_{type(exc).__name__}")
                # HTTP headers are already sent. Terminate the stream; the
                # load generator requires DONE and usage to count success.
            finally:
                await safe_cleanup()

        return ManagedStream(stream(), safe_cleanup, media_type="text/event-stream",
                             headers={"x-request-id": request_id, "cache-control": "no-cache"})

    return app
