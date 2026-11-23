import asyncio
import json

import httpx
import pytest

from admission_lab.app import create_app


class ContentStream(httpx.AsyncByteStream):
    def __init__(self, hang=False):
        self.closed = False
        self.hang = hang

    async def __aiter__(self):
        yield b'data: {"choices":[{"text":"ok","token_ids":[1]}]}\n\n'
        if self.hang:
            await asyncio.sleep(10)
        yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed = True


def payload(request_id="request"):
    return {"model": "admission-7b", "prompt": [1, 2], "max_tokens": 1,
            "stream": True, "request_id": request_id}


async def test_stream_and_error_release_all_budget(tmp_path):
    streams = []
    async def handler(request):
        body = json.loads(request.content)
        if body["request_id"] == "fail":
            return httpx.Response(503, json={"error": "busy"})
        stream = ContentStream()
        streams.append(stream)
        return httpx.Response(200, stream=stream)
    app = create_app({"event_log": str(tmp_path / "events.jsonl")},
                     transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/completions", json=payload())
            assert response.status_code == 200 and "[DONE]" in response.text
            assert streams[0].closed
            failure = await client.post("/v1/completions", json=payload("fail"))
            assert failure.status_code == 503
            state = (await client.get("/debug/state")).json()
            assert not state["paused"]
            assert state["queued"] == state["inflight"] == state["reserved_cost_ms"] == 0


@pytest.mark.parametrize("verified", [False, True])
async def test_explicit_cancel_closes_stream_and_cleans_once(tmp_path, verified):
    stream = ContentStream(hang=True)
    app = create_app({"event_log": str(tmp_path / "events.jsonl"),
                      "cancel_path_verified": verified},
                     transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/v1/completions", json=payload("cancel-me")))
            for _ in range(100):
                if (await app.state.scheduler.snapshot())["inflight"]:
                    break
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.02)
            cancelled = await client.delete("/requests/cancel-me")
            assert cancelled.json()["cancel_requested"]
            await asyncio.wait_for(task, 1)
            state = (await client.get("/debug/state")).json()
            assert stream.closed
            assert state["released"] == 1
            assert state["inflight"] == 0 and state["reserved_cost_ms"] == 0
            assert state["paused"] is (not verified)
