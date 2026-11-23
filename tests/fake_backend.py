"""Protocol test fixture, never a source of GPU performance results."""
import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
active = 0


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(f"vllm:num_requests_running {active}\nvllm:num_requests_waiting 0\n")


@app.post("/v1/completions")
async def complete(request: Request):
    body = await request.json()
    if "fail" in body.get("request_id", ""):
        return JSONResponse({"error": "fixture_failure"}, status_code=503)
    async def stream():
        global active
        active += 1
        try:
            for i in range(body["max_tokens"]):
                await asyncio.sleep(0.003)
                yield "data: " + json.dumps({"choices": [{"text": "a", "token_ids": [1],
                            "finish_reason": "length" if i + 1 == body["max_tokens"] else None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [], "usage": {
                "prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"]}}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            active -= 1
    return StreamingResponse(stream(), media_type="text/event-stream")
