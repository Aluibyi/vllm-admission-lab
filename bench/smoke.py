"""One real generation, kept separate from calibration and measured trials."""
import asyncio
import json
from pathlib import Path
import time

import httpx

from .matrix import wait_idle
from .probe import request_once


async def main():
    prompts = json.loads(Path("artifacts/traces/prompts.json").read_text())
    async with httpx.AsyncClient(timeout=180) as client:
        result = await request_once(client, "http://127.0.0.1:8000/v1/completions", "admission-7b",
                                    prompts["S-0"], 128, f"smoke-{time.time_ns()}")
    await wait_idle()
    result.update(verified=True, scope="Separate real generation; not a calibration or formal trial")
    Path("artifacts/generation_smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
