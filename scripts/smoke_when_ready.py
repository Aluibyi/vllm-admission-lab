import asyncio
from pathlib import Path
import time

import httpx

from bench.smoke import main


async def wait_and_smoke():
    pid = int(Path("artifacts/setup/development_backend.pid").read_text())
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic() - started < 900:
            process = Path(f"/proc/{pid}")
            if not process.exists() or "State:\tZ" in (process / "status").read_text():
                raise RuntimeError("Owned backend wrapper exited; inspect backend logs")
            try:
                response = await client.get("http://127.0.0.1:8000/health")
                if response.status_code == 200:
                    await main()
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Owned backend startup timeout")


asyncio.run(wait_and_smoke())
