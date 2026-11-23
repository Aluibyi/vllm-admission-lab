import asyncio
import json
from pathlib import Path
import time


class Recorder:
    """Bounded, batched JSONL writer; dropped records invalidate a run."""

    def __init__(self, path):
        self.path = Path(path)
        self.queue = asyncio.Queue(maxsize=100000)
        self.dropped = 0
        self.task = None

    async def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task = asyncio.create_task(self._writer())

    def event(self, kind, **data):
        try:
            self.queue.put_nowait({"event": kind, "wall_time": time.time(),
                                   "monotonic": time.perf_counter(), **data})
        except asyncio.QueueFull:
            self.dropped += 1

    def _append(self, rows):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    async def _writer(self):
        batch = []
        while True:
            try:
                row = await asyncio.wait_for(self.queue.get(), 0.2)
            except asyncio.TimeoutError:
                row = "flush"
            if row is None:
                if batch:
                    await asyncio.to_thread(self._append, batch)
                return
            if row != "flush":
                batch.append(row)
            if batch and (row == "flush" or len(batch) >= 100):
                await asyncio.to_thread(self._append, batch)
                batch = []

    async def close(self):
        await self.queue.put(None)
        if self.task:
            await self.task
