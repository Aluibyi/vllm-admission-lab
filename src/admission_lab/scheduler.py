from __future__ import annotations

import asyncio
from collections import deque
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class AdmissionError(Exception):
    def __init__(self, reason: str, status: int = 429):
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass
class Entry:
    request_id: str
    cost: float
    enqueued: float
    sequence: int
    future: asyncio.Future
    admitted: float | None = None
    state: str = "queued"
    reserved: bool = False
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    data: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """One-process scheduler. All reservations and transitions use one condition."""

    def __init__(self, *, policy="cost_aging", max_inflight=16,
                 work_budget_ms=100000.0, queue_limit=256, tau_ms=500.0,
                 aging_threshold_ms=5000.0, queue_timeout_ms=30000.0,
                 clock: Callable[[], float] = time.perf_counter):
        if policy not in {"fcfs", "cost_aging", "cost_only", "passthrough"}:
            raise ValueError(f"Unknown policy: {policy}")
        if min(max_inflight, work_budget_ms, queue_limit, tau_ms,
               aging_threshold_ms, queue_timeout_ms) <= 0:
            raise ValueError("Scheduler limits must be positive")
        self.policy = policy
        self.max_inflight = max_inflight
        self.work_budget_ms = float(work_budget_ms)
        self.queue_limit = queue_limit
        self.tau_ms = tau_ms
        self.aging_threshold_ms = aging_threshold_ms
        self.queue_timeout_ms = queue_timeout_ms
        self.clock = clock
        self.condition = asyncio.Condition()
        self.queue: list[Entry] = []
        self.entries: dict[str, Entry] = {}
        self.seen_ids: set[str] = set()
        self.recent_ids: deque[str] = deque()
        self.inflight = 0
        self.reserved = 0.0
        self.sequence = 0
        self.task: asyncio.Task | None = None
        self.closed = False
        self.paused = False
        self.counters = {k: 0 for k in ("submitted", "admitted", "released",
                                        "cancelled", "queue_timeout", "rejected")}

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="admission-scheduler")

    async def close(self):
        async with self.condition:
            self.closed = True
            for entry in list(self.queue):
                self._remove_queued(entry, "closed", 503)
            self.condition.notify_all()
        if self.task:
            await self.task

    async def submit(self, request_id: str, cost: float) -> Entry:
        async with self.condition:
            if self.closed:
                raise AdmissionError("scheduler_closed", 503)
            if request_id in self.entries or request_id in self.seen_ids:
                raise AdmissionError("duplicate_request_id", 409)
            if not 0 < cost <= self.work_budget_ms:
                raise AdmissionError("request_exceeds_work_budget", 422)
            if len(self.queue) >= self.queue_limit:
                self.counters["rejected"] += 1
                raise AdmissionError("queue_full")
            self.sequence += 1
            entry = Entry(request_id, cost, self.clock(), self.sequence,
                          asyncio.get_running_loop().create_future())
            # Retrieve unobserved exceptions on queue cancellation without hiding
            # them from a route that is awaiting the future.
            entry.future.add_done_callback(
                lambda f: f.exception() if not f.cancelled() else None)
            self.entries[request_id] = entry
            if len(self.recent_ids) >= 100000:
                self.seen_ids.discard(self.recent_ids.popleft())
            self.recent_ids.append(request_id)
            self.seen_ids.add(request_id)
            self.queue.append(entry)
            self.counters["submitted"] += 1
            self.condition.notify_all()
            return entry

    def _remove_queued(self, entry: Entry, reason: str, status: int):
        if entry in self.queue:
            self.queue.remove(entry)
        entry.state = reason
        self.entries.pop(entry.request_id, None)
        if not entry.future.done():
            entry.future.set_exception(AdmissionError(reason, status))

    async def cancel(self, request_id: str) -> bool:
        async with self.condition:
            entry = self.entries.get(request_id)
            if entry is None:
                return False
            if not entry.cancelled.is_set():
                self.counters["cancelled"] += 1
            entry.cancelled.set()
            if entry.state == "queued":
                self._remove_queued(entry, "cancelled", 499)
            # A running request keeps its reservation until upstream cleanup.
            self.condition.notify_all()
            return True

    async def release(self, entry: Entry, state: str):
        async with self.condition:
            if entry.reserved:
                entry.reserved = False
                self.inflight -= 1
                self.reserved -= entry.cost
                if abs(self.reserved) < 1e-7:
                    self.reserved = 0.0
                self.counters["released"] += 1
            if entry in self.queue:
                self.queue.remove(entry)
            entry.state = state
            self.entries.pop(entry.request_id, None)
            self.condition.notify_all()

    async def pause(self):
        async with self.condition:
            self.paused = True
            self.condition.notify_all()

    def _pick(self, now: float) -> Entry | None:
        if self.paused or not self.queue or self.inflight >= self.max_inflight:
            return None
        remaining = self.work_budget_ms - self.reserved
        oldest = self.queue[0]
        if self.policy in {"fcfs", "passthrough"}:
            return oldest if oldest.cost <= remaining + 1e-8 else None
        if (self.policy == "cost_aging" and
                (now - oldest.enqueued) * 1000 >= self.aging_threshold_ms):
            # Reserve the next opportunity for the oldest job: drain instead
            # of letting small new jobs continuously take its capacity.
            return oldest if oldest.cost <= remaining + 1e-8 else None
        candidates = [e for e in self.queue if e.cost <= remaining + 1e-8]
        if not candidates:
            return None
        def rank(e):
            cost = e.cost
            if self.policy == "cost_aging":
                cost /= 1 + max(0.0, now - e.enqueued) * 1000 / self.tau_ms
            return cost, e.sequence
        return min(candidates, key=rank)

    async def _run(self):
        async with self.condition:
            while not self.closed:
                now = self.clock()
                for entry in list(self.queue):
                    if (now - entry.enqueued) * 1000 >= self.queue_timeout_ms:
                        self.counters["queue_timeout"] += 1
                        self._remove_queued(entry, "queue_timeout", 504)
                while (entry := self._pick(now)) is not None:
                    self.queue.remove(entry)
                    entry.state = "admitted"
                    entry.admitted = now
                    entry.reserved = True
                    self.inflight += 1
                    self.reserved += entry.cost
                    self.counters["admitted"] += 1
                    if not entry.future.done():
                        entry.future.set_result(entry)
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=0.02)
                except asyncio.TimeoutError:
                    pass

    async def snapshot(self):
        async with self.condition:
            return {"policy": self.policy, "queued": len(self.queue),
                    "paused": self.paused,
                    "inflight": self.inflight, "reserved_cost_ms": self.reserved,
                    "max_inflight": self.max_inflight,
                    "work_budget_ms": self.work_budget_ms,
                    **self.counters}
