import asyncio

import pytest

from admission_lab.scheduler import AdmissionError, Scheduler


async def admitted(entry):
    return await asyncio.wait_for(asyncio.shield(entry.future), 1)


async def test_concurrent_reservations_and_double_release():
    scheduler = Scheduler(max_inflight=3, work_budget_ms=60, queue_limit=200)
    await scheduler.start()
    try:
        async def request(i):
            entry = await scheduler.submit(str(i), 20)
            await admitted(entry)
            state = await scheduler.snapshot()
            assert 0 <= state["inflight"] <= 3
            assert 0 <= state["reserved_cost_ms"] <= 60
            await asyncio.sleep(0.001)
            await asyncio.gather(scheduler.release(entry, "done"), scheduler.release(entry, "done"))
        await asyncio.gather(*(request(i) for i in range(80)))
        state = await scheduler.snapshot()
        assert (state["queued"], state["inflight"], state["reserved_cost_ms"]) == (0, 0, 0)
        assert state["released"] == state["admitted"] == 80
    finally:
        await scheduler.close()


async def test_cost_order_and_budget_guard():
    scheduler = Scheduler(max_inflight=1, work_budget_ms=100, tau_ms=1000)
    await scheduler.start()
    try:
        blocker = await scheduler.submit("block", 100)
        await admitted(blocker)
        long = await scheduler.submit("long", 80)
        short = await scheduler.submit("short", 10)
        await scheduler.release(blocker, "done")
        await admitted(short)
        assert not long.future.done()
        await scheduler.release(short, "done")
        await admitted(long)
        await scheduler.release(long, "done")
        with pytest.raises(AdmissionError, match="exceeds"):
            await scheduler.submit("oversized", 101)
    finally:
        await scheduler.close()


async def test_aged_request_drains_capacity_before_new_short_jobs():
    logical_time = [100.0]
    scheduler = Scheduler(max_inflight=4, work_budget_ms=100,
                          aging_threshold_ms=10, tau_ms=5, clock=lambda: logical_time[0])
    await scheduler.start()
    try:
        blocker = await scheduler.submit("block", 80)
        await admitted(blocker)
        long = await scheduler.submit("long", 60)
        logical_time[0] += 0.03
        short = await scheduler.submit("short", 10)
        await asyncio.sleep(0.03)
        assert not short.future.done()
        assert not long.future.done()
        await scheduler.release(blocker, "done")
        await admitted(long)
        # Both now fit; the old long request must have been admitted first.
        await admitted(short)
        assert long.admitted <= short.admitted
        await scheduler.release(long, "done")
        await scheduler.release(short, "done")
    finally:
        await scheduler.close()


async def test_queued_cancel_timeout_and_pause():
    scheduler = Scheduler(max_inflight=1, work_budget_ms=100, queue_timeout_ms=40)
    await scheduler.start()
    try:
        blocker = await scheduler.submit("block", 100)
        await admitted(blocker)
        cancelled = await scheduler.submit("cancel", 1)
        assert await scheduler.cancel("cancel")
        with pytest.raises(AdmissionError, match="cancelled"):
            await cancelled.future
        timed = await scheduler.submit("timeout", 1)
        with pytest.raises(AdmissionError, match="queue_timeout"):
            await admitted(timed)
        await scheduler.pause()
        await scheduler.release(blocker, "done")
        pending = await scheduler.submit("paused", 1)
        await asyncio.sleep(0.01)
        assert not pending.future.done()
        await scheduler.cancel("paused")
        assert (await scheduler.snapshot())["reserved_cost_ms"] == 0
    finally:
        await scheduler.close()
