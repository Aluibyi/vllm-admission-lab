from bench.metrics import summarize
from bench.workload import build_trace


def test_failures_and_tail_are_not_hidden_from_goodput():
    common = {"class": "S", "metrics_valid": True}
    rows = [
        {**common, "status": "succeeded", "ttft_ms": 10, "e2e_ms": 100},
        {**common, "status": "succeeded", "ttft_ms": 1000, "e2e_ms": 2000},
        {**common, "status": "timeout", "ttft_ms": None, "e2e_ms": 5000},
        {**common, "status": "failed", "ttft_ms": 2, "e2e_ms": 3},
    ]
    result = summarize(rows, {"S": {"ttft_ms": 100, "e2e_ms": 500}}, 10, 15)
    assert result["offered"] == 4
    assert result["slo_passed"] == 1
    assert result["slo_success_fraction"] == 0.25
    assert result["cohort_goodput"] == 0.1
    assert result["wall_goodput"] == 1 / 15
    assert result["failure_fraction"] == 0.5


def test_trace_is_fixed_and_not_closed_loop():
    first = build_trace(5, 90, 1101, "burst")
    second = build_trace(5, 90, 1101, "burst")
    assert first == second
    assert all(0 <= r["scheduled_seconds"] < 90 for r in first["requests"])
    assert [r["scheduled_seconds"] for r in first["requests"]] == sorted(
        r["scheduled_seconds"] for r in first["requests"])
    assert first != build_trace(5, 90, 1102, "burst")
