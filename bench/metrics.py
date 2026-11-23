from collections import Counter

import numpy as np


def percentile(values, q=95):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(finite, q)) if finite else None


def summarize(rows, slo_ms, arrival_window, total_seconds):
    passing = []
    for row in rows:
        slo = slo_ms[row["class"]]
        passed = (row["status"] == "succeeded" and row.get("metrics_valid", True)
                  and row.get("ttft_ms") is not None and row.get("e2e_ms") is not None
                  and row["ttft_ms"] <= slo["ttft_ms"] and row["e2e_ms"] <= slo["e2e_ms"])
        passing.append(passed)
    successful = [r for r in rows if r["status"] == "succeeded"]
    counts = Counter(r["status"] for r in rows)
    return {"offered": len(rows), "completed": len(successful), "slo_passed": sum(passing),
            "slo_success_fraction": sum(passing) / len(rows) if rows else 0,
            "cohort_goodput": sum(passing) / arrival_window,
            "wall_goodput": sum(passing) / total_seconds,
            "failure_fraction": (len(rows) - len(successful)) / len(rows) if rows else 0,
            "successful_ttft_p50_ms": percentile([r["ttft_ms"] for r in successful], 50),
            "successful_ttft_p95_ms": percentile([r["ttft_ms"] for r in successful], 95),
            "successful_e2e_p95_ms": percentile([r["e2e_ms"] for r in successful], 95),
            "statuses": dict(counts)}
