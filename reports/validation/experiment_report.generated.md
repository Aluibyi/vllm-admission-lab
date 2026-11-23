**GPU experiment report — generated from per-request records**

Completed manifests: 4. Valid runs: 4. Invalid retained: 0.

Expected runs: 4. Missing: 0. Complete matrix: True.

Incomplete policy cells have different seed counts; unpaired medians must not be interpreted as controlled policy effects.

All failures, rejections and timeouts remain in the offered-request denominator. Quantiles are explicitly conditional on success.

| Load | Strategy | Runs | Median cohort goodput | Median wall goodput | Median SLO success |
|---|---|---:|---:|---:|---:|
| near_capacity | B0 | 1 | 2.5000 | 2.3410 | 72.58% |
| near_capacity | B1 | 1 | 2.5000 | 2.3413 | 72.58% |
| near_capacity | B2 | 1 | 0.7556 | 0.7075 | 21.94% |
| near_capacity | P | 1 | 1.0444 | 0.9780 | 30.32% |

**Within-seed paired contrasts (descriptive only)**

Effect = candidate minus baseline; relative effects are calculated within each matched seed before taking the median. n is paired seed runs, not requests. No significance test is claimed.

| Candidate / baseline | Load | Paired n | Median delta req/s | Median relative change |
|---|---|---:|---:|---:|
| B1 / B0 | near_capacity | 1 | +0.0000 | +0.00% |
| B2 / B0 | near_capacity | 1 | -1.7444 | -69.78% |
| P / B2 | near_capacity | 1 | +0.2889 | +38.24% |
| P / B0 | near_capacity | 1 | -1.4556 | -58.22% |

**Interpretation limits**

The SLOs and synthetic arrival traces are experimental, not production service guarantees. Input lengths and output budgets are controlled. Three paired trials are descriptive and do not establish statistical significance. Each backend uses one GPU. If trials run concurrently on separate GPUs, shared host CPU interference is possible and is recorded in the manifests. Policies are paired on the same GPU within each seed.

Inspect the raw per-class results and failure counts before quoting an overall improvement. Native baseline configuration and development tuning attempts are recorded separately. A nonpositive result is retained.

See `run_summary.csv`, `all_results.json`, frozen configuration, run manifests, and original request JSONL for the evidence.
