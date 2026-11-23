**GPU experiment report — generated from per-request records**

Completed manifests: 36. Valid runs: 36. Invalid retained: 0.

Expected runs: 36. Missing: 0. Complete matrix: True.

Incomplete policy cells have different seed counts; unpaired medians must not be interpreted as controlled policy effects.

All failures, rejections and timeouts remain in the offered-request denominator. Quantiles are explicitly conditional on success.

| Load | Strategy | Runs | Median cohort goodput | Median wall goodput | Median SLO success |
|---|---|---:|---:|---:|---:|
| light | B0 | 3 | 1.7000 | 1.6850 | 87.43% |
| light | B1 | 3 | 1.7000 | 1.6849 | 87.43% |
| light | B2 | 3 | 1.6667 | 1.6519 | 85.71% |
| light | P | 3 | 1.6778 | 1.6628 | 86.29% |
| near_capacity | B0 | 3 | 1.2222 | 1.1317 | 30.05% |
| near_capacity | B1 | 3 | 1.2222 | 1.1313 | 30.05% |
| near_capacity | B2 | 3 | 0.2333 | 0.1681 | 6.12% |
| near_capacity | P | 3 | 0.2444 | 0.1765 | 6.41% |
| burst | B0 | 3 | 0.8222 | 0.7825 | 22.22% |
| burst | B1 | 3 | 0.8111 | 0.7719 | 22.22% |
| burst | B2 | 3 | 0.7000 | 0.5371 | 19.50% |
| burst | P | 3 | 0.7111 | 0.5452 | 19.81% |

**Within-seed paired contrasts (descriptive only)**

Effect = candidate minus baseline; relative effects are calculated within each matched seed before taking the median. n is paired seed runs, not requests. No significance test is claimed.

| Candidate / baseline | Load | Paired n | Median delta req/s | Median relative change |
|---|---|---:|---:|---:|
| B1 / B0 | light | 3 | +0.0000 | +0.00% |
| B1 / B0 | near_capacity | 3 | +0.0000 | +0.00% |
| B1 / B0 | burst | 3 | +0.0000 | +0.00% |
| B2 / B0 | light | 3 | -0.0333 | -1.96% |
| B2 / B0 | near_capacity | 3 | -0.9111 | -69.57% |
| B2 / B0 | burst | 3 | -0.0333 | -4.05% |
| P / B2 | light | 3 | +0.0111 | +0.67% |
| P / B2 | near_capacity | 3 | +0.0111 | +4.76% |
| P / B2 | burst | 3 | +0.0000 | +0.00% |
| P / B0 | light | 3 | -0.0222 | -1.31% |
| P / B0 | near_capacity | 3 | -0.7444 | -68.12% |
| P / B0 | burst | 3 | -0.0333 | -4.05% |

**Interpretation limits**

The SLOs and synthetic arrival traces are experimental, not production service guarantees. Input lengths and output budgets are controlled. Three paired trials are descriptive and do not establish statistical significance. Each backend uses one GPU. If trials run concurrently on separate GPUs, shared host CPU interference is possible and is recorded in the manifests. Policies are paired on the same GPU within each seed.

Inspect the raw per-class results and failure counts before quoting an overall improvement. Native baseline configuration and development tuning attempts are recorded separately. A nonpositive result is retained.

See `run_summary.csv`, `all_results.json`, frozen configuration, run manifests, and original request JSONL for the evidence.
