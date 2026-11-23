"""Derive every reported number from per-request records, keeping failures."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .matrix import metric_sum
from .metrics import summarize


STRATEGIES = ["B0", "B1", "B2", "P"]
LOADS = ["light", "near_capacity", "burst"]
COLORS = {"B0": "#64748b", "B1": "#2563eb", "B2": "#d97706", "P": "#059669"}


def load_runs(root, slo_ms, additional_roots=()):
    runs = []
    paths = [p for directory in (root, *additional_roots)
             for p in sorted(Path(directory).glob("runs/*/manifest.json"))]
    seen_ids = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest["run_id"] in seen_ids:
            raise ValueError(f"Duplicate run across evidence roots: {manifest['run_id']}")
        seen_ids.add(manifest["run_id"])
        rows = [json.loads(line) for line in path.with_name("requests.jsonl").read_text().splitlines()]
        if len(rows) != manifest["offered"]:
            raise ValueError(f"Request conservation failed: {path}")
        strategy = manifest["run_id"].split("-", 1)[0]
        stats = summarize(rows, slo_ms, manifest["arrival_window_seconds"],
                          manifest["total_seconds_including_drain"])
        by_class = {cls: summarize([r for r in rows if r["class"] == cls], slo_ms,
                                   manifest["arrival_window_seconds"],
                                   manifest["total_seconds_including_drain"])
                    for cls in ("S", "P", "D")}
        runs.append({"path": path.parent, "manifest": manifest, "rows": rows,
                     "strategy": strategy, "stats": stats, "by_class": by_class})
    # Paired runs must share exactly the same offered trajectory.
    hashes = defaultdict(set)
    for run in runs:
        m = run["manifest"]
        hashes[(m["load"], m["seed"])].add(m["trace_sha256"])
    if any(len(value) != 1 for value in hashes.values()):
        raise ValueError("Paired trajectory hashes differ")
    return runs


def paired_comparisons(runs):
    """Descriptive within-seed effects only; requests are not replicates."""
    lookup = {(r["strategy"], r["manifest"]["load"], r["manifest"]["seed"]): r
              for r in runs if r["manifest"]["valid"]}
    comparisons = []
    for candidate, baseline in (("B1", "B0"), ("B2", "B0"), ("P", "B2"), ("P", "B0")):
        for load in LOADS:
            pairs = []
            seeds = sorted({seed for strategy, label, seed in lookup if label == load})
            for seed in seeds:
                left = lookup.get((candidate, load, seed))
                right = lookup.get((baseline, load, seed))
                if left is None or right is None:
                    continue
                for key in ("gpu_uuid", "trace_sha256", "prompts_sha256", "frozen_config_sha256",
                            "source_manifest_sha256", "model_manifest_sha256", "environment_sha256"):
                    if not left["manifest"].get(key) or left["manifest"][key] != right["manifest"].get(key):
                        raise ValueError(f"Incompatible paired {key}: {candidate}/{baseline}/{load}/{seed}")
                before, after = right["stats"]["cohort_goodput"], left["stats"]["cohort_goodput"]
                pairs.append({"seed": seed, "baseline_goodput": before, "candidate_goodput": after,
                              "difference_req_s": after - before,
                              "relative_change": after / before - 1 if before > 0 else None})
            if pairs:
                differences = [p["difference_req_s"] for p in pairs]
                ratios = [p["relative_change"] for p in pairs if p["relative_change"] is not None]
                comparisons.append({"candidate": candidate, "baseline": baseline, "load": load,
                                    "paired_runs": len(pairs), "pairs": pairs,
                                    "median_paired_difference_req_s": float(np.median(differences)),
                                    "range_paired_difference_req_s": [min(differences), max(differences)],
                                    "median_paired_relative_change": float(np.median(ratios)) if ratios else None})
    return comparisons


def savefig(figure, root, name):
    figure.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(root / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def plot_goodput(runs, root):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for index, strategy in enumerate(STRATEGIES):
        for column, metric in enumerate(("cohort_goodput", "slo_success_fraction")):
            for load_index, load in enumerate(LOADS):
                values = [r["stats"][metric] for r in runs
                          if r["strategy"] == strategy and r["manifest"]["load"] == load]
                if not values:
                    continue
                x = load_index + (index - 1.5) * 0.19
                axes[column].bar(x, np.median(values), width=0.17, color=COLORS[strategy],
                                 alpha=0.75, label=strategy if load_index == 0 else None)
                axes[column].scatter([x] * len(values), values, s=18, color="black", zorder=3)
                axes[column].annotate(f"n={len(values)}", (x, max(values)), xytext=(0, 5),
                                      textcoords="offset points", ha="center", fontsize=7)
    for axis in axes:
        axis.set_xticks(range(3), ["Light", "Near capacity", "Burst"])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("SLO-passing requests / offered second")
    axes[1].set_ylabel("SLO success / all offered requests")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=COLORS[s], label=s, alpha=0.75)
                            for s in STRATEGIES if any(r["strategy"] == s for r in runs)],
                   ncol=4, frameon=False)
    figure.suptitle("Fixed arrival traces; bars: median, dots: paired trials")
    savefig(figure, root, "01_goodput")


def plot_classes(runs, root):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for axis, cls in zip(axes, ("S", "P", "D")):
        for index, strategy in enumerate(STRATEGIES):
            values = [row["ttft_ms"] for r in runs if r["strategy"] == strategy
                      and r["manifest"]["load"] == "near_capacity" for row in r["rows"]
                      if row["class"] == cls and row["status"] == "succeeded" and row["ttft_ms"] is not None]
            subset = [r["by_class"][cls] for r in runs if r["strategy"] == strategy
                      and r["manifest"]["load"] == "near_capacity"]
            if values:
                quantiles = np.percentile(values, [50, 95])
                axis.plot([index, index], quantiles, color=COLORS[strategy], linewidth=4)
                axis.scatter([index, index], quantiles, color=COLORS[strategy], s=[30, 70])
            offered = sum(s["offered"] for s in subset)
            completed = sum(s["completed"] for s in subset)
            if offered:
                axis.text(index, 1.02, f"fail {1-completed/offered:.1%}",
                          transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
        axis.set_xticks(range(4), STRATEGIES)
        axis.set_title(f"Class {cls}", pad=28)
        if cls == "S":
            axis.set_ylabel("Successful-request TTFT (ms), P50 to P95")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Near-capacity workload; pooled descriptive quantiles, failures shown separately")
    savefig(figure, root, "02_class_latency")


def plot_burst(runs, root):
    chosen = [r for r in runs if r["manifest"]["load"] == "burst" and r["manifest"]["seed"] == 1101]
    if not chosen:
        return
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for run in chosen:
        strategy = run["strategy"]
        points = [json.loads(line) for line in (run["path"] / "monitoring.jsonl").read_text().splitlines()]
        x = [s["offset_seconds"] for s in points]
        waiting = [metric_sum(s.get("backend_metrics", ""), "vllm:num_requests_waiting") for s in points]
        axes[0].plot(x, [np.nan if v is None else v for v in waiting], color=COLORS[strategy], label=strategy)
        gateway = [s for s in points if "gateway" in s and "queued" in s["gateway"]]
        if gateway:
            axes[1].plot([s["offset_seconds"] for s in gateway], [s["gateway"]["queued"] for s in gateway],
                         color=COLORS[strategy], label=strategy)
        rows = [r for r in run["rows"] if r["ttft_ms"] is not None]
        axes[2].scatter([r["send_offset_seconds"] for r in rows], [r["ttft_ms"] for r in rows],
                        color=COLORS[strategy], alpha=0.35, s=8, label=strategy)
    for axis in axes:
        axis.axvspan(35, 55, color="#94a3b8", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, ncol=4)
    axes[0].set_ylabel("Backend waiting requests")
    axes[1].set_ylabel("Gateway queued requests\n(B0/B1: not applicable)")
    axes[2].set_ylabel("Client TTFT (ms)")
    axes[2].set_xlabel("Seconds since trace start")
    figure.suptitle("Preselected burst trace, seed 1101; shaded: high arrival-rate interval")
    savefig(figure, root, "03_burst_timeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/formal")
    parser.add_argument("--frozen", default="configs/frozen.yaml")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--additional-root", action="append", default=[])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    frozen = yaml.safe_load(Path(args.frozen).read_text())
    runs = load_runs(args.root, frozen["slo_ms"], args.additional_root)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    valid = [r for r in runs if r["manifest"]["valid"]]
    flat = [{"run_id": r["manifest"]["run_id"], "strategy": r["strategy"],
             "load": r["manifest"]["load"], "seed": r["manifest"]["seed"],
             "valid": r["manifest"]["valid"],
             "invalid_reasons": ";".join(r["manifest"]["invalid_reasons"]),
             **{k: v for k, v in r["stats"].items() if k != "statuses"}} for r in runs]
    with (root / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["run_id"])
        writer.writeheader()
        writer.writerows(flat)
    payload = [{"manifest": r["manifest"], "stats": r["stats"], "by_class": r["by_class"]} for r in runs]
    (root / "all_results.json").write_text(json.dumps(payload, indent=2))
    comparisons = paired_comparisons(runs)
    (root / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))
    from .checkpoint import expected_jobs
    expected = expected_jobs(frozen)
    observed = {r["manifest"]["run_id"] for r in runs}
    completeness = {"expected": len(expected), "completed": len(runs), "valid": len(valid),
                    "missing": sorted(expected - observed), "unexpected": sorted(observed - expected),
                    "complete": observed == expected and len(valid) == len(expected),
                    "evidence_roots": [args.root, *args.additional_root]}
    (root / "completeness.json").write_text(json.dumps(completeness, indent=2))
    if args.require_complete and not completeness["complete"]:
        raise ValueError(f"Experiment matrix is incomplete or invalid: {completeness}")
    if valid:
        plot_goodput(valid, figures)
        plot_classes(valid, figures)
        plot_burst(valid, figures)
    lines = ["**GPU experiment report — generated from per-request records**", "",
             f"Completed manifests: {len(runs)}. Valid runs: {len(valid)}. Invalid retained: {len(runs)-len(valid)}.", "",
             f"Expected runs: {len(expected)}. Missing: {len(expected-observed)}. Complete matrix: {completeness['complete']}.", "",
             "Incomplete policy cells have different seed counts; unpaired medians must not be interpreted as controlled policy effects.", "",
             "All failures, rejections and timeouts remain in the offered-request denominator. Quantiles are explicitly conditional on success.", "",
             "| Load | Strategy | Runs | Median cohort goodput | Median wall goodput | Median SLO success |", "|---|---|---:|---:|---:|---:|"]
    for load in LOADS:
        for strategy in STRATEGIES:
            subset = [r for r in valid if r["strategy"] == strategy and r["manifest"]["load"] == load]
            if subset:
                values = [np.median([r["stats"][metric] for r in subset]) for metric in
                          ("cohort_goodput", "wall_goodput", "slo_success_fraction")]
                lines.append(f"| {load} | {strategy} | {len(subset)} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.2%} |")
    lines.extend(["", "**Within-seed paired contrasts (descriptive only)**", "",
                  "Effect = candidate minus baseline; relative effects are calculated within each matched seed before taking the median. n is paired seed runs, not requests. No significance test is claimed.", "",
                  "| Candidate / baseline | Load | Paired n | Median delta req/s | Median relative change |",
                  "|---|---|---:|---:|---:|"])
    for comparison in comparisons:
        relative = comparison['median_paired_relative_change']
        relative_text = f"{relative:+.2%}" if relative is not None else "undefined (zero baseline)"
        lines.append(f"| {comparison['candidate']} / {comparison['baseline']} | {comparison['load']} | "
                     f"{comparison['paired_runs']} | {comparison['median_paired_difference_req_s']:+.4f} | {relative_text} |")
    lines.extend(["", "**Interpretation limits**", "",
                  "The SLOs and synthetic arrival traces are experimental, not production service guarantees. Input lengths and output budgets are controlled. Three paired trials are descriptive and do not establish statistical significance. Each backend uses one GPU. If trials run concurrently on separate GPUs, shared host CPU interference is possible and is recorded in the manifests. Policies are paired on the same GPU within each seed.", "",
                  "Inspect the raw per-class results and failure counts before quoting an overall improvement. Native baseline configuration and development tuning attempts are recorded separately. A nonpositive result is retained.", "",
                  "See `run_summary.csv`, `all_results.json`, frozen configuration, run manifests, and original request JSONL for the evidence."])
    (root / "experiment_report.generated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "valid": len(valid), "output": str(root)}))


if __name__ == "__main__":
    main()
