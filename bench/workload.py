import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SHAPES = {"S": (512, 128), "P": (8192, 128), "D": (512, 512)}
CLASSES = ["S", "P", "D"]
PROBABILITIES = [0.60, 0.25, 0.15]


def arrivals(rate, duration, rng, burst=False):
    points = []
    position = 0.0
    while position < duration:
        multiplier = 2.0 if burst and 35 <= position < 55 else (0.6 if burst else 1.0)
        # Piecewise homogeneous Poisson process using independent segment
        # generation, not a rate frozen across a boundary-crossing interval.
        boundaries = [35.0, 55.0, duration] if burst else [duration]
        end = min(x for x in boundaries if x > position)
        while True:
            position += float(rng.exponential(1.0 / (rate * multiplier)))
            if position >= end:
                position = end
                break
            points.append(position)
    return points


def build_trace(rate, duration, seed, load):
    rng = np.random.default_rng(seed)
    multiplier = {"light": 0.5, "near_capacity": 0.95, "burst": 1.0,
                  "calibration": 1.0}[load]
    times = arrivals(rate * multiplier, duration, rng, burst=load == "burst")
    rows = []
    for index, offset in enumerate(times):
        cls = str(rng.choice(CLASSES, p=PROBABILITIES))
        prompt, output = SHAPES[cls]
        rows.append({"index": index, "scheduled_seconds": offset, "class": cls,
                     "prompt_key": f"{cls}-{int(rng.integers(0, 16))}",
                     "prompt_tokens": prompt, "max_tokens": output})
    return {"duration_seconds": duration, "seed": seed, "load": load,
            "reference_rate": rate, "requests": rows}


def build_prompts(tokenizer):
    sentences = [
        "A distributed service processes incoming requests and records their completion times. ",
        "The engineering team measures memory traffic, queue lengths, and scheduling overhead. ",
        "Explain the tradeoffs of this system with concrete examples and clear reasoning. ",
        "Long documents and short questions share a single inference accelerator. ",
        "Performance experiments require fixed inputs and reproducible measurement procedures. ",
    ]
    prompts = {}
    for cls, (length, _) in SHAPES.items():
        for i in range(16):
            rng = np.random.default_rng(5000 + i + length)
            text = f"Experiment sample {cls} {i}. " + "".join(
                rng.choice(sentences, size=3000).tolist())
            tokens = tokenizer.encode(text, add_special_tokens=False)[:length]
            if len(tokens) != length:
                raise ValueError("Insufficient prompt tokens")
            prompts[f"{cls}-{i}"] = tokens
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--reference-rate", type=float, required=True)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--output-dir", default="artifacts/traces")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prompt_path = root / "prompts.json"
    prompt_path.write_text(json.dumps(build_prompts(tokenizer)), encoding="utf-8")
    for seed in (1101, 1102, 1103):
        for load in ("light", "near_capacity", "burst"):
            trace = build_trace(args.reference_rate, args.duration, seed, load)
            raw = json.dumps(trace, sort_keys=True)
            (root / f"{load}-{seed}.json").write_text(raw, encoding="utf-8")
    print(json.dumps({"prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                      "output_dir": str(root), "reference_rate": args.reference_rate}))


if __name__ == "__main__":
    main()
