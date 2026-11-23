import copy
from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from bench.analyze import paired_comparisons
from bench.checkpoint import plan_continuation, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"seeds": [1101], "loads": ["light"], "trace_dir": "traces", "prompts": "prompts.json",
              "gateway": {"cancel_path_verified": True},
              "gpu_slots": [{"gpu_uuid": "GPU-test", "seeds": [1101], "backend_port": 8000, "gateway_port": 8080}]}
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump(config))
    trace_row = {"index": 0, "class": "S", "prompt_key": "S-0", "prompt_tokens": 1,
                 "max_tokens": 1, "scheduled_seconds": 0.01}
    trace = {"requests": [trace_row]}
    prompts = {"S-0": [42]}
    write_json(tmp_path / "traces/light-1101.json", trace)
    write_json(tmp_path / "prompts.json", prompts)
    previous = tmp_path / "original"
    provenance = previous / "provenance"
    for name in ("source_manifest", "model_manifest", "environment"):
        write_json(provenance / f"{name}.json", {"id": name, "files": []})
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"\n')
    run_id = "B0-light-1101"
    directory = previous / "runs" / run_id
    manifest = {"run_id": run_id, "load": "light", "seed": 1101, "offered": 1,
                "gpu_uuid": "GPU-test", "valid": True, "frozen_config_sha256": sha256(frozen)}
    for name in ("source_manifest", "model_manifest", "environment"):
        manifest[name + "_sha256"] = sha256(provenance / f"{name}.json")
    for name, data in (("trace", trace), ("prompts", prompts)):
        manifest[name + "_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    write_json(directory / "manifest.json", manifest)
    row = {**trace_row, "request_id": run_id + "-000000", "run_id": run_id,
           "status": "succeeded", "metrics_valid": True, "done": True, "finish_reason": "length",
           "actual_output_tokens": 1, "actual_prompt_tokens": 1}
    (directory / "requests.jsonl").write_text(json.dumps(row) + "\n")
    write_json(directory / "backend_drained.json", {"verified": True})
    interrupted = previous / "runs/P-light-1101"
    interrupted.mkdir()
    (interrupted / "requests.jsonl").write_text("partial evidence\n")
    return previous, frozen, directory


def test_continuation_preserves_completed_and_interrupted(experiment):
    previous, frozen, directory = experiment
    partial = previous / "runs/P-light-1101/requests.jsonl"
    original = partial.read_bytes()
    plan = plan_continuation(previous, frozen)
    assert plan["completed"] == ["B0-light-1101"]
    assert len(plan["missing"]) == 3
    assert "P-light-1101" in plan["missing"]
    assert plan["interrupted"] == ["P-light-1101"]
    assert partial.read_bytes() == original


@pytest.mark.parametrize("tamper", ["config", "trajectory", "gpu", "missing_request", "drain"])
def test_continuation_refuses_incompatible_evidence(experiment, tamper):
    previous, frozen, directory = experiment
    if tamper == "config":
        frozen.write_text(frozen.read_text() + "changed: true\n")
    elif tamper == "trajectory":
        (Path("traces") / "light-1101.json").write_text('{"requests": []}')
    elif tamper == "gpu":
        path = directory / "manifest.json"
        data = json.loads(path.read_text())
        data["gpu_uuid"] = "GPU-other"
        write_json(path, data)
    elif tamper == "missing_request":
        (directory / "requests.jsonl").write_text("")
    else:
        write_json(directory / "backend_drained.json", {"verified": False})
    with pytest.raises(ValueError):
        plan_continuation(previous, frozen)


def test_paired_effects_use_only_matched_seeds():
    def run(strategy, seed, goodput):
        manifest = {"valid": True, "load": "light", "seed": seed,
                    **{key: "same" for key in ("gpu_uuid", "trace_sha256", "prompts_sha256",
                       "frozen_config_sha256", "source_manifest_sha256", "model_manifest_sha256", "environment_sha256")}}
        return {"strategy": strategy, "manifest": manifest, "stats": {"cohort_goodput": goodput}}
    runs = [run("B0", 1, 1), run("B1", 1, 2), run("B0", 2, 100)]
    result = paired_comparisons(runs)[0]
    assert result["paired_runs"] == 1
    assert result["median_paired_difference_req_s"] == 1
    assert result["median_paired_relative_change"] == 1
    incompatible = copy.deepcopy(runs)
    incompatible[1]["manifest"]["gpu_uuid"] = "different"
    with pytest.raises(ValueError, match="gpu_uuid"):
        paired_comparisons(incompatible)


@pytest.mark.asyncio
async def test_matrix_dry_run_selects_missing_jobs_without_starting_backend(experiment, monkeypatch, tmp_path):
    from bench.matrix import Backend, execute
    previous, frozen, directory = experiment
    before = (directory / "manifest.json").read_bytes()
    async def refuse_start(*args, **kwargs):
        pytest.fail("A dry run must never start a GPU backend")
    monkeypatch.setattr(Backend, "start", refuse_start)
    output = tmp_path / "continuation"
    await execute(Namespace(frozen=str(frozen), output=str(output), slot=0,
                            resume_from=str(previous), dry_run=True))
    groups = json.loads((output / "run_order-slot-0.json").read_text())
    jobs = [job for group in groups for job in group["jobs"]]
    assert len(jobs) == 3
    assert all(job[0] != "B0" for job in jobs)
    assert (directory / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="separate output"):
        await execute(Namespace(frozen=str(frozen), output=str(previous), slot=0,
                                resume_from=str(previous), dry_run=True))
