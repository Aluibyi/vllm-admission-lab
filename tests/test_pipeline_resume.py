import json

import pytest

from bench.pipeline import Pipeline


@pytest.mark.parametrize("stage", ["formal_traces", "real_gateway_cancel_verification"])
def test_trace_resume_retains_failure_and_refuses_existing_formal_data(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts/setup").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/frozen.yaml").write_text("reference_rate: 4\n")
    state = {"status": "failed", "current_stage": stage, "error": "retained failure",
             "stages": [{"name": "freeze", "returncode": 0}, {"name": stage, "returncode": 1}]}
    (tmp_path / "artifacts/pipeline_state.json").write_text(json.dumps(state))
    pipeline = Pipeline(formal_mode="single", resume_from=stage)
    assert pipeline.state["status"] == "running"
    assert pipeline.state["stages"][-1]["returncode"] == 1
    archive = "pipeline_before_trace_resume.json" if stage == "formal_traces" else "pipeline_before_gateway_probe_resume.json"
    assert json.loads((tmp_path / "artifacts/setup" / archive).read_text()) == state
    existing = tmp_path / "artifacts/formal/runs/B0-light-1101/requests.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("partial evidence\n")
    with pytest.raises(ValueError, match="no formal data"):
        Pipeline(formal_mode="single", resume_from=stage)
