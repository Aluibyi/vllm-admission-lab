from argparse import Namespace
import json

import yaml

from bench.develop import finalize


def test_validation_stays_on_the_allocated_gpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts/traces").mkdir(parents=True)
    (tmp_path / "configs/engine.yaml").write_text(yaml.safe_dump({
        "gpu_uuid": "GPU-allocated", "concurrent_gpu_trials": False}))
    (tmp_path / "configs/selected_gateway.yaml").write_text("cancel_path_verified: true\n")
    (tmp_path / "configs/selected_native.yaml").write_text("policy: fcfs\n")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"slo_ms": {}, "client_deadline_seconds": 60}))
    finalize(Namespace(engine_config="configs/engine.yaml", calibration=str(calibration),
                       prompts="artifacts/traces/prompts.json", rate=1.0))
    validation = yaml.safe_load((tmp_path / "configs/validation.yaml").read_text())
    assert validation["gpu_uuid"] == "GPU-allocated"
    assert validation["gpu_slots"][0]["gpu_uuid"] == "GPU-allocated"
    assert validation["seeds"] == [4401]
    assert validation["concurrent_gpu_trials"] is False
    trace = json.loads((tmp_path / "artifacts/traces/near_capacity-4401.json").read_text())
    assert trace["seed"] == 4401
