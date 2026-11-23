"""Verify public-export bytes and recompute all 40 runs without a GPU."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.analyze import load_runs, paired_comparisons  # noqa: E402
import yaml  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(path):
    # The replay manifest hashes canonical JSON content, not file formatting.
    value = json.loads(path.read_text(encoding='utf-8'))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def equal(actual, expected, location):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f'{location}: keys differ'
        for key in expected:
            equal(actual[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f'{location}: lengths differ'
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, f'{location}[{index}]')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), f'{location}: {actual} != {expected}'
    else:
        assert actual == expected, f'{location}: values differ'


def main():
    integrity = json.loads((ROOT / 'PUBLIC_INTEGRITY.json').read_text(encoding='utf-8'))
    for entry in integrity['files']:
        path = ROOT / entry['path']
        assert path.is_file(), f'Missing: {entry["path"]}'
        assert digest(path) == entry['sha256'], f'Public bytes changed: {entry["path"]}'
    exported = json.loads((ROOT / 'PUBLIC_EXPORT.json').read_text(encoding='utf-8'))
    for entry in exported['files']:
        assert digest(ROOT / entry['path']) == entry['public_sha256'], f'Export changed: {entry["path"]}'

    total = succeeded = failed = 0
    all_ids = set()
    for stage, count, frozen_name, report_dir in (
        ('formal', 36, 'frozen.yaml', 'reports'),
        ('validation', 4, 'validation.yaml', 'reports/validation'),
    ):
        frozen = yaml.safe_load((ROOT / 'configs' / frozen_name).read_text(encoding='utf-8'))
        runs = load_runs(ROOT / 'artifacts' / stage, frozen['slo_ms'])
        assert len(runs) == count and all(r['manifest']['valid'] for r in runs), f'{stage}: incomplete/invalid'
        expected = json.loads((ROOT / report_dir / 'all_results.json').read_text(encoding='utf-8'))
        lookup = {r['manifest']['run_id']: r for r in expected}
        for run in runs:
            manifest = run['manifest']
            run_id = manifest['run_id']
            equal(run['stats'], lookup[run_id]['stats'], f'{run_id}.stats')
            equal(run['by_class'], lookup[run_id]['by_class'], f'{run_id}.by_class')
            assert manifest['gpu_uuid'] == 'GPU-PUBLIC-DEVICE-0'
            trace_path = ROOT / 'artifacts/traces' / f'{manifest["load"]}-{manifest["seed"]}.json'
            assert content_digest(trace_path) == manifest['trace_sha256'], f'{run_id}: trace mismatch'
            assert content_digest(ROOT / 'artifacts/traces/prompts.json') == manifest['prompts_sha256']
            assert manifest['send_lag_p95_ms'] < 20
            for row in run['rows']:
                assert row['request_id'] not in all_ids, 'Duplicate global request ID'
                all_ids.add(row['request_id'])
                total += 1
                if row['status'] == 'succeeded':
                    succeeded += 1
                    assert row['done'] and row['metrics_valid'], f'{run_id}: incomplete successful stream'
                    assert row['actual_output_tokens'] == row['max_tokens']
                    assert row['actual_prompt_tokens'] == row['prompt_tokens']
                else:
                    failed += 1
            state = run['path'] / 'final_gateway_state.json'
            if state.exists():
                final = json.loads(state.read_text(encoding='utf-8'))
                assert final['inflight'] == 0 and final['queued'] == 0
                assert final['reserved_cost_ms'] == 0
        equal(paired_comparisons(runs), json.loads((ROOT / report_dir / 'paired_comparisons.json').read_text(encoding='utf-8')), f'{stage}.paired')
    assert (total, succeeded, failed) == (11560, 11559, 1)
    probe = json.loads((ROOT / 'artifacts/gateway_cancellation_probe.json').read_text(encoding='utf-8'))
    assert probe['verified'] and probe['observed_running_and_reserved_before_close']
    print(json.dumps({'public_files_verified': len(integrity['files']), 'gpu_runs_recomputed': 40,
                      'offered': total, 'succeeded': succeeded, 'failed_retained': failed,
                      'paired_effects_match': True, 'scope': 'offline public export, not a GPU rerun'}))


if __name__ == '__main__':
    main()
