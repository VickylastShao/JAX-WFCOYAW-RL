#!/usr/bin/env python3
"""Independent CPU-only acceptance of one returned prefix attempt.

No tolerance is used for the prospective bitwise acceptance. Error magnitudes
are descriptive only and cannot convert a failed identity check into a pass.
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
OLD = AUDIT.parent / 'paper1_mainline_revision_20261015'
PACK = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/github_release_ready_20261014/repository/n81_20261014_visual_revision')
STATE_SHA = '1db258cbe44cfe3eedcca2ad82c7965baba35b7f280be520dc74128d838a58be'
ACTOR_SHA = 'be68a370939eaecf3408af4bbbc2db84f0e9677fe46fe8f8189f933ff0a54ca5'
PREFIX_SHA = '1780de4e3c66474825b4880a23f18c2bb941b2efee00d4adc5010bbb45cab7b0'
IMAGE_ID = 'sha256:a72fc0a237072f4177524eab7896b3d5ab8d1e2d7d62eb1bf45ff163e9c5729d'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text())

def identity(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def docker_time(value):
    # Python 3.10 accepts microseconds, whereas Docker emits nanoseconds.
    return datetime.datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', value.replace('Z', '+00:00')))

def main():
    current = load(AUDIT / 'CURRENT_EXECUTION.json')
    execution = Path(current['local_root'])
    returned = execution / 'returned'
    run = returned / 'results/replay'
    records = []
    output_manifest = load(returned / 'OUTPUT_MANIFEST.json')
    for item in output_manifest:
        p = returned / item['path']
        assert p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], item['path']
        records.append(identity(p))
    remote_receipt = load(execution / 'OUTPUT_REMOTE_RECEIPT.json')
    archive = execution / 'replay_outputs.tar.gz'
    assert archive.stat().st_size == remote_receipt['bytes'] and sha(archive) == remote_receipt['sha256']
    bundle = execution / 'frozen_prefix_runtime.tar.gz'
    assert bundle.stat().st_size == current['bundle_bytes'] and sha(bundle) == current['bundle_sha256']
    members = load(execution / 'BUNDLE_MEMBERS.json')
    with tarfile.open(bundle, 'r:gz') as tf:
        assert set(m.name for m in tf.getmembers() if m.isfile()) == set(m['name'] for m in members) | {'BUNDLE_MEMBERS.json'}
        assert tf.extractfile('BUNDLE_MEMBERS.json').read() == (execution / 'BUNDLE_MEMBERS.json').read_bytes()
        for row in members:
            data = tf.extractfile(row['name']).read()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            if row['name'].startswith('pack/source/'):
                src = PACK / row['name'][5:]
                assert src.is_file() and sha(src) == row['sha256']

    readiness = load(returned / 'CPU_DRIVER_READINESS.json')
    local_readiness = load(OLD / 'PREFIX_DRIVER_READINESS.json')
    assert readiness['status'] == 'ready_for_bounded_replay'
    assert readiness['frozen_source_files_verified'] == 63
    assert readiness['common_state_SHA'] == STATE_SHA
    assert readiness['final_Actor_SHA'] == ACTOR_SHA
    assert readiness['inlet_prefix_SHA'] == PREFIX_SHA
    assert readiness['state'] == local_readiness['state']
    for path, expected in [(Path(local_readiness['common_state_path']), STATE_SHA),
                           (PACK / 'actors/actor.pkl', ACTOR_SHA),
                           (Path(local_readiness['inlet_prefix_path']), PREFIX_SHA)]:
        assert sha(path) == expected
        records.append(identity(path))
    plan = load(returned / 'GPU_PLAN.json')
    start = load(returned / 'ALLOCATION_START.json')
    end = load(returned / 'ALLOCATION_EXIT.json')
    container = load(returned / 'GPU_CONTAINER_FINAL_STATE.json')
    worker_exit = load(run / 'EXIT.json')
    comparison = load(run / 'COMPARISON.json')
    assert plan['image_config_id'] == IMAGE_ID and IMAGE_ID in start['argv']
    assert start['gpu_id'] in {'0', '1', '2', '3'}
    assert start['gpu_id'] == end['gpu_id'] and start['gpu_uuid'] == end['gpu_uuid']
    assert not container['Running'] and end['container_stopped_verified'] and end['container_removed_verified']
    assert container['ExitCode'] == worker_exit['exit_code'] == end['exit_code'] == 1
    assert not container['OOMKilled']
    assert not (run / 'RESULT.json').exists()
    assert not (run / 'rl_r1.npz').exists()
    attempts = sorted((AUDIT / 'execution').glob('*/returned/ALLOCATION_START.json'))
    assert len(attempts) == end['GPU_attempts'] == 1
    outer_seconds = float(end['outer_monotonic_seconds_including_create_and_cleanup'])
    gpu_h = outer_seconds / 3600
    assert abs(gpu_h - end['conservative_allocated_GPUh']) < 1e-12
    docker_started = docker_time(container['StartedAt'])
    docker_finished = docker_time(container['FinishedAt'])
    docker_seconds = (docker_finished - docker_started).total_seconds()
    assert 0 < docker_seconds <= outer_seconds < 900
    assert end['unix_before_docker_create'] <= docker_started.timestamp()
    assert docker_finished.timestamp() <= end['unix_after_cleanup']
    assert abs((end['unix_after_cleanup'] - end['unix_before_docker_create']) - outer_seconds) < 0.01

    branches = []
    total_elements = 0
    for record in comparison['branches']:
        branch = record['branch']
        actual_path = run / f'{branch}.npz'
        original_path = PACK / f'data/BC/B_w1/{branch}.npz'
        fields = {}
        with np.load(actual_path, allow_pickle=False) as actual, np.load(original_path, allow_pickle=False) as original:
            assert set(actual.files) == set(original.files)
            for field in actual.files:
                a, r = actual[field], original[field][:8]
                assert a.shape == r.shape and a.dtype == r.dtype == np.dtype('float32')
                assert np.isfinite(a).all() and np.isfinite(r).all()
                difference = np.abs(a.astype(np.float64) - r.astype(np.float64))
                nonzero = np.abs(r.astype(np.float64)) > 0
                relative = difference[nonzero] / np.abs(r.astype(np.float64))[nonzero]
                byte_equal = a.tobytes(order='C') == r.tobytes(order='C')
                # Values here are all nonnegative; signed integer ordering gives ULP distance.
                ulps = np.abs(a.view(np.int32).astype(np.int64) - r.view(np.int32).astype(np.int64))
                fields[field] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'element_count': int(a.size),
                    'bitwise_equal': byte_equal, 'mismatched_element_count': int(np.count_nonzero(a.view(np.uint32) != r.view(np.uint32))),
                    'max_abs_difference': float(difference.max(initial=0)),
                    'max_relative_difference_where_reference_nonzero': float(relative.max(initial=0)),
                    'max_ULP_distance_nonnegative_outputs': int(ulps.max(initial=0))}
                assert fields[field]['max_abs_difference'] == record['comparison']['fields'][field]['max_abs_difference']
                assert byte_equal == record['comparison']['fields'][field]['bitwise_equal']
                total_elements += int(a.size)
            actual_energy = float(np.sum(actual['farm_power_mw'], dtype=np.float64) * 10 / 3600)
            original_energy = float(np.sum(original['farm_power_mw'][:8], dtype=np.float64) * 10 / 3600)
        assert record['final_farm_step_index'] == 7900 and record['final_interaction_index'] == 8
        assert record['global_final_inlet_cursor_exclusive'] == 27900
        branches.append({'branch': branch, 'controls': 8, 'fields': fields,
            'actual_energy_MWh': actual_energy, 'original_energy_MWh': original_energy,
            'energy_difference_MWh': actual_energy-original_energy,
            'relative_energy_difference': actual_energy/original_energy-1,
            'final_farm_step_index': 7900, 'final_interaction_index': 8,
            'global_final_inlet_cursor_exclusive': 27900})
    assert [b['branch'] for b in branches] == ['zero']
    assert total_elements == 1336
    all_equal = all(v['bitwise_equal'] for b in branches for v in b['fields'].values())
    assert not all_equal
    records.extend([identity(returned / 'OUTPUT_MANIFEST.json'), identity(archive), identity(bundle),
        identity(execution / 'OUTPUT_REMOTE_RECEIPT.json'), identity(execution / 'BUNDLE_MEMBERS.json'),
        identity(PACK / 'data/BC/B_w1/RESULT.json')])
    result = {
        'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'fail', 'accepted': False, 'physical_replay_executed': True,
        'failure_class': 'strict_original_prefix_output_identity_mismatch',
        'all_15_output_arrays_bitwise_equal': False,
        'all_executed_output_arrays_bitwise_equal': all_equal,
        'planned_array_count': 15, 'compared_array_count': 7,
        'compared_float32_element_count': total_elements,
        'matching_array_count': sum(v['bitwise_equal'] for b in branches for v in b['fields'].values()),
        'executed_branches': ['zero'], 'completed_controls': {'zero': 8, 'rl_r1': 0},
        'paired_replay_completed': False, 'paired_replay_gain_percent': None,
        'input_identity_checks_pass': True, 'scoring_cursor_checks_pass': True,
        'returned_file_SHA_checks_pass': True, 'returned_manifest_verified_file_count': len(output_manifest),
        'outer_lifecycle_seconds': outer_seconds, 'Docker_running_seconds': docker_seconds,
        'conservative_GPUh': gpu_h, 'cap_satisfied': gpu_h <= 0.25,
        'gpu_attempt_count': len(attempts), 'worker_exit_code': worker_exit['exit_code'],
        'container_exit_code': container['ExitCode'], 'container_stopped_verified': True,
        'container_removed_verified': True, 'OOMKilled': False,
        'image_config_id': IMAGE_ID, 'gpu_id': end['gpu_id'], 'gpu_uuid': end['gpu_uuid'],
        'branches': branches,
        'source_receipts_and_outputs': records,
        'original_result_path': str(PACK / 'data/BC/B_w1/RESULT.json'),
        'RESULT_path': None,
        'EXIT_path': str(run / 'EXIT.json'),
        'ALLOCATION_EXIT_path': str(returned / 'ALLOCATION_EXIT.json'),
        'comparison_path': str(run / 'COMPARISON.json'),
        'scientific_scope': [
            'The prospective strict byte-identity gate failed after the zero-yaw 8-control branch; the RL branch was correctly not run.',
            'Magnitudes are descriptive diagnostics only; no tolerance or pass criterion was relaxed after observing the result.',
            'The failure does not by itself establish that the original full-window B/C gains are false; those traces and aggregate calculations are unchanged.',
            'No claim of completed 8+8 replay, final full-flow-state bitwise identity, or cross-hardware reproducibility is supported.',
            'E1 112-array equivalence is a separate execution experiment and is not evidence of passing this policy replay.',
            'The specific source of the FP32 output differences remains unproven; runtime/compilation/input-layout explanations are hypotheses.',
        ],
    }
    (HERE / 'FINAL_ACCEPTANCE.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status','accepted','compared_array_count','compared_float32_element_count','matching_array_count','conservative_GPUh','cap_satisfied']}))

if __name__ == '__main__':
    main()
