#!/usr/bin/env python3
"""CPU-only independent verification of the corrected-runtime confirmation.

Preserves the earlier primary FAIL. Never edits the prospective acceptance gate.
"""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from verify_returned_prefix import sha, load, identity, docker_time

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
OLD = AUDIT.parent / 'paper1_mainline_revision_20261015'
PACK = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/github_release_ready_20261014/repository/n81_20261014_visual_revision')

def main():
    frozen = load(HERE / 'PRIMARY_ACCEPTANCE_FREEZE.json')
    for row in frozen['files']:
        p = Path(row['path'])
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    primary = load(HERE / 'FINAL_ACCEPTANCE.json')
    assert primary['status'] == 'fail' and primary['accepted'] is False
    protocol = load(AUDIT / 'CORRECTED_RUNTIME_CONFIRMATION_PROTOCOL.json')
    assert protocol['primary_acceptance_sha256'] == sha(HERE / 'FINAL_ACCEPTANCE.json')
    prior_env = Path(protocol['verified_runtime_evidence']['path'])
    assert sha(prior_env) == protocol['verified_runtime_evidence']['sha256']
    superseded = load(AUDIT / 'DIAGNOSTIC_CONTINUATION_SUPERSEDED.json')
    assert superseded['GPU_started'] is False and superseded['RL_only_controls_executed'] == 0
    assert superseded['superseded_decision_sha256'] == sha(AUDIT / 'DIAGNOSTIC_CONTINUATION_DECISION.json')

    current = load(AUDIT / 'CURRENT_CORRECTED_EXECUTION.json')
    root = Path(current['local_root'])
    returned = root / 'returned'
    run = returned / 'results/replay'
    records = []
    output_manifest = load(returned / 'OUTPUT_MANIFEST.json')
    for row in output_manifest:
        p = returned / row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
        records.append(identity(p))
    remote = load(root / 'OUTPUT_REMOTE_RECEIPT.json')
    archive = root / 'replay_outputs.tar.gz'
    assert archive.stat().st_size == remote['bytes'] and sha(archive) == remote['sha256']
    bundle = root / 'frozen_prefix_runtime.tar.gz'
    assert bundle.stat().st_size == current['bundle_bytes'] and sha(bundle) == current['bundle_sha256']
    members = load(root / 'BUNDLE_MEMBERS.json')
    assert sha(bundle) == load(Path(current['primary_execution_root']) / 'LOCAL_PREPARE.json')['bundle_sha256']
    source_count = 0
    with tarfile.open(bundle, 'r:gz') as tf:
        assert {m.name for m in tf.getmembers() if m.isfile()} == {m['name'] for m in members} | {'BUNDLE_MEMBERS.json'}
        assert tf.extractfile('BUNDLE_MEMBERS.json').read() == (root / 'BUNDLE_MEMBERS.json').read_bytes()
        for row in members:
            data = tf.extractfile(row['name']).read()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            if row['name'].startswith('pack/source/'):
                source = PACK / row['name'][5:]
                assert sha(source) == row['sha256']
                if row['name'].endswith('.py'):
                    source_count += 1

    readiness = load(returned / 'CPU_DRIVER_READINESS.json')
    before = load(OLD / 'PREFIX_DRIVER_READINESS.json')
    assert readiness['state'] == before['state']
    for key in ('common_state_SHA', 'final_Actor_SHA', 'inlet_prefix_SHA'):
        assert readiness[key] == before[key]
    assert readiness['frozen_source_files_verified'] == 63
    for path, expected in [(Path(before['common_state_path']), before['common_state_SHA']),
        (PACK / 'actors/actor.pkl', before['final_Actor_SHA']),
        (Path(before['inlet_prefix_path']), before['inlet_prefix_SHA'])]:
        assert sha(path) == expected
        records.append(identity(path))

    start = load(returned / 'ALLOCATION_START.json')
    end = load(returned / 'ALLOCATION_EXIT.json')
    state = load(returned / 'GPU_CONTAINER_FINAL_STATE.json')
    worker = load(run / 'EXIT.json')
    result = load(run / 'RESULT.json')
    contract = load(returned / 'ORIGINAL_RUNTIME_CONTRACT.json')
    plan = load(returned / 'GPU_PLAN.json')
    assert (root / 'PROTOCOL.json').read_bytes() == (AUDIT / 'CORRECTED_RUNTIME_CONFIRMATION_PROTOCOL.json').read_bytes()
    protocol_sha = sha(root / 'PROTOCOL.json')
    assert protocol_sha == start['protocol_sha256'] == end['protocol_sha256'] == current['protocol_sha256']
    assert datetime.datetime.fromisoformat(protocol['recorded_before_GPU_launch_utc']).timestamp() < start['unix_before_docker_create']
    wrapper = root / 'remote_bounded_replay.executed.py'
    assert sha(wrapper) == start['wrapper_sha256_actual'] == current['wrapper_sha256']
    assert end['status'] == 'complete' and end['exit_code'] == state['ExitCode'] == worker['exit_code'] == 0
    assert not state['Running'] and end['container_stopped_verified'] and end['container_removed_verified']
    assert not state['OOMKilled']
    assert contract == plan['runtime_contract']
    assert contract['image'] == plan['image_config_id'] and contract['image'] in start['argv']
    old_env = load(prior_env)['environment_whitelist']
    assert all(contract['environment_whitelist'][k] == v for k, v in old_env.items())
    declared_env = {start['argv'][i+1].partition('=')[0]: start['argv'][i+1].partition('=')[2]
        for i, token in enumerate(start['argv'][:-1]) if token == '-e'}
    assert all(declared_env[k] == v for k, v in contract['environment_whitelist'].items())
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE' not in declared_env and contract['preallocation_key_absent']
    for mount in contract['driver_mounts']:
        assert not mount['RW'] and mount['Type'] == 'bind'
        text = 'type=bind,src='+mount['Source']+',dst='+mount['Destination']+',readonly'
        assert text in start['argv']
    for dev in contract['devices']:
        assert dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions'] in start['argv']
    assert start['gpu_id'] == end['gpu_id'] == '0'
    assert start['gpu_uuid'] == end['gpu_uuid'] == primary['gpu_uuid']
    assert result['status'] == 'paired_original_prefix_bitwise_replay_pass'
    assert result['common_state_sha256'] == before['common_state_SHA']
    assert result['actor_sha256'] == before['final_Actor_SHA']
    assert result['inlet_prefix_sha256'] == before['inlet_prefix_SHA']
    assert result['physical_replay_executed'] is True and result['training_executed'] is False
    assert result == end['result']
    assert result['jax_version'] == plan['runtime']['CPU_framework_versions']['jax']

    fields_count = elements = 0
    branches = []
    for row in result['branches']:
        branch = row['branch']
        actual_path = run / f'{branch}.npz'
        original_path = PACK / f'data/BC/B_w1/{branch}.npz'
        reference_path = OLD / f'REFERENCE_B_W1_PREFIX8_{branch}.npz'
        fields = {}
        with np.load(actual_path, allow_pickle=False) as actual, np.load(original_path, allow_pickle=False) as original, np.load(reference_path, allow_pickle=False) as reference:
            assert set(actual.files) == set(original.files) == set(reference.files)
            for field in actual.files:
                a, r, p = actual[field], original[field][:8], reference[field]
                assert a.dtype == r.dtype == p.dtype == np.dtype('float32')
                assert a.shape == r.shape == p.shape and np.isfinite(a).all()
                exact = a.tobytes(order='C') == r.tobytes(order='C') == p.tobytes(order='C')
                assert exact, (branch, field)
                fields[field] = {'shape': list(a.shape), 'dtype': 'float32', 'element_count': int(a.size),
                    'bitwise_equal_to_original_trace_and_reference': exact,
                    'sha256_C_order_data_bytes': hashlib.sha256(a.tobytes(order='C')).hexdigest(),
                    'max_abs_difference': 0.0}
                fields_count += 1
                elements += int(a.size)
            energy = float(np.sum(actual['farm_power_mw'], dtype=np.float64) * 10 / 3600)
        assert row['final_farm_step_index'] == 7900 and row['final_interaction_index'] == 8
        assert row['global_final_inlet_cursor_exclusive'] == 27900
        assert row['comparison']['all_fields_bitwise_equal'] is True
        branches.append({'branch': branch, 'controls': 8, 'fields': fields, 'energy_MWh': energy,
            'final_farm_step_index': 7900, 'final_interaction_index': 8,
            'global_final_inlet_cursor_exclusive': 27900})
        records.extend([identity(original_path), identity(reference_path)])
    assert [b['branch'] for b in branches] == ['zero', 'rl_r1'] and fields_count == 15 and elements == 3320
    gain = (branches[1]['energy_MWh'] / branches[0]['energy_MWh'] - 1) * 100
    seconds = float(end['outer_monotonic_seconds_including_create_and_cleanup'])
    gpu_h = seconds / 3600
    assert abs(gpu_h - end['conservative_allocated_GPUh']) < 1e-12
    docker_seconds = (docker_time(state['FinishedAt']) - docker_time(state['StartedAt'])).total_seconds()
    assert 0 < docker_seconds <= seconds < protocol['resource_caps']['new_single_GPU_container_lifecycle_seconds_max']
    assert end['unix_before_docker_create'] <= docker_time(state['StartedAt']).timestamp()
    assert docker_time(state['FinishedAt']).timestamp() <= end['unix_after_cleanup']
    assert abs((end['unix_after_cleanup'] - end['unix_before_docker_create']) - seconds) < 0.01
    attempts = sorted((AUDIT / 'execution').glob('*/returned/ALLOCATION_START.json'))
    assert len(attempts) == 2
    total_gpu_h = primary['conservative_GPUh'] + gpu_h
    assert total_gpu_h < 0.25
    records.extend([identity(returned / 'OUTPUT_MANIFEST.json'), identity(archive), identity(bundle),
        identity(root / 'OUTPUT_REMOTE_RECEIPT.json'), identity(root / 'BUNDLE_MEMBERS.json'),
        identity(root / 'PROTOCOL.json'), identity(wrapper), identity(prior_env)])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    corrected = {'schema_version': 1, 'created_utc': now, 'status': 'pass', 'accepted': True,
        'run_role': 'corrected_runtime_confirmation', 'primary_strict_test_status': 'fail_unchanged',
        'physical_replay_executed': True, 'paired_replay_completed': True,
        'all_15_output_arrays_bitwise_equal': True, 'compared_array_count': fields_count,
        'compared_float32_element_count': elements, 'input_identity_checks_pass': True,
        'frozen_scientific_source_files_verified': 63,
        'source_bundle_identical_to_first_attempt': True,
        'scoring_cursor_checks_pass': True, 'returned_file_SHA_checks_pass': True,
        'returned_manifest_verified_file_count': len(output_manifest),
        'recorded_original_runtime_whitelist_matches_declared_launch': True,
        'original_readonly_driver_mounts_and_GPU0_mapping_match_declared_launch': True,
        'runtime_scope': 'Recorded original whitelist, driver binding and container settings restored; not a claim that every unrecorded environment variable is known.',
        'outer_lifecycle_seconds': seconds, 'Docker_running_seconds': docker_seconds,
        'conservative_GPUh': gpu_h, 'cap_satisfied': True, 'gpu_attempt_count_in_this_protocol': 1,
        'worker_exit_code': 0, 'container_exit_code': 0, 'container_stopped_verified': True,
        'container_removed_verified': True, 'OOMKilled': False,
        'branches': branches, 'paired_prefix_gain_percent': gain,
        'paired_prefix_gain_used_as_acceptance_gate': False,
        'RESULT_path': str(run / 'RESULT.json'), 'EXIT_path': str(run / 'EXIT.json'),
        'ALLOCATION_EXIT_path': str(returned / 'ALLOCATION_EXIT.json'),
        'source_receipts_and_outputs': records,
        'scientific_scope': [
            'Exact reproduction of 15 observable output arrays over one 80-second zero/RL prefix under the corrected recorded runtime.',
            'Original first attempt remains a strict FAIL; restoring multiple launch settings together does not isolate a unique cause.',
            'No final full-flow-state byte-identity claim: no independently archived original 8-control final flow exists.',
            'No full-window B/C rerun, cross-hardware validation, new training, or new positive-gain experiment was executed.',
            'The original negative prefix gain is reproduced and retained; it does not replace the full-window B/C results.',
        ]}
    corrected_path = HERE / 'CORRECTED_RUNTIME_ACCEPTANCE.json'
    corrected_path.write_text(json.dumps(corrected, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    diagnostic = {'status': 'not_executed_superseded_before_launch', 'physical_replay_executed': False,
        'RL_only_controls_executed': 0, 'GPUh': 0.0,
        'reason': 'Verified runtime-contract omission led to a separately sealed corrected-runtime confirmation instead.',
        'supersession_receipt': identity(AUDIT / 'DIAGNOSTIC_CONTINUATION_SUPERSEDED.json')}
    (HERE / 'DIAGNOSTIC_ACCEPTANCE.json').write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False) + '\n')
    overall = {'schema_version': 1, 'created_utc': now,
        'status': 'corrected_runtime_confirmation_pass_with_primary_fail_preserved',
        'user_requested_corrected_8_plus_8_completed': True,
        'primary_status': 'fail', 'primary_acceptance': identity(HERE / 'FINAL_ACCEPTANCE.json'),
        'corrected_status': 'pass', 'corrected_acceptance': identity(corrected_path),
        'RL_only_diagnostic_executed': False,
        'gpu_attempt_count': len(attempts), 'no_third_GPU_attempt': True,
        'total_executed_control_intervals': 24,
        'control_accounting': {'primary_zero': 8, 'primary_RL': 0, 'corrected_zero': 8, 'corrected_RL': 8},
        'total_GPU_container_lifecycle_seconds': primary['outer_lifecycle_seconds'] + seconds,
        'conservative_total_GPUh': total_gpu_h, 'user_total_cap_GPUh': 0.25,
        'cap_satisfied': total_gpu_h <= 0.25,
        'all_corrected_15_output_arrays_bitwise_equal': True,
        'corrected_compared_array_count': 15, 'corrected_compared_float32_element_count': 3320,
        'paired_prefix_gain_percent': gain, 'negative_prefix_retained': gain < 0,
        'all_task_owned_GPU_containers_stopped_and_removed': True,
        'scientific_implementation_changed': False, 'statistical_gain_evidence_changed': False,
        'no_field_accuracy_or_cross_hardware_claim': True,
        'scope': 'Successful corrected-runtime output-prefix reproduction after one preserved wrapper-mismatch failure. No retrospective pass of the first protocol and no claim of final full-state bitwise identity.',
        'cost_components': [{'role': 'primary_failed', 'GPUh': primary['conservative_GPUh'],
            'receipt': identity(Path(primary['ALLOCATION_EXIT_path']))},
            {'role': 'corrected_runtime_confirmation', 'GPUh': gpu_h,
            'receipt': identity(returned / 'ALLOCATION_EXIT.json')}],
    }
    (HERE / 'OVERALL_NUMERICAL_REPLAY.json').write_text(json.dumps(overall, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: overall[k] for k in ['status','gpu_attempt_count','total_executed_control_intervals','conservative_total_GPUh','cap_satisfied','corrected_compared_array_count','corrected_compared_float32_element_count','paired_prefix_gain_percent']}))

if __name__ == '__main__':
    main()
