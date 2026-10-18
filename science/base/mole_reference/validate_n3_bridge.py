#!/usr/bin/env python3
"""CPU-only reconstruction and integrity check for the historical N3 bridge.

Standard library only. This validates saved evidence and arithmetic; it does
not rerun a flow solver, a learner, a numerical parity test, or a benchmark.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    def close(name, observed, expected, tolerance=1e-10):
        check(name, math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance))

    def load(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))

    manifest = load('MANIFEST.json')
    actual_files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('MANIFEST.json', 'SHA256SUMS') and '__pycache__' not in p.parts)
    listed = sorted(x['path'] for x in manifest['files'])
    check('manifest exact file set', actual_files == listed)
    expected_sums = {x['path']: x['sha256'] for x in manifest['files']}
    expected_sums['MANIFEST.json'] = digest(root / 'MANIFEST.json')
    read_sums = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split('  ', 1)
        check('checksum list has no duplicate ' + name, name not in read_sums)
        read_sums[name] = value
    check('checksum list equals manifest plus manifest digest', read_sums == expected_sums)
    for row in manifest['files']:
        p = root / row['path']
        check('safe member ' + row['path'], p.resolve().is_relative_to(root.resolve()))
        check('member bytes ' + row['path'], p.stat().st_size == row['bytes'])
        check('member sha ' + row['path'], digest(p) == row['sha256'])

    sources = load('SOURCES.json')
    for row in sources['sources']:
        p = root / row['path']
        check('original frozen byte identity ' + row['path'], p.stat().st_size == row['bytes'] and digest(p) == row['sha256'])
        if p.suffix == '.json':
            d = json.loads(p.read_text())
            if 'payload_sha256_without_hash_field' in d:
                wanted = d.pop('payload_sha256_without_hash_field')
                actual = hashlib.sha256(json.dumps(d, indent=2, sort_keys=True).encode()).hexdigest()
                check('embedded original payload ' + row['path'], actual == wanted)

    def audit(name):
        return load('raw/audit/' + name)

    result = load('N3_BRIDGE_RESULTS.json')
    final_summary = audit('mole_jax_system_efficiency_20260806_summary.json')
    times = []
    expected_targets = [32768, 100352, 1000448]
    all_stage_updates = []
    for stage, target in enumerate(expected_targets, 1):
        pair = [audit(f'mole_jax_system_efficiency_20260806_s{stage}_{rank}.json') for rank in ('gpu205', 'gpu206')]
        validation = audit(f'mole_jax_system_efficiency_20260806_s{stage}_validation.json')
        updates = max(0, target // 512 - math.ceil(32000 / 512) + 1) * 512
        all_stage_updates.append(updates)
        for rank, d in enumerate(pair):
            c, f = d['configuration'], d['final_metrics']
            check(f'stage {stage} rank {rank} original pass checks', d['status'] == 'pass' and all(d['checks'].values()))
            check(f'stage {stage} rank {rank} transitions', c['total_global_transitions_including_compile_warmup'] == target == f['replay_size'])
            check(f'stage {stage} rank {rank} exact replay-gated updates', f['sac_updates'] == updates)
            check(f'stage {stage} rank {rank} context/global work', c['local_environments'] == 16 and c['microbatch'] == 1 and c['global_transitions_per_iteration'] == c['updates_per_iteration'] == 512)
            check(f'stage {stage} rank {rank} finite/replica agreement', f['parameters_finite'] == 1.0 and f['actor_replica_checksum_delta'] == 0.0 and f['replay_replica_checksum_delta'] == 0.0)
            resumed = 0 if stage == 1 else expected_targets[stage - 2]
            check(f'stage {stage} rank {rank} continuous resumed target', c['resumed_global_interactions'] == resumed)
            calculated = resumed + (c['completed_iterations'] + (1 if stage == 1 else 0)) * 512
            check(f'stage {stage} rank {rank} iteration reconstruction', calculated == target)
        seconds = max(d['timing']['total_process_wallclock_seconds'] for d in pair)
        close(f'stage {stage} critical path reconstructed', seconds, validation['critical_path_wallclock_seconds'])
        check(f'stage {stage} original validation pass', validation['status'] == 'pass')
        times.append(seconds)
    total = sum(times)
    hours = total / 3600
    for i, sec in enumerate(times):
        close(f'new stage {i + 1} seconds', result['training']['stage_critical_path_seconds'][i], sec)
    for key, expected in [('summed_stage_critical_path_seconds', total), ('summed_stage_critical_path_hours', hours), ('allocated_training_gpu_hours', hours * 2)]:
        close('new training ' + key, result['training'][key], expected)
    check('new final interactions', result['training']['global_interactions'] == expected_targets[-1])
    check('new final SAC updates', result['training']['sac_updates'] == all_stage_updates[-1] == 968704)
    close('original total hours', hours, final_summary['timing']['staged_critical_path_hours'])
    literature_hours = final_summary['timing']['paper_reported_hours']
    check('published time/node input retained', literature_hours == result['publication_reference']['reported_hours'] == 18.5 and final_summary['resource_accounting']['paper_reported_node_count'] == result['publication_reference']['reported_nodes'] == 34)
    close('cross-system descriptive ratio', result['publication_reference']['descriptive_time_ratio'], literature_hours / hours)
    close('cross-system descriptive reduction', result['publication_reference']['descriptive_time_reduction_percent'], (1 - hours / literature_hours) * 100)

    for mode, steps, directory, summ_name in [
        ('online', 500, 'eval_online_500', 'mole_jax_system_efficiency_20260806_eval_online_500_summary.json'),
        ('replay', 2000, 'eval_replay_2000', 'mole_jax_system_efficiency_20260806_eval_replay_2000_summary_v2.json'),
    ]:
        gains, base_ids = [], []
        for rank in ('gpu205', 'gpu206'):
            suffix = '_v2' if mode == 'replay' and rank == 'gpu205' else ''
            host = load(f'raw/results/mole_jax_system_efficiency_20260806/{directory}/final_{rank}{suffix}.json')
            check(f'{mode} {rank} original integrity', host['status'] == 'pass' and all(host['checks'].values()))
            check(f'{mode} {rank} evaluation length', host['configuration']['eval_interactions'] == steps)
            base_ids.append(host['source_precursor_sha256'])
            sac = host['controllers']['sac']
            zero = host['controllers']['zero']['mean_power_mw_per_environment']
            derived = [100 * (p - z) / z for p, z in zip(sac['mean_power_mw_per_environment'], zero)]
            check(f'{mode} {rank} four states', len(derived) == 4)
            for i, (computed, saved) in enumerate(zip(derived, sac['paired_gain_percent_per_environment'])):
                close(f'{mode} {rank} state {i} gain from power', computed, saved)
            gains.extend(derived)
        saved_summary = audit(summ_name)
        new = result['evaluation'][mode]
        check(f'{mode} two independent source identities', len(set(base_ids)) == 2 and new['independent_base_trajectories'] == 2)
        check(f'{mode} eight paired states and positive signs', len(gains) == new['paired_states'] == new['positive_states'] == 8 and all(x > 0 for x in gains))
        for i, x in enumerate(gains):
            close(f'{mode} new state {i}', new['gain_percent_per_state'][i], x)
        close(f'{mode} arithmetic mean from power', new['mean_paired_gain_percent'], statistics.mean(gains))
        close(f'{mode} original summary mean', statistics.mean(gains), saved_summary['controllers']['sac']['mean_paired_gain_percent'])
        check(f'{mode} original unmet 224-episode gate retained', saved_summary['hypothesis_h2_status'] == 'fail' and not saved_summary['hypothesis_h2_checks']['evaluation_sample_count_matches_mole'])

    measures = [audit(f'mole_sac_update_throughput_{name}_gpu205_v2.json') for name in ('jax', 'torchrl')]
    check('learner benchmark configuration equal', measures[0]['configuration'] == measures[1]['configuration'])
    rates = []
    for name, d in zip(('jax', 'torchrl'), measures):
        c, m = d['configuration'], d['measurement']
        check(name + ' benchmark original pass', d['status'] == 'pass' and m['parameters_finite'])
        check(name + ' timed work exact', c['timed_blocks'] == len(m['block_seconds']) == 20 and c['updates_per_block'] == 512 and m['timed_updates'] == 10240 and c['warmup_blocks'] == 5)
        elapsed = sum(m['block_seconds'])
        close(name + ' elapsed from all blocks', elapsed, m['total_timed_seconds'])
        rate = m['timed_updates'] / elapsed
        close(name + ' update throughput', rate, m['updates_per_second'])
        close(name + ' new update throughput', rate, result['learner_execution'][name + '_updates_per_second'])
        rates.append(rate)
    ratio = rates[0] / rates[1]
    close('new learner execution ratio', ratio, result['learner_execution']['measured_execution_path_ratio'])
    close('original comparison ratio', ratio, audit('mole_sac_update_throughput_comparison_gpu205_v2.json')['measurement']['jax_over_torchrl_update_speedup'])
    source = (root / 'historical_source/scripts/benchmark_mole_sac_update_throughput.py').read_text()
    check('512-scan source retained', 'jax.lax.scan(' in source and 'jax.block_until_ready(last_metrics["critic_loss"])' in source)
    check('four per-update host scalar extractions retained', all(x in source for x in ['float(losses["loss_actor"].detach())', 'float(losses["loss_qvalue"].detach())', 'float(losses["loss_alpha"].detach())', 'float(losses["alpha"].detach())']))

    cal = audit('mole_incompact_jax_wake_response_1000_fp32_v3.json')
    power_errors, gain_errors = [], []
    for name, case in cal['power'].items():
        for i, (p, reference) in enumerate(zip(case['jax_power_by_turbine_watts'], case['reference_power_by_turbine_watts'])):
            error = 100 * (p - reference) / reference
            close(f'calibration {name} turbine {i}', error, case['per_turbine_relative_error_percent'][i])
            power_errors.append(abs(error))
        gain_error = case['jax_gain_vs_zero_percent'] - case['reference_gain_vs_zero_percent']
        close(f'calibration {name} gain error from recorded precision', gain_error, case['gain_error_percentage_points'])
        gain_errors.append(abs(gain_error))
    close('new maximum single-turbine error', max(power_errors), result['deterministic_solver_calibration']['max_absolute_single_turbine_power_error_percent'])
    close('new maximum farm gain error', max(gain_errors), result['deterministic_solver_calibration']['max_absolute_farm_gain_error_percentage_points'])
    check('uniform deterministic calibration original pass', cal['status'] == 'pass' and all(cal['checks'].values()))
    cpu = audit('mole_jax_torchrl_one_update_parity_cpu.json')
    gpu = audit('mole_jax_torchrl_one_update_parity_gpu205.json')
    threshold = {'forward_absolute': 1e-5, 'loss_absolute': 1e-4, 'updated_parameter_absolute': 1e-4}
    check('CPU parity original PASS retained', cpu['status'] == 'pass' and all(cpu['checks'].values()))
    check('strict GPU parity original FAIL retained', gpu['status'] == 'fail' and not gpu['checks']['forward_outputs_within_tolerance'] and not gpu['checks']['losses_within_tolerance'] and gpu['checks']['updated_parameters_within_tolerance'])
    check('CPU/GPU original thresholds unchanged', cpu['thresholds'] == gpu['thresholds'] == result['parity_records']['thresholds_unchanged'] == threshold)
    check('published sample count gate remains false', final_summary['checks']['h2_publication_sample_count'] is False)
    check('new science implementation not introduced', manifest['scope']['new_gpu_hours'] == 0 and manifest['scope']['new_les_runs'] == 0)

    return {'schema': 'mole-n3-bridge-cpu-validation/1.0', 'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'root': str(root), 'manifest_sha256': digest(root / 'MANIFEST.json'), 'results_sha256': digest(root / 'N3_BRIDGE_RESULTS.json'), 'checks_passed': len(checks), 'checks': checks, 'source_files_verified': len(sources['sources']), 'bundle_files_verified': len(listed), 'recomputed': {'training_hours': hours, 'global_interactions': 1000448, 'sac_updates': all_stage_updates[-1], 'learner_execution_ratio': ratio, 'max_single_turbine_power_error_percent': max(power_errors), 'max_farm_gain_error_percentage_points': max(gain_errors)}, 'preserved_negative_records': ['strict GPU one-update parity FAIL', 'online/replay 224-episode matching gate FAIL'], 'scope': 'Saved-record arithmetic, byte integrity and original gate preservation only. No new LES/learner/GPU run or parity retest; no external large-file hash reread.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if args.output and args.output.resolve().is_relative_to(root):
        raise ValueError('Keep validation receipts outside the frozen extension directory.')
    try:
        result = run(root)
        code = 0
    except Exception as e:
        result = {'schema': 'mole-n3-bridge-cpu-validation/1.0', 'status': 'FAIL', 'error': type(e).__name__ + ': ' + str(e), 'root': str(root)}
        code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2, sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
