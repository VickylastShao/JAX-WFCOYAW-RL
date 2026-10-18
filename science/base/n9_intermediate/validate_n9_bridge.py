#!/usr/bin/env python3
"""Independently reconstruct saved N9 results on CPU using only stdlib.

Checks saved evidence; it never imports the scientific code or unpickles an
actor, and never reruns LES, learning, source generation or remote commands.
"""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(root):
    checks = []
    def check(name, yes):
        checks.append({'check': name, 'passed': bool(yes)})
        if not yes:
            raise ValueError(name)
    def close(name, actual, expected, tol=1e-10):
        check(name, math.isclose(actual, expected, rel_tol=tol, abs_tol=tol))
    def load(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))
    manifest = load('MANIFEST.json')
    actual = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('MANIFEST.json', 'SHA256SUMS') and '__pycache__' not in p.parts)
    check('manifest exact member set', actual == sorted(x['path'] for x in manifest['files']))
    for row in manifest['files']:
        p = root / row['path']
        check('safe member ' + row['path'], p.resolve().is_relative_to(root))
        check('bytes ' + row['path'], p.stat().st_size == row['bytes'])
        check('sha ' + row['path'], sha(p) == row['sha256'])
    sums = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split('  ', 1)
        check('unique checksum row ' + name, name not in sums)
        sums[name] = value
    wanted = {x['path']: x['sha256'] for x in manifest['files']}
    wanted['MANIFEST.json'] = sha(root / 'MANIFEST.json')
    check('checksum list exact', sums == wanted)
    sources = load('SOURCES.json')
    for row in sources['sources']:
        p = root / row['path']
        check('original bytes ' + row['path'], p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])
        if row['historical_execution_binding']:
            check('historical source binding ' + row['path'], sha(p) == row['historical_execution_binding']['sha256'])
        if p.suffix == '.json':
            d = json.loads(p.read_text())
            if 'payload_sha256_without_hash_field' in d:
                stored = d.pop('payload_sha256_without_hash_field')
                style = row['embedded_payload_canonicalization']
                if style == 'json-sort-keys-indent2-utf8':
                    serialized = json.dumps(d, indent=2, sort_keys=True).encode('utf-8')
                elif style == 'json-sort-keys-compact-ascii-no-nan':
                    serialized = json.dumps(d, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
                else:
                    raise ValueError('Unknown original payload canonicalization')
                observed = hashlib.sha256(serialized).hexdigest()
                check('original embedded payload ' + row['path'], stored == observed)

    result = load('N9_BRIDGE_RESULTS.json')
    cross = load('CROSS_SOURCE_BINDING.json')
    trains = [load('raw/training/' + rank + '.training.json') for rank in ('gpu205', 'gpu206')]
    records = [load('raw/training/' + rank + '.endpoint_record.v2.json') for rank in ('gpu205', 'gpu206')]
    pair = load('raw/training/endpoint_pair_audit.v2.json')
    check('original paired complete-state audit pass', pair['status'] == 'pass' and all(pair['pair_checks'].values()))
    expected_counts = []
    for rank, d, rec in zip(('gpu205', 'gpu206'), trains, records):
        c, f, meta = d['configuration'], d['final_metrics'], d['checkpoint']['metadata']
        check(rank + ' original training pass', d['status'] == 'pass' and all(d['checks'].values()))
        count = (c['completed_iterations'] + 1) * c['global_transitions_per_iteration']
        check(rank + ' interactions from iteration schedule', count == c['total_global_transitions_including_compile_warmup'] == result['training']['global_interactions'] == 1000448)
        updates = count - c['graph_actor_minimum_replay_size']
        check(rank + ' SAC update work', updates == f['sac_updates'] == result['training']['SAC_updates'] == 967680)
        actor_updates = (updates - c['graph_actor_warmup_updates']) // c['graph_actor_update_period']
        check(rank + ' delayed Actor update work', actor_updates == c['sac_actor_updates_applied_total'] == result['training']['Actor_updates'] == 30112)
        close(rank + ' Actor work summed from complete history', sum(x['sac_actor_updates_applied'] for x in d['history']), actor_updates)
        check(rank + ' finalized/pending n-step replay accounting', f['replay_size'] + 2 * pair['processes'][0]['pending_rows_local'] == count and f['replay_size'] == result['training']['Replay_rows_finalized'])
        check(rank + ' fresh Actor without pretrained checkpoint', meta['resume_mode'] == 'fresh' and meta['initial_actor_checkpoint_sha256'] is None and meta['initial_actor_checkpoint_metadata'] is None and c['resumed_global_interactions'] == 0 and c['resumed_from_full_checkpoint'] is None)
        check(rank + ' 32 histories with microbatch one', c['local_environments'] == 16 and d['environment']['process_count'] == 2 and c['microbatch'] == 1)
        check(rank + ' directional graph and targets', c['training_algorithm'] == 'graph_sac_directional' and c['action_parameterization'] == 'per_turbine_target' and meta['direction_condition'] and not meta['steering_branch_condition'] and meta['reflection_equivariant_actor'])
        check(rank + ' 16-step return and declared reward', c['formal_n_step'] == result['training']['return_horizon'] == 16 and c['formal_reward_name'] == result['training']['reward'] == 'reward_r1')
        check(rank + ' endpoint identity binding', rec['actor_sha256'] == result['selection']['selected_actor_sha256'] and rec['sac_updates'] == updates and rec['configuration']['actual_global_interactions'] == count)
        check(rank + ' finite synchronized endpoint', f['parameters_finite'] == 1 and f['actor_replica_checksum_delta'] == f['replay_replica_checksum_delta'] == 0)
        expected_counts.append(count)
    check('two training implementation maps agree', trains[0]['implementation_sha256'] == trains[1]['implementation_sha256'])
    check('new initialization and architecture description', result['training']['initialization'] == 'fresh_neutral_no_teacher' and result['training']['resume_mode'] == 'fresh' and result['training']['initial_actor_checkpoint_sha256'] is None and result['training']['architecture'] == 'graph_sac_directional')
    meta = trains[0]['checkpoint']['metadata']
    for name in ['action_parameterization', 'direction_condition', 'steering_branch_condition', 'reflection_equivariant_actor', 'graph_geometry']:
        check('new executed policy metadata ' + name, result['executed_policy'][name] == meta[name])
    check('new action and time contract', result['executed_policy']['action_scale'] == 1 and result['executed_policy']['max_yaw_angle_deg'] == 40 and result['executed_policy']['max_yaw_rate_deg_per_s'] == 1 and result['executed_policy']['control_seconds'] == 10 and result['executed_policy']['max_yaw_change_deg_per_control'] == 10)
    for filename, h in trains[0]['implementation_sha256'].items():
        check('executed source ' + filename, sha(root / 'source/training' / filename) == h)
    c = trains[0]['configuration']
    check('LES grid/physical geometry', c['layout_contract']['grid'] == result['training']['grid'] == [193, 41, 175] and c['layout_contract']['num_turbines'] == 9)
    positions = c['layout_contract']['turbine_positions_m']
    close('streamwise spacing 5D', positions[3][0] - positions[0][0], result['training']['streamwise_spacing_D'] * 126)
    close('lateral spacing 5D', positions[1][2] - positions[0][2], result['training']['lateral_spacing_D'] * 126)

    initial_bad = load('raw/initialization/final_selection_freeze.json')
    initial = load('raw/initialization/final_selection_freeze_retry1.json')
    check('initial failed file/payload identity gate preserved', initial_bad['status'] == 'fail' and initial_bad['checks']['p0_payload_hash_identity'] is False)
    check('corrected initial contract retained separately', initial['status'] == 'pass' and all(initial['checks'].values()) and initial['selected']['actor_initialization'] == 'fresh_neutral_no_teacher')
    p0 = load('raw/initialization/p0_table.json')
    table_sha = hashlib.sha256(json.dumps(p0['p0_table'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    check('zero-yaw normalization canonical table identity', table_sha == p0['p0_table_sha256'] == initial['selected']['p0_table_sha256'])
    check('zero-yaw artifact payload separately bound', p0['payload_sha256_without_hash_field'] == initial['selected']['p0_artifact_payload_sha256'])
    selection = load('raw/selection/selection_three_candidates.json')
    freeze = load('raw/selection/freeze.endpoint1000448.json')
    check('three-candidate comparison frozen', selection['status'] == 'pass' and selection['freeze_payload_sha256'] == freeze['payload_sha256_without_hash_field'] and len(selection['candidates']) == 3)
    candidate_gains = []
    for candidate in selection['candidates']:
        records_by_direction = {}
        for row in candidate['records']:
            close('candidate host mean ' + str(candidate['actual_global_interactions']) + ':' + row['host'] + ':' + str(row['angle_deg']), statistics.mean(w['gain_percent'] for w in row['windows']), row['mean_gain_percent'])
            records_by_direction.setdefault(row['angle_deg'], []).append(row['mean_gain_percent'])
            check('candidate action scale retained', row['configuration']['action_scale'] == 1 and row['configuration']['steering_branch'] is None and row['configuration']['steering_branch_mode'] == 'fixed')
        check('candidate development design', len(records_by_direction) == 9 and all(len(v) == 2 for v in records_by_direction.values()))
        mean_gain = statistics.mean(statistics.mean(v) for v in records_by_direction.values())
        close('candidate equal direction mean reconstructed ' + candidate['candidate'], mean_gain, candidate['equal_direction_mean_gain_percent'])
        positivity = all(row['mean_gain_percent'] > 0 for row in candidate['records'])
        check('candidate original positivity gate reconstructed ' + candidate['candidate'], positivity == candidate['checks']['every_direction_positive_on_each_precursor'])
        check('candidate hard gate matches original checks ' + candidate['candidate'], candidate['hard_gate_pass'] == all(candidate['checks'].values()))
        candidate_gains.append(mean_gain)
    check('original early candidate failures preserved', [x['hard_gate_pass'] for x in selection['candidates']] == [False, False, True])
    check('selected endpoint and actor unchanged', selection['selected']['actual_global_interactions'] == 1000448 and selection['selected']['candidate'] == 'step1000448' and selection['selected']['actor_sha256'] == result['selection']['selected_actor_sha256'])
    check('actual selected Actor bytes', sha(root / 'actors/actor_step1000448.pkl') == selection['selected']['actor_sha256'] and (root / 'actors/actor_step1000448.pkl').stat().st_size == 1154217)

    archive = load('raw/evaluation/archive_manifest.json')
    evaluation_freeze = load('raw/evaluation/evaluation_freeze.json')
    release = load('raw/evaluation/release.json')
    archive_audit = load('raw/evaluation/archive_integrity.json')
    integrity = load('raw/evaluation/integrity.json')
    combined = load('raw/evaluation/wp7_combined_summary.json')
    check('one distinct holdout source', archive['source_state']['seed'] == combined['independent_precursor_seed'] == result['evaluation']['holdout_source_seed'] == 20260904)
    dev = freeze['archives']
    check('development source identities distinct', len({x['source_state_sha256'] for x in dev.values()}) == 2)
    check('holdout source identity differs from both development sources', archive['source_state']['sha256'] not in {x['source_state_sha256'] for x in dev.values()})
    check('cross-source summary bound', cross['formal_holdout']['source_state_sha256'] == archive['source_state']['sha256'] and cross['formal_holdout']['archive_manifest_sha256'] == sha(root / 'raw/evaluation/archive_manifest.json'))
    check('both development source bindings retained', cross['training_and_development'] == [{'label': label, 'seed': item['seed'], 'source_state_sha256': item['source_state_sha256'], 'archive_manifest_sha256': item['manifest_sha256']} for label, item in dev.items()])
    check('original archive integrity allows evaluation', archive_audit['status'] == 'pass' and all(archive_audit['checks'].values()) and archive_audit['formal_evaluation_allowed'] and archive_audit['archive_manifest_sha256'] == sha(root / 'raw/evaluation/archive_manifest.json'))
    check('pre-outcome actor freeze matches', evaluation_freeze['identity_sha256']['selected_actor'] == combined['selected_actor_sha256'] == selection['selected']['actor_sha256'])
    check('development selection precedes holdout freeze', datetime.fromisoformat(selection['generated_at_utc']) < datetime.fromisoformat(evaluation_freeze['generated_at_utc']))
    check('holdout evaluation does not reselect', combined['selection_performed'] is False and integrity['selection_performed'] is False and evaluation_freeze['selection_performed'] is False)
    check('release precedes sealed outcome access', release['sealed_material_accessed'] is False and release['selected_actor_sha256'] == selection['selected']['actor_sha256'])
    check('roster deviation original receipt retained', integrity['status'] == 'pass_with_documented_protocol_deviation' and 'ten' in integrity['protocol_deviation'] and len(evaluation_freeze['formal_controller_roster']) == 7)
    for filename, key in [('eval_mole_n9_wp6_formal.py', 'evaluator'), ('summarize_mole_n9_wp7.py', 'summarizer'), ('mole_n9_wp7_contract.py', 'wp7_contract')]:
        check('WP7 explicit executed source ' + filename, sha(root / 'source/evaluation' / filename) == evaluation_freeze['identity_sha256'][key])
    actor_source = (root / 'source/training/jax_mole_graph_sac_directional.py').read_text()
    flow_source = (root / 'source/training/jax_mole_continuous_sac.py').read_text()
    check('recorded action mapping source retained', 'target_yaw - previous_yaw' in flow_source and 'turbine_action * np.float32(config.max_yaw_angle_deg)' in flow_source and 'return jnp.tanh(mean)' in actor_source)

    reports = sorted((root / 'raw/evaluation/directions').glob('*.json'))
    check('seventeen original reports included', len(reports) == 17)
    rows_by_controller = {k: {} for k in evaluation_freeze['formal_controller_roster']}
    starts, angles = [], []
    actual_derived = {row['direction_deg']: row for row in result['evaluation']['direction_rows']}
    for path in reports:
        d = json.loads(path.read_text())
        cfg = d['configuration']; angle = cfg['relative_wind_angle_deg']; angles.append(angle)
        check('report original pass ' + path.name, d['status'] == 'pass')
        check('report original source and actor identity ' + path.name, d['identity']['selected_actor_sha256'] == selection['selected']['actor_sha256'] and d['identity']['archive_manifest_sha256'] == sha(root / 'raw/evaluation/archive_manifest.json') and d['identity']['source_state_sha256'] == archive['source_state']['sha256'])
        check('report paired evaluation contract ' + path.name, cfg['burnin_interactions'] == 150 and cfg['eval_interactions'] == 200 and cfg['offsets'] == [0, 40000] and len(d['windows']) == 2)
        graph = cfg['graph']
        check('report graph branch configuration ' + path.name, graph['conditioned'] and graph['reflection_equivariant_actor'] and graph['steering_branch_conditioned'] is False and graph['flow_selector'] is None and graph['steering_branch'] is None)
        selected_gains = []
        for window in d['windows']:
            check('all seven original controllers retained ' + path.name, set(window['controllers']) == set(evaluation_freeze['formal_controller_roster']))
            zero = window['controllers']['zero_yaw']['mean_farm_power_mw']
            check('positive finite zero-yaw reference ' + path.name, math.isfinite(zero) and zero > 0)
            for name, controller in window['controllers'].items():
                gain = 100 * (controller['mean_farm_power_mw'] - zero) / zero
                close('paired power gain ' + path.name + ':' + str(window['start_plane']) + ':' + name, gain, controller['gain_percent'])
                rows_by_controller[name].setdefault(angle, []).append(gain)
                if name == 'selected_realtime_policy':
                    selected_gains.append(gain)
                    check('selected angle/rate original limits ' + path.name, controller['max_abs_yaw_deg'] <= cfg['max_yaw_angle_deg'] + 1e-6 and controller['peak_yaw_rate_deg_per_s'] <= cfg['max_yaw_rate_deg_per_s'] + 1e-6)
            starts.append((angle, window['start_plane']))
        new = actual_derived[angle]
        for n, (x, y) in enumerate(zip(new['window_gains_percent'], selected_gains)):
            close('new derived window ' + path.name + ':' + str(n), x, y)
        close('new derived direction ' + path.name, new['mean_gain_percent'], statistics.mean(selected_gains))
    expected_angles = [-10 + 1.25 * x for x in range(17)]
    check('full fixed-angle grid exact', sorted(angles) == expected_angles)
    check('34 paired windows exact', len(starts) == len(set(starts)) == result['evaluation']['paired_windows'] == 34)
    all_means = {}
    for controller in combined['controllers']:
        name = controller['controller_id']; grouped = rows_by_controller[name]
        direction_means = [statistics.mean(grouped[angle]) for angle in expected_angles]
        mean = statistics.mean(direction_means); all_means[name] = mean
        close('combined equal-direction controller mean ' + name, mean, controller['equal_direction_mean_gain_percent'])
        for angle, gain in zip(expected_angles, direction_means):
            close('combined direction mean ' + name + ':' + str(angle), gain, controller['direction_mean_gain_percent'][str(float(angle))])
        check('original negative direction count ' + name, sum(x < 0 for x in direction_means) == controller['negative_direction_count'])
        check('original negative window count ' + name, sum(x < 0 for v in grouped.values() for x in v) == controller['negative_window_count'])
    check('all seven summary controllers retained', set(all_means) == set(evaluation_freeze['formal_controller_roster']))
    close('new holdout result from34 paired means', all_means['selected_realtime_policy'], result['evaluation']['equal_direction_mean_gain_percent'])
    close('known original holdout result', all_means['selected_realtime_policy'], 4.4312129851150495)
    check('all selected windows/directions positive', all(x > 0 for v in rows_by_controller['selected_realtime_policy'].values() for x in v) and result['evaluation']['negative_window_count'] == result['evaluation']['negative_direction_count'] == 0)
    retained = result['retained_controller_comparisons']
    check('original full comparison summary retained', retained['paired_comparisons'] == combined['paired_comparisons'] and {x['controller_id'] for x in retained['controllers']} == set(all_means))
    check('all original controller summary values retained', retained['controllers'] == [{k: v for k, v in row.items() if k != 'direction_mean_gain_percent'} for row in combined['controllers']])
    for comparison, saved in combined['paired_comparisons'].items():
        name = comparison.removeprefix('selected_realtime_policy_minus_')
        close('original contrast ' + name, all_means['selected_realtime_policy'] - all_means[name], saved['equal_direction_mean_gain_difference_percentage_points'])
    check('negative lookup comparison preserved', all_means['selected_realtime_policy'] < all_means['static_les_lookup'])
    check('one-source inference boundary retained', result['evaluation']['independent_holdout_sources'] == 1 and not result['evaluation']['population_confidence_interval_reported'] and combined['inferential_interval_reported'] is False)
    check('no new GPU experiment', result['new_GPU_hours'] == result['new_LES_runs'] == manifest['scope']['new_GPU_hours'] == 0)
    return {'schema': 'n9-intermediate-cpu-validation/1.0', 'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'python': sys.version, 'checks_passed': len(checks), 'checks': checks, 'manifest_sha256': sha(root / 'MANIFEST.json'), 'results_sha256': sha(root / 'N9_BRIDGE_RESULTS.json'), 'original_files_verified': len(sources['sources']), 'extension_files_verified': len(manifest['files']) + 2, 'training_execution_source_matches': 14, 'WP7_explicit_source_matches': 3, 'recomputed': {'interactions': 1000448, 'SAC_updates': 967680, 'Actor_updates': 30112, 'paired_windows': len(starts), 'directions': len(angles), 'holdout_mean_gain_percent': all_means['selected_realtime_policy'], 'lookup_mean_gain_percent': all_means['static_les_lookup'], 'RL_minus_lookup_pp': all_means['selected_realtime_policy'] - all_means['static_les_lookup']}, 'preserved_failures_and_deviations': ['initial p0 table/payload identity FAIL', 'early saved-candidate positivity hard-gate FAIL', 'formal roster10-to7 deviation', 'RL below lookup'], 'scope': 'Original small-file integrity and arithmetic from saved per-window mean powers. No new physical/learning/parity replay, actor unpickle, large source-state read or population inference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.output and args.output.resolve().is_relative_to(root):
        raise ValueError('Keep CPU validation output outside the frozen extension.')
    try:
        result = validate(root); code = 0
    except Exception as error:
        result = {'schema': 'n9-intermediate-cpu-validation/1.0', 'status': 'FAIL', 'root': str(root), 'error': type(error).__name__ + ': ' + str(error)}; code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2, sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
