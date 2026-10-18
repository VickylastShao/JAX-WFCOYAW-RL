#!/usr/bin/env python3
"""Standard-library reanalysis of the two unchanged N9 source records.

This checks archived, already measured values; it does not execute LES or RL.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
ORDER = ['zero_yaw', 'neutral_actor_step32768', 'static_les_lookup',
         'selected_policy_direction_mean', 'selected_realtime_policy',
         'selected_probe_mean_ablation', 'cross_precursor_action_replay']
LABELS = ['Zero yaw', 'Neutral actor', 'Static LES lookup', 'Direction-mean action',
          'Realtime policy', 'Mean-probe input', 'Cross-source action replay']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(actual, expected, label, atol=1e-10):
    if not math.isclose(actual, expected, abs_tol=atol, rel_tol=1e-12):
        raise AssertionError((label, actual, expected))

def analyze(root=ROOT):
    checks = 0
    def ck(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
    identities = json.loads((root/'SOURCE_IDENTITIES.json').read_text())
    for source in identities['sources']:
        p = root/source['relative_path']
        ck(p.stat().st_size == source['bytes'], 'source size')
        ck(sha(p) == source['source_sha256'], 'source SHA')
    raw = json.loads((root/'data/wp7_combined_summary.json').read_text())
    lookup = json.loads((root/'data/static_lookup_final.json').read_text())
    directions = [-10 + 1.25*i for i in range(17)]
    offsets = [0, 40000]
    rows = raw['window_rows']
    ck(len(rows) == 7*17*2, 'complete 238-window record')
    ck(raw['independent_precursor_seed'] == 20260904, 'precursor seed')
    ck(raw['inferential_interval_reported'] is False, 'no population interval')
    by_key = {(r['controller_id'], r['direction_deg'], r['start_plane']): r for r in rows}
    ck(len(by_key) == len(rows), 'unique controller/direction/offset')
    zero = {(d, o): by_key[('zero_yaw', d, o)]['mean_farm_power_mw']
            for d in directions for o in offsets}
    output_rows = []
    controller_output = []
    by_controller = {c['controller_id']: c for c in raw['controllers']}
    for controller_id, label in zip(ORDER, LABELS):
        means = []
        gains_by_offset = []
        negative = []
        for d in directions:
            pair = []
            for o in offsets:
                row = by_key[(controller_id, d, o)]
                gain = 100*(row['mean_farm_power_mw']/zero[(d, o)] - 1)
                close(gain, row['gain_percent'], 'gain from paired power'); checks += 1
                ck(math.isfinite(gain), 'finite gain')
                pair.append(gain)
                output_rows.append({'controller_id': controller_id, 'direction_deg': d,
                                    'start_plane': o, 'gain_percent': gain,
                                    'training_anchor': d/2.5 == int(d/2.5)})
                if gain < 0:
                    negative.append({'direction_deg': d, 'start_plane': o, 'gain_percent': gain})
            mean = statistics.mean(pair)
            close(mean, by_controller[controller_id]['direction_mean_gain_percent'][str(d)], 'direction mean'); checks += 1
            means.append(mean); gains_by_offset.append(pair)
        close(statistics.mean(means), by_controller[controller_id]['equal_direction_mean_gain_percent'], 'equal direction mean'); checks += 1
        close(min(means), by_controller[controller_id]['worst_direction_mean_gain_percent'], 'worst direction'); checks += 1
        ck(len(negative) == by_controller[controller_id]['negative_window_count'], 'negative window count')
        controller_output.append({'controller_id': controller_id, 'display_label': label,
                                  'direction_mean_gain_percent': means,
                                  'window_gain_percent_by_direction': gains_by_offset,
                                  'equal_direction_mean_gain_percent': statistics.mean(means),
                                  'worst_direction_mean_gain_percent': min(means),
                                  'negative_window_count': len(negative),
                                  'negative_windows': negative})
    table = {r['relative_wind_angle_deg']: r['grouped_target_yaw_deg'] for r in lookup['lookup_table']}
    ck(table[0.] == [-20., -10., 0.], '0 degree lookup branch')
    ck(table[2.5] == [20., 10., 0.], '2.5 degree lookup branch')
    interpolation = [(a+b)/2 for a,b in zip(table[0.], table[2.5])]
    ck(interpolation == [0.,0.,0.], 'linear interpolation cancellation at 1.25 degrees')
    midpoint_lookup = [by_key[('static_les_lookup', 1.25, o)] for o in offsets]
    for row in midpoint_lookup:
        ck(row['mean_yaw_deg'] == [0.]*9 and row['gain_percent'] == 0., 'recorded midpoint zero yaw and gain')
    realtime = next(c for c in controller_output if c['controller_id'] == 'selected_realtime_policy')
    static = next(c for c in controller_output if c['controller_id'] == 'static_les_lookup')
    wins = [d for d,r,l in zip(directions,realtime['direction_mean_gain_percent'],static['direction_mean_gain_percent']) if r>l]
    ck(wins == [-10.,1.25], 'RL greater than lookup at exactly two direction means')
    derived = {'schema': 'n9-visual-data/1.0',
               'directions_deg': directions, 'start_planes': offsets,
               'independent_precursor_seed': 20260904,
               'statistical_unit': 'one precursor seed; two deterministic windows per fixed direction',
               'gain_definition': '100*(controller mean farm power / paired zero-yaw mean farm power - 1)',
               'direction_mean_definition': 'arithmetic mean of the two paired window gains',
               'global_mean_definition': 'equal-weight arithmetic mean over 17 direction means',
               'controllers': controller_output,
               'window_rows': output_rows,
               'lookup_interpolation': {'directions_deg': [0.,1.25,2.5],
                   'group_targets_deg': [table[0.], interpolation, table[2.5]],
                   'definition': 'componentwise linear interpolation of frozen three-group yaw targets'},
               'realtime_greater_than_lookup_direction_means_deg': wins,
               'source_sha256': {p['relative_path']:p['source_sha256'] for p in identities['sources']},
               'new_GPUh': 0}
    profile = {'n_window_rows':len(rows), 'n_controllers':7, 'n_directions':17,
               'n_windows_per_direction':2, 'n_independent_precursor_seeds':1,
               'grouping': ['controller categorical', 'fixed direction ordered continuous', 'start plane nested deterministic window'],
               'missing_values':0, 'nonfinite_gains':0,
               'gain_range_percent':[min(r['gain_percent'] for r in output_rows),max(r['gain_percent'] for r in output_rows)],
               'negative_windows':[dict(controller_id=c['controller_id'], **r) for c in controller_output for r in c['negative_windows']],
               'gains_all_directions_mean_nonnegative':all(min(c['direction_mean_gain_percent'])>=0 for c in controller_output),
               'no_outliers_removed':True, 'no_confidence_interval_or_significance':True}
    return derived, {'status':'PASS','checks':checks,'data_profile':profile,'scope':'Archived results reanalysis only; no LES or policy rerun.'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--write-derived', action='store_true')
    args = ap.parse_args()
    derived, report = analyze(args.root)
    path = args.root/'data/N9_VISUAL_DATA.json'
    if args.write_derived:
        path.write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
    else:
        if json.loads(path.read_text()) != derived:
            raise AssertionError('derived data does not match raw power reanalysis')
    report['derived_sha256'] = sha(path)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':report['checks'], 'rows':238,'new_GPUh':0}))

if __name__ == '__main__':
    main()
