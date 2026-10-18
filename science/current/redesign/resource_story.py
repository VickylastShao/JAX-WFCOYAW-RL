#!/usr/bin/env python3
"""Source-bound resource-choice figures, CPU-only and cwd independent.

Preview first; set WIND_FIG_FINAL=1 only after reviewing the generated color
and grayscale PNGs. Raw scientific inputs and historical filenames stay intact.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from story_common import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INPUTS = {}


def readj(rel):
    path = ROOT / rel
    INPUTS[str(rel)] = sha(path)
    return json.loads(path.read_text())


def point(ax, x, y, *, dx=0, dy=0, **kwargs):
    return ax.scatter([x], [y], transform=offset_copy(ax.transData, fig=ax.figure,
        x=dx, y=dy, units='points'), clip_on=False, **kwargs)


def label(ax, x, y, text, *, dx=0, dy=6, color=DARK, ha='center', va='bottom', **kw):
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
        fontsize=8, color=color, ha=ha, va=va, clip_on=False, **kw)


def common_payload():
    return {'input_sha256': dict(INPUTS), 'new_GPUh': 0,
        'no_new_simulation_or_measurement': True,
        'confidence_intervals_or_significance_tests_added': False,
        'data_coordinates_preserved_under_screen_offsets': True}


def microbatch():
    e = readj('figure_sources/data_figures/data/E1_COMPARISON.json')
    bs = [1, 2, 4]
    reports = [e['reports'][str(b)] for b in bs]
    peak = np.array([r['device_memory']['peak_bytes_in_use'] for r in reports], dtype=np.int64)
    workspace = np.array([r['compiled_kernel_memory']['temp_size_in_bytes'] for r in reports], dtype=np.int64)
    persistent = np.array([r['persistent_state_bytes'] for r in reports], dtype=np.int64)
    times = [r['retained_pass_seconds'] for r in reports]
    med = np.array([np.median(t) for t in times])
    assert [r['persistent_contexts'] for r in reports] == [8, 8, 8]
    assert np.all(persistent == persistent[0]) and all(len(t) == 3 for t in times)
    comparisons = e['comparisons']
    leaves = [leaf for comparison in comparisons for leaf in comparison['leaves']]
    assert len(comparisons) == 16 and len(leaves) == 112
    assert all(v['exact'] and v['finite'] and v['max_abs_difference'] == 0 for v in leaves)
    assert sum(bool(v['shape']) for v in leaves) == 96
    pg, wg, ps = peak / 1e9, workspace / 1e9, persistent[0] / 1e9
    fig = plt.figure(figsize=(WIDTH, 88 / 25.4))
    a = fig.add_axes([.09, .29, .39, .50])
    b = fig.add_axes([.61, .29, .36, .50])
    ys = np.array([2., 1., 0.])
    for y, w, p in zip(ys, wg, pg, strict=True):
        a.barh(y+.11, w, height=.14, color=ORANGE, alpha=.72, edgecolor='none')
        a.scatter(p, y-.10, s=28, c=BLUE, edgecolors='white', linewidths=.45, zorder=4)
        label(a, w, y+.11, f'{w:.2f}', dy=5, color=ORANGE)
        label(a, p, y-.10, f'{p:.2f}', dx=5, dy=0, ha='left', va='center')
    a.axvline(ps, c=GRAY, ls=(0, (2, 2)), lw=.9)
    a.set(xlim=(0, 36.5), ylim=(-.55, 2.65), xlabel='Memory (GB)')
    a.set_yticks(ys, [f'b = {b}' for b in bs]); a.set_xticks([0, 10, 20, 30])
    panel(a, 'a', 'Retain histories; reuse workspace'); grid(a, 'x')
    for bsize, p, raw, median in zip(bs, pg, times, med, strict=True):
        for dx, value in zip([-4, 0, 4], raw, strict=True):
            point(b, p, value, dx=dx, s=20, c=BLUE, edgecolors='white', linewidths=.35, zorder=4)
        point(b, p, median, s=35, marker='D', facecolors='none', edgecolors=DARK, linewidths=.9, zorder=5)
        label(b, p, max(raw), f'b = {bsize}', dy=13)
        label(b, p, median, f'{median:.3f} s', dy=-13, va='top')
    b.set(xlim=(0, 38), ylim=(10.55, 14.15), xlabel='Allocator peak (GB)', ylabel='Eight-history pass time (s)')
    b.set_xticks([0, 10, 20, 30]); b.set_yticks([11, 12, 13, 14]); grid(b)
    panel(b, 'b', 'Observed memory–time choice')
    handles = [Patch(facecolor=ORANGE, alpha=.72, label='Compiler workspace'),
        Line2D([], [], marker='o', c=BLUE, ls='none', ms=4, label='Allocator peak / raw time'),
        Line2D([], [], c=GRAY, ls=(0, (2, 2)), lw=.9, label=f'Persistent states: {ps:.2f} GB'),
        Line2D([], [], marker='D', c=DARK, mfc='none', ls='none', ms=4, label='Median time')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .055),
        ncol=2, frameon=False, fontsize=8, handlelength=1.3, columnspacing=1.8, labelspacing=.45)
    fig.text(.09, .925, 'E = 8 retained histories', fontsize=9, color=DARK)
    fig.text(.97, .925, '112 state leaves unchanged', fontsize=9, ha='right', color=DARK)
    payload = common_payload() | {'microbatch_order': bs, 'persistent_state_bytes': persistent,
        'compiler_workspace_bytes': workspace, 'allocator_peak_bytes': peak,
        'retained_pass_seconds': times, 'median_pass_seconds': med,
        'persistent_contexts_per_configuration': 8, 'raw_timing_points': 9,
        'full_state_comparisons': 16, 'exact_state_leaves': 112,
        'exact_float_arrays': 96, 'exact_integer_scalars': 16,
        'peak_reduction_b4_to_b1': float(1-peak[0]/peak[2]),
        'median_time_reduction_b4_to_b1': float(1-med[0]/med[2]),
        'memory_unit': 'GB = 10^9 bytes', 'memory_measures_are_not_additive': True,
        'raw_time_screen_dx_points': [-4, 0, 4],
        'scope': reports[0]['scope'],
        'caption_suggestion': 'Eight-history flow test on one H20. (a) Compiler workspace, allocator peak and persistent-state storage are separate memory measures (GB = 10^9 bytes). (b) All three sequential 50-substep pass times for each active microbatch; offsets separate raw marks and open diamonds denote exact-coordinate medians. All 112 state leaves match after 200 substeps per history. Learning is excluded.'}
    finish(fig, 'microbatch_effect', payload, [
        'Short identity markers replace three bottom explanatory lines.',
        'The fixed-history test, separate memory measures and all nine timings remain visible.',
        'Statistical and execution definitions belong to the figure caption.'])


def grouped_data():
    base = 'figure_sources/resource_uplift_20261016/campaign/'
    protocol = readj(base+'PROTOCOL.json')
    pred = readj(base+'PROSPECTIVE_PREDICTIONS.json')
    assert pred['heldout_branch_not_started'] is True
    rows = []
    for name, mode, group in [('dense_b1', 'dense', 81), ('stream_g1_b1', 'stream', 1),
                            ('stream_g3_b1', 'stream', 3), ('stream_g9_b1', 'stream', 9)]:
        path = base+name+'/RESULT.json'; z = readj(path)
        assert z['status'] == 'completed' and z['mode'] == mode and z['group'] == group
        assert z['case']['batch'] == 1 and not z['source_only'] and not z['donation']
        assert z['substeps'] == protocol['substeps'] == 50
        assert z['core_sha256'] == protocol['source_core_sha256'] and z['runner_sha256'] == protocol['runner_sha256']
        assert all(z['finite_outputs'].values())
        if mode == 'stream':
            assert z['comparison']['state_velocity']['relative_l2'] < 1e-4
            assert z['comparison']['power_trace']['relative_l2'] < 1e-3
        raw = z['run_seconds'][1:]
        assert len(raw) == 2 and np.isclose(np.median(raw), z['steady_median_seconds'], rtol=0, atol=1e-9)
        m = z['compiled_memory_analysis']
        budget = sum(int(m[k]) for k in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes'])-int(m['alias_size_in_bytes'])
        row = {'branch': name, 'group': group, 'compiler_buffer_bytes': budget,
            'allocator_peak_bytes': z['device_memory_stats_after_execution']['peak_bytes_in_use'],
            'retained_pass_seconds': raw, 'median_seconds': float(np.median(raw)),
            'source_result_sha256': INPUTS[path], 'input_configuration': z['case'],
            'device': z['device'], 'inputs': z['inputs']}
        rows.append(row)
    assert all(rows[i]['input_configuration'] == rows[0]['input_configuration'] and rows[i]['inputs'] == rows[0]['inputs'] for i in range(1,4))
    assert len({r['compiler_buffer_bytes'] for r in rows[1:]}) == 1
    for group, row in [(1, rows[1]), (9, rows[3])]:
        assert pred['calibration'][str(group)]['sha256'] == row['source_result_sha256']
        assert pred['calibration'][str(group)]['compiled_live_bytes'] == row['compiler_buffer_bytes']
        assert pred['calibration'][str(group)]['steady_seconds'] == row['median_seconds']
    return rows, pred


def tradeoff():
    rows, pred = grouped_data()
    mem = np.array([r['compiler_buffer_bytes'] for r in rows]) / 2**30
    peak = np.array([r['allocator_peak_bytes'] for r in rows]) / 2**30
    med = np.array([r['median_seconds'] for r in rows])
    colors = [GRAY, ORANGE, BLUE, TEAL]; markers = ['o', '^', 's', 'D']
    names = ['Dense', 'g = 1', 'g = 3', 'g = 9']
    fig = plt.figure(figsize=(WIDTH, 91 / 25.4))
    a = fig.add_axes([.078, .25, .23, .56])
    b = fig.add_axes([.415, .25, .245, .56])
    c = fig.add_axes([.765, .25, .22, .56])
    ys = np.arange(3, -1, -1)
    for y, value, color, marker in zip(ys, mem, colors, markers, strict=True):
        a.scatter(value, y, c=color, marker=marker, s=30, edgecolors='white', linewidths=.45, zorder=4)
        label(a, value, y, f'{value:.3f}', dx=5, dy=0, ha='left', va='center', color=color)
    a.set(xlim=(0, 10.3), ylim=(-.5, 3.55), xlabel='Compiler budget (GiB)')
    a.set_yticks(ys, names); a.set_xticks([0, 4, 8]); grid(a, 'x'); panel(a, 'a', 'Workspace floor')
    raw_positions = []
    for i, (x, r, col, marker) in enumerate(zip(peak, rows, colors, markers, strict=True)):
        for dx, t in zip([-3, 3], r['retained_pass_seconds'], strict=True):
            point(b, x, t, dx=dx, c=col, marker=marker, s=22, edgecolors='white', linewidths=.4, zorder=4)
            raw_positions.append([float(x), float(t)])
        point(b, x, r['median_seconds'], s=42, marker='D', facecolors='none', edgecolors=DARK, linewidths=.8, zorder=5)
        label(b, x, r['median_seconds'], names[i], dx=-6 if i == 0 else 6,
            dy=10 if i == 0 else 7, ha='right' if i == 0 else 'left', color=col)
    b.set(xlim=(0, 15.4), ylim=(2.65, 4.12), xlabel='Allocator peak (GiB)', ylabel='50-substep time (s)')
    b.set_xticks([0, 5, 10, 15]); b.set_yticks([2.8, 3.2, 3.6, 4.0]); grid(b)
    panel(b, 'b', 'Memory–time choice')
    C, D = [pred['time_parameters'][v] for v in ['C', 'D']]
    counts = np.arange(9, 82)
    c.plot(counts, C+D*counts, c=GRAY, lw=1, label='Frozen group-count model')
    for i, count in [(1,81), (2,27), (3,9)]:
        row = rows[i]
        for dx, t in zip([-3, 3], row['retained_pass_seconds'], strict=True):
            point(c, count, t, dx=dx, c=colors[i], marker=markers[i], s=23,
                edgecolors='white', linewidths=.4, zorder=4)
        point(c, count, row['median_seconds'], s=42, marker='D', facecolors='none', edgecolors=DARK, linewidths=.8, zorder=5)
    signed = (pred['predicted_stream_g3_b1']['steady_seconds']-med[2])/med[2]*100
    label(c, 81, med[1], 'g = 1', dx=-3, dy=9, ha='right', color=ORANGE)
    label(c, 9, med[3], 'g = 9', dx=3, dy=-11, ha='left', va='top', color=TEAL)
    c.annotate(f'g = 3 holdout\n{signed:+.2f}%', (27, med[2]), xytext=(43,3.035),
        textcoords='data', fontsize=8, ha='left', va='center', color=BLUE,
        arrowprops={'arrowstyle':'-', 'color':BLUE, 'lw':.65})
    c.set(xlim=(0, 96), ylim=(2.65, 4.12), xlabel='Actuator groups\nper substep', ylabel='50-substep time (s)')
    c.set_xticks([9, 27, 81]); c.set_yticks([2.8, 3.2, 3.6, 4.0]); grid(c)
    panel(c, 'c', 'Held-out prediction')
    fig.text(.078, .925, 'Full-grid initialized-state diagnostic · one RTX 4090', fontsize=9)
    legend = [Line2D([], [], c=DARK, marker='o', ls='none', ms=4, label='Sequential timing'),
        Line2D([], [], c=DARK, marker='D', mfc='none', ls='none', ms=4.5, label='Median'),
        Line2D([], [], c=GRAY, lw=1, label=r'Frozen $T=C+D\lceil81/g\rceil$')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.53, .04), ncol=3,
        frameon=False, fontsize=8, handlelength=1.25, handletextpad=.5, columnspacing=1.2)
    payload = common_payload() | {'rows': rows, 'compiler_budget_GiB': mem,
        'allocator_peak_GiB': peak, 'retained_median_seconds': med,
        'raw_timing_points_in_memory_time_panel': raw_positions,
        'raw_timing_points_per_configuration': 2, 'raw_timing_points_total': 8,
        'memory_unit': 'GiB = 2^30 bytes',
        'compiler_budget_formula': 'argument+output+temporary-alias',
        'allocator_peak_scope': 'peak_bytes_in_use since process startup; distinct from compiler budget',
        'memory_and_allocator_are_not_additive': True,
        'frozen_prediction': pred, 'calibration_groups': [1, 9], 'holdout_group': 3,
        'model_domain_actuator_groups_per_substep': [9,81],
        'predicted_minus_observed_relative_percent': float(signed),
        'category_points_not_connected_in_memory_time_plane': True,
        'model_only_connecting_curve': True, 'raw_screen_dx_points': [-3,3],
        'caption_suggestion': 'Full-grid initialized-state diagnostic on one RTX 4090. (a) Compiler buffer budget. (b) Allocator peak versus the two retained sequential 50-substep timings and their median for each implementation. (c) Saved group-count time model calibrated at g=1 and 9 and tested at g=3; the annotation gives prediction-minus-observation relative to the observed median. Offsets separate raw points; open diamonds mark medians. Compiler budget and allocator peak are separate measures, in GiB = 2^30 bytes. This initialized-state execution diagnostic excludes learning.'}
    finish(fig, 'resource_tradeoff', payload, [
        'Five panels condensed into compiler floor, resource choice and held-out model.',
        'Every configuration and both retained timings remain; direct labels replace a four-color legend.',
        'The signed held-out residual is attached to g=3 without a separate residual panel.',
        'No categorical Pareto line, confidence interval or complete-training speedup is added.'])


def scaling():
    source = readj('figure_sources/data_figures/data/reference_mechanism_summary.json')
    ps = np.array([1,2,4])
    rows = [source['efficiency']['topologies'][str(p)] for p in ps]
    collect = np.array([r['normal_collector_seconds_median'] for r in rows])
    learn = np.array([r['normal_learner_including_communication_seconds_median'] for r in rows])
    ordinary = [r['normal_block_critical_rank_seconds'] for r in rows]
    med = np.array([r['normal_block_seconds_median'] for r in rows])
    elapsed = np.array([r['total_including_cold_reset_learner_sync_and_checkpoint_seconds'] for r in rows])
    cost = np.array([source['costs']['efficiency_allocated_gpu_hours_by_topology'][str(p)] for p in ps])
    assert all(r['actual_global_interactions'] == 3072 for r in rows)
    assert [len(t) for t in ordinary] == [10,10,10]
    assert np.allclose([np.median(t) for t in ordinary], med, rtol=0, atol=1e-10)
    colors = [GRAY, BLUE, TEAL]; markers = ['o','s','^']
    fig = plt.figure(figsize=(WIDTH, 113 / 25.4))
    a = fig.add_axes([.09, .59, .36, .26])
    b = fig.add_axes([.60, .59, .37, .26])
    c = fig.add_axes([.13, .15, .80, .25])
    pp = np.linspace(1,4,101)
    a.plot(pp, 1/pp, c=GRAY, lw=.9, ls=(0,(3,2)), label='Ideal 1/P')
    for component, col, marker, dx in [(collect/collect[0],BLUE,'o',-2),(learn/learn[0],ORANGE,'s',2)]:
        for p, v in zip(ps, component, strict=True):
            point(a, p, v, dx=dx, c=col, marker=marker, s=27,
                edgecolors='white', linewidths=.45, zorder=4)
    a.set(xlim=(.78,4.7), ylim=(0,1.14), xlabel='GPUs, P', ylabel='Time / own 1-GPU time')
    a.set_xticks(ps); a.set_yticks([0,.25,.5,.75,1]); grid(a); panel(a,'a','Where scaling slows')
    label(a, 4, learn[-1]/learn[0], 'Learning +\ncommunication', dx=6, dy=3, ha='left', color=ORANGE)
    label(a, 4, collect[-1]/collect[0], 'LES', dx=6, dy=-8, ha='left', color=BLUE)
    a.text(.04,.07,'Ideal 1/P',transform=a.transAxes,fontsize=8,color=GRAY)
    offsets=np.linspace(-11.5,11.5,10)
    for p, raw, median, col, marker in zip(ps,ordinary,med,colors,markers,strict=True):
        for dx, t in zip(offsets,raw,strict=True):
            point(b,p,t,dx=dx,c=col,marker=marker,s=17,edgecolors='white',linewidths=.35,zorder=4)
        point(b,p,median,dx=18,c=DARK,marker='|',s=50,linewidths=1.25,zorder=5)
        label(b,p,max(raw),f'{median:.2f} s',dy=7)
    b.set(xlim=(.55,4.7),ylim=(0,505),xlabel='GPUs, P',ylabel='Ordinary block time (s)')
    b.set_xticks(ps);b.set_yticks([0,100,200,300,400,500]);grid(b);panel(b,'b','All ordinary-block timings')
    b.text(.04,.06,'10 sequential blocks per profile',transform=b.transAxes,fontsize=8)
    for p, elapsed_s, h, col, marker in zip(ps,elapsed,cost,colors,markers,strict=True):
        x = elapsed_s/60
        c.scatter(x,h,c=col,marker=marker,s=43,edgecolors='white',linewidths=.5,zorder=4)
        label(c,x,h,f'{p} GPU'+('s' if p>1 else ''),dy=9 if p!=4 else -10,va='bottom' if p!=4 else 'top',color=col)
        label(c,x,h,f'{x:.2f} min · {h:.3f} GPU h',dy=-11 if p!=4 else 10,
            va='top' if p!=4 else 'bottom',color=col)
    c.set(xlim=(20,238),ylim=(3.12,4.02),xlabel='Complete-profile elapsed time (min)',ylabel='Allocated GPU hours')
    c.set_xticks([50,100,150,200]); c.set_yticks([3.2,3.4,3.6,3.8,4.0]);grid(c)
    panel(c,'c','Choose elapsed time and allocated cost')
    fig.text(.09,.95,'Same 32 histories · 3,072 fixed-command interactions · equal global learner work',fontsize=9)
    fig.text(.13,.025,f'4 GPUs: {elapsed[0]/elapsed[-1]:.3f}× profile speedup; {(cost[-1]/cost[0]-1)*100:.2f}% more allocated GPU hours.',fontsize=8)
    payload = common_payload() | {'topology_order_GPUs':ps, 'collector_median_seconds':collect,
        'learner_including_communication_median_seconds':learn,
        'component_relative_remaining_time': {'collector': collect/collect[0], 'learner_including_communication':learn/learn[0]},
        'component_medians_not_stacked':True, 'ideal_reference':'mathematical 1/P, not measured',
        'all_ordinary_block_seconds':ordinary,'ordinary_block_median_seconds':med,
        'ordinary_raw_points':30,'ordinary_screen_dx_points':offsets,
        'complete_profile_seconds':elapsed,'supervisor_allocated_GPU_hours':cost,
        'profile_speedups':elapsed[0]/elapsed,'allocated_cost_ratios':cost/cost[0],
        'resource_decision_points_are_discrete_unconnected':True,
        'evidence_boundary':source['efficiency']['evidence_boundary'],
        'uncertainty':source['efficiency']['uncertainty'],
        'allocation_timer_note':source['costs']['timer_note'],
        'caption_suggestion': 'Matched-work 1/2/4-GPU profiles. (a) Collection and inclusive learning/communication medians relative to their own one-GPU value; the dashed curve is the mathematical 1/P reference. (b) All ten sequential ordinary-block times per topology; black ticks denote medians and display offsets separate raw marks. (c) Complete-profile elapsed time versus supervisor-accounted allocated GPU hours. Each reset-heavy short profile executes 3072 fixed-command interactions with 32 histories and equal global learner work; these profiles do not measure full-training time-to-gain.'}
    finish(fig,'scaling_breakdown',payload,[
        'Own-baseline component normalization exposes learner scaling without stacking medians.',
        'All thirty sequential ordinary-block times and exact medians remain.',
        'Three directly labeled resource-choice points replace the allocation-area lanes.',
        'Two-GPU intermediate choice and separate supervisor cost accounting remain explicit.'])


def profile():
    """Run the SciPilot numerical profiler on every retained timing point."""
    csv_path = VALID/'RESOURCE_STORY_TIMING_POINTS.csv'
    rows=[]
    e=readj('figure_sources/data_figures/data/E1_COMPARISON.json')
    for b in [1,2,4]:
        for i,t in enumerate(e['reports'][str(b)]['retained_pass_seconds']):
            rows.append({'figure':'microbatch','configuration':f'b{b}','repeat':i+1,'seconds':t})
    grouped,_=grouped_data()
    for r in grouped:
        for i,t in enumerate(r['retained_pass_seconds']):
            rows.append({'figure':'actuator_recomputation','configuration':r['branch'],'repeat':i+1,'seconds':t})
    s=readj('figure_sources/data_figures/data/reference_mechanism_summary.json')
    for p in [1,2,4]:
        for i,t in enumerate(s['efficiency']['topologies'][str(p)]['normal_block_critical_rank_seconds']):
            rows.append({'figure':'scaling','configuration':f'P{p}','repeat':i+1,'seconds':t})
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['figure','configuration','repeat','seconds']);writer.writeheader();writer.writerows(rows)
    spec=importlib.util.spec_from_file_location('scipilot_resource_profile',ROOT/'figure_sources/data_figures/vendor/profile_data.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=mod.profile_data(str(csv_path),group_cols=['figure','configuration'])
    def clean(value):
        if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
        if isinstance(value,np.generic):return value.item()
        return value
    dump(VALID/'RESOURCE_STORY_DATA_PROFILE.json',{'scipilot':clean(result),
        'interpretation':{'total_retained_timings':47,'microbatch':9,'actuator_recomputation':8,'scaling':30,
        'replication':'Sequential timings within fixed configurations; not independent scientific or training replicates.',
        'missing_points':0,'removed_outliers':0,'continuous_frontier_fitted':False},
        'input_sha256':INPUTS,'new_GPUh':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='*',choices=['microbatch','tradeoff','scaling'])
    args=parser.parse_args()
    for directory in [OUT,VALID,PREVIEW]:directory.mkdir(parents=True,exist_ok=True)
    profile()
    for name,fn in [('microbatch',microbatch),('tradeoff',tradeoff),('scaling',scaling)]:
        if not args.only or name in args.only:fn()
    dump(VALID/'RESOURCE_STORY_INPUT_IDENTITIES.json',{'sources':INPUTS,'new_GPUh':0,
        'raw_sources_modified':False,'plot_script_sha256':sha(__file__)})
