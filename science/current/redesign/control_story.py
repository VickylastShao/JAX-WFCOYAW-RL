"""N81 policy development, independent evaluation and complete command records.

CPU-only redraw from preserved records. Preview first; set WIND_FIG_FINAL=1
only after visual review. No temporal smoothing or hidden sample selection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from story_common import *

sys.path.insert(0, str(ROOT / 'figure_sources/data_figures'))
import plot_revision as raw

SIGNED = LinearSegmentedColormap.from_list('signed_yaw', [BLUE, '#F8F7F3', ORANGE])
SOURCE_ORDER = ['B', 'C']
TIME = (np.arange(2000) + 1) * 10 / 3600


def physical_rows(sp):
    """Resolve rows from actual x coordinates, not an assumed roster reshape."""
    xyz = np.asarray(sp['geometry']['turbine_positions_m'], dtype=np.float64)
    xs = np.unique(xyz[:, 0])
    rows = [np.flatnonzero(xyz[:, 0] == x) for x in xs]
    assert len(rows) == 9 and all(len(row) == 9 for row in rows)
    assert np.array_equal(np.sort(np.concatenate(rows)), np.arange(81))
    return xyz, xs, rows


def row_means(yaw, rows):
    assert yaw.shape == (2000, 81)
    means = np.stack([np.mean(yaw[:, row], axis=1) for row in rows])
    assert means.shape == (9, 2000)
    assert np.allclose(means.mean(axis=0), yaw.mean(axis=1), rtol=0, atol=1e-12)
    return means


def process(s, sp, p2, traces):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 162 / 25.4),
                             gridspec_kw={'height_ratios': [0.86, 1.04, 0.86]})
    fig.subplots_adjust(left=.093, right=.886, bottom=.09, top=.922,
                        wspace=.22, hspace=.56)
    a, b = axes[0]
    actual = []
    for node in p2['nodes']:
        step = node['actual_training_interactions']
        name = 'zero.npz' if step == 0 else f'rl_step{step}.npz'
        z = raw.trace('P2/' + name)
        assert raw.INPUTS['data/P2/' + name] == node['trace']['sha256']
        energy = float(z['farm_power_mw'].sum() * 10 / 3600)
        raw.checkequal(energy, node['modeled_gross_ADM_energy_mwh'], 'Source A energy')
        actual.append(energy)
    gains = 100 * (np.asarray(actual) / actual[0] - 1)
    steps = np.asarray([v['actual_training_interactions'] for v in p2['nodes']])
    raw.checkequal(gains, [n['modeled_gross_ADM_energy_gain_percent'] for n in p2['nodes']], 'Source A gains')
    a.axhline(0, color=GRAY, lw=.7, ls='--')
    a.plot(steps[1:] / 1e6, gains[1:], color=BLUE, lw=1.1, marker='o', ms=4)
    a.scatter([0], [0], marker='^', color=GRAY, s=25, zorder=4)
    a.annotate('Zero-yaw reference', (0, 0), xytext=(5, 7),
               textcoords='offset points', fontsize=7.5)
    a.scatter([steps[-1] / 1e6], [gains[-1]], s=83, facecolors='none',
              edgecolors=DARK, lw=.9, zorder=5)
    a.annotate('Final policy', (steps[-1] / 1e6, gains[-1]),
               xytext=(-5, -17), textcoords='offset points', ha='right', fontsize=7.5)
    a.set(xlim=(-.04, 1.06), ylim=(-.8, 25), xlabel='Training interactions (millions)',
          ylabel='Energy gain (%)')
    a.set_xticks([0, .131072, .524288, 1.000448], ['0', '.131', '.524', '1.000'])
    a.set_yticks([0, 10, 20]); grid(a)
    panel(a, 'a', 'Source A · saved-policy evaluation')

    window_gains, aggregates = [], {}
    for i, win in enumerate(s['windows']):
        src = win['source_id']; num = win['window_number']
        z, r = traces[f'{src}_w{num}']
        gain = float(100 * (r['farm_power_mw'].sum() / z['farm_power_mw'].sum() - 1))
        raw.checkequal(gain, win['gain_percent'], 'B/C window gain')
        window_gains.append(gain)
        b.scatter([i], [gain], s=27, color=SRC[src], marker='o' if src == 'B' else 's',
                  edgecolors='white', lw=.35, zorder=3)
    for src, left in [('B', -.3), ('C', 3.7)]:
        ez = sum(traces[f'{src}_w{j}'][0]['farm_power_mw'].sum() for j in range(1, 5))
        er = sum(traces[f'{src}_w{j}'][1]['farm_power_mw'].sum() for j in range(1, 5))
        gain = float(100 * (er / ez - 1)); aggregates[src] = gain
        raw.checkequal(gain, next(v['gain_percent'] for v in s['sources'] if v['source_id'] == src), 'Source aggregate')
        b.hlines(gain, left, left + 3.6, color=SRC[src], lw=.8, ls='--')
        b.scatter([left + 1.8], [gain], marker='D', s=38, color=SRC[src],
                  edgecolors='white', lw=.45, zorder=4)
        b.text(left + 1.8, 22, f'{src}: {gain:.2f}%', ha='center', fontsize=8, color=SRC[src])
    b.axhline(0, color=GRAY, lw=.7); b.axvline(3.5, color=GRID, lw=.8)
    b.set(xlim=(-.5, 7.5), ylim=(-.8, 25), xlabel='Window within independent source')
    b.set_xticks(range(8), [v['source_id'] + str(v['window_number']) for v in s['windows']])
    b.set_yticks([0, 10, 20]); b.tick_params(labelleft=False); grid(b)
    panel(b, 'b', 'Sources B/C · frozen final actor')
    b.legend([Line2D([], [], color=DARK, marker='o', lw=0, ms=3),
              Line2D([], [], color=DARK, marker='D', ls='--', ms=3, lw=.7)],
             ['Window', 'Source aggregate'], frameon=False, loc='lower right',
             fontsize=7.5, handlelength=1.4, labelspacing=.3)

    xyz, xs, rows = physical_rows(sp)
    early = TIME * 60 <= 12
    allcum = {ctx: np.cumsum((r['farm_power_mw'] - z['farm_power_mw']).ravel()) * 10 / 3600
              for ctx, (z, r) in traces.items()}
    low = min(0., min(float(v.min()) for v in allcum.values())) - 1
    high = max(float(v.max()) for v in allcum.values()) * 1.06
    early_values = np.concatenate([allcum[f'{src}_w1'][early] for src in SOURCE_ORDER])
    e_min, e_max = float(early_values.min()), float(early_values.max())
    pad = (e_max - e_min) * .15
    startup, mean_payload = [], {}
    for col, src in enumerate(SOURCE_ORDER):
        c, d = axes[1:, col]; cum = allcum[f'{src}_w1']
        yaw = traces[f'{src}_w1'][1]['yaw_deg'][:, 0, :]
        means = row_means(yaw, rows); mean_payload[src] = means
        c.plot(TIME, cum, color=SRC[src], lw=1.15)
        c.axhline(0, color=GRAY, lw=.7)
        c.set_ylim(low, high); c.set_yticks([0, 20, 40, 60, 80]); grid(c)
        panel(c, 'cd'[col], f'Source {src} · first scoring window')
        if col == 0: c.set_ylabel('Cumulative energy\ndifference (MWh)')
        inset = c.inset_axes([.18, .69, .39, .285])
        inset.plot(TIME[early] * 60, cum[early], color=SRC[src], lw=.95)
        inset.axhline(0, color=GRAY, lw=.55)
        inset.set(xlim=(0, 12), ylim=(e_min - pad, e_max + pad))
        inset.set_xticks([0, 6, 12]); inset.set_yticks([-.5, 0, .5])
        inset.tick_params(labelsize=7.5, length=2, pad=1.5, width=.5)
        inset.set_xlabel('Time (min)', fontsize=7.5, labelpad=1)
        inset.set_ylabel('ΔE (MWh)', fontsize=7.5, labelpad=1)
        inset.set_title('First 12 min', fontsize=7.5, pad=3)
        for spine in inset.spines.values():
            spine.set_visible(True); spine.set_linewidth(.55); spine.set_edgecolor('#AAB3B9')
        mi = int(np.argmin(cum)); ri = int(np.flatnonzero((np.arange(len(cum)) > mi) & (cum >= 0))[0])
        inset.scatter([TIME[mi] * 60], [cum[mi]], s=12, color=SRC[src], zorder=4)
        inset.annotate(f'{cum[mi]:.3f}', (TIME[mi] * 60, cum[mi]), xytext=(.5, -.21),
                       textcoords='data', fontsize=7.5, ha='left', va='center',
                       arrowprops={'arrowstyle': '-', 'lw': .5, 'color': DARK})
        inset.scatter([TIME[ri] * 60], [cum[ri]], s=13, color=DARK, marker='d', zorder=4)
        inset.annotate(f'{TIME[ri] * 3600:.0f} s', (TIME[ri] * 60, cum[ri]),
                       xytext=(5.5, .38), textcoords='data', ha='center', fontsize=7.5,
                       arrowprops={'arrowstyle': '-', 'lw': .5, 'color': DARK})
        im = d.imshow(means, origin='lower', aspect='auto', extent=[0, 20000 / 3600, .5, 9.5],
                      cmap=SIGNED, norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40), interpolation='none')
        d.set_yticks([1, 3, 5, 7, 9]); d.set_xlabel('Scoring time (h)')
        panel(d, 'ef'[col], 'Row-mean yaw · nine turbines per row')
        if col == 0: d.set_ylabel('Streamwise row')
        for ax in [c, d]:
            ax.set_xlim(0, 20000 / 3600); ax.set_xticks(range(6))
            ax.tick_params(labelbottom=ax is d)
            if col: ax.tick_params(labelleft=False)
        startup.append({'source': src, 'main_controls': 2000, 'inset_controls': int(early.sum()),
                        'minimum_MWh': float(cum[mi]), 'first_nonnegative_seconds': float(TIME[ri] * 3600)})
    cbax = fig.add_axes([.912, .09, .014, .238]); cb = fig.colorbar(im, cax=cbax)
    cb.set_label('Row-mean yaw (°)', labelpad=5); cb.set_ticks([-40, -20, 0, 20, 40]); cb.outline.set_linewidth(.5)
    finish(fig, 'control_process', {
        'source_A': {'interactions': steps, 'gain_percent': gains, 'zero_reference_is_not_untrained_policy': True,
                     'final_actor_is_budget_endpoint_not_best_node': True},
        'all_eight_BC_gain_percent': window_gains, 'source_aggregate_gain_percent': aggregates,
        'source_aggregate_formula': '100*(sum RL energy / sum paired zero energy - 1), not mean window percentages',
        'independent_sources': 2, 'nested_windows_per_source': 4, 'optimization_seeds': 1,
        'startup': startup, 'physical_row_x_m': xs, 'physical_row_turbine_indices': rows,
        'physical_row_mean_yaw_deg': mean_payload, 'averaging': 'Arithmetic mean across nine actual same-x turbines; no temporal averaging; not a physically executed common-row command.',
        'complete_individual_commands': 'Supplementary Figure S9; all eight windows × 81 turbines × 2000 samples',
        'input_sha256': dict(raw.INPUTS)}, [
        'Separate saved-policy development, complete independent-source evaluation, startup response and row command pattern.',
        'Keep all evaluation points, pooled source ratios, negative startup values and all 2000 temporal samples.',
        'Replace 81-row miniature texture with nine physical row means; preserve all individual commands in S9.',
        'Use a shared zero-centered signed color scale for actual positive and negative yaw.'])


def all_yaw(sp, traces):
    xyz, xs, rows = physical_rows(sp)
    fig, axes = plt.subplots(4, 2, figsize=(WIDTH, 177 / 25.4))
    fig.subplots_adjust(left=.085, right=.88, top=.94, bottom=.078, wspace=.16, hspace=.46)
    payload = {}
    for win in range(1, 5):
        for col, src in enumerate(SOURCE_ORDER):
            ax = axes[win - 1, col]; ctx = f'{src}_w{win}'
            yaw = traces[ctx][1]['yaw_deg'][:, 0, :].T
            assert yaw.shape == (81, 2000)
            means = row_means(yaw.T, rows)
            summary = json.loads((ROOT / 'figure_sources/data_figures/data/reference_mechanism_summary.json').read_text())
            summary_window = next(v for v in summary['windows']
                                  if v['source_id'] == src and v['window_number'] == win)
            raw.checkequal(means.mean(axis=1), summary_window['row_mean_yaw_deg'], ctx + ' row mean')
            payload[ctx] = {'shape': yaw.shape, 'minimum_deg': float(yaw.min()), 'maximum_deg': float(yaw.max()),
                            'sha256_float64_le': hashlib.sha256(yaw.astype('<f8').tobytes()).hexdigest(),
                            'negative_samples': int(np.sum(yaw < 0)), 'samples': int(yaw.size),
                            'row_mean_shape': list(means.shape),
                            'row_mean_sha256_float64_le': hashlib.sha256(means.astype('<f8').tobytes()).hexdigest(),
                            'time_mean_row_yaw_deg': means.mean(axis=1)}
            im = ax.imshow(yaw, origin='lower', aspect='auto', extent=[0, 20000 / 3600, 0, 81],
                           cmap=SIGNED, norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40), interpolation='none')
            for boundary in range(9, 81, 9): ax.axhline(boundary, color='#9DA6AA', lw=.35, alpha=.6)
            ax.set(xlim=(0, 20000 / 3600), ylim=(0, 81))
            ax.set_xticks(range(6)); ax.set_yticks([4.5, 22.5, 40.5, 58.5, 76.5], ['1', '3', '5', '7', '9'])
            ax.tick_params(length=2.5, labelbottom=win == 4, labelleft=col == 0)
            if col == 0: ax.set_ylabel('Streamwise row\n(9 turbines each)')
            if win == 4: ax.set_xlabel('Scoring time (h)')
            panel(ax, chr(97 + (win - 1) * 2 + col), f'Source {src} · window {win}')
    cbax = fig.add_axes([.91, .255, .015, .49]); cb = fig.colorbar(im, cax=cbax)
    cb.set_ticks([-40, -20, 0, 20, 40]); cb.set_label('Applied turbine yaw (°)', labelpad=5)
    cb.outline.set_linewidth(.5)
    finish(fig, 'all_turbine_yaw', {'windows': payload, 'independent_sources': 2, 'nested_windows_per_source': 4,
        'optimization_seeds': 1, 'samples_total': 8 * 81 * 2000, 'time_seconds': [10, 20000],
        'sampling': 'Every 10 s control-boundary sample, no temporal/spatial averaging; each heatmap raster stores all 81×2000 samples.',
        'roster': 'Actual row-major source roster: increasing streamwise x, then increasing lateral z.',
        'physical_row_turbine_indices': rows, 'physical_row_x_m': xs, 'shared_clim_deg': [-40, 40],
        'source_sha256': {k: v for k, v in raw.INPUTS.items() if k.startswith('data/BC/')}}, [
        'Show all eight complete per-turbine command records, rather than only two illustrative windows.',
        'Use the same signed zero-centered color scale as main Figure 7, with unaltered 81×2000 arrays.'])


def reference_learner():
    data = ROOT / 'figure_sources/reference_learner/data'
    inp = {}; records = {}
    for backend in ['jax', 'torchrl']:
        path = data / f'mole_sac_update_throughput_{backend}_gpu205_v2.json'
        records[backend] = json.loads(path.read_text()); inp[str(path.relative_to(ROOT))] = sha(path)
    path = data / 'N3_BRIDGE_RESULTS.json'; bridge = json.loads(path.read_text()); inp[str(path.relative_to(ROOT))] = sha(path)
    cfg = records['jax']['configuration']; assert cfg == records['torchrl']['configuration']
    assert cfg['updates_per_block'] == 512 and cfg['timed_blocks'] == 20
    assert cfg['batch_size'] == 256 and cfg['observation_dim'] == 234 and cfg['action_dim'] == 3
    times = {}; rates = {}
    for backend in ['jax', 'torchrl']:
        m = records[backend]['measurement']; t = np.asarray(m['block_seconds'])
        assert t.shape == (20,) and np.isfinite(t).all() and np.all(t > 0)
        raw.checkequal(t.sum(), m['total_timed_seconds'], backend + ' total seconds', atol=1e-12)
        rates[backend] = float(10240 / t.sum()); times[backend] = t
        raw.checkequal(rates[backend], m['updates_per_second'], backend + ' throughput', atol=1e-10)
    ratio = rates['jax'] / rates['torchrl']
    raw.checkequal(ratio, 42.027189446462145, 'throughput ratio', atol=1e-10)
    stages = np.asarray(bridge['training']['stage_critical_path_seconds']) / 3600
    total = float(stages.sum()); raw.checkequal(total, bridge['training']['summed_stage_critical_path_hours'], 'N3 total')

    fig = plt.figure(figsize=(WIDTH, 102 / 25.4))
    a = fig.add_axes([.105, .255, .385, .61]); b = fig.add_axes([.64, .255, .335, .61])
    # Two independent batches of serial blocks: visually offset points, not pairs or training seeds.
    jitter = np.linspace(-.38, .38, 20)
    for y, backend, color, marker in [(1, 'jax', BLUE, 'o'), (0, 'torchrl', ORANGE, 's')]:
        a.scatter(times[backend], y + jitter, s=8, color=color, marker=marker,
                  edgecolors='white', lw=.3, zorder=3)
        a.text(.047, y + .46, f'{rates[backend]:,.1f} updates s⁻¹', fontsize=8, color=color)
    a.set_xscale('log'); a.set_xlim(.045, 5); a.set_ylim(-.5, 1.7)
    a.set_xticks([.05, .1, .5, 1, 5], ['0.05', '0.1', '0.5', '1', '5'])
    a.set_yticks([0, 1], ['Local\nTorchRL', 'JAX'])
    a.set_xlabel('Time per 512-update block (s; log scale)'); grid(a, axis='x')
    a.text(.5, -.37, f'Configured learner paths: {ratio:.2f}× throughput ratio',
           transform=a.transAxes, ha='center', fontsize=8)
    panel(a, 'a', 'Same RTX 4090 · learner only')

    y = np.arange(3)
    b.barh(y, stages, height=.38, color=[GRAY, TEAL, BLUE], alpha=.9)
    for j, value in enumerate(stages):
        b.text(value + .15, j, f'{value:.3f} h', va='center', fontsize=8)
    b.set(xlim=(0, 11.1), ylim=(2.6, -.6), xlabel='Stage critical path (h)')
    b.set_xticks([0, 3, 6, 9]); b.set_yticks(y, ['I', 'II', 'III'])
    b.set_ylabel('Training stage'); grid(b, axis='x')
    b.text(.5, -.37, f'Summed stages: {total:.4f} h · two RTX 4090 GPUs',
           transform=b.transAxes, ha='center', fontsize=8)
    panel(b, 'b', 'Three-turbine direct-LES reference run')
    finish(fig, 'reference_learner_paths', {
        'block_seconds': times, 'displayed_blocks_per_backend': 20,
        'timed_updates_per_backend': 10240, 'updates_per_block': 512,
        'updates_per_second': rates, 'throughput_ratio': ratio,
        'throughput_definition': '10240/sum(all 20 block seconds), not median block-rate ratio',
        'sample_unit': 'Serial post-warmup timing blocks in each of two separate runs; points are unpaired, not independent training seeds.',
        'visual_jitter': 'Twenty distinct vertical offsets per category prevent occlusion; vertical position carries no quantitative meaning.',
        'learning_workload': cfg, 'environments': {k: v['environment'] for k, v in records.items()},
        'implementation': 'Locally constructed TorchRL 0.3.1/PyTorch 2.2.1+cu121 learner; not unchanged upstream Wind-RL TorchRL 0.2.1/CPU PyTorch 2.0.1.',
        'excluded_from_learner_timing': ['LES environment', 'initial compilation', 'warmup'],
        'not_inferred': ['Framework-independent advantage', 'policy or LES physical equivalence from timing', 'end-to-end LES speedup from learner ratio'],
        'N3_stage_critical_path_hours': stages, 'N3_summed_stage_hours': total,
        'N3_stage_rule': 'For each stage, maximum of two process runtimes; sum of three stage maxima. One staged run, not three repetitions.',
        'N3_timing_scope': bridge['training']['timing_scope'], 'source_sha256': inp}, [
        'Expose every measured 512-update block on a logarithmic time axis; avoid a single opaque 42× bar.',
        'Keep the three-turbine end-to-end run as a separate, explicitly scoped single-observation stage account.',
        'No new chart compares the published ARCHER2 wall clock as a matched-hardware benchmark.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--figures', nargs='*',
        default=['control_process', 'all_turbine_yaw', 'reference_learner_paths'])
    args = parser.parse_args()
    jobs = set(args.figures)
    if jobs & {'control_process', 'all_turbine_yaw'}:
        summary, spatial, _, p2, traces, records, _ = raw.load()
        if 'control_process' in jobs: process(summary, spatial, p2, traces)
        if 'all_turbine_yaw' in jobs: all_yaw(spatial, traces)
    if 'reference_learner_paths' in jobs: reference_learner()
