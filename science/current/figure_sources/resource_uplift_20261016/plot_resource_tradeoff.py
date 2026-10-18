#!/usr/bin/env python3
"""Source-bound, CPU-only dot plots of the completed GPU branch measurements.

Missing measurements produce a pending receipt and no figure. No synthetic,
interpolated or predicted point is drawn. Repeats are sequential timings of one
physical prefix, not independent training or physical samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import warnings

BRANCHES = [('dense_b1', 'Dense', 'dense', 81),
            ('stream_g1_b1', 'g = 1', 'stream', 1),
            ('stream_g3_b1', 'g = 3', 'stream', 3),
            ('stream_g9_b1', 'g = 9', 'stream', 9)]
DEFAULT_FONT_DIR = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/wind_energy_navigation_closure_20261016/figure_sources/fonts/STIXTwoText')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_measurements(campaign):
    required = [campaign / 'PROTOCOL.json'] + [campaign / name / 'RESULT.json' for name, *_ in BRANCHES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {'status': 'pending', 'missing_measurements': missing,
                      'figure_created': False, 'synthetic_points': False}
    protocol = json.loads(required[0].read_text())
    rows = []
    for name, label, mode, group in BRANCHES:
        path = campaign / name / 'RESULT.json'
        row = json.loads(path.read_text())
        if (row['status'] != 'completed' or row['mode'] != mode or row['group'] != group or
                row['case']['batch'] != 1 or row['source_only'] or row['donation']):
            raise ValueError('Unexpected scientific execution scope: ' + name)
        if (row['substeps'] != protocol['substeps'] or row['core_sha256'] != protocol['source_core_sha256'] or
                row['runner_sha256'] != protocol['runner_sha256']):
            raise ValueError('Protocol/source/runner mismatch: ' + name)
        if not all(row['finite_outputs'].values()):
            raise ValueError('Nonfinite output: ' + name)
        if mode == 'stream':
            comparison = row['comparison']
            if (not comparison or comparison['state_velocity']['relative_l2'] >= 1e-4 or
                    comparison['power_trace']['relative_l2'] >= 1e-3):
                raise ValueError('Numerical diagnostic did not pass: ' + name)
        raw = [float(x) for x in row['run_seconds']]
        if len(raw) < 2 or any(x <= 0 for x in raw):
            raise ValueError('First/cold plus at least one steady execution required: ' + name)
        steady = raw[1:]
        if abs(statistics.median(steady) - row['steady_median_seconds']) > 1e-9:
            raise ValueError('Retained timing median mismatch: ' + name)
        analysis = row['compiled_memory_analysis']
        buffer_bytes = sum(int(analysis[k]) for k in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes']) - int(analysis['alias_size_in_bytes'])
        peak = (row.get('device_memory_stats_after_execution') or {}).get('peak_bytes_in_use')
        if buffer_bytes <= 0 or peak is not None and peak <= 0:
            raise ValueError('Invalid memory measurement: ' + name)
        rows.append({'branch': name, 'label': label, 'source_result': str(path), 'result_sha256': sha(path),
                     'compiler_buffer_bytes': buffer_bytes, 'allocator_peak_bytes': peak,
                     'cold_first_seconds': raw[0], 'steady_run_seconds': steady,
                     'median_seconds': statistics.median(steady),
                     'case': row['case'], 'device': row['device'], 'jax_version': row['jax_version'],
                     'numpy_version': row['numpy_version'], 'xla_flags': row['xla_flags'],
                     'inputs': row['inputs'], 'weights_setup_implementation': row.get('weights_setup_implementation')})
    reference = rows[0]
    for row in rows[1:]:
        for key in ['case', 'device', 'jax_version', 'numpy_version', 'xla_flags', 'inputs', 'weights_setup_implementation']:
            if row[key] != reference[key]:
                raise ValueError('Branches differ in matched input or runtime: ' + row['branch'] + '/' + key)
    return rows, {'status': 'real_measurements_validated', 'protocol_sha256': sha(required[0]),
                  'substeps': protocol['substeps'], 'gpu_identity': protocol.get('gpu_identity'),
                  'campaign_purpose': protocol.get('purpose'), 'campaign_scope': protocol.get('scope'),
                  'diagnostic_input_metadata': protocol.get('diagnostic_input_metadata'),
                  'data_rows': len(rows), 'raw_steady_values': sum(len(r['steady_run_seconds']) for r in rows),
                  'missing_data': False, 'outliers_removed': False,
                  'memory_unit': 'GiB = 2^30 bytes',
                  'memory_definitions': {'compiler': 'argument+output+temporary-alias compiled buffer budget',
                                         'allocator': 'peak_bytes_in_use since process startup; includes setup and runs; may be unavailable'},
                  'replication': 'one source/state and prescribed yaw; within-process sequential timing repeats',
                  'inference': 'descriptive raw timing points and median; no CI, p-value or training-speed claim',
                  'chart_choice': 'aligned dot plots of implementation versus memory and time; small-n raw points, no bars or category-joining line',
                  'alternative_considered': '2D memory-time scatter; rejected here because nearby timing replicates and implementation labels can overlap at final column width'}


def configure_style(args, plt, font_manager):
    if args.font_file:
        files = [args.font_file]
    else:
        font_dir = args.font_dir or (Path(__file__).parent / 'fonts/STIXTwoText')
        if not font_dir.exists():
            font_dir = DEFAULT_FONT_DIR
        files = [font_dir / f'STIXTwoText-{name}.otf' for name in ['Regular', 'Bold', 'Italic', 'BoldItalic']]
    for p in files:
        if not p.is_file():
            raise FileNotFoundError('Required font missing; supply --font-file or --font-dir: ' + str(p))
        font_manager.fontManager.addfont(str(p))
    family = font_manager.FontProperties(fname=str(files[0])).get_name()
    plt.rcParams.update({'font.family': [family], 'font.serif': [family], 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
                         'axes.linewidth': .65, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#7B8188', 'text.color': '#202A35',
                         'axes.labelcolor': '#202A35', 'xtick.color': '#202A35', 'ytick.color': '#202A35',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white', 'svg.hashsalt': 'paper1_resource_tradeoff_20261016'})
    resolved = Path(font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False))
    return {'family': family, 'resolved_font': str(resolved), 'font_files': {str(p): sha(p) for p in files}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stem', default='resource_tradeoff')
    p.add_argument('--font-file', type=Path, help='e.g. Windows times.ttf; otherwise preserve current STIX Two Text')
    p.add_argument('--font-dir', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, report = read_measurements(args.campaign)
    if rows is None:
        save(args.output / 'PLOT_PENDING.json', report)
        print(json.dumps(report), flush=True)
        return 3
    if any((args.output / (args.stem + '.' + ext)).exists() for ext in ['pdf', 'png', 'svg']):
        raise FileExistsError('Refuse to overwrite an existing scientific figure')
    save(args.output / 'DATA_PROFILE.json', report)
    save(args.output / 'PLOT_DATA.json', rows)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from PIL import Image, ImageOps
    sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
    from visual_qa import audit_layout
    from profile_data import profile_data
    timing_csv = args.output / 'STEADY_TIMING_POINTS.csv'
    with timing_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['implementation', 'retained_run', 'seconds'])
        writer.writeheader()
        for row in rows:
            for i, seconds in enumerate(row['steady_run_seconds'], 1):
                writer.writerow({'implementation': row['label'], 'retained_run': i, 'seconds': seconds})
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return clean(value.item())
        return value
    save(args.output / 'SCIPILOT_PROFILE.json', clean(profile_data(str(timing_csv), group_cols=['implementation'])))
    typography = configure_style(args, plt, font_manager)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.235, top=.85, wspace=.25)
    blue, orange, dark = '#0072B2', '#D55E00', '#202A35'
    positions = np.arange(len(rows))[::-1]
    no_allocator = []
    for pos, row in zip(positions, rows, strict=True):
        axes[0].scatter(row['compiler_buffer_bytes'] / 2**30, pos + .075,
                        s=33, marker='o', color=blue, edgecolors='white', linewidths=.5, zorder=3)
        if row['allocator_peak_bytes'] is not None:
            axes[0].scatter(row['allocator_peak_bytes'] / 2**30, pos - .075,
                            s=34, marker='D', facecolors='white', edgecolors=orange, linewidths=.9, zorder=3)
        else:
            no_allocator.append(row['branch'])
        times = np.asarray(row['steady_run_seconds'])
        offsets = np.linspace(-.07, .07, len(times)) if len(times) > 1 else [0.]
        axes[1].scatter(times, pos + np.asarray(offsets), s=24, marker='o',
                        color=blue, edgecolors='white', linewidths=.5, zorder=3)
        axes[1].plot(row['median_seconds'], pos, marker='|', markersize=13,
                     markeredgewidth=1.2, color=dark, linestyle='none', zorder=4)
    axes[0].set_yticks(positions, [row['label'] for row in rows])
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    axes[0].set_xlabel('Memory (GiB)')
    axes[1].set_xlabel(f"Time for {report['substeps']} LES substeps (s)")
    axes[0].set_title('a  Memory footprint', loc='left', pad=10)
    axes[1].set_title('b  Matched-work execution', loc='left', pad=10)
    for ax in axes:
        ax.set_ylim(-.45, len(rows) - .55)
        ax.set_xlim(left=0)
        ax.margins(x=.13)
        ax.grid(axis='x', color='#E5E7EB', linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)
    handles = [Line2D([], [], color=blue, marker='o', linestyle='none', markersize=4.5, label='Compiler buffers'),
               Line2D([], [], color=orange, marker='D', markerfacecolor='white', linestyle='none', markersize=4.5, label='Allocator peak'),
               Line2D([], [], color=blue, marker='o', linestyle='none', markersize=4, label='Steady runs'),
               Line2D([], [], color=dark, marker='|', linestyle='none', markersize=9, label='Median')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.515,.015), ncol=4,
               frameon=False, handlelength=1, handletextpad=.4, columnspacing=1.0)
    fig.canvas.draw()
    issues = [list(x) for x in audit_layout(fig)]
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    renderer = fig.canvas.get_renderer()
    clipped = [{'text': t.get_text(), 'bbox_px': list(t.get_window_extent(renderer).bounds)} for t in texts
               if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or
               not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    sizes = sorted(set(t.get_fontsize() for t in texts))
    if clipped or min(sizes) < 9 or any(x[0] in ['FAIL', 'WARN'] for x in issues):
        save(args.output / 'VISUAL_QA_FAILURE.json', {'issues': issues, 'clipped_text': clipped, 'font_sizes_pt': sizes})
        fig.savefig(args.output / (args.stem + '_failed_preview.png'), dpi=180)
        raise AssertionError('Layout/font self-check failed; inspect retained failure preview')
    preview = args.output / (args.stem + '_preview.png')
    fig.savefig(preview, dpi=180)
    with Image.open(preview) as im:
        ImageOps.grayscale(im).save(args.output / (args.stem + '_gray_preview.png'), dpi=(180,180))
    files = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for ext in ['pdf', 'svg', 'png']:
            destination = args.output / (args.stem + '.' + ext)
            kw = {'metadata': {'Creator': 'Measured full-support LES resource diagnostic', 'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {}
            fig.savefig(destination, dpi=600, **kw)
            files.append({'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': sha(destination)})
    if any('missing' in str(w.message).lower() for w in caught):
        raise AssertionError('Missing-glyph warning during export')
    with Image.open(args.output / (args.stem + '.png')) as im:
        ImageOps.grayscale(im).save(args.output / (args.stem + '_grayscale.png'), dpi=(600,600))
        png_size = list(im.size)
    caption = (
        'Measured memory and execution trade-off for the same 81-turbine LES state, '
        f"prescribed yaw and {report['substeps']}-substep inlet prefix on one GPU. "
        '(a) Compiler buffer budget (arguments plus outputs plus temporaries minus aliases) '
        'and allocator peak since process startup are distinct measurements. '
        '(b) Every retained sequential run is shown; the first execution is excluded and '
        'vertical marks denote medians. These timings are descriptive repeats of one '
        'physical prefix, without inferential intervals. Memory uses GiB (2^30 bytes). '
        'The candidates recompute full-support actuator weights within each LES substep.'
    )
    if report.get('campaign_purpose') == 'execution_and_resource_diagnostic_nonuniform_initializer':
        caption += (' The execution fixture uses a deterministic nonuniform initialization '
                    'and a prescribed fixed inlet on the original grid; it is not a qualified turbulent scoring window.')
    (args.output / 'CAPTION.en.txt').write_text(caption + '\n')
    save(args.output / 'VISUAL_QA.json', {
        'status': 'programmatic_checks_pass_visual_review_pending', 'issues': issues,
        'font': typography, 'font_sizes_pt': sizes, 'figure_inches': [6.3,3.0],
        'png_size_pixels': png_size, 'clipped_text': clipped,
        'allocator_unavailable': no_allocator, 'exports': files,
        'profile': report, 'script_sha256': sha(__file__), 'caption': caption,
        'AI_visual_review': 'pending; human/root agent must inspect color and grayscale previews',
        'no_GPU_execution': True, 'no_hardcoded_figure_number': True,
    })
    plt.close(fig)
    print(json.dumps({'status': 'rendered_visual_review_pending', 'files': files}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
