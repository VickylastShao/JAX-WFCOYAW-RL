"""Read-only verification of saved large-scale precursor states (CPU only).

This verifies files and numerical fields. It does not declare statistical
stationarity or choose a control policy using held-out generation results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def verify(path, spec):
    path = Path(path)
    before = path.stat()
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    case = meta['case']
    actual = digest(path)
    checks = {'metadata_hash': actual == meta['npz_sha256'],
              'frozen_expected_hash': actual == spec['expected_sha256'],
              'seed': meta['seed'] == spec['seed'],
              'steps': meta['completed_steps'] == spec['saved_steps'],
              'case': case == spec['case'],
              'provenance': meta.get('provenance') == spec['provenance']}
    required = ('velocity', 'rhs_previous', 'rhs_previous_2',
                'filtered_disk_speed', 'bottom_pressure_gradient',
                'open_x_pressure_gradient', 'step_index')
    shape = (case['batch'], 3, case['nx'], case['ny'], case['nz'])
    summaries = {}
    with np.load(path, allow_pickle=False) as saved:
        checks['required_fields'] = set(required).issubset(saved.files)
        for name in required:
            arr = saved[name]
            finite = bool(np.isfinite(arr).all())
            checks['finite_' + name] = finite
            if name in ('velocity', 'rhs_previous', 'rhs_previous_2'):
                checks['shape_' + name] = arr.shape == shape
                checks['dtype_' + name] = arr.dtype == np.dtype('float32')
            if name == 'bottom_pressure_gradient':
                checks['shape_' + name] = arr.shape == (case['batch'], 2, case['nx'], case['nz'])
            if name == 'open_x_pressure_gradient':
                checks['shape_' + name] = arr.shape == (case['batch'], 2, 2, case['ny'], case['nz'])
            summaries[name] = {'shape': list(arr.shape), 'dtype': str(arr.dtype),
                               'finite': finite}
            if finite and arr.size:
                summaries[name].update(minimum=float(arr.min()), maximum=float(arr.max()))
            if name == 'step_index':
                checks['step_index_matches'] = bool(np.all(arr == meta['completed_steps']))
            if name == 'velocity':
                hub = round(case['hub_height'] / (case['domain_ly'] / (case['ny'] - 1)))
                summaries[name]['mean_hub_u'] = float(np.mean(arr[:, 0, :, hub, :], dtype=np.float64))
                summaries[name]['bottom_normal_abs_max'] = float(np.max(np.abs(arr[:, 1, :, 0, :])))
                summaries[name]['top_normal_abs_max'] = float(np.max(np.abs(arr[:, 1, :, -1, :])))
            del arr
    after = path.stat()
    checks['unchanged_during_read'] = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    return {'source_id': spec['source_id'], 'path': str(path),
            'status': 'pass' if all(checks.values()) else 'fail',
            'sha256': actual, 'bytes': before.st_size, 'checks': checks,
            'fields': summaries, 'saved_steps': meta['completed_steps'],
            'case': case, 'provenance': meta.get('provenance'),
            'diagnostic_history': meta.get('diagnostic_history', []),
            'physical_maturity_qualified': False,
            'boundary': 'Full field and file verification; maturity requires separate physical assessment.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    results = []
    started = time.monotonic()
    for spec in plan['sources']:
        row = verify(spec['original_state'], spec)
        results.append(row)
        print(json.dumps({'source_id': spec['source_id'], 'status': row['status'], 'checks': row['checks']}), flush=True)
    status = 'pass' if all(row['status'] == 'pass' for row in results) else 'fail'
    atomic(args.output, {'status': status, 'results': results,
                         'elapsed_seconds': time.monotonic() - started,
                         'numpy_version': np.__version__, 'training_started': False})
    return 0 if status == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
