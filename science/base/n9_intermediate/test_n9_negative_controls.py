#!/usr/bin/env python3
"""Fresh N9 negative controls on temporary copies; stdlib only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seal(root):
    p = root / 'MANIFEST.json'
    m = json.loads(p.read_text())
    for row in m['files']:
        target = root / row['path']
        row.update(bytes=target.stat().st_size, sha256=sha(target))
    p.write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
    (root / 'SHA256SUMS').write_text(''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in m['files']) + sha(p) + '  MANIFEST.json\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.output:
        if args.output.resolve().is_relative_to(root):
            raise ValueError('Keep receipts outside the frozen extension.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    cases = []
    with tempfile.TemporaryDirectory(prefix='n9_negative_', dir=args.output.parent if args.output else None) as temporary:
        for name, category in [('original_byte_tamper', 'original-byte integrity'), ('derived_gain_tamper', 'derived scientific semantics'), ('remove_lookup_comparison', 'derived scientific semantics')]:
            copy = Path(temporary) / name
            shutil.copytree(root, copy)
            if name == 'original_byte_tamper':
                q = copy / 'raw/evaluation/wp7_combined_summary.json'
                q.write_bytes(q.read_bytes() + b'\n')
            else:
                q = copy / 'N9_BRIDGE_RESULTS.json'; d = json.loads(q.read_text())
                if name == 'derived_gain_tamper':
                    d['evaluation']['equal_direction_mean_gain_percent'] += 1
                else:
                    d['retained_controller_comparisons']['controllers'] = [x for x in d['retained_controller_comparisons']['controllers'] if x['controller_id'] != 'static_les_lookup']
                q.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')
                # Recompute packaging hashes to make the semantic check do work.
                seal(copy)
            run = subprocess.run([sys.executable, '-B', str(copy / 'validate_n9_bridge.py'), '--root', str(copy)], capture_output=True, text=True, timeout=120)
            try:
                receipt = json.loads(run.stdout)
            except json.JSONDecodeError:
                receipt = {'status': 'ERROR', 'error': run.stdout + run.stderr}
            cases.append({'case': name, 'class': category, 'expected_exit_code': 1, 'observed_exit_code': run.returncode, 'observed_status': receipt['status'], 'diagnostic': receipt.get('error'), 'expected_failure_observed': run.returncode == 1 and receipt['status'] == 'FAIL'})
    final = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    passed = all(x['expected_failure_observed'] for x in cases) and baseline == final
    result = {'schema': 'n9-intermediate-negative-controls/1.0', 'status': 'PASS' if passed else 'FAIL', 'root': str(root), 'classes': 2, 'cases_count': len(cases), 'cases': cases, 'frozen_input_bytes_unchanged': baseline == final, 'temporary_copies_removed': True, 'new_GPU_hours': 0, 'scope': 'New temporary-copy tamper controls; original-byte integrity and independently resealed derived-semantics mutations. No LES, learner, actor execution or original data modification.'}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
