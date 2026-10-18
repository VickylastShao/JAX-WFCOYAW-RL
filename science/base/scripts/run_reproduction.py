#!/usr/bin/env python3
"""Single CPU entry point. Frozen payload is never used as an output directory."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from verify_final_package import verify


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Run every CPU check (also the default)')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='New directory outside frozen package')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Output must be a new directory outside the frozen package')
    if shutil.disk_usage(output.parent).free < (40 << 30) + (1 << 30):
        raise RuntimeError('Insufficient capacity while retaining 40 GiB reserve')
    output.mkdir(parents=True)
    temporary = output/'temporary'
    temporary.mkdir()
    start = time.monotonic()
    commands = []
    dump(output / 'JOB_HANDLE.json', {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'pid': os.getpid(), 'command': sys.argv, 'root': str(root), 'output': str(output),
         'resource_scope': 'CPU only; each child timeout 300 seconds; no network/GPU',
         'completion_criteria': 'all identity, L1 959, L1 negative controls, closure, prefix 12 tests pass'})

    def run(label, script_args, cwd):
        command = [sys.executable, '-B', *map(str, script_args)]
        before = time.monotonic()
        handle = {'command': command, 'cwd': str(cwd), 'timeout_seconds': 300,
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        dump(output / (label + '.HANDLE.json'), handle)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                   TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
        with (output / (label + '.stdout')).open('w', encoding='utf-8') as stdout, (output / (label + '.stderr')).open('w', encoding='utf-8') as stderr:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=300)
        row = dict(handle, exit_code=result.returncode, wall_seconds=time.monotonic()-before)
        commands.append(row)
        dump(output / (label + '.EXIT.json'), row)
        if result.returncode:
            raise RuntimeError(label + ' failed; inspect retained stdout/stderr')

    try:
        before = verify(root)
        visual = root / 'payload/visual_revision'
        prefix = root / 'payload/prefix_replay'
        run('L1_959', [visual/'scripts/reanalyse.py', '--root', visual, '--output', output/'L1_959.json'], output)
        l1 = json.loads((output/'L1_959.json').read_text(encoding='utf-8'))
        assert l1['status'] == 'pass' and l1['expected_values_checked'] == 959
        run('L1_negative_controls', [visual/'scripts/reanalyse.py', '--root', visual, '--self-test', '--output', output/'L1_negative_controls.json'], output)
        negative = json.loads((output/'L1_negative_controls.json').read_text(encoding='utf-8'))
        assert negative['status'] == 'pass' and negative['source_package_unchanged']
        run('prefix_integrity', [prefix/'scripts/verify_prefix_extension.py', '--root', prefix], output)
        run('prefix_12_tests', ['-m', 'unittest', 'discover', '-s', prefix/'prior_mainline/scripts', '-p', 'test_*.py'], output)
        test_text = (output/'prefix_12_tests.stderr').read_text(encoding='utf-8')
        assert re.search(r'Ran 12 tests\b', test_text) and 'OK' in test_text
        closure_work = output/'closure_work'
        shutil.copytree(root/'payload/wind_energy_closure', closure_work)
        run('closure_reanalysis', [closure_work/'reanalyse_closure.py'], output)
        actual = json.loads((closure_work/'results/REANALYSIS.json').read_text(encoding='utf-8'))
        reference = json.loads((root/'payload/wind_energy_closure/results/REANALYSIS.json').read_text(encoding='utf-8'))
        assert actual['state'] == 'complete' and actual['equivalent_arrays'] == 112 and actual['state_identity_files'] == 24
        assert same(reference, actual)
        after = verify(root)
        import numpy
        report = {'status': 'pass', 'started_from_frozen_package': str(root),
            'python': sys.version, 'numpy': numpy.__version__, 'commands': commands,
            'L1_expected_values_checked': 959, 'L1_numerical_tolerance': l1['numerical_comparison'],
            'L1_negative_controls': negative, 'prefix_CPU_tests_passed': 12,
            'closure': {'state': actual['state'], 'equivalent_arrays': 112, 'state_identity_files': 24,
                'reference_matches': True, 'comparison_abs_tolerance': 1e-10, 'comparison_relative_tolerance': 1e-12,
                'dt02_gain_percent': actual['numerics']['02']['gain_percent'],
                'dt01_gain_percent': actual['numerics']['01']['gain_percent'],
                'timestep_comparison': actual['timestep_comparison']},
            'frozen_package_integrity_before': before, 'frozen_package_integrity_after': after,
            'elapsed_seconds': time.monotonic()-start, 'new_GPUh': 0, 'network_used': False,
            'physical_replay_executed': False, 'full_training_or_window_replay_validated': False}
        dump(output/'CPU_REPRODUCTION.json', report)
        dump(output/'JOB_EXIT.json', {'status': 'complete', 'exit_code': 0, 'elapsed_seconds': time.monotonic()-start})
        print(json.dumps({k:v for k,v in report.items() if k not in ['commands','L1_negative_controls']}, indent=2))
    except BaseException as error:
        dump(output/'JOB_EXIT.json', {'status': 'failed', 'exit_code': 1, 'error': str(error), 'elapsed_seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    main()
