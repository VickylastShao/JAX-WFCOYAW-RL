#!/usr/bin/env python3
"""Run the bounded saved-record CPU suite from an arbitrary package location.

No physical LES, Actor inference, training, network, plotting or GPU is used.
The only non-stdlib requirement is NumPy 2.3.5 (Python >=3.11).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import site
import subprocess
import sys
import time

sys.dont_write_bytecode = True

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Extracted package root')
    parser.add_argument('--output', type=Path, required=True, help='New directory outside package')
    parser.add_argument('--timeout-seconds', type=int, default=120, help='Each child; range 30..120')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not 30 <= args.timeout_seconds <= 120:
        raise ValueError('Each child timeout must lie in 30..120 seconds')
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Output must be a new directory outside the frozen package')
    if not output.parent.is_dir():
        raise ValueError('Create the output parent directory first')
    import numpy
    if numpy.__version__ != '2.3.5':
        raise ValueError('Use the pinned NumPy 2.3.5 environment')
    output.mkdir()
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    commands = []
    dump(output/'JOB_HANDLE.json', dict(pid=os.getpid(), argv=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completion_criteria='16 CPU commands: 13 baseline, current roster disclosure, 143 current evidence references, current navigation; source manifest unchanged',
        resource_scope='CPU only; no network; each child bounded by timeout_seconds',
        timeout_seconds=args.timeout_seconds, package_root=str(root)))
    extension = root/'resource_uplift_20261016/extension'
    visuals = [
        ('inherited', root/'n9_visual_story_20261016/figure_sources/n9_visual_story'),
        ('policy', root/'execution_focus_20261016/figure_sources/n9_policy_focus')]
    suite = [
        ('inherited_CPU_all', ['scripts/run_reproduction.py', '--all', '--root', root,
                               '--output', output/'inherited_cpu'], root),
        ('n3_bridge', ['mole_reference/validate_n3_bridge.py', '--root', root/'mole_reference',
                       '--output', output/'n3_validation.json'], root),
        ('n9_bridge', ['n9_intermediate/validate_n9_bridge.py', '--root', root/'n9_intermediate',
                       '--output', output/'n9_validation.json'], root),
        ('n9_negative_controls', ['n9_intermediate/test_n9_negative_controls.py', '--root', root/'n9_intermediate',
                                  '--output', output/'n9_negative_controls.json'], root),
        ('n9_clarifications', ['n9_review_closure_20261015/validate_clarifications.py', '--root', root,
                               '--output', output/'n9_clarifications.json'], root)]
    for label, source in visuals:
        suite.extend([
            (label+'_visual', [source/'validate_n9_visual_data.py', '--report', output/(label+'_visual.json')], root),
            (label+'_visual_negatives', [source/'test_n9_visual_negative_controls.py',
                                         '--report', output/(label+'_visual_negatives.json')], root)])
    suite.extend([
        ('existing_resource_portable', [extension/'log_analysis/run_existing_resources_portable.py',
             '--root', extension, '--output', output/'existing_resources'], root),
        ('new_resource_portable', [extension/'log_analysis/run_new_resources_portable.py',
             '--root', extension, '--output', output/'new_resources'], root),
        ('resource_eight_unit_tests', ['-m', 'unittest', 'test_new_campaign_analysis.py',
             'test_validate_resource_export.py'], extension/'log_analysis'),
        ('exporter_six_negative_controls', [extension/'low_workspace/test_export_small_results.py',
             '--exporter', extension/'exporters/export_small_results_v2_local_negative_controls.py',
             '--output', output/'exporter_negative_controls'], root)])
    roster_script = root/'n9_intermediate/scripts/validate_roster_change.py'
    if roster_script.is_file():
        suite.append(('current_roster_disclosure', [roster_script, '--output',
                      output/'current_roster_disclosure.json'], root))
    paths_script=root/'minor_fixes_20261016/access/validate_current_claim_paths.py'
    if paths_script.is_file():
        suite.append(('current_claim_paths',[paths_script,'--root',root,'--output',
                      output/'current_claim_paths.json'],root))
    navigation_script=root/'minor_fixes_20261016/access/validate_current_navigation.py'
    if not navigation_script.is_file():
        raise FileNotFoundError('Current navigation verifier is required')
    suite.append(('current_navigation',[navigation_script,'--root',root,'--output',output/'current_navigation.json'],root))
    if len(suite)!=16:
        raise ValueError('All sixteen current CPU checks are required')
    manifest_before = hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest()
    try:
        for label, argv, cwd in suite:
            command = [sys.executable, '-B', *map(str, argv)]
            handle = dict(label=label, command=command, cwd=str(cwd),
                timeout_seconds=args.timeout_seconds,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            dump(output/(label+'.HANDLE.json'), handle)
            then = time.monotonic()
            with (output/(label+'.stdout')).open('w') as stdout, (output/(label+'.stderr')).open('w') as stderr:
                completed = subprocess.run(command, cwd=cwd, env=env, stdout=stdout,
                    stderr=stderr, timeout=args.timeout_seconds)
            row = dict(handle, exit_code=completed.returncode, elapsed_seconds=time.monotonic()-then)
            commands.append(row)
            dump(output/(label+'.EXIT.json'), row)
            print(json.dumps(dict(label=label, exit_code=completed.returncode,
                                  elapsed_seconds=row['elapsed_seconds'])), flush=True)
            if completed.returncode:
                raise RuntimeError(label+' failed; inspect its retained stdout/stderr')
        inherited = json.loads((output/'inherited_cpu/CPU_REPRODUCTION.json').read_text())
        if inherited['status'] != 'pass' or inherited['L1_expected_values_checked'] != 959:
            raise ValueError('Inherited saved-result checks differ')
        if inherited['prefix_CPU_tests_passed'] != 12:
            raise ValueError('Prefix CPU contract test count differs')
        unit_text = (output/'resource_eight_unit_tests.stderr').read_text()
        if not re.search(r'Ran 8 tests\b', unit_text) or 'OK' not in unit_text:
            raise ValueError('Resource unit-test count differs')
        if hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest() != manifest_before:
            raise ValueError('Package manifest changed during CPU suite')
        # run_reproduction verifies every manifest-bound member before and after.
        report = dict(schema='review-closure-CPU-acceptance/1', status='pass',
            commands=commands, CPU_command_count=len(commands), baseline_CPU_command_count=13,
            current_roster_disclosure_checked=roster_script.is_file(), python=sys.version,
            current_claim_evidence_paths_checked=paths_script.is_file(),
            current_navigation_identities_checked=True,
            python_executable=sys.executable, numpy_version=numpy.__version__,
            numpy_import_path=numpy.__file__, user_site_enabled=site.ENABLE_USER_SITE,
            L1_expected_values_checked=959, prefix_CPU_tests_passed=12,
            resource_unit_tests_passed=8, exporter_negative_control_cases=6,
            source_manifest_sha256=manifest_before, source_unchanged_after_checks=True,
            new_GPU_hours=0, network_used=False, physical_LES_replay_executed=False,
            full_training_or_window_replay_validated=False,
            scope='saved-result arithmetic and structural negative controls; no physical rerun',
            elapsed_seconds=time.monotonic()-started)
        dump(output/'CPU_ACCEPTANCE.json', report)
        dump(output/'JOB_EXIT.json', dict(status='complete', exit_code=0,
             elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        dump(output/'JOB_EXIT.json', dict(status='failed', exit_code=1, error=str(error),
             elapsed_seconds=time.monotonic()-started))
        raise

if __name__ == '__main__':
    main()
