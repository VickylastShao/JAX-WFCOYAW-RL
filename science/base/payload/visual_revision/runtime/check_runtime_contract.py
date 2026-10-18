#!/usr/bin/env python3
"""Cheap meaningful checks for the new adapter, not GPU/LES validation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from resolve_run import PACK, build_spec, execute, parser_from_frozen_source


def options(job, base, context='n81_B_w1'):
    return argparse.Namespace(job=job, input_root=base/'external_inputs',
        output_root=base/'results', context=context, gpus=[0,1,2,3],
        coordinator='127.0.0.1:17381', python=sys.executable,
        wall_seconds=600, verify_input_hashes=False, run=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks = {}
    contexts = json.loads((PACK/'configs/runtime_evaluation.json').read_text())['contexts']
    with tempfile.TemporaryDirectory(prefix='n81-runtime-contract-') as temp:
        root = Path(temp)
        for context in contexts:
            spec = build_spec(options('L2', root, context['context_id']))
            checks[context['context_id']+'_frozen_arguments'] = (
                len(spec['jobs'])==1 and spec['argparse_checks'][0]['all_arguments_parsed_by_frozen_registration'])
            checks[context['context_id']+'_source_pending'] = (
                spec['input_contract']['status']=='needs_external_input' and not spec['LES_execution_verified'])
        opts = options('L3', root)
        spec = build_spec(opts)
        checks['four_rank_arguments_parsed'] = len(spec['jobs'])==4 and all(
            c['all_arguments_parsed_by_frozen_registration'] for c in spec['argparse_checks'])
        checks['global_work_unchanged'] = all(
            c['resolved_defaults']['planned_global_interactions']==1000448 and
            c['resolved_defaults']['iterations']==3907 and
            c['resolved_defaults']['local_environments']==8 and
            c['resolved_defaults']['sample_size']==64 and
            c['resolved_defaults']['collection_steps']==8
            for c in spec['argparse_checks'])
        checks['rank_window_offsets_unchanged'] = [c['resolved_defaults']['plane_offset_add']
                                                 for c in spec['argparse_checks']] == [0,160000,320000,480000]
        checks['no_original_machine_dependency_in_argv'] = all(
            '/data/' not in value and 'libnvidia-ml' not in value and 'launch_mole_n9' not in value
            for job in spec['jobs'] for value in job['argv'])
        checks['default_does_not_create_outputs'] = not (root/'results').exists()
        # Fail closed before any subprocess spawn or device access when input bytes are absent.
        def forbidden(*a, **kw):
            raise AssertionError('missing inputs reached subprocess execution')
        real_run, real_popen = subprocess.run, subprocess.Popen
        subprocess.run = subprocess.Popen = forbidden
        try:
            try:
                execute(spec, opts)
            except ValueError as error:
                checks['missing_inputs_fail_before_subprocess'] = 'input bytes' in str(error)
        finally:
            subprocess.run, subprocess.Popen = real_run, real_popen
        # A malformed source identity must not be accepted simply because files exist.
        source = json.loads((PACK/'configs/runtime_sources.json').read_text())['sources'][0]
        archive = opts.input_root / source['archive_relative_path']
        archive.mkdir(parents=True)
        original = json.loads((PACK/'source/provenance/n81_train_A.manifest.json').read_text())
        original['provenance']['target_layout_sha256'] = 'wrong-layout'
        (archive/'manifest.json').write_text(json.dumps(original))
        changed = build_spec(opts)
        checks['changed_manifest_and_layout_rejected'] = (
            'manifest SHA differs from frozen scientific source' in changed['input_contract']['issues'] and
            'canonical 81-layout identity differs' in changed['input_contract']['issues'])
    result={'status':'portable_runtime_contract_pass' if all(checks.values()) else 'fail',
            'checks':checks,'GPU_executed':False,'LES_executed':False,
            'scope':'Actual frozen argparse registrations for eight L2 contexts/four L3 ranks, portable root substitution, dry-run/no-output and missing-input fail-closed checks.'}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
