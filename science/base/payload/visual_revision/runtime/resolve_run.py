#!/usr/bin/env python3
"""Resolve and validate frozen N81 jobs. Default: no GPU, process or directory creation."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PACK = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def parser_from_frozen_source(path):
    """Execute only the actual ArgumentParser construction, with no source imports/main."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name in ('parse_args', 'main'))
    body = []
    for node in fn.body:
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'parser'
               and n.func.attr == 'parse_args' for n in ast.walk(node)):
            break
        body.append(node)
    else:
        raise ValueError('cannot isolate frozen argparse boundary: ' + str(path))
    body.append(ast.Return(value=ast.Name(id='parser', ctx=ast.Load())))
    module = ast.Module(body=[ast.FunctionDef(name='_parser', args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])], type_ignores=[])
    scope = {'argparse': argparse, 'Path': Path, '__doc__': ast.get_docstring(tree)}
    # This immutable option registry is needed only when validating spinup recipes.
    if '_PRECURSOR_SPEED_CONDITIONS' in path.read_text():
        speed_node = next((n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                           and '_PRECURSOR_SPEED_CONDITIONS' in ast.unparse(n).split('=')[0]), None)
        if speed_node is not None and isinstance(speed_node.value, ast.Dict):
            scope['_PRECURSOR_SPEED_CONDITIONS'] = {
                ast.literal_eval(key): None for key in speed_node.value.keys}
        elif speed_node is not None and isinstance(speed_node.value, ast.DictComp):
            keys = [ast.literal_eval(keyword.value)
                    for node in ast.walk(speed_node.value) if isinstance(node, ast.Call)
                    for keyword in node.keywords if keyword.arg == 'condition_id']
            if not keys:
                raise ValueError('cannot statically resolve immutable precursor option registry')
            scope['_PRECURSOR_SPEED_CONDITIONS'] = dict.fromkeys(keys)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope['_parser']()


def source_checks():
    manifest = read(PACK / 'source/FROZEN_SOURCE_MANIFEST.json')
    for row in manifest['files']:
        f = PACK / 'source' / row['path']
        if not f.is_file() or f.stat().st_size != row['bytes'] or digest(f) != row['sha256']:
            raise ValueError('frozen implementation identity mismatch: ' + row['path'])
    return {'frozen_py_files': len(manifest['files']),
            'formal_core_files': len(manifest['formal_core_implementation_sha256']),
            'byte_identity_verified': True}


def archive_checks(source, input_root, verify_hashes):
    archive = input_root / source['archive_relative_path']
    issues = []
    manifest_path, receipt_path = archive / 'manifest.json', archive / 'VERIFIED.json'
    if not manifest_path.is_file():
        return {'source_id': source['source_id'], 'status': 'needs_external_input',
                'issues': ['missing ' + str(manifest_path)], 'actual_hashes_verified': False}
    manifest = read(manifest_path)
    if digest(manifest_path) != source['manifest_sha256']:
        issues.append('manifest SHA differs from frozen scientific source')
    if manifest.get('source_state', {}).get('sha256') != source['source_state_sha256']:
        issues.append('qualified source-state identity differs')
    if manifest.get('provenance', {}).get('target_layout_sha256') != source['layout_sha256']:
        issues.append('canonical 81-layout identity differs')
    if manifest.get('status') != 'complete' or manifest.get('completed_planes') != source['completed_planes']:
        issues.append('archive is incomplete or has a different horizon')
    if manifest.get('dtype') != 'float32' or manifest.get('plane_shape') != source['plane_shape']:
        issues.append('archive transverse shape/dtype differs')
    if (archive / 'extension_manifest.json').exists():
        issues.append('unexpected extension changes frozen archive; no silent override')
    checked = 0
    for chunk in manifest.get('chunks', []):
        path = archive / chunk['path']
        try:
            path.resolve().relative_to(archive.resolve())
        except ValueError:
            issues.append('chunk escapes archive root'); continue
        if not path.is_file():
            issues.append('missing chunk ' + chunk['path']); continue
        if path.stat().st_size != chunk['bytes']:
            issues.append('chunk size differs ' + chunk['path']); continue
        if verify_hashes:
            if digest(path) != chunk['sha256']:
                issues.append('chunk SHA differs ' + chunk['path']); continue
            checked += 1
    if not receipt_path.is_file():
        issues.append('missing complete archive verification receipt')
    else:
        receipt = read(receipt_path)
        if (receipt.get('manifest_sha256') != source['manifest_sha256'] or
                not receipt.get('all_chunk_hashes_verified') or
                receipt.get('extension_manifest_sha256') is not None):
            issues.append('verification receipt does not bind the frozen complete archive')
    return {'source_id': source['source_id'],
            'status': 'input_contract_pass' if not issues else 'needs_external_input',
            'issues': issues, 'chunk_count': len(manifest.get('chunks', [])),
            'hash_checked_chunks_this_invocation': checked,
            'actual_hashes_verified': bool(verify_hashes and not issues and checked == source['chunk_files'])}


def build_spec(args):
    sources = read(PACK / 'configs/runtime_sources.json')['sources']
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    implementation = source_checks()
    jobs = []
    if args.job == 'L2':
        conf = read(PACK / 'configs/runtime_evaluation.json')
        context = next((c for c in conf['contexts'] if c['context_id'] == args.context), None)
        if context is None:
            raise ValueError('context must be one of the eight frozen B/C windows')
        source = next(s for s in sources if s['source_id'] == context['source_id'])
        archive = input_root / source['archive_relative_path']
        out = output_root / context['context_id']
        actor = PACK / conf['actor_path']
        if digest(actor) != conf['actor_sha256']:
            raise ValueError('final Actor identity mismatch')
        e = conf['evaluation']
        params = ['--target-turbines', '81', '--plane-archive', str(archive),
                  '--verified-archive', str(archive / 'VERIFIED.json'), '--actor', str(actor),
                  '--plane-start', str(context['window_start']), '--context-id', args.context,
                  '--output-dir', str(out), '--burnin', str(e['burnin']),
                  '--interactions', str(e['scored_interactions']), '--block-interactions',
                  str(e['block_interactions']), '--seed', str(e['seed']),
                  '--required-actor-interactions', '1000448']
        if context['window_end_exclusive'] != context['window_start'] + 50 * (e['burnin'] + e['scored_interactions']):
            raise ValueError('frozen evaluation input horizon mismatch')
        jobs.append({'rank': 0, 'physical_gpu': args.gpus[0],
                     'entrypoint': 'codes/eval_mole_large_scale_paired.py', 'parameters': params,
                     'exclusive_outputs': [str(out)]})
    else:
        conf = read(PACK / 'configs/runtime_training.json')
        source = next(s for s in sources if s['source_id'] == 'A')
        if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
            raise ValueError('L3 requires exactly four distinct explicit GPU ids')
        run_root = output_root / 'n81_train_r1'
        for row in conf['ranks']:
            params = list(row['argv'][3:])
            for index, value in enumerate(params):
                if value.startswith('/batch/archives/'):
                    params[index] = str(input_root / value[len('/batch/'):])
                elif value.startswith('/batch/runs/n81_train_r1/'):
                    params[index] = str(run_root / value.rsplit('/', 1)[1])
            params[params.index('--coordinator') + 1] = args.coordinator
            # An explicit process wall bound is an execution guard, not a change to physics/reward.
            if args.wall_seconds is not None:
                params.extend(['--stop-after-seconds', str(args.wall_seconds)])
            jobs.append({'rank': row['rank'], 'physical_gpu': args.gpus[row['rank']],
                         'entrypoint': 'codes/jax_mole_microbatch_distributed.py',
                         'parameters': params, 'exclusive_outputs': [str(run_root)]})
    parsed = []
    for job in jobs:
        entry = PACK / 'source' / job['entrypoint']
        frozen_parser = parser_from_frozen_source(entry)
        ns = frozen_parser.parse_args(job['parameters'])
        parsed.append({'rank': job['rank'], 'all_arguments_parsed_by_frozen_registration': True,
                       'resolved_defaults': json_value(vars(ns))})
        job['argv'] = [args.python, '-u', str(entry), *job['parameters']]
        job['environment'] = {'JAX_PLATFORMS': 'cuda', 'JAX_ENABLE_X64': '0',
                              'XLA_PYTHON_CLIENT_MEM_FRACTION': '.60',
                              'CUDA_VISIBLE_DEVICES': str(job['physical_gpu']),
                              'PYTHONPATH': str(PACK / 'source/codes'),
                              'PYTHONDONTWRITEBYTECODE': '1'}
        del job['parameters']
    inputs = archive_checks(source, input_root, args.verify_input_hashes or args.run)
    existing_outputs = sorted({out for job in jobs for out in job['exclusive_outputs'] if Path(out).exists()})
    return {'schema_version': 1, 'job': args.job, 'execution_requested': args.run,
            'default_is_dry_run': True, 'pack_root': str(PACK),
            'working_directory': str(PACK / 'source'), 'input_root': str(input_root),
            'output_root': str(output_root), 'wall_seconds_limit': args.wall_seconds,
            'allocated_gpu_hour_upper_bound': args.wall_seconds * len(jobs) / 3600 if args.wall_seconds else None,
            'implementation': implementation, 'argparse_checks': parsed, 'jobs': jobs,
            'input_contract': inputs, 'existing_outputs_refused': existing_outputs,
            'LES_execution_verified': False,
            'readiness': 'static_contract_valid_external_inputs_pending' if inputs['issues'] else 'static_contract_valid_gpu_untested',
            'image_identity': {'formal_local_image_id': 'sha256:a72fc0a237072f4177524eab7896b3d5ab8d1e2d7d62eb1bf45ff163e9c5729d',
                               'verified_registry_repo_digest': None,
                               'runtime': 'direct Python in a provisioned matching environment; no Docker factory or fixed NVML mount'}}


def execute(spec, args):
    if args.wall_seconds is None or args.wall_seconds <= 0:
        raise ValueError('--run requires an explicit positive --wall-seconds')
    if spec['input_contract']['issues'] or not spec['input_contract']['actual_hashes_verified']:
        raise ValueError('--run refused: complete frozen input bytes must be hash verified now')
    if spec['existing_outputs_refused']:
        raise FileExistsError('refuse existing output directories: ' + repr(spec['existing_outputs_refused']))
    runtime_probe = subprocess.run([args.python, '-c',
        'import importlib.metadata as m,json,platform; '
        'print(json.dumps({"python":platform.python_version(),'
        '"dependencies":{k:m.version(k) for k in ("jax","jaxlib","flax","optax","numpy","scipy")}}))'],
        env=dict(os.environ, JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1'),
        check=True, capture_output=True, text=True, timeout=30)
    actual_runtime = json.loads(runtime_probe.stdout)
    required_dependencies = {'jax':'0.9.0.1','jaxlib':'0.9.0.1','flax':'0.12.4',
                             'optax':'0.2.6','numpy':'2.3.5','scipy':'1.16.3'}
    if actual_runtime['python'] != '3.12.12' or actual_runtime['dependencies'] != required_dependencies:
        raise ValueError('frozen runtime policy refused mismatched Python/dependencies: ' + repr(actual_runtime))
    # Hardware allocation is an operator responsibility; never stop foreign processes.
    selected_gpus = ','.join(str(j['physical_gpu']) for j in spec['jobs'])
    rows = subprocess.run(['nvidia-smi', '-i', selected_gpus, '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()
    if rows:
        raise ValueError('GPU execution refused while compute processes are present; no process stopped')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for job in spec['jobs']:
            for out in job['exclusive_outputs']:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, **job['environment'])
            log_dir = args.output_root / '_runtime_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            logfile = log_dir / f'{spec["job"]}.rank{job["rank"]}.log'
            handle = logfile.open('x')
            logs.append(handle)
            processes.append(subprocess.Popen(job['argv'], env=env, cwd=spec['working_directory'],
                                              stdout=handle, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            if time.monotonic() - started >= args.wall_seconds:
                raise TimeoutError('explicit process wall budget exhausted')
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('one owned rank failed; stop only this launcher process group')
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('nonzero owned process exit')
        return {'processes_exit_zero': True, 'elapsed_wall_seconds': time.monotonic()-started,
                'scientific_artifact_validation': 'not_implemented_here; independently inspect frozen reports'}
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            if p.poll() is None:
                try:
                    p.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
        for handle in logs:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('L2', 'L3'), required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--context', default='n81_B_w1')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--coordinator', default='127.0.0.1:17381')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--wall-seconds', type=float)
    parser.add_argument('--verify-input-hashes', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-spec', type=Path)
    args = parser.parse_args()
    if not args.gpus or any(g < 0 for g in args.gpus):
        parser.error('GPU ids must be nonnegative')
    spec = build_spec(args)
    if args.output_spec:
        encoded = json.dumps(spec, indent=2, allow_nan=False) + '\n'
        with args.output_spec.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    if args.run:
        spec['execution_result'] = execute(spec, args)
    print(json.dumps(spec, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
