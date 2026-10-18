#!/usr/bin/env python3
"""Verify/extract the manifest-bound scientific derivative and run its bounded CPU suite."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--workdir', required=True, type=Path, help='New directory on a Linux native filesystem, outside payload')
    args = parser.parse_args()
    root, work = args.root.resolve(strict=True), args.workdir.resolve()
    if os.name != 'posix' or sys.platform != 'linux':
        raise ValueError('The unchanged archival validators were qualified on Linux; use WSL/Linux')
    if work.exists() or work.is_relative_to(root) or str(work).startswith('/mnt/'):
        raise ValueError('Use a new workdir on a Linux native filesystem outside the delivery')
    import numpy
    if numpy.__version__ != '2.3.5':
        raise ValueError('CPU arithmetic acceptance requires the pinned NumPy 2.3.5 environment')
    work.mkdir(parents=True)
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    dump(work / 'JOB_HANDLE.json', dict(pid=os.getpid(), argv=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), timeout_seconds=1200,
        scientific_scope='Saved-record CPU arithmetic and structural checks only; no physical LES or training',
        completion_criteria='Package SHA, manifest-bound derivative extraction and all sixteen retained saved-record CPU commands pass'))
    try:
        subprocess.run([sys.executable, '-B', root / 'verify_package.py', '--root', root,
                        '--output', work / 'PACKAGE_SHA.json'], check=True, env=env, timeout=240)
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        binding = manifest['immutable_base']
        archive = root / binding['path']
        if sha(archive) != binding['sha256']:
            raise ValueError('Scientific-derivative SHA differs')
        target = work / 'historical_base'
        target.mkdir()
        extracted = []
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe/nonregular historical member: ' + member.name)
                if member.isdir():
                    (target / member.name).mkdir(parents=True, exist_ok=True)
                else:
                    destination = target / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, destination.open('wb') as output:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                            output.write(block)
                extracted.append(member.name)
        base = target / binding['top_directory']
        entry = base / 'minor_fixes_20261016/access/run_cpu_acceptance.py'
        command = [sys.executable, '-B', str(entry), '--root', str(base), '--output', str(work / 'results')]
        dump(work / 'BASE_EXTRACTION.json', dict(status='pass', scientific_derivative_sha256=binding['sha256'],
             members_extracted=len(extracted), base=str(base), command=command, validators_modified=False))
        with (work / 'CPU.stdout').open('w') as out, (work / 'CPU.stderr').open('w') as err:
            result = subprocess.run(command, cwd=base, env=env, stdout=out, stderr=err, timeout=720)
        if result.returncode:
            raise RuntimeError('The unchanged CPU suite failed; inspect CPU.stdout, CPU.stderr and results/')
        report = json.loads((work / 'results/CPU_ACCEPTANCE.json').read_text())
        if report['status'] != 'pass' or report['CPU_command_count'] != 16:
            raise ValueError('All sixteen historical checks are required')
        subprocess.run([sys.executable, '-B', root / 'verify_current_figures.py', '--root', root,
                        '--output', work / 'CURRENT_FIGURES.json'], check=True, env=env, timeout=120)
        dump(work / 'JOB_EXIT.json', dict(status='complete', exit_code=0,
            elapsed_seconds=time.monotonic()-started, CPU_command_count=16,
            validators_modified=False, new_GPU_hours=0, physical_LES_replay_executed=False))
        print(json.dumps(dict(status='pass', CPU_command_count=16, workdir=str(work), new_GPU_hours=0)), flush=True)
    except BaseException as error:
        dump(work / 'JOB_EXIT.json', dict(status='failed', exit_code=1, error=str(error),
             elapsed_seconds=time.monotonic()-started, new_GPU_hours=0))
        raise


if __name__ == '__main__':
    main()
