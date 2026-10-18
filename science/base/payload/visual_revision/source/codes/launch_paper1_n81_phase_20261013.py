"""Materialize a new immutable evidence phase and launch its finite supervisor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--materials', type=Path, required=True)
    p.add_argument('--materials-sha256', required=True)
    p.add_argument('--closure-root', type=Path, required=True)
    p.add_argument('--base-stage', type=Path, required=True)
    p.add_argument('--legacy-root', type=Path, required=True)
    a = p.parse_args()
    if digest(a.materials) != a.materials_sha256:
        raise ValueError('material package checksum differs')
    with tarfile.open(a.materials, 'r:gz') as t:
        members = t.getmembers()
        if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise ValueError('unsafe material member')
        plan = json.load(t.extractfile('PLAN.json'))
        phase = plan['phase_id']
        if phase not in ('P3', 'P4'):
            raise ValueError('only the approved remaining phases are supported')
        b = a.closure_root / 'jobs' / phase
        b.mkdir()  # Refuse overwrite, reset, or duplicate dispatch.
        (b / 'logs').mkdir()
        shutil.copytree(a.base_stage, b / 'stage')
        for m in members:
            target = b / ('stage/' + m.name if m.name.startswith('codes/') else m.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as f:
                shutil.copyfileobj(t.extractfile(m), f)
    overlay = json.loads((b / 'NEW_CODE_MANIFEST.json').read_text())
    for name, expected in overlay.items():
        if digest(b / 'stage' / name) != expected:
            raise ValueError('overlay checksum differs: ' + name)
    manifest = {str(p.relative_to(b / 'stage')): digest(p)
                for p in sorted((b / 'stage').rglob('*')) if p.is_file()}
    (b / 'CODE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    argv = ['python3', '-u', str(b / 'stage/codes/run_paper1_n81_budgeted_job_20261013.py'),
            '--batch', str(b), '--legacy-root', str(a.legacy_root)]
    with (b / 'logs/supervisor.log').open('xb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'pid': child.pid, 'argv': argv, 'batch': str(b), 'started_unix': time.time(),
              'plan_sha256': digest(b / 'PLAN.json'), 'stage_files': len(manifest),
              'budget_gpu_hours': plan['initial_stage_gpu_hours_limit'],
              'cpu_before_gpu': plan['steps'][0]['kind'] == 'cpu'}
    (b / 'DISPATCH.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
