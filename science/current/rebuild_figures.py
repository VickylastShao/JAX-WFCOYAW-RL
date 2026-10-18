#!/usr/bin/env python3
"""Rebuild seven current scientific figure groups from any working directory.

Run on a writable copy. Default is preview-only; --final replaces formal
PDF/SVG/PNG figures. Rebuild output requires local visual review.
No GPU, LES, training, installation or network access is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'redesign/story_common.py').is_file():
        raise ValueError('Use the package-root copy of rebuild_figures.py')
    for rel in ['validation', 'previews', 'manuscript/figures']:
        (root / rel).mkdir(parents=True, exist_ok=True)
    original = {str(path.relative_to(root)): sha(path)
                for path in (root / 'figure_sources').rglob('*')
                if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in path.parts}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               WIND_FIG_FINAL='1' if args.final else '0', WIND_FIG_REBUILD='1')
    env.pop('PYTHONPATH', None)
    start = time.monotonic()
    commands = []
    for script in ['method_story.py', 'resource_story.py', 'control_story.py']:
        command = [sys.executable, '-B', str(root / 'redesign' / script)]
        result = subprocess.run(command, env=env, check=True, timeout=360)
        commands.append({'script': 'redesign/' + script, 'exit_code': result.returncode})
    assert all(sha(root / rel) == digest for rel, digest in original.items()), 'A scientific input changed'
    report = {'status': 'pass', 'scope': 'seven current figure groups',
              'cwd_independent_paths': True, 'invoked_cwd': str(Path.cwd()),
              'final_exports_requested': args.final, 'commands': commands,
              'scientific_inputs_unchanged_sha_count': len(original),
              'elapsed_seconds': time.monotonic() - start,
              'new_GPU_hours': 0, 'network_used': False,
              'byte_identical_exports_across_platforms_claimed': False}
    (root / 'validation/FIGURE_REBUILD.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
