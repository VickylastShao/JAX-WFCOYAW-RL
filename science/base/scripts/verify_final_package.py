#!/usr/bin/env python3
"""Verify the final package and optional streamed archive using stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for part in iter(lambda: stream.read(4 << 20), b''):
        h.update(part)
    return h.hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe path: ' + name)


def verify(root, archive=None):
    root = root.resolve()
    manifest_path = root / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['files']
    expected = {row['path']: row for row in rows}
    if len(expected) != len(rows) or 'MANIFEST.json' in expected:
        raise ValueError('Duplicate/recursive manifest')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlink in package')
        if p.is_file() and p != manifest_path:
            actual.add(p.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError('Stage members differ from manifest')
    for name, row in expected.items():
        safe(name)
        p = root / name
        if p.stat().st_size != row['bytes']:
            raise ValueError('Size mismatch: ' + name)
        with p.open('rb') as f:
            if digest(f) != row['sha256']:
                raise ValueError('SHA mismatch: ' + name)
    tar_count = None
    if archive:
        wanted = {root.name + '/' + k: v for k, v in expected.items()}
        with manifest_path.open('rb') as f:
            wanted[root.name + '/MANIFEST.json'] = {
                'bytes': manifest_path.stat().st_size, 'sha256': digest(f)}
        found = set()
        with tarfile.open(archive, 'r|gz') as tar:
            for m in tar:
                safe(m.name)
                if m.isdir():
                    if m.name.rstrip('/') != root.name and not any(k.startswith(m.name.rstrip('/') + '/') for k in wanted):
                        raise ValueError('Unexpected directory')
                    continue
                if not m.isfile() or m.name not in wanted or m.name in found:
                    raise ValueError('Unexpected, duplicate or nonregular tar member')
                row = wanted[m.name]
                with tar.extractfile(m) as f:
                    if m.size != row['bytes'] or digest(f) != row['sha256']:
                        raise ValueError('Tar member identity differs: ' + m.name)
                found.add(m.name)
        if found != set(wanted):
            raise ValueError('Incomplete tar coverage')
        tar_count = len(found)
    return {'status': 'pass', 'files_excluding_top_manifest': len(rows),
            'tar_files_including_top_manifest': tar_count,
            'scope': 'file identity only; CPU reanalysis and physical replay are separate'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.archive), indent=2))
