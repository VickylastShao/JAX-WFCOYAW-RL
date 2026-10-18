#!/usr/bin/env python3
"""Verify one extracted visual-story delivery; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    expected = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        digest, rel = line.split('  ', 1)
        path = PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or '\\' in rel:
            raise ValueError('Unsafe checksum path: ' + rel)
        expected[rel] = digest
    failures = []
    for rel, digest in expected.items():
        path = root / rel
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            failures.append(rel)
    records = {record['path']: record for record in manifest['files']}
    for rel, record in records.items():
        path = root / rel
        if rel not in expected or path.stat().st_size != record['bytes'] or expected[rel] != record['sha256']:
            failures.append('manifest:' + rel)
    allowed = set(expected) | {'SHA256SUMS.txt'}
    # The immutable payload is checked before generating any reproduction output.
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in root.rglob('*') if p.is_file()}
    extra = sorted(actual - allowed)
    report = dict(schema='final-polish-package-sha/1', status='pass' if not failures and not extra else 'fail',
                  files_checked=len(expected), payload_files=len(records), failures=failures,
                  unexpected_files=extra, manifest_sha256=sha(root / 'MANIFEST.json'),
                  new_GPU_hours=0, network_used=False)
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(root):
            raise ValueError('Write the report outside the immutable package')
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
