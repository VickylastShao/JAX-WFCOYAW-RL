#!/usr/bin/env python3
"""Check a downloaded release asset against the supplied RELEASE_FILES.json."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', type=Path, default=Path('RELEASE_FILES.json'))
    p.add_argument('--directory', type=Path, default=Path('.'))
    p.add_argument('--name', help='Check one listed filename; default all listed assets')
    args = p.parse_args()
    record = json.loads(args.files.read_text(encoding='utf-8'))
    rows = record['files']
    if args.name:
        rows = [row for row in rows if row['name'] == args.name]
        if len(rows) != 1:
            raise ValueError('Requested asset is not uniquely listed')
    results = []
    for row in rows:
        name = Path(row['name'])
        if name.name != str(name) or name.is_absolute():
            raise ValueError('Expected a basename in file inventory')
        path = args.directory/name
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 << 20), b''):
                h.update(chunk)
        okay = path.stat().st_size == row['bytes'] and h.hexdigest() == row['sha256']
        results.append(dict(name=row['name'], status='pass' if okay else 'fail',
                            observed_bytes=path.stat().st_size, observed_sha256=h.hexdigest()))
    print(json.dumps(dict(status='pass' if all(r['status']=='pass' for r in results) else 'fail',
                          files=results), indent=2))
    if not all(r['status']=='pass' for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
