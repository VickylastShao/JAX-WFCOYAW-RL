#!/usr/bin/env python3
"""Copy the nine frozen small JSON inputs; never rewrite original evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ORIGINAL_REPO=Path.cwd()
ORIGINAL_SUBMISSION=Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/wind_energy_navigation_closure_20261016')
PACK_NAME='n81_20261016_navigation_closure_repro'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'portable_inputs')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Portable input snapshot is immutable')
    manifest=json.loads((HERE/'INPUT_MANIFEST.json').read_text())
    rows=[]
    for label,record in manifest['sources'].items():
        source=Path(record['path'])
        if source.is_relative_to(ORIGINAL_SUBMISSION):
            relative=Path('submission')/source.relative_to(ORIGINAL_SUBMISSION)
        elif source.is_relative_to(ORIGINAL_REPO):
            relative=Path('repo')/source.relative_to(ORIGINAL_REPO)
        else:raise ValueError('Unexpected original root')
        if source.stat().st_size!=record['bytes'] or sha(source)!=record['sha256']:
            raise ValueError('Original input identity changed: '+label)
        rows.append(dict(label=label,relative_path=relative.as_posix(),
                         bytes=record['bytes'],sha256=record['sha256']))
    # All sources are verified before any copy is written.
    args.output.mkdir(parents=True)
    for row in rows:
        source=Path(manifest['sources'][row['label']]['path'])
        target=args.output/row['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if sha(target)!=row['sha256'] or sha(source)!=row['sha256']:
            raise ValueError('Copy identity mismatch: '+row['label'])
    frozen=dict(schema='portable_existing_resource_inputs_v1',pack_name=PACK_NAME,
        original_input_manifest_sha256=sha(HERE/'INPUT_MANIFEST.json'),
        file_count=len(rows),total_bytes=sum(row['bytes'] for row in rows),
        retained_original_bytes=True,inputs=rows)
    (args.output/'PORTABLE_INPUT_MANIFEST.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status='pass',file_count=frozen['file_count'],total_bytes=frozen['total_bytes'],
                         every_original_and_copy_SHA_verified=True)))


if __name__=='__main__':main()
