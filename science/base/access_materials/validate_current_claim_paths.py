#!/usr/bin/env python3
"""Resolve current claim evidence against the declared extracted-package root."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):
            h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    output=args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Keep outputs outside the immutable package')
    mapping=json.loads((root/'CLAIM_EVIDENCE_MAP.json').read_text())
    binding=mapping['current_bundle_path_resolution']
    if binding['base']!='this_extracted_package_root':
        raise ValueError('Unexpected resolution base')
    prefix=binding['delivery_prefix']
    if prefix!='repro-pack/n81_20261016_minor_fixes_repro/':
        raise ValueError('Unexpected current delivery prefix')
    def resolve(value):
        # Current manuscript-side metadata uses one ../ before its outer root.
        if value.startswith('../'+prefix):value=value[3:]
        if not value.startswith(prefix):
            raise ValueError('Current evidence points outside the declared current delivery: '+value)
        relative=PurePosixPath(value[len(prefix):])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe current evidence suffix')
        target=root/relative
        if not target.resolve().is_relative_to(root):
            raise ValueError('Current evidence escapes package')
        return relative,target
    checked=[]
    for claim in mapping['claims']:
        for row in claim.get('final_bundle_evidence',[]):
            relative,target=resolve(row['path'])
            if row['kind']=='directory':
                if not target.is_dir():raise FileNotFoundError(target)
                actual=None
            elif row['kind']=='file':
                if not target.is_file():raise FileNotFoundError(target)
                actual=sha(target)
                if actual!=row['sha256']:
                    raise ValueError('Current evidence SHA differs: '+str(relative))
                if 'bytes' in row and target.stat().st_size!=row['bytes']:
                    raise ValueError('Current evidence length differs: '+str(relative))
            else:
                raise ValueError('Unknown evidence kind')
            checked.append(dict(claim=claim['id'],declared_path=row['path'],
                resolved_package_path=relative.as_posix(),kind=row['kind'],sha256=actual,status='pass'))
    if len(mapping['claims'])!=23 or len(checked)!=143:
        raise ValueError('Current fixed-version claim/evidence count differs')
    top=[]
    for field in ['source_pack_root','revision_root','current_visual_revision_root']:
        relative,target=resolve(mapping[field])
        if not target.is_dir():raise FileNotFoundError(target)
        top.append(dict(field=field,resolved_package_path=relative.as_posix(),status='pass'))
    for field in ['evidence_roots','current_visual_roots']:
        for label,value in mapping[field].items():
            relative,target=resolve(value)
            if not target.is_dir():raise FileNotFoundError(target)
            top.append(dict(field=field+'.'+label,resolved_package_path=relative.as_posix(),status='pass'))
    for field in ['support_compute_ledger','resource_compute_ledger']:
        row=mapping[field]
        relative,target=resolve(row['path'])
        if not target.is_file() or sha(target)!=row['sha256']:
            raise ValueError('Current top-level ledger identity differs: '+field)
        if 'bytes' in row and target.stat().st_size!=row['bytes']:
            raise ValueError('Current top-level ledger length differs: '+field)
        top.append(dict(field=field,resolved_package_path=relative.as_posix(),status='pass'))
    report=dict(schema='review-closure-current-claim-path-check/1',status='pass',
        claims=23,current_evidence_references_checked=len(checked),
        file_references_checked=sum(r['kind']=='file' for r in checked),
        directory_references_checked=sum(r['kind']=='directory' for r in checked),
        current_top_level_paths_checked=len(top),current_top_level_path_checks=top,
        root_resolution=binding,checks=checked,new_GPU_hours=0,network_used=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
