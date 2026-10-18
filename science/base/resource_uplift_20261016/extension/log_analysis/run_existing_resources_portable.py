#!/usr/bin/env python3
"""Replay historical resource arithmetic from bound portable JSONs, stdlib only.

Usage: python3 run_existing_resources_portable.py --root EXTENSION --output /tmp/fresh-output
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent,
                        help='Extension containing log_analysis and sibling GPU campaigns')
    parser.add_argument('--output',type=Path,required=True,help='New external output directory')
    parser.add_argument('--pack-name',default=None,help='Default: frozen portable input manifest pack_name')
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.is_relative_to(root):raise ValueError('Output must be outside the immutable extension')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be new or empty')
    directory=root/'log_analysis';inputs=directory/'portable_inputs'
    manifest=json.loads((inputs/'PORTABLE_INPUT_MANIFEST.json').read_text())
    checks=[]
    for row in manifest['inputs']:
        relative=Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unexpected portable relative path')
        path=inputs/relative
        actual=sha(path)
        okay=path.stat().st_size==row['bytes'] and actual==row['sha256']
        checks.append(dict(label=row['label'],relative_path=relative.as_posix(),
                           expected_sha256=row['sha256'],observed_sha256=actual,pass_check=okay))
    if len(checks)!=manifest['file_count'] or not all(r['pass_check'] for r in checks):
        raise ValueError('Portable inputs failed identity checks')
    historical=directory/'analyze_existing_resources.py'
    spec=importlib.util.spec_from_file_location('frozen_existing_resources',historical)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Historical script remains byte-identical. Only explicit configuration is
    # rebound to the frozen portable directory for this invocation.
    module.DEFAULT_ROOT=inputs/'submission'
    module.DEFAULT_REPO=inputs/'repo'
    module.PACK_NAME=args.pack_name or manifest['pack_name']
    previous_argv=sys.argv
    try:
        sys.argv=[str(historical),'--submission-root',str(module.DEFAULT_ROOT),
                  '--repo-root',str(module.DEFAULT_REPO),'--output',str(out)]
        module.main()
    finally:sys.argv=previous_argv
    receipt=dict(schema='portable_resource_analysis_binding_v1',status='pass',
        input_manifest_sha256=sha(inputs/'PORTABLE_INPUT_MANIFEST.json'),
        historical_analysis_script_sha256=sha(historical),
        wrapper_sha256=sha(Path(__file__)),pack_name=module.PACK_NAME,
        inputs_checked=checks,file_count=len(checks),no_GPU_or_network=True,
        original_historical_inputs_not_rewritten=True,
        source_paths_in_new_output_rebased_to_this_portable_copy=True)
    (out/'PORTABLE_SOURCE_BINDING.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
