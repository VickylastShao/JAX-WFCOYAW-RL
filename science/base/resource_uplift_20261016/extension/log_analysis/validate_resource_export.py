#!/usr/bin/env python3
"""Validate returned norms and complete small traces with stdlib only.

Full velocity arrays remain remote. This validates SHA/membership, recomputes
relative norms from the independently exported sufficient statistics, and
independently recomputes full returned power/probe trace differences. It does
not claim to recompute complete velocity-field differences locally.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npy(data):
    if data[:6]!=b'\x93NUMPY':raise ValueError('Invalid NPY magic')
    version=data[6:8]
    if version==b'\x01\x00':length=struct.unpack('<H',data[8:10])[0];offset=10
    elif version in (b'\x02\x00',b'\x03\x00'):length=struct.unpack('<I',data[8:12])[0];offset=12
    else:raise ValueError('Unsupported NPY version')
    header=ast.literal_eval(data[offset:offset+length].decode('utf-8' if version==b'\x03\x00' else 'latin1').strip())
    if header['fortran_order']:raise ValueError('Fortran-order export not supported by this C-order checksum audit')
    dtype=header['descr'];shape=list(header['shape']);count=math.prod(shape)
    formats={'<f4':('f',4),'<f8':('d',8),'<i4':('i',4),'<i8':('q',8),'|u1':('B',1),'|i1':('b',1),'|b1':('?',1)}
    if dtype not in formats:raise ValueError('Unsafe or unsupported dtype: '+repr(dtype))
    code,size=formats[dtype];body=data[offset+length:]
    if len(body)!=count*size:raise ValueError('NPY shape/body size mismatch')
    values=[row[0] for row in struct.iter_unpack('<'+code,body)]
    return dict(shape=shape,dtype=dtype,values=values,raw=body)


def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-14)


def validate(root,expected_branch_count=5):
    manifest=json.loads((root/'SMALL_EXPORT_MANIFEST.json').read_text())
    file_checks=[]
    for row in manifest:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        path=root/relative
        file_checks.append(dict(path=row['path'],size_matches=path.stat().st_size==row['bytes'],SHA_matches=sha(path)==row['sha256']))
    expected={r['path'] for r in manifest}|{'SMALL_EXPORT_MANIFEST.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    stats=json.loads((root/'INDEPENDENT_ARRAY_CHECK.json').read_text())
    branches=stats['expected_branches'];fields=stats['expected_fields']
    pairs={(r['branch'],r['field']) for r in stats['arrays']}
    coverage=(pairs=={(b,f) for b in branches for f in fields} and len(pairs)==len(stats['arrays']))
    recalculated=[]
    for row in stats['arrays']:
        relative=math.sqrt(row['sum_squared_difference'])/max(math.sqrt(row['sum_squared_reference']),1e-30)
        recalculated.append(dict(branch=row['branch'],field=row['field'],relative_L2_recomputed=relative,
            relative_L2_exported=row['relative_l2'],norm_arithmetic_matches=close(relative,row['relative_l2']),
            runner_norm_matches=(close(relative,row['runner_relative_l2']) if row['runner_relative_l2'] is not None else row['branch']=='dense_b1'),
            finite_record=row['finite'],array_sha256=row['array_sha256'],max_absolute_error=row['max_abs']))
    with zipfile.ZipFile(root/'SCORING_AND_FIELD_SAMPLES.npz') as archive:
        arrays={name[:-4]:read_npy(archive.read(name)) for name in archive.namelist() if name.endswith('.npy')}
    trace_checks=[];sample_checks=[]
    by_pair={(r['branch'],r['field']):r for r in stats['arrays']}
    for branch in branches:
        for field in ('power_trace','final_probes'):
            key=branch+'__'+field
            a=arrays[key];reference=arrays['dense_b1__'+field]
            values=reference['values']
            if branch.endswith('_b2'):
                if field=='power_trace':
                    per_time=math.prod(reference['shape'][1:])
                    values=[value for step in range(reference['shape'][0])
                        for _ in range(2) for value in values[step*per_time:(step+1)*per_time]]
                else:values=values*2
            if len(values)!=len(a['values']):raise ValueError('Tiled small trace size mismatch')
            differences=[x-y for x,y in zip(a['values'],values)]
            ssr=math.fsum(x*x for x in values);ssd=math.fsum(x*x for x in differences)
            relative=math.sqrt(ssd)/max(math.sqrt(ssr),1e-30);maximum=max(abs(x) for x in differences)
            row=by_pair[(branch,field)]
            trace_checks.append(dict(branch=branch,field=field,shape=a['shape'],
              raw_C_order_SHA_matches=hashlib.sha256(a['raw']).hexdigest()==row['array_sha256'],
              all_values_finite=all(math.isfinite(x) for x in a['values']),
              max_absolute_error_recomputed=maximum,max_absolute_error_matches=maximum==row['max_abs'],
              relative_L2_recomputed=relative,relative_L2_matches=close(relative,row['relative_l2']),
              sum_squared_difference_matches=close(ssd,row['sum_squared_difference']),
              scope='all returned elements; independently recomputed locally with Python math.fsum'))
        for field in fields:
            idx=arrays[branch+'__'+field+'__sample_indices'];sample=arrays[branch+'__'+field+'__samples']
            size=math.prod(by_pair[(branch,field)]['shape'])
            okay=len(idx['values'])==len(sample['values']) and all(0<=v<size for v in idx['values'])
            sample_checks.append(dict(branch=branch,field=field,samples=len(sample['values']),
                indices_valid=okay,finite=all(math.isfinite(v) for v in sample['values']),
                complete_field_available_locally=field in ('power_trace','final_probes')))
    gates=dict(export_manifest_all_file_SHA_and_sizes=all(r['size_matches'] and r['SHA_matches'] for r in file_checks),
      export_membership_matches_manifest=actual==expected,
      expected_branch_ten_field_coverage=coverage and len(branches)==expected_branch_count and len(fields)==10,
      all_norm_arithmetic_matches=all(r['norm_arithmetic_matches'] for r in recalculated),
      all_recomputed_norms_match_runner=all(r['runner_norm_matches'] for r in recalculated),
      all_complete_returned_trace_SHA_match=all(r['raw_C_order_SHA_matches'] for r in trace_checks) and len(trace_checks)==2*expected_branch_count,
      all_trace_differences_and_norms_match=all(r['max_absolute_error_matches'] and r['relative_L2_matches'] and r['sum_squared_difference_matches'] for r in trace_checks),
      returned_samples_indices_and_values_valid=all(r['indices_valid'] and r['finite'] for r in sample_checks))
    return dict(schema='returned_resource_export_validation/1',status='pass' if all(gates.values()) else 'fail',
       checks=gates,file_checks=file_checks,full_array_norm_arithmetic=recalculated,
       expected_branch_count=expected_branch_count,
       complete_small_trace_reanalysis=trace_checks,sample_checks=sample_checks,
       external_full_field_files=stats['raw_output_files'],
       external_full_field_bytes=sum(r['bytes'] for r in stats['raw_output_files']),
       locally_recomputed_complete_field_differences=False,
       locally_recomputed_full_returned_power_and_probe_differences=True,
       full_field_norms_origin='independent CPU export on original server; sufficient-statistic arithmetic checked locally',
       no_GPU_or_network=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-branch-count',type=int,default=5,help='Preregistered main has 5; separate loaded follow-up has 2')
    args=parser.parse_args();result=validate(args.root,args.expected_branch_count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],
             external_full_field_bytes=result['external_full_field_bytes'])))
    raise SystemExit(0 if result['status']=='pass' else 1)


if __name__=='__main__':main()
