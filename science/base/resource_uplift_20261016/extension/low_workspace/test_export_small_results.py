#!/usr/bin/env python3
"""Artificial CPU fixtures for export acceptance, never scientific data.

All test inputs/outputs are confined to an explicitly separate fixture directory.
The production exporter is imported unchanged with its ROOT redirected in this
process. No actual remote export, JAX import or GPU operation occurs.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import json
import time

import numpy as np

BRANCHES = ['dense_b1', 'stream_g1_b1', 'stream_g9_b1', 'stream_g3_b1', 'stream_g3_b2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_arrays():
    velocity = np.linspace(.2, 2., 3*48*48*48, dtype=np.float32).reshape(1,3,48,48,48)
    return {'state_velocity': velocity,
            'state_rhs_previous': np.array(velocity * np.float32(.1)),
            'state_rhs_previous_2': np.array(velocity * np.float32(.07)),
            'state_filtered_disk_speed': np.linspace(.5,1.5,3,dtype=np.float32)[None],
            'state_bottom_pressure_gradient': np.ones((1,2,2,2),dtype=np.float32),
            'state_open_x_pressure_gradient': np.ones((1,2,2,2,2),dtype=np.float32),
            'state_step_index': np.asarray(50,dtype=np.int32),
            'power_trace': np.arange(12,dtype=np.float32).reshape(4,1,3)+1,
            'final_probes': np.linspace(.5,2.,3*77,dtype=np.float32).reshape(1,3,77),
            'final_divergence': np.linspace(-1e-6,1e-6,8,dtype=np.float32).reshape(1,2,2,2)}


def create_fixture(root, kind, compare):
    root.mkdir(parents=True)
    (root/'FAKE_FIXTURE_ONLY.json').write_text(json.dumps({'purpose':'artificial exporter negative-control fixture, not LES results','kind':kind}))
    (root/'WORKFLOW_EXIT.json').write_text(json.dumps({'status':'completed'}))
    reference = reference_arrays()
    for i, branch in enumerate(BRANCHES):
        if kind=='missing_branch' and branch=='stream_g3_b2':
            continue
        if kind=='missing_reference' and branch=='dense_b1':
            continue
        directory = root/'gpu_main'/branch
        directory.mkdir(parents=True)
        values, comparisons = {}, {}
        for key, value in reference.items():
            target = np.repeat(value,2,axis=1 if key=='power_trace' else 0) if branch.endswith('_b2') and value.ndim else value
            actual = np.array(target)
            if i and key!='state_step_index':
                actual += np.float32(i*1e-7)
            values[key] = actual
            comparisons[key] = compare(actual,target,np)
        result = {'status':'completed','comparison':comparisons if i else None}
        if kind=='missing_comparison' and branch=='stream_g3_b1':
            result.pop('comparison')
        if kind=='missing_comparison_field' and branch=='stream_g3_b2':
            result['comparison'].pop('power_trace')
        if kind=='corrupt_runner_summary' and branch=='stream_g1_b1':
            result['comparison']['power_trace']['relative_l2'] = .123
        (directory/'RESULT.json').write_text(json.dumps(result,indent=2))
        np.savez(directory/'OUTPUT.npz',**values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exporter',type=Path,default=Path(__file__).resolve().parents[1]/'export_small_results.py')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    exporter = load_module(args.exporter,'isolated_exporter_fixture')
    runner = load_module(Path(__file__).parent/'benchmark_full_flow.py','frozen_runner_comparison_fixture')
    records=[]
    started=time.monotonic()
    for kind in ['positive','missing_branch','missing_comparison','missing_comparison_field','corrupt_runner_summary','missing_reference']:
        root=args.output/kind
        create_fixture(root,kind,runner.compare)
        exporter.ROOT=root
        with (root/'EXPORT_STDOUT.log').open('w') as log,contextlib.redirect_stdout(log):
            exporter.main()
        summary=json.loads((root/'SMALL_EXPORT/INDEPENDENT_ARRAY_CHECK.json').read_text())
        receipt=json.loads((root/'SMALL_EXPORT_RECEIPT.json').read_text())
        valid=(summary['coverage_complete'] and summary['all_finite'] and summary['all_runner_summaries_agree'])
        complete=summary['status']=='independent_array_summary_complete'
        # A complete acceptance label must not survive missing/mismatched runner
        # comparison evidence; actual array coverage is retained separately.
        expected=(kind=='positive')
        checks={'acceptance_matches_expectation':valid==expected,
                'completion_label_matches_expectation':complete==expected,
                'receipt_agreement_matches_expectation':receipt['all_runner_summaries_agree']==expected}
        if expected:
            checks['all_five_branches_ten_fields']=len(summary['arrays'])==50
            with np.load(root/'SMALL_EXPORT/SCORING_AND_FIELD_SAMPLES.npz',allow_pickle=False) as small:
                checks['complete_power_and_probe_traces_retained']=all(
                    branch+'__'+key in small.files for branch in BRANCHES for key in ['power_trace','final_probes'])
                checks['scalar_state_sample_valid']=small['stream_g3_b2__state_step_index__samples'].tolist()==[50]
        records.append({'fixture':kind,'observed_status':summary['status'],
                        'coverage_complete':summary['coverage_complete'],
                        'all_runner_summaries_agree':summary['all_runner_summaries_agree'],
                        'array_records':len(summary['arrays']),'checks':checks,
                        'pass':all(checks.values())})
    value={'status':'pass' if all(r['pass'] for r in records) else 'regression_findings',
           'exporter':str(args.exporter),'exporter_sha256':sha(args.exporter),
           'test_script_sha256':sha(__file__),'fixtures':records,
           'elapsed_cpu_wall_seconds':time.monotonic()-started,
           'artificial_fixture_only':True,'real_remote_export_executed':False,
           'JAX_or_GPU_initialized':False}
    (args.output/'EXPORT_NEGATIVE_CONTROL_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2),flush=True)
    return 0 if value['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
