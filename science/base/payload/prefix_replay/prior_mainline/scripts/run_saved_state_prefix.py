#!/usr/bin/env python3
"""B_w1 common-state/original-input replay. Default: CPU preflight only.

Scientific functions are imported unchanged from the SHA-frozen package.
--run requires separately confirmed allocation; one paired 8-control prefix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import numpy as np
from saved_prefix_cpu_contract import (sha256,trusted_structural_load,
    validate_saved_state,validate_actor_payload,STATE_SHA,ACTOR_SHA,PREFIX_SHA)

SOURCE_MANIFEST_SHA='3610b1b059068511ff2f5821c437720da6e1600b5047067c108e73d684fdf980'
REFERENCE_SHAS={'zero':'9494a0573e99a216ad02582f2d2f5cbc769699a4625ad4dfd5b509097c055dfb',
                'rl_r1':'bd84f65d6e5b643b8a33e06a3b54157c1adb6bced2fb40cb48979f9843db9f95'}


def save(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def preflight(args):
    manifest=args.pack_root/'source/FROZEN_SOURCE_MANIFEST.json'
    if sha256(manifest)!=SOURCE_MANIFEST_SHA:raise ValueError('frozen source manifest identity differs')
    source=json.loads(manifest.read_text())
    for row in source['files']:
        p=args.pack_root/'source'/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
            raise ValueError('frozen scientific source differs: '+row['path'])
    code=args.pack_root/'source/codes'
    state=trusted_structural_load(args.common_state,STATE_SHA,code)
    state_report=validate_saved_state(state)
    actor_path=args.pack_root/'actors/actor.pkl'
    actor=trusted_structural_load(actor_path,ACTOR_SHA,code)
    actor_report=validate_actor_payload(actor)
    receipt=json.loads(args.source_binding_receipt.read_text())
    expected={'source_manifest_sha256_actual':'4b42d5793764bffb2a577f9f4bd8e448873123a008cf5b93d6cdf9ea8f522b30',
              'source_chunk_sha256_actual':'f5165e220d0b3ccd17a588dec2e1c807e6e00dad90b24114c2cbcf0c8c8b99d3',
              'source_chunk_bytes_actual':4752720128,'global_start':27500,'global_stop_exclusive':27900,
              'common_state_sha256':STATE_SHA,'original_actor_sha256':ACTOR_SHA,'sha256':PREFIX_SHA}
    for k,v in expected.items():
        if receipt.get(k)!=v:raise ValueError('original inlet binding differs: '+k)
    if sha256(args.inlet_prefix)!=PREFIX_SHA or args.inlet_prefix.stat().st_size!=95054528:
        raise ValueError('actual original prefix SHA/size differs')
    inlet=np.load(args.inlet_prefix,mmap_mode='r',allow_pickle=False)
    if inlet.shape!=(400,3,41,483) or inlet.dtype!=np.dtype('float32') or not np.isfinite(inlet).all():
        raise ValueError('actual original prefix shape/dtype/finiteness differs')
    refs={}
    for branch,expected_sha in REFERENCE_SHAS.items():
        p=args.reference_root/f'REFERENCE_B_W1_PREFIX8_{branch}.npz'
        if sha256(p)!=expected_sha:raise ValueError('frozen prefix reference differs: '+branch)
        with np.load(p,allow_pickle=False) as ref:refs[branch]={k:np.array(ref[k]) for k in ref.files}
    checks=[]
    for branch,ref in refs.items():
        for k,v in ref.items():
            if v.shape[0]!=8 or v.dtype!=np.dtype('float32') or not np.isfinite(v).all():
                raise ValueError('reference field contract differs: '+branch+'/'+k)
        # CPU reductions can round differently from JAX; record differences, no new tolerance/gain gate.
        mean_u=inlet[:,0].reshape(8,50,41,483).mean(axis=(1,2,3),dtype=np.float32)
        delta=np.sqrt(np.mean((inlet[49::50]-inlet[0::50])**2,axis=(1,2,3),dtype=np.float32))
        checks.append({'branch':branch,'CPU_mean_u_vs_original_max_abs_m_s':float(np.max(np.abs(mean_u-ref['inlet_mean_u'][:,0]))),
                       'CPU_inlet_delta_rms_vs_original_max_abs_m_s':float(np.max(np.abs(delta-ref['inlet_delta_rms'][:,0])))})
    return {'status':'ready_for_bounded_replay','frozen_source_files_verified':len(source['files']),
            'common_state_path':str(args.common_state),'common_state_SHA':STATE_SHA,
            'common_state_bytes':args.common_state.stat().st_size,'state':state_report,'Actor':actor_report,
            'final_Actor_SHA':ACTOR_SHA,
            'Actor_seed_field_present':bool('seed' in actor['metadata']),
            'inlet_prefix_path':str(args.inlet_prefix),'inlet_prefix_SHA':PREFIX_SHA,
            'CPU_original_inlet_summary_differences':checks,'controls_per_branch':8,'branches':['zero','rl_r1'],
            'physical_replay_executed':False,'native_JAX_used_in_CPU_preflight':False,
            'first_observation_policy':'consume saved observation/frames unchanged; no initialization or cache regeneration',
            'scientific_functions':['make_scaled_mole_training_case','build_flow_functions','load_actor_checkpoint',
                                    'validate_actor_metadata','build_wake_graph','make_step','make_scan'],
            'prospective_cost_limits':{'devices':1,'allocated_GPUh_upper_bound':0.25,'wall_seconds':900,'total_compile_seconds':600},
            'scope':'preflight readiness, not GPU availability/allocation or completed physical replay'}


def compare_trace(actual,reference):
    if set(actual)!=set(reference):raise ValueError('output field set differs')
    rows={}
    for k,a in actual.items():
        r=reference[k]
        if a.shape!=r.shape or a.dtype!=r.dtype or not np.isfinite(a).all():
            raise ValueError('output structure/finiteness differs: '+k)
        difference=np.abs(a.astype(np.float64)-r.astype(np.float64))
        rows[k]={'bitwise_equal':bool(a.tobytes(order='C')==r.tobytes(order='C')),
                 'numerically_equal':bool(np.array_equal(a,r)),
                 'max_abs_difference':float(difference.max(initial=0))}
    return {'all_fields_bitwise_equal':all(x['bitwise_equal'] for x in rows.values()),'fields':rows}


def worker(args):
    sys.dont_write_bytecode=True
    import jax
    import jax.numpy as jnp
    if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise RuntimeError('exactly one authorized GPU must be visible')
    sys.path.insert(0,str(args.pack_root/'source/codes'))
    from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case,build_flow_functions
    from jax_mole_continuous_sac import ContinuousControlConfig,load_actor_checkpoint
    from jax_mole_layouts import canonical_scale_layouts
    from jax_mole_graph_sac_v2 import build_wake_graph
    from mole_large_scale_evaluation import validate_actor_metadata,make_step,make_scan
    layout=canonical_scale_layouts()[81]
    case=make_scaled_mole_training_case(layout,1,compact_solver='dense_inverse',rhs_assembly='tensor')
    config=ContinuousControlConfig(inlet_mode='time_planes')
    actor,meta=load_actor_checkpoint(args.pack_root/'actors/actor.pkl',expected_case=case,expected_action_parameterization='per_turbine')
    validate_actor_metadata(meta,case,config,1000448)
    with args.common_state.open('rb') as f:common=pickle.load(f)
    validate_saved_state(common)
    # Directly carry every saved field into the original scan. Do not reconstruct observations.
    common=jax.tree.map(jnp.asarray,common)
    planes=jnp.asarray(np.load(args.inlet_prefix,allow_pickle=False)[None])
    graph=build_wake_graph(layout.turbine_positions_m,layout.diameter_m)
    farm=build_flow_functions(case,periodic_x=False,include_turbines=True)
    precursor={'case':case,'external_time_planes':True}
    records=[]
    total_compile_seconds=0.0
    for branch,parameters in [('zero',None),('rl_r1',actor)]:
        step=make_step(case,config,precursor,farm,actor=parameters,graph=graph if parameters is not None else None)
        scan=make_scan(step,8)
        save(args.output/'STATUS.json',{'phase':'compile','branch':branch,'phase_start_unix':time.time(),'completed_compile_seconds':total_compile_seconds})
        started=time.monotonic();compiled=scan.lower(common,planes,common.farm_flow.step_index).compile()
        compile_seconds=time.monotonic()-started;total_compile_seconds+=compile_seconds
        save(args.output/'STATUS.json',{'phase':'advance','branch':branch,'phase_start_unix':time.time(),'completed_compile_seconds':total_compile_seconds})
        started=time.monotonic();final,metrics=compiled(common,planes,common.farm_flow.step_index);jax.block_until_ready(metrics)
        runtime_seconds=time.monotonic()-started
        host=jax.tree.map(np.asarray,metrics)
        reference_path=args.reference_root/f'REFERENCE_B_W1_PREFIX8_{branch}.npz'
        with np.load(reference_path,allow_pickle=False) as ref:comparison=compare_trace(host,{k:ref[k] for k in ref.files})
        with (args.output/f'{branch}.npz').open('xb') as f:np.savez(f,**host)
        record={'branch':branch,'compile_seconds':compile_seconds,'advance_seconds':runtime_seconds,
                'final_farm_step_index':int(np.asarray(final.farm_flow.step_index)),
                'final_interaction_index':int(np.asarray(final.interaction_index)),
                'global_final_inlet_cursor_exclusive':27900,'comparison':comparison}
        records.append(record);save(args.output/'COMPARISON.json',{'branches':records})
        if record['final_farm_step_index']!=7900 or record['final_interaction_index']!=8:
            raise ValueError('actual final cursor differs')
        if not comparison['all_fields_bitwise_equal']:
            raise ValueError('original prefix output identity differs; stop before any next branch')
    save(args.output/'RESULT.json',{'status':'paired_original_prefix_bitwise_replay_pass','controls_per_branch':8,
        'branches':records,'common_state_sha256':STATE_SHA,'actor_sha256':ACTOR_SHA,'inlet_prefix_sha256':PREFIX_SHA,
        'layout_sha256':layout.payload_sha256,'device':[str(x) for x in jax.devices()],
        'jax_version':jax.__version__,'physical_replay_executed':True,'training_executed':False,
        'scope':'one original 80-second paired prefix, not full-window gain or cross-hardware reproducibility'})


def supervise(args):
    if not args.allocation_confirmed:raise ValueError('explicit confirmed allocation required; flag does not reserve a GPU')
    if args.output.exists():raise FileExistsError('refuse existing replay output')
    args.output.mkdir(parents=True)
    argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    started=time.time()
    with (args.output/'worker.log').open('xb') as log:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(args.output/'HANDLE.json',{'pid':process.pid,'argv':argv,'started_unix':started,
                                      'wall_seconds':890,'termination_reserve_seconds':10,
                                      'compile_seconds':600,'log':str(args.output/'worker.log')})
        reason=None
        while process.poll() is None:
            phase_path=args.output/'STATUS.json'
            phase=json.loads(phase_path.read_text()) if phase_path.exists() else {}
            if time.time()-started>890:reason='890-second worker bound plus termination reserve within 900-second target'
            if phase.get('phase')=='compile' and phase.get('completed_compile_seconds',0)+time.time()-phase['phase_start_unix']>600:
                reason='600-second total compile bound'
            if reason:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                break
            time.sleep(1)
        save(args.output/'EXIT.json',{'pid':process.pid,'exit_code':process.returncode,'elapsed_seconds':time.time()-started,
                                      'stopping_reason':reason,'allocated_GPUh':(time.time()-started)/3600})
    if process.returncode!=0:raise RuntimeError('bounded replay worker failed; inspect retained log and comparison')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ['pack-root','common-state','inlet-prefix','source-binding-receipt','reference-root']:
        p.add_argument('--'+option,type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--run',action='store_true');p.add_argument('--allocation-confirmed',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if not args.run or not args.allocation_confirmed or args.output is None:raise ValueError('supervised worker flags required')
        worker(args);return
    readiness=preflight(args)
    if args.run:
        if args.output is None:raise ValueError('new output directory required')
        supervise(args)
    else:print(json.dumps(readiness,indent=2,allow_nan=False))


if __name__=='__main__':main()
