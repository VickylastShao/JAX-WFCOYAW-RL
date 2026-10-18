#!/usr/bin/env python3
"""Audit and analyze the bounded low-workspace GPU campaign, using stdlib only.

The script never launches GPU work or imports JAX/NumPy. Missing evidence stays
missing, diagnostic failures stay false, and failed pilot costs are retained.
It analyzes only one main campaign and never pools H20 or CPU timing records.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIELDS=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def live_bytes(result):
    m=result.get('compiled_memory_analysis')
    if not isinstance(m,dict) or not all(isinstance(m.get(k),(int,float)) for k in FIELDS):
        return None
    return m[FIELDS[0]]+m[FIELDS[1]]+m[FIELDS[2]]-m[FIELDS[3]]


def timestamp(value):
    if isinstance(value,(int,float)):
        return float(value)
    if not isinstance(value,str):
        return None
    try:
        return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except ValueError:
        return None


def numeric(value):
    return isinstance(value,(float,int)) and math.isfinite(value)


def close(a,b):
    return numeric(a) and numeric(b) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-7)


def relative_error(predicted,observed):
    if not numeric(predicted) or not numeric(observed) or observed<=0:
        return None
    return abs(predicted/observed-1)


def branch_path(root,name):
    if not isinstance(name,str) or Path(name).name!=name:
        raise ValueError('Unexpected branch name')
    return root/name


def read_optional(path,manifest):
    if not path.exists():
        return None
    value=json.loads(path.read_text())
    manifest[str(path.resolve())]=dict(bytes=path.stat().st_size,sha256=sha(path),
                                      modification_unix=path.stat().st_mtime)
    return value


def export_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def hardware_identity(protocol):
    if not isinstance(protocol,dict):
        return None
    for key in ('gpu_identity','hardware_identity','hardware'):
        value=protocol.get(key)
        if value:
            return value
    return None


def normalized_identity(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')) if value is not None else None


def budget(campaigns,manifest):
    rows=[];checks={}; total=0.0; complete=True
    for directory in campaigns:
        ledger=read_optional(directory/'BRANCH_LEDGER.json',manifest)
        proto=read_optional(directory/'PROTOCOL.json',manifest)
        terminal=read_optional(directory/'EXIT.json',manifest)
        if ledger is None:
            rows.append(dict(campaign=directory.name,status='ledger_missing',
                             elapsed_gpu_seconds=None,elapsed_gpu_hours=None,
                             exit_status=(terminal or {}).get('status')))
            complete=False
            continue
        if not isinstance(ledger,list):
            raise TypeError('BRANCH_LEDGER must be a list: '+str(directory))
        elapsed=[]
        for record in ledger:
            seconds=record.get('elapsed_seconds')
            if not numeric(seconds) or seconds<0:
                raise ValueError('Invalid branch elapsed time')
            elapsed.append(seconds)
            rows.append(dict(campaign=directory.name,branch=record.get('name'),
                  exit_code=record.get('exit_code'),stopped_reason=record.get('stopped_reason'),
                  elapsed_gpu_seconds=seconds,elapsed_gpu_hours=seconds/3600,
                  status='failed_cost_retained' if record.get('exit_code')!=0 else 'completed_cost_retained',
                  exit_status=(terminal or {}).get('status')))
        subtotal=sum(elapsed); total+=subtotal
        checks[directory.name+'_terminal_cost_matches_ledger']=(
            close((terminal or {}).get('gpu_hours'),subtotal/3600) if terminal is not None else None)
        cap=proto.get('budget_gpu_seconds',proto.get('campaign_gpu_seconds_cap')) if proto else None
        checks[directory.name+'_recorded_budget_respected']=(subtotal<=cap if numeric(cap) else None)
        if terminal is None:
            complete=False
    # One CUDA-visible device is reserved by the supervisor per branch. This is
    # conservative allocated wall time, not integrated GPU utilization/energy.
    return dict(rows=rows,checks=checks,accounted_gpu_seconds=total,
         accounted_gpu_hours=total/3600,authorized_gpu_seconds=7200,
         combined_budget_respected=total<=7200,
         all_requested_campaigns_have_terminal_receipts=complete,
         cost_scope='one allocated CUDA device per child; elapsed includes setup, compile, execution and failed attempts; supervisor CPU-only setup outside children excluded')


def analyze(main,pilots,out,source_override=None,generation_path=None,cost_only=()):
    manifest={}; proto=read_optional(main/'PROTOCOL.json',manifest)
    terminal=read_optional(main/'EXIT.json',manifest)
    predictions=read_optional(main/'PROSPECTIVE_PREDICTIONS.json',manifest)
    ledger=read_optional(main/'BRANCH_LEDGER.json',manifest)
    if generation_path is None:
        generation_path=next((p for p in [main/'GENERATION_RESULT.json',main.parent/'GENERATION_RESULT.json',
            main.parent/'inputs/GENERATION_RESULT.json',main.parent/'protocol/GENERATION_RESULT.json'] if p.exists()),None)
    generation=read_optional(generation_path,manifest) if generation_path is not None else None
    account=budget([*pilots,main,*cost_only],manifest)
    if cost_only:
        original=budget([*pilots,main],manifest)
        account['original_resource_campaign_and_pilots_GPU_seconds']=original['accounted_gpu_seconds']
        account['cost_only_followup_GPU_seconds']=account['accounted_gpu_seconds']-original['accounted_gpu_seconds']
        account['cost_only_campaigns']=[str(p) for p in cost_only]
        account['cost_only_timings_excluded_from_resource_model']=True
    if proto is None:
        report=dict(schema='new_resource_campaign_analysis_v1',status='pending_main_protocol',
                    main_campaign=str(main.resolve()),input_manifest=manifest,budget=account,
                    no_results_invented=True)
        save(out/'NEW_CAMPAIGN_ANALYSIS.json',report)
        return report
    definitions=proto.get('branches',[])
    records={r['name']:r for r in (ledger or [])}
    results={}; result_paths={}; table=[]; numerical=[]
    for definition in definitions:
        name=definition[0] if isinstance(definition,(list,tuple)) else definition['name']
        path=branch_path(main,name)/'RESULT.json'
        value=read_optional(path,manifest)
        if value is None:
            continue
        results[name]=value;result_paths[name]=path
        memory=value.get('compiled_memory_analysis') or {}
        before=value.get('device_memory_stats_before_compile') or {}
        after=value.get('device_memory_stats_after_execution') or {}
        case=value.get('case') or {}
        times=value.get('run_seconds') or []
        record=records.get(name) or {}
        samples=[x.get('total_device_used_mib') for x in record.get('device_samples',[]) if numeric(x.get('total_device_used_mib'))]
        row=dict(branch=name,status=value.get('status'),mode=value.get('mode'),
                 group=value.get('group'),batch=case.get('batch'),
                 grid_x=case.get('nx'),grid_y=case.get('ny'),grid_z=case.get('nz'),
                 substeps=value.get('substeps'),compiled_live_bytes=live_bytes(value),
                 compiler_workspace_bytes=memory.get('temp_size_in_bytes'),
                 compiler_arguments_bytes=memory.get('argument_size_in_bytes'),
                 compiler_outputs_bytes=memory.get('output_size_in_bytes'),
                 compiler_alias_bytes=memory.get('alias_size_in_bytes'),
                 compiler_peak_memory_field_bytes=memory.get('peak_memory_in_bytes'),
                 allocator_peak_before_compile_bytes=before.get('peak_bytes_in_use'),
                 allocator_peak_after_execution_bytes=after.get('peak_bytes_in_use'),
                 desktop_sampled_total_device_peak_MiB=max(samples) if samples else None,
                 setup_seconds=value.get('setup_seconds'),compile_seconds=value.get('compile_seconds'),
                 first_execution_seconds=times[0] if times else None,
                 steady_median_seconds=value.get('steady_median_seconds'),
                 retained_execution_count=max(0,len(times)-1),
                 elapsed_allocation_seconds=record.get('elapsed_seconds'),
                 one_history_tiled_for_batch=value.get('history_replication') is not None,
                 device_record=';'.join(value.get('device') or []),
                 comparison_recorded=value.get('comparison') is not None)
        table.append(row)
        for field,details in (value.get('comparison') or {}).items():
            numerical.append(dict(branch=name,field=field,
              max_absolute_error=details.get('max_abs'),relative_L2_error=details.get('relative_l2'),
              finite=details.get('finite'),bitwise_equal=details.get('bitwise_equal'),
              allclose_rtol1e4_atol1e5=details.get('allclose_rtol1e-4_atol1e-5'),
              provenance='branch RESULT.json comparison statistics; raw NPZ difference not re-executed by stdlib audit'))
    checks={
      'main_exit_completed': terminal.get('status')=='completed' if terminal else None,
      'all_five_main_branch_results_present':len(results)==len(definitions)==5,
      'all_main_ledger_branches_success':all(r.get('exit_code')==0 for r in (ledger or [])) if ledger else None,
      'all_result_core_hashes_match_protocol':all(r.get('core_sha256')==proto.get('source_core_sha256') for r in results.values()) if results else None,
      'all_result_runner_hashes_match_protocol':all(r.get('runner_sha256')==proto.get('runner_sha256') for r in results.values()) if results else None,
      'all_recorded_results_finite':all(all(r.get('finite_outputs',{}).values()) and bool(r.get('finite_outputs')) for r in results.values()) if results else None,
      'all_main_substeps_match_protocol':all(r.get('substeps')==proto.get('substeps') for r in results.values()) if results else None,
      'single_hardware_identity_declared':hardware_identity(proto) is not None,
      'supervisor_snapshot_SHA_matches_protocol':(sha(main/'SUPERVISOR.py')==proto.get('supervisor_sha256')) if (main/'SUPERVISOR.py').exists() else None,
    }
    expected_generation_sha=next((proto[k] for k in ['input_generation_receipt_sha256','generation_metadata_sha256','generated_input_metadata_sha256','generation_result_sha256'] if proto.get(k)),None)
    checks['generation_metadata_SHA_matches_protocol']=(sha(generation_path)==expected_generation_sha) if generation_path is not None and expected_generation_sha is not None else None
    if generation is not None:
        seed=generation.get('seed',generation.get('initialization_seed'))
        checks['generated_initialization_seed_81731_recorded']=(seed==81731) if seed is not None else None
    source_root=source_override if source_override is not None else (Path(proto['source_dir']) if proto.get('source_dir') else None)
    frozen_sources=proto.get('frozen_source_files')
    if frozen_sources and source_root:
        source_results=[]
        for item in frozen_sources:
            relative=Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unexpected frozen source path')
            path=source_root/relative
            source_results.append(sha(path)==item['sha256'] if path.exists() else None)
        checks['all_frozen_source_SHA_currently_match']=(False if False in source_results else None if None in source_results else True)
        checks['frozen_source_file_count']=len(source_results)
        checks['frozen_source_files_available_here']=sum(v is not None for v in source_results)
    else:
        checks['all_frozen_source_SHA_currently_match']=None
    for name,result in results.items():
        times=result.get('run_seconds') or []
        checks[name+'_steady_median_recomputed']=close(statistics.median(times[1:] or times),result.get('steady_median_seconds')) if times else False
        recorded=(result.get('inputs') or {})
        expected_state=proto.get('common_state_gzip_sha256')
        expected_inlet=proto.get('original_inlet_prefix_sha256')
        checks[name+'_physical_start_hash_matches_protocol']=((recorded.get('common_state') or {}).get('sha256')==expected_state) if expected_state is not None else None
        checks[name+'_inlet_hash_matches_protocol']=((recorded.get('inlet_prefix') or {}).get('sha256')==expected_inlet) if expected_inlet is not None else None
        if result.get('mode')=='stream':
            checks[name+'_candidate_hash_matches_protocol']=(result.get('provenance') or {}).get('candidate_module_sha256')==proto.get('candidate_sha256')
    prediction_checks={}; heldouts=[]; recomputed_model=None
    if predictions is not None:
        contract=proto.get('prospective_model_contract') or {}
        targets=predictions.get('diagnostic_targets') or {}
        prediction_checks['memory_target_matches_frozen_protocol']=close(targets.get('memory_relative_error'),contract.get('memory_error_target'))
        prediction_checks['time_target_matches_frozen_protocol']=close(targets.get('time_relative_error'),contract.get('time_error_target'))
        for group in (1,9):
            name=f'stream_g{group}_b1';cal=(predictions.get('calibration') or {}).get(str(group)) or {}
            prediction_checks[name+'_calibration_file_SHA_matches']=(sha(result_paths[name])==cal.get('sha256')) if name in result_paths else None
            prediction_checks[name+'_calibration_live_bytes_match']=(close(live_bytes(results[name]),cal.get('compiled_live_bytes'))) if name in results else None
            prediction_checks[name+'_calibration_time_matches']=(close(results[name].get('steady_median_seconds'),cal.get('steady_seconds'))) if name in results else None
        created=timestamp(predictions.get('created_utc'))
        first_start=min([records[n]['started_unix'] for n in ['stream_g3_b1','stream_g3_b2'] if n in records],default=None)
        prediction_checks['declared_prediction_creation_precedes_any_g3_start']=(created<first_start) if created is not None and first_start is not None else None
        prediction_checks['prediction_file_mtime_precedes_any_g3_start']=(main/'PROSPECTIVE_PREDICTIONS.json').stat().st_mtime<first_start if first_start is not None else None
        prediction_checks['heldout_not_started_flag_recorded']=predictions.get('heldout_branch_not_started') is True
        for group in (1,9):
            record=records.get(f'stream_g{group}_b1')
            prediction_checks[f'g{group}_completed_before_prediction_creation']=(record['started_unix']+record['elapsed_seconds']<=created) if record and created is not None else None
        if all(f'stream_g{g}_b1' in results for g in (1,9)):
            c1,c9=(results[f'stream_g{g}_b1'] for g in (1,9))
            m1,m9=live_bytes(c1),live_bytes(c9)
            t1,t9=c1['steady_median_seconds'],c9['steady_median_seconds']
            B=(m9-m1)/8; A=m1-B; D=(t1-t9)/72; C=t1-D*81
            recomputed_model=dict(memory_A=A,memory_B=B,time_C=C,time_D=D,
                                  g3b1_live_bytes=A+3*B,g3b1_seconds=C+27*D,
                                  g3b2_live_bytes=2*(A+3*B))
            prediction_checks['saved_memory_parameters_match_calibration']=close(A,predictions['memory_parameters']['A']) and close(B,predictions['memory_parameters']['B'])
            prediction_checks['saved_time_parameters_match_calibration']=close(C,predictions['time_parameters']['C']) and close(D,predictions['time_parameters']['D'])
            for batch in (1,2):
                name=f'stream_g3_b{batch}';saved=predictions.get('predicted_'+name) or {}
                if name not in results:
                    heldouts.append(dict(branch=name,status='result_missing',memory_target_pass=None,time_target_pass=None));continue
                value=results[name]; observed_m=live_bytes(value); predicted_m=recomputed_model[f'g3b{batch}_live_bytes']
                mem_error=relative_error(predicted_m,observed_m)
                memory_limit=(predictions.get('diagnostic_targets') or {}).get('memory_relative_error')
                row=dict(branch=name,status='evaluated',memory_prediction=predicted_m,
                    memory_observation=observed_m,memory_absolute_relative_error=mem_error,
                    memory_diagnostic_limit=memory_limit,
                    memory_target_pass=(mem_error<=memory_limit) if mem_error is not None and numeric(memory_limit) else None)
                prediction_checks[name+'_saved_memory_prediction_recomputed']=close(predicted_m,saved.get('compiled_live_bytes'))
                if batch==1:
                    pred_t=recomputed_model['g3b1_seconds'];obs_t=value.get('steady_median_seconds');err_t=relative_error(pred_t,obs_t)
                    limit_t=(predictions.get('diagnostic_targets') or {}).get('time_relative_error')
                    row.update(time_prediction=pred_t,time_observation=obs_t,time_absolute_relative_error=err_t,
                        time_diagnostic_limit=limit_t,time_target_pass=(err_t<=limit_t) if err_t is not None and numeric(limit_t) else None)
                    prediction_checks[name+'_saved_time_prediction_recomputed']=close(pred_t,saved.get('steady_seconds'))
                else:
                    row.update(time_prediction=None,time_observation=value.get('steady_median_seconds'),
                               time_absolute_relative_error=None,time_target_pass=None,
                               time_prediction_scope='No b2 time prediction was preregistered')
                heldouts.append(row)
    else:
        prediction_checks['prospective_prediction_file_present']=False

    numerical_summary={}
    for field in ('state_velocity','power_trace','final_divergence'):
        rows=[r for r in numerical if r['field']==field]
        numerical_summary[field]=dict(
            evaluated_branches=[r['branch'] for r in rows],
            maximum_absolute_error=max([r['max_absolute_error'] for r in rows if numeric(r['max_absolute_error'])],default=None),
            maximum_relative_L2_error=max([r['relative_L2_error'] for r in rows if numeric(r['relative_L2_error'])],default=None))
    velocity=numerical_summary['state_velocity']['maximum_relative_L2_error'];power=numerical_summary['power_trace']['maximum_relative_L2_error']
    numerical_summary['velocity_gate_pass']=velocity<1e-4 if velocity is not None else None
    numerical_summary['power_gate_pass']=power<1e-3 if power is not None else None
    numerical_summary['b2_physical_comparison_status']=('compared_to_tiled_dense_b1_reference' if results.get('stream_g3_b2',{}).get('comparison') else 'not_compared_or_not_available')
    numerical_summary['near_zero_residual_relative_error_is_not_a_gate']=True
    dense=next((r for r in table if r['mode']=='dense' and r['batch']==1),None)
    pairs=[]
    if dense:
        for r in table:
            if r['mode']=='stream' and r['batch']==1:
                pairs.append(dict(branch=r['branch'],baseline=dense['branch'],
                 same_batch=True,compiled_live_reduction_percent=100*(1-r['compiled_live_bytes']/dense['compiled_live_bytes']),
                 compiler_workspace_reduction_percent=100*(1-r['compiler_workspace_bytes']/dense['compiler_workspace_bytes']),
                 steady_runtime_ratio_vs_dense=r['steady_median_seconds']/dense['steady_median_seconds'],
                 steady_time_difference_seconds=r['steady_median_seconds']-dense['steady_median_seconds'],
                 evidence_scope='same frozen campaign hardware, 50-substep full LES operation; not H20 training or policy-performance comparison'))
    selector=select_candidates(proto,table,results)
    complete=bool(terminal) and terminal.get('status')=='completed' and len(results)==len(definitions)
    analysis=dict(schema='new_resource_campaign_analysis_v1',status='complete_with_recorded_diagnostics' if complete else 'partial_or_pending',
         main_campaign=str(main.resolve()),hardware_identity=hardware_identity(proto),
         physical_input_scope=proto.get('physical_input_contract',proto.get('initialization',proto.get('scope'))),
         generated_input_metadata=generation,
         protocol_checks=checks,prediction_checks=prediction_checks,
         compiled_live_definition='compiled buffer budget = argument + output + temporary - alias; historical JSON field compiled_live_bytes; neither compiler peak_memory field nor total device allocator peak',
         resource_table=table,dense_vs_stream=pairs,heldout_diagnostics=heldouts,
         recomputed_calibration_model=recomputed_model,numerical_summary=numerical_summary,
         numerical_by_field=numerical,finite_candidate_selector=selector,budget=account,
         input_manifest=manifest,
         evidence_boundaries=[
           'N81 full-grid execution diagnostic and prescribed yaw; physical input origin is the recorded campaign protocol, never inferred from historical B_w1 records. No new policy, wind regime, training seed or long-window gain test.',
           'All repetitions begin at the identical physical state. Retained passes are correlated timings, not independent scientific replicates.',
           'The b2 history is tiled from one history, not independent physical replication.',
           'Low-workspace summation order differs; absence of bitwise equality is reported, not hidden.',
           'Numerical difference statistics here are re-read from RESULT.json. Full output NPZ comparisons are performed by branch runner, not independently recomputed by this standard-library audit.',
           'Compiler live bytes, allocator peak and desktop-wide sampled memory are different quantities and are never substituted.',
           'Failed pilot allocation is retained. No GPU timing is pooled with old H20 data or CPU mini-grid tests.',
           'False heldout targets remain false and restrict the resource-model claim; script completion does not imply model validation.',
           'Source and calibration SHA establish file identity; timestamps establish the recorded same-host chronology, not an independent external timestamp attestation.',
         ])
    save(out/'NEW_CAMPAIGN_ANALYSIS.json',analysis)
    export_csv(out/'NEW_CAMPAIGN_RESOURCE_TABLE.csv',table)
    export_csv(out/'NEW_CAMPAIGN_DENSE_STREAM.csv',pairs)
    export_csv(out/'NEW_CAMPAIGN_HELDOUT.csv',heldouts)
    export_csv(out/'NEW_CAMPAIGN_NUMERICAL.csv',numerical)
    export_csv(out/'NEW_CAMPAIGN_BUDGET.csv',account['rows'])
    write_text(out,analysis)
    return analysis


def select_candidates(protocol,table,results):
    scenarios=protocol.get('selector_scenarios')
    if not scenarios:
        return dict(status='pending_predeclared_scenarios',decisions=[],
                    reason='No capacity/reserve/latency scenarios frozen in PROTOCOL.json; no scenarios invented after observing results.')
    identity=hardware_identity(protocol)
    if isinstance(scenarios,dict):
        # Expand only the capacity/reserve/latency set frozen before any run.
        # These are illustrative component-job budgets, not turbine deadlines.
        scenarios=[dict(capacity_bytes=cap*(2**30),reserve_bytes=scenarios['reserve_gib']*(2**30),
                        latency_seconds=latency,batch=1,memory_metric='compiled_live_bytes',
                        capacity_gib=cap,reserve_gib=scenarios['reserve_gib'],
                        diagnostic_latency_seconds=latency,scope=scenarios.get('scope'))
                   for cap in scenarios['capacity_gib'] for latency in scenarios['diagnostic_latency_seconds']]
    decisions=[]
    for scenario in scenarios:
        requested=scenario.get('hardware_identity',identity)
        metric=scenario.get('memory_metric','allocator_peak_after_execution_bytes')
        if metric not in ('allocator_peak_after_execution_bytes','compiled_live_bytes'):
            raise ValueError('Unsupported selector memory metric')
        capacity=scenario.get('capacity_bytes');reserve=scenario.get('reserve_bytes');latency=scenario.get('latency_seconds');batch=scenario.get('batch',1)
        valid=all(numeric(x) and x>=0 for x in (capacity,reserve,latency)) and capacity>reserve
        rejected=[];feasible=[]
        for row in table:
            reasons=[]
            if identity is None or normalized_identity(requested)!=normalized_identity(identity):reasons.append('hardware_identity_missing_or_mismatch')
            if not valid:reasons.append('invalid_predeclared_scenario')
            if row['batch']!=batch:reasons.append('different_batch_work')
            if row['status']!='completed':reasons.append('incomplete')
            memory=row.get(metric)
            if not numeric(memory):reasons.append('selected_memory_metric_missing')
            elif valid and memory+reserve>capacity:reasons.append('capacity_plus_reserve')
            seconds=row.get('steady_median_seconds')
            if not numeric(seconds):reasons.append('latency_missing')
            elif valid and seconds>latency:reasons.append('latency')
            result=results[row['branch']]
            compare=result.get('comparison')
            if row['mode']=='stream' and row['batch']==1:
                if not compare:reasons.append('numerical_comparison_missing')
                elif not(compare['state_velocity']['relative_l2']<1e-4 and compare['power_trace']['relative_l2']<1e-3):reasons.append('numerical_gate')
            if reasons:rejected.append(dict(branch=row['branch'],reasons=reasons))
            else:feasible.append(row)
        winner=min(feasible,key=lambda row:row['steady_median_seconds']) if feasible else None
        decisions.append(dict(scenario=scenario,hardware_identity_verified=identity is not None,
            selected_branch=winner['branch'] if winner else None,
            selected_latency_seconds=winner['steady_median_seconds'] if winner else None,
            feasible_branches=[r['branch'] for r in feasible],rejected=rejected,
            memory_metric=metric,
            guarantee_scope='measured finite candidates only; compiler-live choice is compiled-program eligibility, not total device capacity guarantee'))
    return dict(status='evaluated_predeclared_scenarios',decisions=decisions,
       same_hardware_and_same_batch_only=True,
       objective='fastest measured steady execution satisfying predeclared memory+reserve and latency; no interpolation or old hardware pooling')


def write_text(out,data):
    lines=['# 新增低工作区GPU诊断分析','',f"状态：{data['status']}。诊断指标的true/false按原先目标保存，分析脚本完成不代表所有模型预测均通过。",'',
       '## 数据口径','',data['compiled_live_definition'],'',
       f"仅分析本次记录硬件（{data['hardware_identity']}）上的完整81机网格短操作；不与H20旧计时或CPU小网格结果混合。",'',
       '| 分支 | b | 编译buffer budget/GB | 编译工作区/GB | 分配器峰值/GB | 稳态/s |','|---|---|---|---|---|---|']
    for r in data['resource_table']:
        peak=r['allocator_peak_after_execution_bytes'];peak='未提供' if peak is None else f'{peak/1e9:.5f}'
        lines.append(f"| {r['branch']} | {r['batch']} | {r['compiled_live_bytes']/1e9:.5f} | {r['compiler_workspace_bytes']/1e9:.5f} | {peak} | {r['steady_median_seconds']:.6f} |")
    lines+=['','## 前瞻校准与留出','',
      'g=1/9用于校准；g=3,b=1用于内存及时间留出，g=3,b=2仅有内存预报。核对校准文件SHA、预测公式与参数、预测时间先于g3开始；文件mtime单独记录，不把缺失当通过。','']
    for name,value in data['prediction_checks'].items():lines.append(f'- {name}: {value}')
    for r in data['heldout_diagnostics']:
        if r['status']=='evaluated':
            lines.append(f"- {r['branch']}：内存绝对相对误差{100*r['memory_absolute_relative_error']:.4f}%，目标通过={r['memory_target_pass']}；时间目标通过={r.get('time_target_pass')}。")
        else:lines.append(f"- {r['branch']}：尚无结果。")
    lines+=['','## 数值一致性','',json.dumps(data['numerical_summary'],ensure_ascii=False,indent=2),'',
      '近零散度的相对误差不作为门槛，须同时报告绝对差。b2重复同一物理历史，仅验证执行批处理；若没有dense b2对照，明确列为未比较。','',
      '## 成本','',f"已计入分支分配时间{data['budget']['accounted_gpu_seconds']:.3f}秒，即{data['budget']['accounted_gpu_hours']:.6f} GPUh。包含失败pilot，是否在7200 GPU秒内：{data['budget']['combined_budget_respected']}。所有要求的terminal收据齐全：{data['budget']['all_requested_campaigns_have_terminal_receipts']}。",'',
      '## 有限配置选择','',json.dumps(data['finite_candidate_selector'],ensure_ascii=False,indent=2),'',
      '只从同一硬件、同一批量工作量的实际候选选择。编译live并不包含所有运行时常量/桌面分配，若使用该指标作容量判断，只能声称编译程序范围内的候选资格。','',
      '未重新运行NPZ差值、训练、策略收益、跨网格或跨硬件实验。']
    (out/'NEW_CAMPAIGN_ANALYSIS.zh-CN.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,
                        help='Portable extension containing gpu_main and gpu_pilot* siblings')
    parser.add_argument('--main',type=Path,default=None,
                        help='Default: ROOT/gpu_main; timing candidates are only taken from this directory')
    parser.add_argument('--pilot',type=Path,action='append',default=None,
                        help='Repeat for every attempt, including failures. Default: all sibling gpu_pilot* directories.')
    parser.add_argument('--pilots',type=Path,nargs='+',default=None,
                        help='Alternative explicit list of all pilot attempt directories; cannot combine with --pilot')
    parser.add_argument('--source-dir',type=Path,default=None,
                        help='Optional local relocated copy of the frozen source files; hashes must match PROTOCOL')
    parser.add_argument('--generation-metadata',type=Path,default=None,
                        help='Relocated GENERATION_RESULT.json; validate against frozen input_generation_receipt_sha256 / generation_metadata_sha256')
    parser.add_argument('--cost-only',type=Path,action='append',default=[],
                        help='Additional follow-up ledger charged to total budget; never included in timing/model tables')
    parser.add_argument('--output',type=Path,default=HERE/'new_campaign_analysis')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.pilot is not None and args.pilots is not None:parser.error('Use --pilot or --pilots, not both')
    main_directory=args.main if args.main is not None else args.root/'gpu_main'
    pilots=args.pilot if args.pilot is not None else args.pilots
    if pilots is None:pilots=sorted(p for p in main_directory.parent.glob('gpu_pilot*') if p.is_dir())
    report=analyze(main_directory,pilots,args.output,args.source_dir,args.generation_metadata,args.cost_only)
    print(json.dumps(dict(status=report['status'],output=str(args.output.resolve()),
      budget_gpu_hours=report['budget']['accounted_gpu_hours'],
      new_results_invented=False),ensure_ascii=True))


if __name__=='__main__':
    main()
