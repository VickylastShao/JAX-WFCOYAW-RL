#!/usr/bin/env python3
"""Read-only, standard-library resource analysis of frozen N81 evidence.

No simulator or ML libraries are imported. Existing configurations are a
retrospective backtest, never a prospective unseen experiment. Input artifacts
are not edited. The output directory must be this script's directory or a new
directory supplied explicitly by the caller.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


DEFAULT_ROOT = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/wind_energy_navigation_closure_20261016')
DEFAULT_REPO = Path.cwd()
PACK_NAME = 'n81_20261016_navigation_closure_repro'


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    assert rows, path
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def describe(values):
    assert values and all(finite(v) for v in values)
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), mean=statistics.mean(values), total=sum(values))


def inverse_affine(p1, y1, p2, y2):
    assert p1 > 0 and p2 > 0 and p1 != p2
    slope = (y1 - y2) / (1 / p1 - 1 / p2)
    intercept = y1 - slope / p1
    return dict(inverse_coefficient=slope, intercept=intercept)


def affine(x1, y1, x2, y2):
    assert x1 != x2
    slope = (y2 - y1) / (x2 - x1)
    return dict(slope=slope, intercept=y1 - slope * x1)


def prediction_record(name, predicted, observed, calibration, evaluation, unit, boundary):
    assert observed > 0 and all(finite(x) for x in [predicted, observed])
    return dict(name=name, prediction=predicted, observation=observed,
                signed_error=predicted-observed,
                signed_relative_error_percent=100 * (predicted / observed - 1),
                absolute_relative_error_percent=100 * abs(predicted / observed - 1),
                calibration=calibration, evaluation=evaluation, unit=unit,
                evaluation_status='retrospective_backtest_not_prospective_validation',
                boundary=boundary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--repo-root', type=Path, default=DEFAULT_REPO)
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pack = args.submission_root / 'repro-pack' / PACK_NAME
    sources = {}

    def load(label, path):
        data = json.loads(path.read_text())
        sources[label] = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                              sha256=sha(path))
        return data

    e1 = load('microbatch_component', pack/'payload/wind_energy_closure/results/E1_COMPARISON.json')
    scale = load('matched_work_summary', pack/'payload/visual_revision/data/timing/SCALING_SUMMARY.json')
    previous = load('cost_receipt_summary', pack/'payload/visual_revision/data/reference_mechanism_summary.json')
    proto = load('matched_work_protocol', args.repo_root/'audit/paper1_n81_evidence_closure_20261013/existing/efficiency/efficiency/PROFILE.json')
    formal = [load('formal_rank%d' % rank, args.repo_root/('audit/paper1_n81_evidence_closure_20261013/existing/source/runs/n81_train_r1/rank%d.json' % rank)) for rank in range(4)]
    pack_formal = load('portable_formal_rank0', pack/'payload/visual_revision/source/provenance/formal_rank0.json')
    checks = dict(
        formal_rank0_matches_current_package=sources['formal_rank0']['sha256'] == sources['portable_formal_rank0']['sha256'],
        microbatch_complete_and_full_state_equivalence=e1['status']=='complete' and e1['all_equivalent'] is True,
        matched_work_profile_complete=scale['status']=='matched_profiles_complete',
        matched_work_protocol_hash=scale['protocol_sha256'] == sources['matched_work_protocol']['sha256'],
        formal_four_ranks_complete=all(x['status']=='pass' and x['declared_budget_completed'] for x in formal),
        formal_four_ranks_all_recorded_checks=all(all(x['checks'].values()) for x in formal),
        formal_four_ranks_same_iteration_count=all(len(x['history'])==3907 for x in formal),
        formal_interaction_count=all(x['declared_global_interactions']==1000448 for x in formal),
        profile_global_contract=proto['global_environments']==32 and proto['microbatch']==1 and proto['global_sample_size']==256 and proto['global_collected_interactions']==3072,
    )
    if not all(checks.values()):
        raise ValueError('Source contract checks failed: '+repr(checks))

    micro = []
    micro_timing = []
    grid = formal[0]['configuration']['layout_contract']['grid']
    cells = math.prod(grid)
    dense_pair_bytes_per_b = 2 * 81 * cells * 4
    for b in (1, 2, 4):
        q = e1['reports'][str(b)]
        assert q['microbatch']==b and q['persistent_contexts']==8
        assert q['substeps_per_context']==200 and q['retained_substeps_per_context']==150
        assert len(q['retained_pass_seconds'])==3
        m=q['compiled_kernel_memory']; peak=q['device_memory']['peak_bytes_in_use']
        micro.append(dict(microbatch=b, persistent_contexts=8,
                          persistent_state_bytes=q['persistent_state_bytes'],
                          persistent_state_bytes_per_context=q['persistent_state_bytes']/8,
                          compiled_argument_bytes=m['argument_size_in_bytes'],
                          compiled_output_bytes=m['output_size_in_bytes'],
                          compiled_alias_bytes=m['alias_size_in_bytes'],
                          compiled_workspace_bytes=m['temp_size_in_bytes'],
                          measured_active_component_peak_bytes=peak,
                          measured_active_component_peak_GB=peak/1e9,
                          analytical_two_dense_tensors_bytes=dense_pair_bytes_per_b*b,
                          workspace_minus_two_dense_tensors_bytes=m['temp_size_in_bytes']-dense_pair_bytes_per_b*b,
                          compile_seconds=q['compile_seconds'],
                          warm_pass_seconds=q['warm_pass_seconds'],
                          retained_pass_median_seconds=statistics.median(q['retained_pass_seconds']),
                          retained_pass_min_seconds=min(q['retained_pass_seconds']),
                          retained_pass_max_seconds=max(q['retained_pass_seconds']),
                          scope='LES component; fixed 8 real histories; learner and Replay excluded'))
        for index, seconds in enumerate(q['retained_pass_seconds'], 1):
            micro_timing.append(dict(microbatch=b, sequential_repeat=index, seconds=seconds,
                                     contexts_per_pass=8, substeps_per_context_per_pass=50,
                                     independent_job_replicate=False))
    write_csv(out/'microbatch_component.csv', micro)
    write_csv(out/'microbatch_retained_timings.csv', micro_timing)

    topologies=[]; blocks=[]
    for p in (1,2,4):
        q=scale['topologies'][str(p)]
        times=q['normal_block_critical_rank_seconds']
        assert len(times)==10 and statistics.median(times)==q['normal_block_seconds_median']
        total=q['total_including_cold_reset_learner_sync_and_checkpoint_seconds']
        allocated=previous['costs']['efficiency_allocated_gpu_hours_by_topology'][str(p)]
        topologies.append(dict(
            devices=p, contexts_per_device=32//p, global_contexts=32, microbatch=1,
            interactions=3072, learner_updates=3072, global_minibatch=256,
            ordinary_median_seconds=q['normal_block_seconds_median'],
            ordinary_min_seconds=min(times), ordinary_max_seconds=max(times),
            collector_median_seconds=q['normal_collector_seconds_median'],
            learner_and_communication_median_seconds=q['normal_learner_including_communication_seconds_median'],
            ordinary_collector_fraction_from_component_medians=q['normal_collector_seconds_median']/q['normal_block_seconds_median'],
            reset_block_seconds=q['reset_block_seconds'], reset_fraction_of_profile_total=q['reset_block_seconds']/total,
            profile_total_seconds=total, profile_total_minutes=total/60,
            rank_timer_gpu_hours=q['rank_accounted_gpu_hours'],
            critical_path_times_devices_gpu_hours=p*total/3600,
            supervisor_allocated_gpu_hours=allocated,
            ordinary_speedup=q['regular_block_speedup'], profile_total_speedup=q['bounded_profile_total_speedup'],
            measured_full_profile_allocator_peak_bytes=q['peak_allocator_bytes'],
            checkpoint_host_conversion_critical_rank_seconds=q['checkpoint_host_conversion_seconds_critical_rank'],
            full_checkpoint_io_critical_rank_seconds=q['checkpoint_io_seconds_critical_rank'],
            final_checkpoint_barrier_critical_rank_seconds=q['final_checkpoint_barrier_seconds_critical_rank'],
            scope='bounded fixed-action matched-work profile; not scientific training time-to-quality'))
        for index,seconds in enumerate(times,1):
            blocks.append(dict(devices=p, recorded_ordinary_block=index, seconds=seconds,
                               independent_run=False, reset_included=False))
    write_csv(out/'matched_work_topologies.csv',topologies)
    write_csv(out/'matched_work_ordinary_blocks.csv',blocks)

    rank_summary=[]; formal_blocks=[]
    for rank,x in enumerate(formal):
        assert [r['iteration'] for r in x['history']]==list(range(1,3908))
        rows=x['history']; cfg=x['configuration']; timing=x['timing']
        assert cfg['local_environments']==8 and cfg['microbatch']==1 and cfg['collection_steps_per_environment']==8
        keys=['wallclock_seconds','collector_seconds','farm_collection_seconds','learner_seconds','precursor_generation_seconds']
        assert all(all(finite(r[k]) and r[k]>=0 for k in keys) for r in rows)
        rank_summary.append(dict(rank=rank, iterations=len(rows),
             total_process_wallclock_seconds=timing['total_process_wallclock_seconds'],
             history_block_seconds_sum=sum(r['wallclock_seconds'] for r in rows),
             history_collector_seconds_sum=sum(r['collector_seconds'] for r in rows),
             history_farm_seconds_sum=sum(r['farm_collection_seconds'] for r in rows),
             history_learner_seconds_sum=sum(r['learner_seconds'] for r in rows),
             history_precursor_read_generation_seconds_sum=sum(r['precursor_generation_seconds'] for r in rows),
             total_minus_block_seconds=timing['total_process_wallclock_seconds']-sum(r['wallclock_seconds'] for r in rows),
             initial_burnin_seconds=timing['zero_yaw_burnin_seconds'],
             compile_warm_actor_seconds=timing['compile_and_warm_actor_seconds'],
             compile_first_learner_seconds=timing['compile_and_first_learner_seconds'],
             checkpoint_records=len(x['rolling_checkpoint_records']),
             rolling_checkpoint_io_seconds_sum=sum(r['checkpoint_io_seconds'] for r in x['rolling_checkpoint_records']),
             real_reset_blocks=sum(bool(r['physical_reset_interactions_per_environment']) for r in rows)))
    # Keep rank0 components additive. Per-iteration max across ranks is a separate
    # conservative synchronization diagnostic, not an exact integrated makespan.
    for index,r in enumerate(formal[0]['history']):
        reset=bool(r['physical_reset_interactions_per_environment'])
        active=r['replay_size']>=32000
        phase=('learned_reset' if reset else 'learned_ordinary') if active else ('warmup_reset' if reset else 'warmup_ordinary')
        formal_blocks.append(dict(iteration=r['iteration'], phase=phase,
             real_reset=reset, replay_learning_active=active,
             rank0_wallclock_seconds=r['wallclock_seconds'],
             rank0_collector_seconds=r['collector_seconds'],
             rank0_farm_seconds=r['farm_collection_seconds'],
             rank0_learner_seconds=r['learner_seconds'],
             rank0_precursor_read_seconds=r['precursor_generation_seconds'],
             rank0_other_seconds=r['wallclock_seconds']-r['collector_seconds']-r['learner_seconds'],
             conservative_iteration_max_rank_seconds=max(x['history'][index]['wallclock_seconds'] for x in formal),
             reset_controls_per_environment=r['physical_reset_interactions_per_environment'],
             checkpoint_record_on_global_interactions=any(q['global_interactions']==r['iteration']*256 for q in formal[0]['rolling_checkpoint_records'])))
    categories={}
    for phase in sorted({r['phase'] for r in formal_blocks}):
        rows=[r for r in formal_blocks if r['phase']==phase]
        categories[phase]={k:describe([r[k] for r in rows]) for k in ['rank0_wallclock_seconds','rank0_collector_seconds','rank0_farm_seconds','rank0_learner_seconds','rank0_other_seconds']}
    write_csv(out/'formal_rank_summary.csv',rank_summary)
    write_csv(out/'formal_blocks.csv',formal_blocks)

    # Existing-data leave-configuration-out checks. Because all data already
    # existed and were available during model choice, they are retrospective.
    backtests=[]; models={}
    for field in ['ordinary_median_seconds','collector_median_seconds','learner_and_communication_median_seconds','reset_block_seconds','profile_total_seconds','measured_full_profile_allocator_peak_bytes']:
        a,b,c=topologies
        model=inverse_affine(1,a[field],2,b[field]);models['profile_'+field]=model
        pred=model['inverse_coefficient']/4+model['intercept']
        backtests.append(prediction_record('P4_'+field,pred,c[field], 'P=1,2 existing fixed-work profile','P=4 existing profile', 'bytes' if field.endswith('bytes') else 'seconds', 'same H20, Nt=81, Ng=9822288, Eglobal=32, b=1, fixed small-Replay work'))
    for field in ['compiled_workspace_bytes','measured_active_component_peak_bytes']:
        a,b,c=micro
        model=affine(1,a[field],4,c[field]);models['component_'+field]=model
        pred=model['slope']*2+model['intercept']
        backtests.append(prediction_record('b2_'+field,pred,b[field],'b=1,4 existing component','b=2 existing component','bytes','fixed 8 contexts, no learner or Replay; only interpolation at fixed N/grid'))

    # Contiguous chronological split of a single training run tests temporal
    # stability only, not unseen hardware, configuration, seed, or physics.
    split=1953
    fit=[r for r in formal_blocks if r['iteration']<=split]
    later=[r for r in formal_blocks if r['iteration']>split]
    phase_models={p:statistics.median(r['rank0_wallclock_seconds'] for r in fit if r['phase']==p)
                  for p in sorted({r['phase'] for r in fit})}
    for phase in ['learned_ordinary','learned_reset']:
        target=[r for r in later if r['phase']==phase]
        observed=sum(r['rank0_wallclock_seconds'] for r in target)
        predicted=phase_models[phase]*len(target)
        backtests.append(prediction_record('formal_later_'+phase,predicted,observed,
             'formal rank0 iterations 1..1953, phase median', 'formal rank0 iterations 1954..3907 same run',
             'seconds','chronological stability only; phase count observed, not forecast; checkpoints outside block excluded'))
    models['formal_rank0_first_half_phase_medians_seconds']=phase_models
    write_csv(out/'model_backtests.csv',backtests)

    ordinary=[r for r in formal_blocks if r['phase']=='learned_ordinary']
    resets=[r for r in formal_blocks if r['real_reset']]
    known_peak_reduction=100*(1-micro[0]['measured_active_component_peak_bytes']/micro[2]['measured_active_component_peak_bytes'])
    process_wall=max(r['total_process_wallclock_seconds'] for r in rank_summary)
    allocation_increase=100*(topologies[2]['supervisor_allocated_gpu_hours']/topologies[0]['supervisor_allocated_gpu_hours']-1)
    derived=dict(grid=grid, grid_cells=cells, dense_pair_bytes_per_active_member=dense_pair_bytes_per_b,
                 component_peak_reduction_b4_to_b1_percent=known_peak_reduction,
                 component_peak_is_not_full_training_peak=True,
                 formal_training_critical_rank_hours=process_wall/3600,
                 formal_training_sum_rank_timer_gpu_hours=sum(x['timing']['total_process_wallclock_seconds'] for x in formal)/3600,
                 formal_rank0_history_block_fraction_of_process_wall=sum(r['rank0_wallclock_seconds'] for r in formal_blocks)/process_wall,
                 formal_real_reset_blocks=len(resets),
                 formal_reset_block_fraction=len(resets)/len(formal_blocks),
                 formal_reset_block_time_fraction_of_process_wall=sum(r['rank0_wallclock_seconds'] for r in resets)/process_wall,
                 formal_learned_ordinary_rank0_median_seconds=statistics.median(r['rank0_wallclock_seconds'] for r in ordinary),
                 formal_learned_ordinary_collector_fraction_from_component_medians=statistics.median(r['rank0_collector_seconds'] for r in ordinary)/statistics.median(r['rank0_wallclock_seconds'] for r in ordinary),
                 formal_learned_ordinary_learner_fraction_from_component_medians=statistics.median(r['rank0_learner_seconds'] for r in ordinary)/statistics.median(r['rank0_wallclock_seconds'] for r in ordinary),
                 profile_P4_supervisor_GPUh_increase_vs_P1_percent=allocation_increase)
    boundary=[
      'All existing-data model checks are retrospective, even when a configuration is omitted from fitting.',
      'Single hardware family and fixed Nt=81/grid: no cross-grid, cross-hardware or cross-turbine generalization measured here.',
      'E1 timing has three sequential passes per configuration, not independent job replicates.',
      'Matched-work ordinary blocks are correlated serial timings; no iid significance test or confidence interval.',
      'Formal four ranks form one synchronized optimization seed, not four independent training replicates.',
      'Profile learner plus communication is inclusive; network, all-gather, pmean and synchronization were not separately timed.',
      'Per-component critical-rank maxima need not occur on the same rank; do not sum maxima as an exact checkpoint critical path.',
      'Supervisor allocated GPUh, sum of rank timers and critical-path times device count are distinct accounting measures.',
      'Profile clock offset forces a reset in 12 collected blocks; formal training resets approximately every 62.5 blocks.',
      'No same-quality training speedup or learning/sample-efficiency improvement is inferred.',
      'Original P2/P4 profile per-rank raw detail is not required for these summary calculations; it was not independently reconstructed here.',
      'Formal checkpoint IO is outside reported block timing; total-minus-block includes initialization, checkpoint and unassigned host work, not a pure IO measurement.',
    ]
    data=dict(schema='paper1_existing_resource_analysis_v1',status='pass',sources=sources,
              checks=checks,derived=derived,microbatch_component=micro,
              matched_work_topologies=topologies,formal_rank_summary=rank_summary,
              formal_phase_summaries=categories,models=models,backtests=backtests,
              evidence_boundaries=boundary,
              prospective_validation_requirements=dict(
                first='Freeze model form, input contract, predictions and acceptance tolerance before new measurements.',
                component='Keep fixed N81 grid and independent physical starts; vary E and b separately; reserve one unmeasured configuration. Compare dense vs low-workspace implementation separately.',
                timing='Choose a small fixed-work retained pass and synchronize device results. Do not use a new full training run.',
                memory='Record analytical persistent bytes, compiler argument/output/alias/temp bytes, allocator active peak, total peak including learner, and allocator preallocation separately.',
                quality='If mathematical summation order changes, record direct force/ADM outputs and bounded physical trajectory discrepancies before timing claims.',
                cost='GPU-second budget includes compile/warmup/failed attempts; no speculative extrapolation to 100 turbines or other hardware.'))
    dump_json(out/'RESOURCE_ANALYSIS.json',data)
    dump_json(out/'INPUT_MANIFEST.json',dict(schema='resource_log_input_manifest_v1', sources=sources))
    # Explicit selector on measured candidates only; dimensions absent from the
    # data are never interpolated. This is executable decision support, not an
    # externally validated universal configuration law.
    choices=[]
    for cap_gb in (32,40,48):
        eligible=[r for r in topologies if r['measured_full_profile_allocator_peak_bytes']<=cap_gb*1e9]
        if eligible:
            for deadline_min in (60,120,240):
                feasible=[r for r in eligible if r['profile_total_minutes']<=deadline_min]
                winner=min(feasible,key=lambda r:r['supervisor_allocated_gpu_hours']) if feasible else None
                choices.append(dict(available_device_memory_GB=cap_gb,deadline_minutes=deadline_min,
                   chosen_devices=(winner['devices'] if winner else None),
                   allocated_gpu_hours=(winner['supervisor_allocated_gpu_hours'] if winner else None),
                   status='measured_profile_configuration_only' if winner else 'no_measured_feasible_configuration',
                   scope='bounded profile only; nominal active peak; no production headroom or full training feasibility guarantee'))
    dump_json(out/'MEASURED_PROFILE_SELECTOR.json',dict(schema='measured_profile_selector_v1',
          purpose='Choose minimum allocated GPUh among measured configurations satisfying the supplied nominal peak and bounded-profile deadline.',
          scenarios=choices,not_a_full_training_capacity_certificate=True))
    write_report(out,data,choices)
    receipts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())
              if p.is_file() and p.suffix in ('.py','.csv','.md','.json') and p.name!='ANALYSIS_RECEIPT.json'}
    dump_json(out/'ANALYSIS_RECEIPT.json',dict(status='pass',python=sys.version,
         no_GPU_execution=True,no_network=True,no_input_mutation=True,
         source_SHA_rechecked_after_analysis=all(sha(Path(v['path']))==v['sha256'] for v in sources.values()),
         outputs=receipts))
    print(json.dumps(dict(status='pass',output=str(out),checks=checks,derived=derived),ensure_ascii=True))


def write_report(out,data,choices):
    d=data['derived']; tops=data['matched_work_topologies']; b=data['backtests']
    report='''# 81风机既有日志资源分析

本分析只读取冻结日志，不启动LES、训练或GPU计算。采用标准库脚本重建数值；没有把历史其他硬件、风场规模或任务拼成性能曲线。

## 结论

已有数据支持一个有物理结构的局部资源模型：持久状态随驻留上下文数增长，稠密ADM工作区随活动微批量增长；固定全局工作时，采集时间近似随设备数的倒数下降，但学习与同步形成非零时间下限。它足以形成待检验的资源选择假设。现有三配置均已被看过，因此回算结果不是新的前瞻验证，也没有跨网格或跨硬件普适性。

## 实测配置

| GPU数 | 每卡上下文 | 普通块/s | 采集/s | 学习含通信/s | 真实重置块/s | 总profile/min | 分配GPUh | 完整profile峰值/GB |
|---|---|---|---|---|---|---|---|---|
'''
    for r in tops:
        report+=f"| {r['devices']} | {r['contexts_per_device']} | {r['ordinary_median_seconds']:.3f} | {r['collector_median_seconds']:.3f} | {r['learner_and_communication_median_seconds']:.3f} | {r['reset_block_seconds']:.3f} | {r['profile_total_minutes']:.3f} | {r['supervisor_allocated_gpu_hours']:.6f} | {r['measured_full_profile_allocator_peak_bytes']/1e9:.3f} |\n"
    report+='''
该表为固定32个上下文、b=1、全局minibatch=256、3072次采样及相同真实SAC更新的短profile。采集采用固定物理±20°动作，强制重置发生在12块内；Replay只有3072条。它不是训练到相同策略质量的对照。完整profile峰值包括该小Replay和学习路径；不能拿它与E1纯流动组件峰值直接作同口径减法。

## 瓶颈与正式训练

'''
    report+=f"- 正式训练共3907块，每卡8个上下文，完成1000448次交互。临界rank总时长为{d['formal_training_critical_rank_hours']:.6f}小时，四rank计时合计{d['formal_training_sum_rank_timer_gpu_hours']:.6f} GPUh。这里不包含独立完整入流档案生成及容器启动成本。\n"
    report+=f"- 学习启用后的普通块，rank0中位耗时{d['formal_learned_ordinary_rank0_median_seconds']:.3f}秒；以组件中位数比值描述，采集约{100*d['formal_learned_ordinary_collector_fraction_from_component_medians']:.1f}%，学习含通信约{100*d['formal_learned_ordinary_learner_fraction_from_component_medians']:.1f}%。网络和集体通信没有独立计时，不能把学习时间全归为通信开销。\n"
    report+=f"- 真实重置共{d['formal_real_reset_blocks']}块，占块数{100*d['formal_reset_block_fraction']:.3f}%，占正式进程时间约{100*d['formal_reset_block_time_fraction_of_process_wall']:.2f}%。短profile重置占总时长54–60%，因此短profile总加速比不能直接预测正式训练总加速比。\n"
    report+='''- 正式块计时不包含所有检查点、初始编译/预热和宿主工作；总时长减块计时是“未分配到块内的时间”，不能全部标为IO。检查点转换、写盘与屏障的最大值也未必来自同一卡，不应相加当作精确临界路径。

## 活动微批量与内存

| b | 固定上下文 | 持久状态/GB | 编译工作区/GB | 流动组件实测峰值/GB | 8上下文保留pass中位数/s |
|---|---|---|---|---|---|
'''
    for r in data['microbatch_component']:
        report+=f"| {r['microbatch']} | 8 | {r['persistent_state_bytes']/1e9:.6f} | {r['compiled_workspace_bytes']/1e9:.6f} | {r['measured_active_component_peak_GB']:.6f} | {r['retained_pass_median_seconds']:.6f} |\n"
    report+=f"\n网格为{d['grid']}，共{d['grid_cells']}点。两个FP32稠密b×81×Ng数组每单位b合计{d['dense_pair_bytes_per_active_member']/1e9:.6f} GB，与编译工作区的主量级一致。b=4到1的组件峰值下降{d['component_peak_reduction_b4_to_b1_percent']:.4f}%，但b=2与4时间并不单调；三次连续保留pass不足以证明普适的最优b。\n"
    report+='''
## 模型候选和事后留出回算

1. 内存候选：`M(E,b)=E*m_state + W_impl(b,Nt,Ng) + M_other`。当前稠密实现中`W_dense`的主项为`2*b*Nt*Ng*4 bytes`。完整训练还需Replay、Actor、优化器、编译常量及同时存活数组；低工作区实现会改变`W_impl`，必须分别测量，不能把当前系数当普适规律。
2. 固定任务、固定b的时间候选：`T_ordinary(P)=A/P+B`。采集分量近似`C/P`；学习含同步分量可以先拟合`a/P+l0`。仅有三个P无法再识别log(P)通信项、网络拓扑项或调度复杂度。
3. 重置候选：`T_reset(P)=R/P+r0`。完整作业用普通块数、实际重置次数、初始编译预热和检查点日程分别核算，不能把短profile一个重置的占比套入百万步训练。
4. 选择目标：在可用设备、显存余量和时间约束内，最小化实际分配GPUh。当前4卡最快，但与单卡相比短profile分配GPUh增加'''
    report+=f"{d['profile_P4_supervisor_GPUh_increase_vs_P1_percent']:.2f}%。\n\n"
    report+='| 回算目标 | 拟合配置 | 评价配置 | 预测 | 实测 | 绝对相对误差 |\n|---|---|---|---|---|---|\n'
    for r in b:
        if r['name'] in ['P4_ordinary_median_seconds','P4_collector_median_seconds','P4_learner_and_communication_median_seconds','P4_reset_block_seconds','P4_profile_total_seconds','b2_compiled_workspace_bytes','b2_measured_active_component_peak_bytes','formal_later_learned_ordinary','formal_later_learned_reset']:
            scale=1e9 if r['unit']=='bytes' else 1
            unit='GB' if scale==1e9 else 's'
            report+=f"| {r['name']} | {r['calibration']} | {r['evaluation']} | {r['prediction']/scale:.5f} {unit} | {r['observation']/scale:.5f} {unit} | {r['absolute_relative_error_percent']:.5f}% |\n"
    report+='''
这些误差描述结构与现有数据的一致性。P=4、b=2在模型选择前已经公开可见；正式训练后半段也来自同一次运行。不得把它们改写成盲法前瞻验证、独立重复或统计置信区间。

## 下一步最小验证设计

- 在新测量前冻结模型形式、输入字段、明确预测与容许误差，预先保留一个未测配置。不能看完新结果后再选择最漂亮的公式。
- E与b分开改变，否则持久状态和活动工作区无法辨识。可用固定物理起点的短pass，不需完整训练。新配置仅需回答是否能预测显存和时间，不承担新的收益评价任务。
- 稠密与流式/低工作区实现分层测量：先直接力场、盘平均量和短轨迹数值等价核验，再编译工作区、实测活动峰值及同步计时。改变归约次序时不得承诺逐位等价。
- 学习工作、Replay大小、编译环境、设备型号、网格、输出/持久状态数和计时同步必须记录。E1纯流动组件结果与完整训练容量分开。
- 尽量以“当前机器上满足显存和时限约束的配置选择”作为可检验命题；不外推到100机、其他GPU或新网格。

## 已执行检查与未覆盖

已重算三个E1配置、九次保留pass、三种拓扑的30个普通块、全部四个正式rank的15628条块记录；核对当前复现包rank0与原始rank0逐字节SHA一致，核对profile协议SHA及正式完成/内部检查，分别保留分配GPUh与rank计时口径。CSV逐行保留数据与分类，JSON记录源路径、SHA和所有模型系数。输出前再次核对所有源SHA未变化。

本次未重新运行LES、未测量任何新GPU点、未重构P=2/4原始每rank profile明细、未复验跨硬件或跨网格泛化、未检验训练到相同收益所需时间。当前资源选择器仅在三条已测短profile之间选择，且没有生产显存余量保证。
'''
    (out/'ANALYSIS.zh-CN.md').write_text(report)


if __name__=='__main__':
    main()
