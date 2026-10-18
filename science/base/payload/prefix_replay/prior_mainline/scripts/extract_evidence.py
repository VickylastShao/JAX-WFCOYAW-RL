"""Read-only small-record audit; no JAX import, GPUs, or large LES states."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import statistics
import numpy as np

OUT = Path(__file__).resolve().parent
LOCAL = Path.cwd() / 'audit'
BACKUP = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL')
BASE = BACKUP / 'github_release_ready_20261014/repository/n81_20261014_visual_revision'
CODES = BASE / 'source/codes'
SOURCES = BACKUP / 'n81_20261013/source_metadata'
CLOSURE = BACKUP / 'wind_energy_closure_20261014'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p, field=None):
    return {'path': str(p), 'sha256': digest(p), 'field': field}


def read(p):
    return json.loads(Path(p).read_text())


def relative_change(first, last):
    return 100 * (last - first) / first


def source_records():
    result = {}
    temporal_root = LOCAL / 'paper1_n81_evidence_closure_20261013/existing/source/archives'
    for letter, name in [('A', 'n81_train_A'), ('B', 'n81_heldout_B'), ('C', 'n81_heldout_C')]:
        mp, ap = SOURCES / name / 'state.meta.json', SOURCES / name / 'ADMISSION.json'
        meta, admission = read(mp), read(ap)
        nodes = []
        for index, node in enumerate(meta['diagnostic_history']):
            m = node['metrics']
            spectra = {}
            for axis, key, dx in [('streamwise', 'hub_streamwise_spectrum_normalized', meta['case']['domain_lx'] / meta['case']['nx']),
                                  ('spanwise', 'hub_spanwise_spectrum_normalized', meta['case']['domain_lz'] / meta['case']['nz'])]:
                spectrum = m[key]
                peak = max(range(1, len(spectrum)), key=lambda k: spectrum[k])
                length = meta['case']['domain_lx'] if axis == 'streamwise' else meta['case']['domain_lz']
                spectra[axis] = {
                    'bins': len(spectrum), 'sum': math.fsum(spectrum),
                    'peak_nonzero_bin': peak, 'peak_wavelength_m': length / peak,
                    'normalized_energy': spectrum,
                    'definition': 'Mean squared orthonormal one-sided spatial rFFT coefficients; normalized by their stored sum. No one-sided doubling. Not a temporal spectrum or verified inertial-range law.',
                    'field': f'diagnostic_history[{index}].metrics.{key}',
                }
            nodes.append({
                'completed_steps': node['completed_steps'],
                'simulated_seconds': node['completed_steps'] * meta['case']['dt'],
                'mean_hub_speed_m_per_s': m['mean_hub_speed_m_per_s'],
                'spatial_ti_proxy': m['mean_hub_turbulence_intensity'],
                'spatial_tke_m2_per_s2': m['hub_turbulent_kinetic_energy_m2_per_s2'],
                'interior_divergence_rms_per_s': m['interior_divergence_rms'],
                'finite': m['finite'], 'spectra': spectra,
                'field': f'diagnostic_history[{index}].metrics',
            })
        first, last = nodes[0], nodes[-1]
        late = [n for n in nodes if n['completed_steps'] >= 160000]
        summaries = []
        for p in sorted((temporal_root / name / 'temporal').glob('block_*.json')):
            d = read(p)
            assert d['source_sha256'] == meta['npz_sha256']
            summaries.append({'record': ref(p), **{k: d[k] for k in (
                'start', 'stop', 'duration_seconds', 'mean_hub_u_m_per_s',
                'mean_spatial_ti', 'std_plane_mean_hub_u_m_per_s')}})
        assert len(summaries) in (135, 26)
        total_duration = math.fsum(s['duration_seconds'] for s in summaries)
        mean_u = math.fsum(s['mean_hub_u_m_per_s'] * s['duration_seconds'] for s in summaries) / total_duration
        mean_ti = math.fsum(s['mean_spatial_ti'] * s['duration_seconds'] for s in summaries) / total_duration
        result[letter] = {
            'source_name': name, 'meta_record': ref(mp), 'admission_record': ref(ap),
            'seed': meta['seed'], 'original_state_sha256': meta['npz_sha256'],
            'completed_steps': meta['completed_steps'], 'hub_sample_y_m': round(meta['case']['hub_height'] / (meta['case']['domain_ly'] / (meta['case']['ny'] - 1))) * (meta['case']['domain_ly'] / (meta['case']['ny'] - 1)),
            'admitted_for_archive': admission['admitted_for_archive'],
            'statistical_stationarity_proven': admission['statistical_stationarity_proven'],
            'qualification_definition': 'Finite state and source-physics/geometry identity; not a statistical stationarity gate.',
            'nodes': nodes,
            'snapshot_drift': {
                'first_to_final_mean_speed_change_m_s': last['mean_hub_speed_m_per_s'] - first['mean_hub_speed_m_per_s'],
                'first_to_final_mean_speed_change_percent': relative_change(first['mean_hub_speed_m_per_s'], last['mean_hub_speed_m_per_s']),
                'last_three_mean_speed_range_m_s': [min(n['mean_hub_speed_m_per_s'] for n in late), max(n['mean_hub_speed_m_per_s'] for n in late)],
                'last_three_spatial_ti_range': [min(n['spatial_ti_proxy'] for n in late), max(n['spatial_ti_proxy'] for n in late)],
                'last_three_spatial_tke_range_m2_per_s2': [min(n['spatial_tke_m2_per_s2'] for n in late), max(n['spatial_tke_m2_per_s2'] for n in late)],
                'last_three_mean_speed_change_m_s': late[-1]['mean_hub_speed_m_per_s'] - late[0]['mean_hub_speed_m_per_s'],
                'scope': 'Descriptive snapshot drift only; snapshots are neither independent samples nor temporal-point TI.'
            },
            'archive_small_metadata_reanalysis': {
                'blocks': len(summaries), 'duration_seconds': total_duration,
                'hub_mean_u_m_s': mean_u, 'mean_spatial_ti_proxy': mean_ti,
                'hub_block_mean_range_m_s': [min(s['mean_hub_u_m_per_s'] for s in summaries), max(s['mean_hub_u_m_per_s'] for s in summaries)],
                'first_to_last_block_change_m_s': summaries[-1]['mean_hub_u_m_per_s'] - summaries[0]['mean_hub_u_m_per_s'],
                'plane_mean_time_sd_range_m_s': [min(s['std_plane_mean_hub_u_m_per_s'] for s in summaries), max(s['std_plane_mean_hub_u_m_per_s'] for s in summaries)],
                'records': summaries,
                'scope': 'Only previously saved small temporal JSONs read. Original inlet arrays were not rehashed or loaded in this audit. Temporal SD of spatially averaged planes is not a fixed-point temporal TI.'
            }
        }
    return result


def disk_cpu_check():
    """Execute the exact frozen two function bodies using NumPy on a small grid."""
    p = CODES / 'jax_mole_case_benchmark.py'
    source = p.read_text()
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('disk_weights', 'disk_source')]
    assert len(functions) == 2
    functions.sort(key=lambda n: n.lineno)
    # Bound CPU arithmetic only: this is a deliberately reduced synthetic grid,
    # not the 81-turbine LES or a numerical-convergence experiment.
    case = SimpleNamespace(lz=5922.0, diameter=126.0, density=1.2,
                           dx=6174.0/495, dy=12.5, dz=5922.0/483)
    nx, ny, nz = 40, 21, 40
    volume = np.float32(case.dx * case.dy * case.dz)
    ns = {
        'np': np, 'jnp': np, 'Array': np.ndarray, 'case': case,
        'cell_volume': volume, 'disk_area': np.float32(math.pi * case.diameter**2 / 4),
        'ct_prime': np.float32(.75 / (1-.17095)**2),
        'relaxation_alpha': 1 - math.exp(-.2 / 1.1291),
        'x_grid': np.arange(nx, dtype=np.float32)[None,None,:,None,None] * np.float32(case.dx),
        'y_grid': np.arange(ny, dtype=np.float32)[None,None,None,:,None] * np.float32(case.dy),
        'z_grid': (np.arange(nz, dtype=np.float32)[None,None,None,None,:] - np.float32(nz//2)) * np.float32(case.dz) + np.float32(2961),
        'turbine_x': np.asarray([126,252,378], dtype=np.float32),
        'turbine_y': np.asarray([90,90,90], dtype=np.float32),
        'turbine_z': np.asarray([2861,2961,3061], dtype=np.float32),
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(p), 'exec'), ns)
    velocity = np.zeros((1,3,nx,ny,nz), dtype=np.float32)
    velocity[:,0] = 8.0
    receipts = []
    for value in (-40,-20,0,20,40):
        yaw = np.asarray([[value,-value,value/2]], dtype=np.float32)
        weights = ns['disk_weights'](yaw)
        acceleration, speed, power, thrust = ns['disk_source'](velocity, yaw, weights, np.full((1,3),7.,np.float32),np.asarray(False))
        normal = np.stack((np.cos(np.deg2rad(yaw)),np.zeros_like(yaw),-np.sin(np.deg2rad(yaw))),axis=2)
        force = np.float32(case.density) * np.sum(acceleration,axis=(-3,-2,-1)) * volume
        expected = -np.sum(thrust[:,:,None]*normal,axis=1)
        relative = float(np.max(np.abs(force-expected))/max(float(np.max(np.abs(expected))),1.0))
        receipts.append({'yaw_degrees':yaw.tolist(), 'weight_sum_max_abs_error':float(np.max(np.abs(np.sum(weights,axis=(-3,-2,-1))-1.))),
            'integrated_force_N':force.tolist(), 'expected_force_N':expected.tolist(),
            'force_relative_error':relative, 'power_identity_relative_error':float(np.max(np.abs(power-thrust*speed))/max(float(np.max(np.abs(power))),1.0)),
            'all_finite':bool(np.isfinite(acceleration).all()), 'pass':relative < 2e-5})
    assert all(r['pass'] and r['all_finite'] for r in receipts)
    return {'status':'pass', 'implementation':ref(p,'disk_weights lines 432-467; disk_source lines 469-521'),
            'function_body_sha256':hashlib.sha256('\n'.join(ast.get_source_segment(source,n) for n in functions).encode()).hexdigest(),
            'backend':'NumPy '+np.__version__, 'synthetic_grid':[nx,ny,nz], 'synthetic_turbines':3,
            'cases':receipts, 'max_force_relative_error':max(r['force_relative_error'] for r in receipts),
            'force_density_units':'rho * acceleration has N/m^3; acceleration = -T*w*n/(rho*cell_volume) has m/s^2; w is dimensionless and sums to 1.',
            'identity':'sum_cells rho * acceleration * cell_volume = -sum_turbines T*n',
            'scope':'Source-expression and discrete-normalization arithmetic only; not a full JAX run, LES solver validation, spatial-convergence result, or field validation.'}


def hardware():
    preflight = LOCAL / 'paper1_n81_efficiency_corrected_20261012/PREFLIGHT_REMOTE.json'
    d = json.loads(read(preflight)['result']['stdout'])
    rows = []
    for line in d['gpus']['stdout'].splitlines():
        index, name, total, used, free, util = [s.strip() for s in line.split(',')]
        rows.append({'index':int(index),'model':name,'memory_total_MiB':int(total),
                     'memory_total_GiB':int(total)/1024,'memory_total_decimal_GB':int(total)*2**20/1e9})
    training = BASE / 'configs/runtime_training.json'
    td = read(training)
    launch = CODES / 'run_mole_large_scale_initial_stage.py'
    health = LOCAL / 'paper1_n81_efficiency_corrected_20261012/ADMISSION_RUNTIME_CHECK.json'
    hd = json.loads(read(health)['result']['stdout'])
    profile = LOCAL / 'paper1_n81_evidence_closure_20261013/existing/efficiency/efficiency/PROFILE.json'
    pd = read(profile)
    return {'direct_gpu_query': {'record':ref(preflight,'result.stdout -> gpus.stdout'), 'checked_at_cst':d['checked_at_cst'],'gpus':rows,
                               'experiment_scope':'2026-10-12 corrected efficiency profile preflight',
                               'formal_training_variant_link':'unknown: no GPU UUID/model chain to formal training is retained in these audited records'},
            'placement': {'training_record':ref(training,'formal_environment; ranks[*].physical_gpu / argv'),
                          'observed_training_environment':td['formal_environment'],
                          'recorded_rank_to_gpu':[{k:r[k] for k in ('rank','physical_gpu')} for r in td['ranks']],
                          'observed_profile_health_record':ref(health,'result.stdout -> STATUS.json.jobs'),
                          'observed_profile_rank_to_gpu':{k:v['gpu'] for k,v in hd['STATUS.json']['jobs'].items()},
                          'profile_protocol_record':ref(profile,'ranks[*].trainer_argv --coordinator / --process-id / --num-processes'),
                          'interpretation':'Single-host four-process placement; one exposed physical GPU per container and localhost coordinator.'},
            'allocator': {'frozen_launcher':ref(launch,'prefix: lines 50-93'),
                          'explicit_memory_fraction':'.60',
                          'memory_fraction_evidence_type':'Forced Docker -e entry in frozen launcher source; effective running-container environment value was not retained. Pool matching is corroboration only.',
                          'explicit_preallocate_setting':'unknown: not set in audited launcher, and runtime effective variable not retained',
                          'explicit_allocator_setting':'unknown: not set in audited launcher, and runtime effective variable not retained',
                          'effective_default_not_inferred':True,
                          'observed_pool_comes_from_E1_device_memory':True,
                          'warning':'Device reservation pool is distinct from peak live bytes; do not equate nvidia-smi memory.used with live scientific working set.'},
            'interconnect': {'topology':'unknown: no nvidia-smi topo -m receipt in the checked experiment records',
                             'NVLink_bandwidth':'unknown; not measured', 'PCIe_bandwidth':'unknown; not measured',
                             'communication_backend':'JAX distributed collectives recorded; physical routing not instrumented',
                             'scope':'No attribution of speedup to an unrecorded NVLink/PCIe topology.'}}


def e1():
    p = CLOSURE / 'results/E1_COMPARISON.json'
    ident = CLOSURE / 'results/E1_FINAL_STATE_IDENTITIES.json'
    d, identities = read(p), read(ident)
    leaves = [l for c in d['comparisons'] for l in c['leaves']]
    assert len(d['comparisons']) == 16 and len(leaves) == 112
    assert all(l['exact'] and l['finite'] and l['pass'] and l['max_abs_difference'] == 0 for l in leaves)
    per_context = {}
    for f in identities['files']:
        per_context.setdefault(f['context_index'],[]).append(f['sha256'])
    assert all(len(hs)==3 and len(set(hs))==1 for hs in per_context.values())
    assert digest(p)==identities['comparison_sha256']
    return {'comparison_record':ref(p), 'identity_record':ref(ident),
            'checked_comparison_identity_matches':True,
            'reference_microbatch':1,'alternative_microbatches':[2,4],
            'contexts':8,'comparisons':16,'arrays':112,
            'checked_fields':[l['path'] for l in d['comparisons'][0]['leaves']],
            'all_arrays_finite_and_bitwise_equal':True,'maximum_absolute_difference':0.0,
            'reported_tolerance':{'float_rtol':1e-4,'float_atol':1e-5,'integer':'exact'},
            'final_state_file_receipt_count':24,'same_sha_per_context_across_microbatch':True,
            'source':ref(CLOSURE/'source/executed/codes/paper1_wind_energy_closure_20261014.py','microbatch and compare functions'),
            'reports':d['reports'],
            'verification_scope':'Same eight saved B/C zero-yaw warmup endpoints, saved fixed initial inlet per context, zero yaw. Fifty substeps per pass, one warm pass and three retained passes; final comparison after 200 substeps. Seven flow/history fields only; no observation, Actor, Replay, RNG, optimizer, gradient or policy-gain equivalence claim.',
            'new_audit_scope':'This audit recomputes small comparison receipts and verifies their binding to the saved SHA receipt. The 24 multi-GB final states were not loaded or newly rehashed locally.'}


def main():
    result = {'schema':'paper1_mainline_evidence_supplement_v1','date':'2026-10-15','new_GPU_hours':0,
              'mutated_scientific_implementation':False,'read_large_LES_states':False,
              'source_qualification':source_records(), 'disk_force_arithmetic':disk_cpu_check(),
              'hardware':hardware(),'E1_validation':e1(),
              'script_sha256':digest(__file__)}
    (OUT/'EVIDENCE_SUPPLEMENT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    write_markdown(result)
    print(json.dumps({'status':'pass','sources':3,'source_snapshot_nodes':30,
                     'archive_metadata_blocks':sum(x['archive_small_metadata_reanalysis']['blocks'] for x in result['source_qualification'].values()),
                     'disk_force_max_relative_error':result['disk_force_arithmetic']['max_force_relative_error'],
                     'E1_exact_arrays':result['E1_validation']['arrays'],'output':str(OUT/'EVIDENCE_SUPPLEMENT.json')}))


def write_markdown(result):
    lines = ['# 81台主线修订：小记录与CPU证据补强', '',
        '本轮新增GPU成本为0。仅读取小型JSON、冻结源码；未载入压缩大状态，未改科学实现、原稿或旧冻结包。数字来源及完整SHA、字段见同目录 `EVIDENCE_SUPPLEMENT.json`。', '',
        '## 1. 湍流源资格与统计口径', '',
        'A/B/C三个源的 `ADMISSION.json` 均为 `admitted_for_archive=true`，同时明确 `statistical_stationarity_proven=false`。qualified表示状态有限及物理/几何来源一致；它不表示通过了严格时间平稳性检验。', '',
        '状态诊断的近hub采样面为87.5m。TI为该面空间标准差/空间均值；TKE为三个速度分量相对于该面空间均值的波动平方和的一半。它们均不是单点时间TI。', '',
        '|源|step|模拟时间(s)|近hub均值(m/s)|空间TI|空间TKE(m²/s²)|空间散度RMS(s⁻¹)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for letter,r in result['source_qualification'].items():
        for n in r['nodes']:
            lines.append(f"|{letter}|{n['completed_steps']}|{n['simulated_seconds']:.0f}|{n['mean_hub_speed_m_per_s']:.6f}|{n['spatial_ti_proxy']:.6f}|{n['spatial_tke_m2_per_s2']:.6f}|{n['interior_divergence_rms_per_s']:.3e}|")
    lines += ['', '表内直接记录位于各 `state.meta.json:diagnostic_history[i].metrics`；时间由step×0.2s派生。三个文件的位置与SHA：', '']
    for letter,r in result['source_qualification'].items():
        for key in ('meta_record','admission_record'):
            f=r[key];lines.append(f"- {letter} {key}: [{Path(f['path']).name}]({f['path']})；SHA `{f['sha256']}`。")
    lines += ['', '源spinup早期节点仍有明显均值漂移，不能仅根据后段几次快照宣布平稳。最后三个节点160k/180k/200k的速度、空间TI、TKE范围及均值变化如下，均为描述性CPU派生：', '',
        '|源|均值范围(m/s)|空间TI范围|空间TKE范围(m²/s²)|160k→200k速度变化(m/s)|',
        '|---|---:|---:|---:|---:|']
    for letter,r in result['source_qualification'].items():
        s=r['snapshot_drift'];a,b=s['last_three_mean_speed_range_m_s'];c,d=s['last_three_spatial_ti_range'];e,f=s['last_three_spatial_tke_range_m2_per_s2']
        lines.append(f"|{letter}|{a:.4f}–{b:.4f}|{c:.5f}–{d:.5f}|{e:.5f}–{f:.5f}|{s['last_three_mean_speed_change_m_s']:+.5f}|")
    lines += ['', '每个节点还保存1153个streamwise与242个spanwise归一化空间rFFT能量bin。完整数组、峰值非零bin、峰波长及字段已保存在本轮JSON；本轮仅核查其存在、归一化和值有限。不得把它们解释成时间谱、惯性区拟合或独立平稳性证明。定义来自冻结 `generate_mole_precursor_resumable.py:_normalized_energy_spectrum`，SHA `b142cd50aeb4637cb49ea5e590462954cd063dd091b38d6bff665e7fb9922bc0`。', '',
        '187份已保存的归档小JSON给出另一阶段的入流统计。它们来自源spinup完成后的归档过程，与上述200k状态快照不能混用。', '',
        '|源|4000s块数|归档时长(s)|近hub时间平均(m/s)|平均空间TI|块均值范围(m/s)|首块→末块变化(m/s)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for letter,r in result['source_qualification'].items():
        a=r['archive_small_metadata_reanalysis'];lo,hi=a['hub_block_mean_range_m_s']
        lines.append(f"|{letter}|{a['blocks']}|{a['duration_seconds']:.0f}|{a['hub_mean_u_m_s']:.6f}|{a['mean_spatial_ti_proxy']:.6f}|{lo:.6f}–{hi:.6f}|{a['first_to_last_block_change_m_s']:+.6f}|")
    lines += ['', '上述块来自同一源内连续轨迹，彼此相关。每份JSON的路径、SHA和fields保存在 `archive_small_metadata_reanalysis.records`。本轮没有重新读取或验证0.889TB入流数组；源SHA与所有187份记录绑定关系已核对。', '',
        '## 2. 实际硬件与allocator的可报告范围', '',
        '2026-10-12 17:40:08 +08:00 corrected efficiency profile预检记录4张NVIDIA H20-3e，各自143771MiB（140.401367GiB / 150.754820GB十进制）。此处是该次profile的实测盘点。缺少GPU UUID/型号贯通链，因此不把这个variant及容量无条件追认成正式训练的配置。', '',
        '正式训练记录 `runtime_training.json:formal_environment` 确認hostname=h20、process_count=4、global_devices=cuda:0–3，四rank对应物理GPU0–3。profile健康记录确认四个容器分别绑定GPU0–3，协议coordinator为127.0.0.1；可报告单机四进程、每个进程一张GPU。', '',
        '冻结启动器 `run_mole_large_scale_initial_stage.py:prefix` 强制添加Docker `-e XLA_PYTHON_CLIENT_MEM_FRACTION=.60`，并非setdefault。运行时有效环境变量未被保留；E1报告pool=90067190016bytes可作一致性旁证，不能当作环境变量实测。PREALLOCATE/ALLOCATOR有效设置、NVLink/PCIe拓扑与带宽均未知，不补写厂商标称参数。', '',
        '显存pool预留不等于peak_bytes_in_use；E1内存主张以live allocator峰值及compiler workspace为依据。硬件/launcher/placement所有路径、SHA与字段见本轮JSON的 `hardware`。', '',
        '## 3. 离散力源单位和归一化CPU核查', '',
        '冻结 `disk_weights` 的离散权重w无量纲，Σw=1；`disk_source`返回加速度a=-ΣTwn/(ρΔV)，单位m/s²。实际积分恒等式是ΣcellsρaΔV=-ΣturbinesTn；若正文的f表示力密度，则f=ρa、单位N/m³，必须明确定义。功率定义P=T·u_relax。', '',
        f"独立CPU审计直接提取冻结文件中的两函数AST，用NumPy {np.__version__}执行40×21×40合成小网格、3个合成disk、5组yaw，有限与Σw/力积分/功率恒等式通过。最大积分相对误差 `{result['disk_force_arithmetic']['max_force_relative_error']:.12e}`。这是源码表达式的低成本量纲与离散归一化核查，不是JAX/全81台LES复跑，不是空间收敛或实测物理验证。", '',
        '冻结文件SHA `eaa0717fc1655c411f6e1003cddf339d86f571fbc05f1a860bf1ca07f6fbaffe`；函数所在432–521行；AST函数体SHA、各case细节见 `disk_force_arithmetic`。', '',
        '## 4. E1验证文件定位和边界', '',
        '原E1对同8个B/C warmup起点，固定各自saved initial inlet、零yaw，比较microbatch1与2/4。每pass50substeps，1个warm pass+3个retained passes，200substeps后的最终7类flow/history字段比较16次，共112数组全部finite且逐位一致，max_abs_difference=0。原float容差1e−4/1e−5保留；实际结果更严格，不因容差通过而称逐位一致。', '',
        '字段为velocity、rhs_previous、rhs_previous_2、filtered_disk_speed、bottom_pressure_gradient、open_x_pressure_gradient、step_index。24份最终state的远端SHA收据显示每个context跨3个microbatch完全相同。', '',
        '可直接定位文件：', '',
        '- `E1_COMPARISON.json`，SHA `1d0b2fa8cfc126606ede68a7c6bd070c48ca44b663e772a5773611c01820f402`。',
        '- `E1_FINAL_STATE_IDENTITIES.json`，SHA `f905c7bdd6e0e154bb3df8afec524167d0ec387720bf39d285235946a71b36ab`。',
        '- 目录 `/mnt/d/research-backups/JAX-WFCOYAW-RL/wind_energy_closure_20261014/results`；新GitHub候选repository内同名results也保留。', '',
        '本轮独立核对小记录中的112项exact/finite/pass、0差值、24份SHA相等关系及comparison_sha256绑定；未重新载入或重hash远端8.54GB最终状态。E1范围不包含Actor、Replay、RNG、optimizer、梯度、学习质量或B/C收益复跑。', '',
        '## 5. 可直接使用的最短英文段落与不编号tabular', '',
        '插入Supplement源定义段后，可用无caption的tabular；不会改变既有S表编号。以下均为提案，未修改稿件。', '',
        '```latex',
        r'Source qualification verified finite fields and matching physical and geometric provenance. It did not certify temporal stationarity: all three admission records retain \texttt{statistical\_stationarity\_proven=false}. Near-hub diagnostics sample $y=87.5$ m. Their TI and TKE are spatial snapshot statistics, distinct from temporal point statistics. The ranges below summarize the saved 160k, 180k and 200k snapshots; subsequent inlet-archive averages describe a separate continuation period.',
        '\\begin{center}', '\\small', '\\begin{tabular}{lccc}', '\\toprule',
        'Source & $\\bar u_h$ (m s$^{-1}$) & Spatial TI & Spatial TKE (m$^2$ s$^{-2}$) \\\\', '\\midrule']
    for letter,r in result['source_qualification'].items():
        s=r['snapshot_drift'];a,b=s['last_three_mean_speed_range_m_s'];c,d=s['last_three_spatial_ti_range'];e,f=s['last_three_spatial_tke_range_m2_per_s2']
        lines.append(f"{letter} & {a:.4f}--{b:.4f} & {c:.5f}--{d:.5f} & {e:.5f}--{f:.5f} \\\\")
    lines += ['\\bottomrule','\\end{tabular}','\\end{center}','```','',
        r'注意：上段的 `\texttt` 应在LaTeX中保留反斜线；root可直接使用本报告中的字面文本。', '',
        '**力源单位段：**', '',
        'The solver right-hand side stores acceleration, $\\mathbf a_j=-\\sum_k T_k w_{kj}\\mathbf n_k/(\\rho\\Delta V)$, where the dimensionless discrete weights satisfy $\\sum_j w_{kj}=1$. Consequently, $\\sum_j\\rho\\mathbf a_j\\Delta V=-\\sum_kT_k\\mathbf n_k$. A bounded CPU audit of the frozen source expressions confirmed this discrete identity to $1.06\\times10^{-7}$ relative error on synthetic inputs; this check concerns source normalization rather than LES or field validation.', '',
        '**硬件段（仅profile variant可限定实测）：**', '',
        'The corrected efficiency-profile preflight on 12 October 2026 recorded four NVIDIA H20-3e devices, each reporting 143771 MiB. The recorded training configuration used four processes on one host, with one exposed GPU per process and a localhost coordinator. The frozen launcher explicitly supplied \\texttt{XLA\\_PYTHON\\_CLIENT\\_MEM\\_FRACTION=.60}; effective preallocation and allocator settings and the physical interconnect topology were not retained. Device-pool reservation is distinguished from live allocator peaks. The available records do not establish a GPU-UUID chain tying the preflight hardware variant to the formal training run.', '',
        '**E1证据定位段：**', '',
        'The released \\texttt{E1\\_COMPARISON.json} records 16 comparisons of seven flow/history fields across eight contexts, while \\texttt{E1\\_FINAL\\_STATE\\_IDENTITIES.json} binds their 24 final-state exports by SHA-256. All 112 compared arrays were finite and bitwise equal after 200 substeps. This equivalence is restricted to the fixed-inlet, zero-yaw LES component workload; it does not cover policy observations, Replay, optimization state or controlled-energy replay.', '']
    (OUT/'EVIDENCE_SUPPLEMENT.zh-CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
