"""Independent small-output reanalysis, preserving raw result/metadata records."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/'results'/name).read_text())
E=load('E1_COMPARISON.json');S=load('E1_FINAL_STATE_IDENTITIES.json');N0=load('N0_COMMON_STATES.json')
assert E['all_equivalent'] and len(E['comparisons'])==16
leaves=[l for c in E['comparisons'] for l in c['leaves']]
assert len(leaves)==112 and all(l['exact'] and l['finite'] and l['max_abs_difference']==0 for l in leaves)
assert S['comparison_sha256']==hashlib.sha256((ROOT/'results/E1_COMPARISON.json').read_bytes()).hexdigest()
assert len(S['files'])==24
for i in range(8):assert len({p['sha256'] for p in S['files'] if p['context_index']==i})==1
micro={}
for m in (1,2,4):
 r=E['reports'][str(m)];assert len(r['retained_pass_seconds'])==3
 assert np.isclose(np.median(r['retained_pass_seconds']),r['median_seconds'],rtol=0,atol=1e-12)
 micro[str(m)]={'median_seconds':r['median_seconds'],'peak_GB':r['device_memory']['peak_bytes_in_use']/1e9,'workspace_GB':r['compiled_kernel_memory']['temp_size_in_bytes']/1e9}
flux=[]
for r in N0['records']:
 f=r['flux'];expected=(f['outlet_m3_s'][0]-f['inlet_m3_s'][0]+f['top_m3_s'][0]-f['bottom_m3_s'][0])/abs(f['inlet_m3_s'][0]);assert abs(expected-f['relative_net_to_inlet'][0])<1e-12;assert r['finite'];flux.append(abs(expected)*100)
num={}
for dt in ['02','01']:
 p=ROOT/f'results/N1_DT{dt}_RESULT.json'
 if not p.exists():continue
 result=json.loads(p.read_text());scores={};diags={};rows_energy={}
 c=result['contract'];actual=c['scored_controls']*c['control_dt_seconds'];assert actual==1280
 assert (c['final_plane_cursor_exclusive']-c['scoring_plane_cursor'])*c['archive_dt_seconds']==actual
 for controller in ['ZERO','RL']:
  with np.load(ROOT/f'results/N1_DT{dt}_{controller}.npz',allow_pickle=False) as trace:
   power=trace['farm_power_mw'].astype(np.float64);turbines=trace['turbine_power_mw'].astype(np.float64);yaw=trace['yaw_deg'].astype(np.float64)
   assert power.shape==(128,1) and yaw.shape==(128,1,81);np.testing.assert_allclose(power,turbines.sum(axis=-1),rtol=2e-6,atol=1e-5)
   assert all(np.isfinite(trace[k]).all() for k in trace.files)
   delta=np.diff(np.concatenate([np.zeros((1,1,81)),yaw]),axis=0)
   assert np.abs(delta).max()/10<=1+1e-6 and np.abs(yaw).max()<=40+1e-5
   positions=np.asarray(result['contract']['probe_layout_contract']['turbine_positions_m']);xrows=np.unique(positions[:,0]);assert len(xrows)==9
   rowvals=[float(turbines[:,:,positions[:,0]==x].sum()*10/3600) for x in xrows]
   energy=float(power.sum()*10/3600);key='zero' if controller=='ZERO' else 'rl_final'
   assert abs(energy-result['scores'][key]['generation_energy_mwh'])<1e-10;scores[key]=energy
   rows_energy[key]=rowvals
   diags[key]={}
   for name in ['geometric_flux_relative','raw_divergence_rms','raw_interior_rms','reference_subtracted_rms','max_velocity']:
    a=trace['qualification_'+name].astype(np.float64);diags[key][name]={'min':float(a.min()),'mean':float(a.mean()),'max':float(a.max()),'max_abs':float(np.abs(a).max())}
 gain=100*(scores['rl_final']/scores['zero']-1);assert abs(gain-result['pair']['generation_gain_percent'])<1e-10
 num[dt]={'energies_mwh':scores,'gain_percent':gain,'diagnostics':diags,'row_energies_mwh':rows_energy,'actual_scoring_seconds':actual,'raw_legacy_duration_field':c['scoring_physical_seconds']}
summary={'state':'partial' if len(num)!=2 else 'complete','microbatch':micro,'equivalent_arrays':112,'state_identity_files':24,'starting_flux_abs_percent_range':[min(flux),max(flux)],'numerics':num}
if len(num)==2:
 differences={k:100*(num['01']['energies_mwh'][k]/num['02']['energies_mwh'][k]-1) for k in ['zero','rl_final']};gain_delta=num['01']['gain_percent']-num['02']['gain_percent'];worst=max(v['geometric_flux_relative']['max_abs'] for n in num.values() for v in n['diagnostics'].values())
 row_gains={dt:100*(np.array(n['row_energies_mwh']['rl_final'])/np.array(n['row_energies_mwh']['zero'])-1) for dt,n in num.items()};row_delta=row_gains['01']-row_gains['02']
 summary['timestep_comparison']={'row_gain_changes_pp':row_delta.tolist(),'row_gain_change_range_pp':[float(row_delta.min()),float(row_delta.max())],'gain_difference_pp':gain_delta,'absolute_energy_changes_percent':differences,'max_abs_flux_percent':100*worst,'investigation_triggered':abs(gain_delta)>1 or any(abs(v)>3 for v in differences.values()) or worst>=.01,'trigger_scope':'prospective investigation thresholds only, not a universal LES accuracy certificate'}
(ROOT/'results/REANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
