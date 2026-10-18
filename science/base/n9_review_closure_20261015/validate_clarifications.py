#!/usr/bin/env python3
"""CPU stdlib validation of added scientific notes against unchanged source records."""
from pathlib import Path
import argparse, hashlib, json, math

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();out=a.output.resolve();assert not out.is_relative_to(root) and not out.exists()
 rev=root/'n9_review_closure_20261015';n9=root/'n9_intermediate'
 doc=json.loads((rev/'SCIENTIFIC_CLARIFICATIONS.json').read_text());reference=json.loads((n9/'N9_BRIDGE_RESULTS.json').read_text())
 checks=0
 def check(ok):
  nonlocal checks
  assert ok;checks+=1
 def close(x,y):check(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-10))
 for row in doc['source_evidence']:
  q=root/row['path'];check(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'])
 freeze=json.loads((n9/'raw/evaluation/evaluation_freeze.json').read_text())
 for row in json.loads((rev/'ADDED_SOURCE_IDENTITIES.json').read_text())['files']:
  q=rev/row['path'];key='actor_dependent_freeze' if q.name.startswith('actor_') else 'static_lookup'
  check(sha(q)==freeze['identity_sha256'][key]==row['sha256'])
 cfg=json.loads((n9/'raw/training/gpu205.training.json').read_text())['configuration']
 l=doc['learning_contract'];check(l['reward']['name']==cfg['formal_reward_name']=='reward_r1')
 check(l['return']['n_step']==cfg['formal_n_step']==16);check(l['return']['episode_controls']==cfg['episode_interactions']==500)
 check(l['return']['reset_controls_excluded']==cfg['reset_interactions']==150)
 ep=l['return']['endpoint'];check(ep['raw_interactions']-ep['finalized_replay_global']==ep['pending_per_rank']*2==480)
 for key,rawkey in [('actor_warmup_learner_updates','graph_actor_warmup_updates'),('actor_update_period','graph_actor_update_period'),('global_sample_size','effective_global_sample_size'),('sample_per_rank','sample_size_per_host'),('global_transitions_per_iteration','global_transitions_per_iteration'),('updates_per_iteration','updates_per_iteration')]:
  check(l['learner'][key]==cfg[rawkey])
 check((l['learner']['sac_updates']-l['learner']['actor_warmup_learner_updates'])//l['learner']['actor_update_period']==l['learner']['actor_updates']==cfg['sac_actor_updates_applied_total'])
 close(l['optimizer_temperature']['actor_learning_rate'],cfg['learning_rate']);close(l['optimizer_temperature']['critic_learning_rate'],cfg['learning_rate'])
 close(l['optimizer_temperature']['entropy_scale'],cfg['entropy_scale']);close(l['optimizer_temperature']['initial_alpha'],cfg['initial_alpha'])
 scope=doc['scope'];check(scope['independent_holdout_precursors']==1 and scope['direction_cases']==17 and scope['nested_windows']==34)
 check(not scope['mean_energy_feedback_advantage_established'] and not scope['RL_lookup_superiority_established'] and not scope['population_robustness_ranking_established'])
 check(scope['lookup_is_three_streamwise_row_group_target_table'] and not scope['lookup_is_later_nine_dimensional_search_table'])
 report_list=[]
 for row in reference['evaluation']['direction_rows']:
  j=json.loads((n9/row['report']).read_text());check(len(j['windows'])==2);report_list.append(j)
 controllers=doc['controller_comparisons']['controllers'];check(len(controllers)==7)
 expected={row['controller_id']:row for row in reference['retained_controller_comparisons']['controllers']}
 for row in controllers:
  key=row['controller_id'];direction_means=[];window_gains=[]
  for report in report_list:
   values=[]
   for win in report['windows']:
    z=win['controllers']['zero_yaw']['mean_farm_power_mw'];q=win['controllers'][key]
    gain=(q['mean_farm_power_mw']/z-1)*100
    close(gain,q['gain_percent']);values.append(gain);window_gains.append(gain)
   direction_means.append(sum(values)/len(values))
  close(sum(direction_means)/len(direction_means),row['equal_direction_mean_gain_percent'])
  close(min(direction_means),row['worst_direction_mean_gain_percent'])
  check(sum(x<0 for x in window_gains)==row['negative_window_count'])
  check(sum(x<0 for x in direction_means)==row['negative_direction_count'])
  for metric in ['equal_direction_mean_gain_percent','worst_direction_mean_gain_percent','negative_window_count']:
   close(row[metric],expected[key][metric])
 current={row['controller_id']:row for row in controllers};rt=current['selected_realtime_policy']
 for key,pair in doc['controller_comparisons']['paired_comparisons'].items():
  other=key.removeprefix('selected_realtime_policy_minus_')
  close(rt['equal_direction_mean_gain_percent']-current[other]['equal_direction_mean_gain_percent'],pair['equal_direction_mean_gain_difference_percentage_points'])
 check(rt['negative_window_count']==0 and rt['worst_direction_mean_gain_percent']>0)
 check(all(rt['equal_direction_mean_gain_percent']<current[key]['equal_direction_mean_gain_percent'] for key in ['static_les_lookup','selected_policy_direction_mean','selected_probe_mean_ablation','cross_precursor_action_replay']))
 check(all(rt['worst_direction_mean_gain_percent']>current[key]['worst_direction_mean_gain_percent'] for key in ['static_les_lookup','selected_policy_direction_mean','selected_probe_mean_ablation','cross_precursor_action_replay']))
 check(doc['historical_metadata_conflict']['preserved'] is True)
 report={'status':'PASS','checks_passed':checks,'controller_windows_recomputed':7*34,'source_files_SHA_verified':len(doc['source_evidence']),
  'added_sources_bound_to_original_WP7_freeze':2,'definitions_and_seven_controller_results_match':True,'historical_metadata_conflict_preserved_and_explained':True,
  'comparison_tolerance':{'absolute':1e-10,'relative':1e-12},'new_GPUh':0,'new_LES_runs':0,'scope':'CPU saved-record/document consistency; no new physical validation or training reproduction'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
