#!/usr/bin/env python3
"""Synthetic-fixture regression tests; values are not experimental evidence."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from datetime import datetime, timezone

from analyze_new_campaign import analyze, save, sha


HERE=Path(__file__).resolve().parent


class CampaignAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic_campaign_test_')
        self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self):
        main=self.root/'gpu_main';pilot=self.root/'gpu_pilot';out=self.root/'analysis'
        main.mkdir();pilot.mkdir();out.mkdir()
        before=time.time()-100
        definitions=[['dense_b1','dense',81,1],['stream_g1_b1','stream',1,1],
             ['stream_g9_b1','stream',9,1],['stream_g3_b1','stream',3,1],['stream_g3_b2','stream',3,2]]
        proto=dict(branches=definitions,substeps=50,source_core_sha256='synthetic_core',
          runner_sha256='synthetic_runner',candidate_sha256='synthetic_candidate',
          common_state_gzip_sha256='synthetic_state',original_inlet_prefix_sha256='synthetic_inlet',
          gpu_identity='SYNTHETIC RTX; TEST ONLY',budget_gpu_seconds=6660,
          prospective_model_contract=dict(memory_error_target=.10,time_error_target=.20),
          selector_scenarios=dict(capacity_gib=[4,8,10],reserve_gib=1,diagnostic_latency_seconds=[5,15,60]))
        save(main/'PROTOCOL.json',proto)
        rows=[]
        for index,(name,mode,group,batch) in enumerate(definitions):
            directory=main/name;directory.mkdir()
            live={'dense_b1':9000,'stream_g1_b1':1000,'stream_g9_b1':1800,
                 'stream_g3_b1':2400,'stream_g3_b2':2700}[name]
            seconds={'dense_b1':10,'stream_g1_b1':81,'stream_g9_b1':9,
                 'stream_g3_b1':40,'stream_g3_b2':70}[name]
            value=dict(status='completed',mode=mode,group=group,case=dict(batch=batch,nx=496,ny=41,nz=483),
              compiled_memory_analysis=dict(argument_size_in_bytes=100,output_size_in_bytes=100,
                temp_size_in_bytes=live-200,alias_size_in_bytes=0),
              device_memory_stats_after_execution=dict(peak_bytes_in_use=live+1000),
              run_seconds=[seconds*1.1,seconds,seconds],steady_median_seconds=seconds,
              substeps=50,core_sha256='synthetic_core',runner_sha256='synthetic_runner',
              finite_outputs=dict(state_velocity=True,power_trace=True),
              inputs=dict(common_state=dict(sha256='synthetic_state'),inlet_prefix=dict(sha256='synthetic_inlet')),
              provenance=dict(candidate_module_sha256='synthetic_candidate'),
              comparison=(None if mode=='dense' else {
                 key:dict(max_abs=1e-7,relative_l2=1e-8,finite=True,bitwise_equal=False,
                   **{'allclose_rtol1e-4_atol1e-5':True}) for key in ['state_velocity','power_trace','final_divergence']}))
            save(directory/'RESULT.json',value)
            rows.append(dict(name=name,started_unix=before+index*10,elapsed_seconds=1,exit_code=0))
        save(main/'BRANCH_LEDGER.json',rows)
        save(main/'EXIT.json',dict(status='completed',gpu_hours=5/3600))
        save(pilot/'PROTOCOL.json',dict(budget_gpu_seconds=540))
        save(pilot/'BRANCH_LEDGER.json',[dict(name='dense_b1',elapsed_seconds=4.133981142,exit_code=1)])
        save(pilot/'EXIT.json',dict(status='stopped_branch_failure',gpu_hours=4.133981142/3600))
        created=before+25
        pred=dict(created_utc=datetime.fromtimestamp(created,timezone.utc).isoformat(),
           calibration={str(g):dict(sha256=sha(main/f'stream_g{g}_b1/RESULT.json'),
                 compiled_live_bytes={1:1000,9:1800}[g],steady_seconds={1:81,9:9}[g]) for g in (1,9)},
           heldout_branch_not_started=True,memory_parameters=dict(A=900,B=100),time_parameters=dict(C=0,D=1),
           predicted_stream_g3_b1=dict(compiled_live_bytes=1200,steady_seconds=27),
           predicted_stream_g3_b2=dict(compiled_live_bytes=2400),
           diagnostic_targets=dict(memory_relative_error=.10,time_relative_error=.20))
        save(main/'PROSPECTIVE_PREDICTIONS.json',pred)
        os.utime(main/'PROSPECTIVE_PREDICTIONS.json',(created,created))
        return main,pilot,out

    def test_bad_heldout_model_stays_false_and_failed_pilot_cost_is_retained(self):
        main,pilot,out=self.fixture()
        result=analyze(main,[pilot],out)
        self.assertEqual(result['status'],'complete_with_recorded_diagnostics')
        self.assertTrue(result['prediction_checks']['stream_g1_b1_calibration_file_SHA_matches'])
        self.assertTrue(result['prediction_checks']['declared_prediction_creation_precedes_any_g3_start'])
        self.assertFalse(result['heldout_diagnostics'][0]['memory_target_pass'])
        self.assertFalse(result['heldout_diagnostics'][0]['time_target_pass'])
        self.assertFalse(result['heldout_diagnostics'][1]['memory_target_pass'])
        self.assertIsNone(result['heldout_diagnostics'][1]['time_target_pass'])
        self.assertAlmostEqual(result['budget']['accounted_gpu_seconds'],9.133981142)
        self.assertEqual(result['budget']['rows'][0]['status'],'failed_cost_retained')
        decisions=result['finite_candidate_selector']['decisions']
        self.assertEqual(len(decisions),9)
        self.assertEqual(decisions[1]['selected_branch'],'stream_g9_b1')

    def test_calibration_tamper_and_chronology_failure_are_not_hidden(self):
        main,pilot,out=self.fixture()
        path=main/'stream_g1_b1/RESULT.json';value=json.loads(path.read_text());value['synthetic_tamper']=True;save(path,value)
        predpath=main/'PROSPECTIVE_PREDICTIONS.json';pred=json.loads(predpath.read_text())
        pred['created_utc']=datetime.now(timezone.utc).isoformat();save(predpath,pred)
        result=analyze(main,[pilot],out)
        self.assertFalse(result['prediction_checks']['stream_g1_b1_calibration_file_SHA_matches'])
        self.assertFalse(result['prediction_checks']['declared_prediction_creation_precedes_any_g3_start'])

    def test_missing_main_is_pending_without_invented_rows(self):
        out=self.root/'output';out.mkdir()
        result=analyze(self.root/'not_started',[],out)
        self.assertEqual(result['status'],'pending_main_protocol')
        self.assertNotIn('resource_table',result)
        self.assertTrue(result['no_results_invented'])

    def test_hardware_mismatch_never_selects_candidate(self):
        main,pilot,out=self.fixture()
        path=main/'PROTOCOL.json';proto=json.loads(path.read_text())
        proto['selector_scenarios']=[dict(hardware_identity='OTHER SYNTHETIC CARD',
             capacity_bytes=10**9,reserve_bytes=0,latency_seconds=100,batch=1)]
        save(path,proto)
        result=analyze(main,[pilot],out)
        decision=result['finite_candidate_selector']['decisions'][0]
        self.assertIsNone(decision['selected_branch'])
        self.assertTrue(all('hardware_identity_missing_or_mismatch' in row['reasons'] for row in decision['rejected']))

    def test_generated_input_metadata_is_bound_and_absent_hash_is_not_pass(self):
        main,pilot,out=self.fixture()
        generation=main.parent/'GENERATION_RESULT.json'
        save(generation,dict(seed=81731,scope='SYNTHETIC generation metadata fixture'))
        path=main/'PROTOCOL.json';proto=json.loads(path.read_text())
        proto['input_generation_receipt_sha256']=sha(generation)
        proto.pop('common_state_gzip_sha256');proto.pop('original_inlet_prefix_sha256')
        save(path,proto)
        result=analyze(main,[pilot],out)
        self.assertTrue(result['protocol_checks']['generation_metadata_SHA_matches_protocol'])
        self.assertTrue(result['protocol_checks']['generated_initialization_seed_81731_recorded'])
        self.assertIsNone(result['protocol_checks']['dense_b1_physical_start_hash_matches_protocol'])
        self.assertIsNone(result['protocol_checks']['dense_b1_inlet_hash_matches_protocol'])
        save(generation,dict(seed=81732,scope='SYNTHETIC tamper'))
        result=analyze(main,[pilot],out)
        self.assertFalse(result['protocol_checks']['generation_metadata_SHA_matches_protocol'])
        self.assertFalse(result['protocol_checks']['generated_initialization_seed_81731_recorded'])


if __name__=='__main__':
    unittest.main()
