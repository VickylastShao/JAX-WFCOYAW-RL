"""Negative cases for scientific/engineering stage admission, CPU only."""
import copy
import unittest

from run_mole_large_scale_source_to_training import assess_engineering, engineering_argv, estimate_formal_budget


def fixtures():
    rows = [{'wallclock_seconds': 10., 'physical_reset_interactions_per_environment': 0,
             'sac_actor_gradient_norm': .2, 'mean_abs_yaw_deg': .1} for _ in range(11)]
    rows.append(dict(rows[-1], wallclock_seconds=160., physical_reset_interactions_per_environment=150))
    fresh = {'status':'pass', 'configuration':{'total_global_transitions_including_compile_warmup':512},
             'checks':{'finite':True},'timing':{'total_process_wallclock_seconds':100.}}
    resumed = {'status':'pass','configuration':{'total_global_transitions_including_compile_warmup':16640,
               'resumed_global_interactions':512,'resumed_from_full_checkpoint':'fresh.pkl',
               'sac_actor_updates_applied_total':1,'replay_capacity_per_host':262144,'local_environments':8},
               'history':rows,'checks':{'finite':True},'rolling_checkpoint_records':[{'checkpoint_io_seconds':2.}]}
    return [(copy.deepcopy(fresh),copy.deepcopy(resumed)) for _ in range(4)]


class WorkflowAdmissionTest(unittest.TestCase):
    def test_missing_reset_cannot_pass(self):
        data=fixtures()
        for row in data[2][1]['history']:row['physical_reset_interactions_per_environment']=0
        self.assertFalse(assess_engineering(data)['rank2_reset_exercised'])

    def test_wrong_resume_counter_cannot_pass(self):
        data=fixtures();data[1][1]['configuration']['resumed_global_interactions']=0
        self.assertFalse(assess_engineering(data)['rank1_full_resume'])

    def test_missing_measured_checkpoint_cost_rejected(self):
        data=fixtures();data[0][1]['rolling_checkpoint_records']=[]
        with self.assertRaises(ValueError):estimate_formal_budget(data)

    def test_engineering_resume_keeps_original_input_checkpoint(self):
        argv=engineering_argv(['python','trainer.py','--save-initial-checkpoint'],2,True)
        self.assertNotIn('--save-initial-checkpoint',argv)
        self.assertIn('/batch/engineering/n81/fresh_full_state.process2.pkl',argv)
        self.assertIn('/batch/engineering/n81/resume_full_state.pkl',argv)


if __name__=='__main__':unittest.main()
