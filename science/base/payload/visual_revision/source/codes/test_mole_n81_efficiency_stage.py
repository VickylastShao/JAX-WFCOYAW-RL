"""Stdlib-only budget, admission and terminal-state checks; no Docker or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import run_mole_n81_efficiency_stage as module


def plan_fixture():
    return {'task_id': 'fixture-owned-efficiency', 'initial_stage_gpu_hours_limit': 15.,
            'wall_seconds_limit': 100000., 'minimum_free_gib': 100.,
            'sources': [{'source_id': 'A', 'manifest_sha256': 'immutable-A-hash',
                         'archive_name': 'n81_train_A'}],
            'efficiency': {'gpu_hours_limit': 15., 'execution_order': [4, 2, 1],
                           'compute_protocol': {
                               'topologies': [1, 2, 4], 'topology_gpu_hour_cap': 5.,
                               'E': 32, 'global_sample': 256, 'replay': 262144,
                               'microbatch': 1, 'episode': 500,
                               'physical_reset_controls': 150, 'initial_burnin': 0,
                               'synthetic_initial_episode_clock': 404,
                               'controls_per_environment': 96,
                               'real_LES_steps_per_environment': 12300,
                               'real_SAC_updates': 3072}}}


def protocol_fixture(plan):
    work = plan['efficiency']['compute_protocol']
    protocol = {value: work[key] for key, value in module.WORK_FIELDS.items()}
    protocol.update(topologies=[1, 2, 4], total_gpu_hour_cap=15.,
                    topology_gpu_hour_cap=5., checkpoint_save_reserve_seconds=600,
                    global_collected_interactions=3072, collection_steps=8,
                    source_archive='/inputs/archives/n81_train_A',
                    source_manifest_sha256='immutable-A-hash', ranks=[])
    for topology in (1, 2, 4):
        for rank in range(topology):
            flags = {'--stop-after-seconds': 18000 / topology - 600,
                     '--local-environments': 32 // topology, '--sample-size': 256 // topology,
                     '--num-processes': topology, '--process-id': rank,
                     '--microbatch': 1, '--replay-capacity': 262144,
                     '--collection-steps': 8, '--episode-interactions': 500,
                     '--reset-interactions': 150, '--burnin-interactions': 0,
                     '--planned-global-interactions': 3072}
            argv = ['python', '-u', 'fixture_trainer.py']
            for name, value in flags.items():
                argv.extend((name, str(value)))
            protocol['ranks'].append({'topology': topology, 'rank': rank, 'gpu': rank,
                                      'hard_wall_seconds': 18000 / topology,
                                      'trainer_argv': argv})
    return protocol


def stage_fixture(batch):
    stage = module.Efficiency.__new__(module.Efficiency)
    stage.batch, stage.plan = batch, plan_fixture()
    stage.task = stage.plan['task_id']
    stage.started = stage.accounted = 0.
    stage.active, stage.gpu_seconds = 4, 4.4 * 3600
    stage.phase = 'matched_compute_profiles'
    stage.topology, stage.topology_started_gpu_seconds = 4, 0.
    stage.total_gpu_hour_cap, stage.topology_gpu_hour_cap, stage.execution_order = module.plan_settings(stage.plan)
    stage.jobs, stage.cpu_jobs = {}, {}
    stage.cancel, stage.lock = threading.Event(), threading.Lock()
    return stage


class EfficiencyStageContractTest(unittest.TestCase):
    def test_order_does_not_change_matched_topologies_or_work(self):
        plan = plan_fixture()
        protocol = protocol_fixture(plan)
        self.assertEqual(module.validate_protocol(plan, protocol), (4, 2, 1))
        self.assertEqual(protocol['topologies'], [1, 2, 4])
        for order in ([4, 4, 1], [4, 2], [4, 2, 3], [4, 2, True]):
            invalid = copy.deepcopy(plan)
            invalid['efficiency']['execution_order'] = order
            with self.assertRaises(ValueError):
                module.plan_settings(invalid)

    def test_prepare_rejects_budget_work_and_source_drift_before_launch(self):
        plan = plan_fixture()
        protocol = protocol_fixture(plan)
        for field, bad in [('topology_gpu_hour_cap', 4.), ('total_gpu_hour_cap', 12.),
                           ('global_environments', 16), ('global_learner_updates', 0),
                           ('source_manifest_sha256', 'different-source'),
                           ('source_archive', '/inputs/archives/n81_heldout_B')]:
            invalid = copy.deepcopy(protocol)
            invalid[field] = bad
            with self.assertRaises(ValueError):
                module.validate_protocol(plan, invalid)
        invalid = copy.deepcopy(protocol)
        invalid['ranks'][-1]['hard_wall_seconds'] = 3600
        with self.assertRaises(ValueError):
            module.validate_protocol(plan, invalid)
        invalid = copy.deepcopy(protocol)
        argv = invalid['ranks'][-1]['trainer_argv']
        argv[argv.index('--stop-after-seconds') + 1] = '4380'
        with self.assertRaises(ValueError):
            module.validate_protocol(plan, invalid)

    def test_topology_cancels_at_frozen_five_gpu_hours_not_four(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = stage_fixture(Path(temporary))
            with mock.patch.object(module.time, 'monotonic', return_value=10.):
                stage.status('running')
            self.assertFalse(stage.cancel.is_set())
            self.assertAlmostEqual(stage.gpu_seconds, 4.4 * 3600 + 40.)
            stage.gpu_seconds = 5 * 3600 - 40.
            with mock.patch.object(module.time, 'monotonic', return_value=20.):
                stage.status('running')
            self.assertTrue(stage.cancel.is_set())
            state = json.loads((stage.batch / 'STATUS.json').read_text())
            self.assertEqual(state['allocated_gpu_hours'], 5.)
            self.assertIn('topology', state['budget_stop_reason'])
            self.assertFalse(state['heldout_evaluation'])

    def test_total_budget_still_cancels_outside_topology_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = stage_fixture(Path(temporary))
            stage.phase = 'efficiency_summary'
            stage.gpu_seconds = 15 * 3600
            stage.active = 0
            with mock.patch.object(module.time, 'monotonic', return_value=10.):
                stage.status('running')
            self.assertTrue(stage.cancel.is_set())

    def test_failure_is_terminal_and_preserves_rank_exit_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            stage = stage_fixture(batch)
            stage.active = 0
            stage.jobs = {'p4-rank0': {'status': 'stopped_or_failed', 'exit_code': 143,
                                     'container_id': 'only-owned-container'}}
            evidence = batch / 'failed-checkpoint.pkl'
            evidence.write_bytes(b'preserve unfinished profile state')
            with mock.patch.object(module.time, 'monotonic', return_value=10.):
                module.write_failure_status(batch, stage, RuntimeError('owned rank exit 143'))
            result = json.loads((batch / 'STATUS.json').read_text())
            self.assertTrue(result['terminal'])
            self.assertEqual(result['status'], 'failed_or_budget_stopped')
            self.assertEqual(result['jobs']['p4-rank0']['exit_code'], 143)
            self.assertEqual(result['active_containers'], 0)
            self.assertEqual(evidence.read_bytes(), b'preserve unfinished profile state')

    def test_main_failed_exit_agrees_with_terminal_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            stage = stage_fixture(batch)
            stage.active = 0
            stage.execute = mock.Mock(side_effect=ValueError('prepared work mismatch'))
            args = ['fixture-stage', '--batch', str(batch), '--legacy-root', '/unused-readonly']
            with mock.patch.object(sys, 'argv', args), mock.patch.object(module, 'Efficiency', return_value=stage), \
                    mock.patch.object(module.time, 'monotonic', return_value=10.):
                with self.assertRaises(ValueError):
                    module.main()
            self.assertEqual((batch / 'supervisor.exit').read_text(), '1\n')
            self.assertEqual(json.loads((batch / 'WORKFLOW_RESULT.json').read_text())['status'],
                             'failed_or_budget_stopped')
            self.assertTrue(json.loads((batch / 'STATUS.json').read_text())['terminal'])

    def test_disk_reserve_is_checked_before_dependency_or_gpu_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = stage_fixture(Path(temporary))
            with mock.patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=99 * 1024**3)):
                with self.assertRaisesRegex(ValueError, 'disk reserve'):
                    stage.execute()


if __name__ == '__main__':
    unittest.main()
