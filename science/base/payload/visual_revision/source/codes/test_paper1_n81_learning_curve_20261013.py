"""CPU regressions for diagnostic milestone identity and unchanged final gates."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import paper1_n81_learning_curve_20261013 as curve


def actor_fixture():
    def linear(n, m):
        return {'w': np.zeros((n, m), np.float32), 'b': np.zeros((m,), np.float32)}
    return {'body': {'encoder': linear(82, 256),
                     'layers': tuple({'self': linear(256, 256), 'message': linear(259, 256)} for _ in range(2))},
            'head': linear(256, 2), 'neutral_scale_contract': ()}


class LearningCurveContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.layout = {'name': 'test-fixture-layout', 'num_turbines': 81}
        self.metadata = {
            'policy_architecture': 'graph_sac_v2', 'training_algorithm': 'graph_sac_v2',
            'actual_global_interactions': 131072, 'target_turbines': 81,
            'action_dim': 81, 'action_parameterization': 'per_turbine',
            'graph_geometry': 'canonical', 'inlet_mode': 'time_planes',
            'episode_interactions': 500, 'reset_interactions': 150,
            'source_precursor_sha256': curve.SOURCE_STATE_SHA256,
            'plane_archive_sha256': curve.ARCHIVE_SHA256,
            'direction_condition': False, 'steering_branch_condition': False,
            'initial_actor_checkpoint_sha256': None,
            'layout_contract': self.layout, 'seed': curve.OPTIMIZATION_SEED,
        }

    def tearDown(self):
        self.directory.cleanup()

    def load(self, metadata=None, actor=None, node=131072, digest=None):
        path = self.root / 'actor.pkl'
        with path.open('wb') as stream:
            pickle.dump({'schema_version': 1, 'actor': actor_fixture() if actor is None else actor,
                         'metadata': self.metadata if metadata is None else metadata}, stream)
        return curve.load_diagnostic_actor(path, curve.sha256_file(path) if digest is None else digest,
                                           node, self.layout)

    def test_sparse_metadata_is_bound_but_never_rewritten(self):
        actor, receipt = self.load()
        self.assertEqual(len(receipt['parameter_shapes']), 12)
        self.assertNotIn('learner_observation_dim', receipt['checkpoint_metadata'])
        self.assertNotIn('compact_solver', receipt['checkpoint_metadata'])
        self.assertFalse(receipt['derived_diagnostic_contract']['checkpoint_metadata_was_modified'])
        self.assertEqual(actor['head']['w'].shape, (256, 2))

    def test_hash_mismatch_rejected_before_unpickle(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.load(digest='0' * 64)

    def test_identity_source_seed_and_node_mismatches_rejected(self):
        for field, value in (('seed', 17), ('actual_global_interactions', 524288),
                             ('source_precursor_sha256', 'x'), ('graph_geometry', 'changed'),
                             ('direction_condition', True), ('action_parameterization', 'target'),
                             ('compact_solver', 'cute_fused_41'), ('learner_observation_dim', 123)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.load(metadata={**self.metadata, field: value})

    def test_shape_dtype_nonfinite_and_feedback_rejected(self):
        bad = actor_fixture()
        bad['body']['encoder']['w'] = np.zeros((81, 256), np.float32)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            self.load(actor=bad)
        bad = actor_fixture()
        bad['head']['w'] = bad['head']['w'].astype(np.float64)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            self.load(actor=bad)
        bad = actor_fixture()
        bad['body']['layers'][1]['message']['w'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.load(actor=bad)
        bad = actor_fixture()
        bad['feedback_self'] = {}
        with self.assertRaisesRegex(ValueError, 'branch/structure'):
            self.load(actor=bad)

    def test_initial_node_requires_mathematically_zero_mean_head(self):
        metadata = {**self.metadata, 'actual_global_interactions': 0}
        _, receipt = self.load(metadata=metadata, node=0)
        self.assertTrue(receipt['initial_zero_equivalence'])
        bad = actor_fixture()
        bad['head']['b'][0] = .01
        with self.assertRaisesRegex(ValueError, 'identically zero'):
            self.load(metadata=metadata, actor=bad, node=0)

    def test_formal_report_binds_milestone_references_and_solver(self):
        config = {
            'target_turbines': 81, 'training_algorithm': 'graph_sac_v2',
            'action_parameterization': 'per_turbine', 'action_dim': 81,
            'hidden_dim': 256, 'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor',
            'inlet_mode': 'time_planes', 'direction_condition': False,
            'steering_branch_condition': False, 'effective_reflection_equivariant_actor': False,
            'episode_interactions': 500, 'reset_interactions': 150,
            'total_global_transitions_including_compile_warmup': 1000448,
            'source_precursor_sha256': curve.SOURCE_STATE_SHA256,
            'plane_archive_sha256': curve.ARCHIVE_SHA256,
            'initial_actor_checkpoint': None, 'conservative_actor_init': True,
            'terminated_early_nonfinite': False, 'layout_contract': self.layout,
        }
        report = {'status': 'pass', 'configuration': config, 'checkpoint': {
            'path_process0_only': '/runs/actor.pkl', 'initial': {'actor_process0_only': '/runs/actor_step0.pkl'},
            'milestones_process0_only': [
                {'actual_global_interactions': node, 'actor_process0_only': '/runs/' + curve.ACTOR_IDENTITIES[node][0]}
                for node in (131072, 524288)],
        }}
        curve.validate_formal_report(report, self.layout)
        for changed in (copy.deepcopy(report), copy.deepcopy(report)):
            changed['configuration']['rhs_assembly'] = 'componentwise'
            with self.assertRaisesRegex(ValueError, 'rhs_assembly'):
                curve.validate_formal_report(changed, self.layout)
        changed = copy.deepcopy(report)
        changed['checkpoint']['milestones_process0_only'][0]['actor_process0_only'] = '/runs/wrong.pkl'
        with self.assertRaisesRegex(ValueError, 'milestone_131072'):
            curve.validate_formal_report(changed, self.layout)

    def test_cpu_active_graph_zero_proof_and_final_validator_stays_strict(self):
        # Real policy code, without constructing an LES flow or using a GPU.
        import jax
        import jax.numpy as jnp
        from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
        from jax_mole_continuous_sac import ContinuousControlConfig
        from jax_mole_graph_sac_v2 import build_wake_graph, graph_v2_actor_deterministic
        from jax_mole_layouts import canonical_scale_layouts
        from mole_large_scale_evaluation import validate_actor_metadata
        self.assertEqual(jax.default_backend(), 'cpu', 'run contracts with JAX_PLATFORMS=cpu')
        layout = canonical_scale_layouts()[81]
        graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
        actor = jax.tree.map(jnp.asarray, actor_fixture())
        observations = jax.random.normal(jax.random.PRNGKey(21), (3, 6318))
        actions = np.asarray(graph_v2_actor_deterministic(actor, observations, graph))
        self.assertEqual(actions.shape, (3, 81))
        np.testing.assert_array_equal(actions, 0.)
        case = make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse', rhs_assembly='tensor')
        with self.assertRaisesRegex(ValueError, 'observation width'):
            validate_actor_metadata({'policy_architecture': 'graph_sac_v2', 'actual_global_interactions': 1000448},
                                    case, ContinuousControlConfig(inlet_mode='time_planes'), 1000448)


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2).result
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'status': 'pass', 'tests': result.testsRun, 'backend': 'CPU',
                      'scope': 'checkpoint and policy contracts; no LES gain or runtime speed measured'}))
