"""CPU-only temporal/inlet/reuse regression contracts; no LES execution."""
import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from paper1_n81_timestep_sensitivity_20261013 import (
    temporal_contract, sample_frozen_inlet, physical_control_config,
    validate_cold_state, validate_reference, FINAL_ACTOR_SHA256,
)


def actual_config_class_without_jax():
    # Load the real declaration, not a hand-written replica. The host's local
    # NumPy runtime has no JAX; only GPU runtime uses the full module imports.
    tree = ast.parse(Path(__file__).with_name('jax_mole_continuous_sac.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'ContinuousControlConfig')
    namespace = {'dataclass': dataclass}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-control-config>', 'exec'), namespace)
    return namespace['ContinuousControlConfig']


class PhysicalTimeContracts(unittest.TestCase):
    def test_true_seconds_override_real_legacy_class(self):
        base = actual_config_class_without_jax()
        self.assertEqual(base(les_steps_per_interaction=100).control_dt_seconds, 20.)
        coarse, fine = (physical_control_config(dt, base) for dt in (.2, .1))
        self.assertEqual((coarse.les_steps_per_interaction, fine.les_steps_per_interaction), (50, 100))
        for config in (coarse, fine):
            self.assertEqual(config.control_dt_seconds, 10.)
            self.assertEqual(config.max_yaw_speed_deg_per_s * config.control_dt_seconds, 10.)
            self.assertEqual(config.max_yaw_angle_deg, 40.)
            self.assertEqual(config.frame_stack, 1)

    def test_physical_cursor_is_independent_of_les_timestep(self):
        left, right = temporal_contract(.2), temporal_contract(.1)
        for key in ('plane_start', 'scoring_plane_cursor', 'final_plane_cursor_exclusive',
                    'required_archive_end_exclusive', 'burnin_physical_seconds', 'scoring_physical_seconds'):
            self.assertEqual(left[key], right[key])
        self.assertEqual(left['scoring_plane_cursor'], 27500)
        self.assertEqual(left['final_plane_cursor_exclusive'], 47500)
        self.assertEqual(right['required_archive_end_exclusive'], 47501)
        self.assertNotEqual(left['native_filter_cadence_seconds'], right['native_filter_cadence_seconds'])
        for dt in (0., .05, .3):
            with self.assertRaises(ValueError):
                temporal_contract(dt)

    def test_same_nonlinear_knots_and_all_component_midpoints(self):
        knots = np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2)
        knots[1] *= 2
        coarse, fine = sample_frozen_inlet(knots, .2), sample_frozen_inlet(knots, .1)
        np.testing.assert_array_equal(coarse, knots[:-1])
        np.testing.assert_array_equal(fine[::2], coarse)
        np.testing.assert_array_equal(fine[1::2], .5 * (knots[:-1] + knots[1:]))
        self.assertEqual(fine.shape[0], 2 * coarse.shape[0])

    def test_block_overlap_has_no_duplicate_or_missing_time(self):
        knots = (np.arange(21, dtype=np.float32) ** 2)[:, None, None, None]
        knots = np.repeat(knots, 3, axis=1)
        for dt in (.2, .1):
            full = sample_frozen_inlet(knots, dt)
            split = np.concatenate([sample_frozen_inlet(knots[:9], dt),
                                    sample_frozen_inlet(knots[8:17], dt),
                                    sample_frozen_inlet(knots[16:], dt)])
            np.testing.assert_array_equal(split, full)

    def test_missing_endpoint_bad_dtype_and_nonfinite_are_rejected(self):
        for values in (np.zeros((1, 3, 2, 2), np.float32),
                       np.zeros((2, 3, 2, 2), np.float64),
                       np.full((2, 3, 2, 2), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                sample_frozen_inlet(values, .1)

    def test_warm_pressure_history_or_les_clock_is_rejected(self):
        flow = SimpleNamespace(velocity=np.ones((1, 3, 2, 2, 2), np.float32),
                               rhs_previous=np.ones((1, 3, 2, 2, 2), np.float32),
                               rhs_previous_2=np.ones((1, 3, 2, 2, 2), np.float32),
                               filtered_disk_speed=np.zeros((1, 81), np.float32),
                               bottom_pressure_gradient=np.zeros((1, 2, 2, 2), np.float32),
                               open_x_pressure_gradient=np.zeros((1, 2, 2, 2, 2), np.float32),
                               step_index=np.asarray(0, np.int32))
        flow._fields = tuple(key for key in vars(flow) if key != '_fields')
        state = SimpleNamespace(farm_flow=flow, previous_yaw=np.zeros((1, 81), np.float32),
                                interaction_index=np.asarray(0, np.int32))
        self.assertTrue(validate_cold_state(state)['velocity']['finite'])
        for field, value in (('step_index', np.asarray(7500)),
                             ('open_x_pressure_gradient', np.ones((1, 2, 2, 2, 2)))):
            bad = copy.deepcopy(state)
            setattr(bad.farm_flow, field, value)
            with self.assertRaises(ValueError):
                validate_cold_state(bad)


class ReferenceReuseContracts(unittest.TestCase):
    def reference(self, controls=400):
        record = {'common_state_sha256': 'common', 'scoring_cursor': 27500,
                  'final_cursor_exclusive': 27500 + controls * 50,
                  'trace': '/former/pilot_A/trace.npz', 'trace_sha256': 'trace'}
        return {'status': 'complete', 'turbines': 81, 'layout_sha256': 'layout',
                'archive_manifest_sha256': 'archive', 'common_state_sha256': 'common',
                'window_start': 20000, 'source_state': {'sha256': 'source'},
                'evaluation_numerics': {'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor'},
                'common_burnin': {'controls': 150, 'duration_seconds': 1500.,
                                 'initial_plane_cursor': 20000, 'final_plane_cursor_exclusive': 27500,
                                 'final_farm_step_index': 7500},
                'scores': {'zero': {'controls': controls, 'duration_seconds': controls * 10.},
                           'rl_r1': {'controls': controls, 'duration_seconds': controls * 10.}},
                'controllers': {'zero': {**record, 'actor': None},
                                'rl_r1': {**record, 'actor': {'sha256': FINAL_ACTOR_SHA256}}}}

    def validate(self, result):
        return validate_reference(result, common_hash='common', layout_hash='layout',
                                  archive_hash='archive', extension_hash=None,
                                  source_hash='source', actor_hash=FINAL_ACTOR_SHA256)

    def test_400_reference_or_fixed_2000_prefix_can_be_bound(self):
        for controls in (400, 2000):
            self.assertEqual(set(self.validate(self.reference(controls))), {'zero', 'rl_final'})

    def test_same_path_is_not_seed_numerics_or_state_evidence(self):
        mutations = [('common_state_sha256', 'other-initial-seed'),
                     ('source_state', {'sha256': 'other'}),
                     ('evaluation_numerics', {'compact_solver': 'cute_fused_41', 'rhs_assembly': 'tensor'})]
        for key, value in mutations:
            result = self.reference()
            result[key] = value
            with self.assertRaises(ValueError):
                self.validate(result)

    def test_wrong_actor_horizon_and_pressure_step_are_rejected(self):
        result = self.reference()
        result['controllers']['rl_r1']['actor']['sha256'] = 'otheractor'
        with self.assertRaises(ValueError):
            self.validate(result)
        result = self.reference(399)
        with self.assertRaises(ValueError):
            self.validate(result)
        result = self.reference()
        result['common_burnin']['final_farm_step_index'] = 15000
        with self.assertRaises(ValueError):
            self.validate(result)


if __name__ == '__main__':
    unittest.main()
