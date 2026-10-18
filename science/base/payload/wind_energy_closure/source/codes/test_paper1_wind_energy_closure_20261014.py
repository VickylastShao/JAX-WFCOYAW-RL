"""Meaningful conservation quadrature and diagnostic filter-order regressions."""
import unittest
from dataclasses import replace
import numpy as np
from paper1_wind_energy_closure_20261014 import face_flux, filter_stride_builder


class FluxTests(unittest.TestCase):
    def test_periodic_last_sample_is_not_outward_face(self):
        v = np.zeros((1, 3, 7, 5, 8), np.float32)
        v[:, 0] = 8.
        v[:, 2] = np.sin(np.arange(8) * 2*np.pi/8)[None, None, None, :]
        result = face_flux(v, (6., 4., 8.))
        self.assertAlmostEqual(result['inlet_m3_s'][0], 256.)
        self.assertEqual(result['net_outward_m3_s'], [0.])

    def test_all_open_faces_are_accounted(self):
        v = np.zeros((1, 3, 7, 5, 8), np.float32)
        v[:, 0] = 8.
        v[:, 0, -1] = 9.
        v[:, 1, :, -1] = -2./3.
        result = face_flux(v, (6., 4., 8.))
        self.assertAlmostEqual(result['net_outward_m3_s'][0], 0., places=5)

    def test_equal_arithmetic_means_do_not_establish_integrated_flux(self):
        v = np.zeros((1, 3, 7, 5, 8), np.float32)
        v[:, 0] = 8.
        v[:, 0, -1, 0] += 1.
        v[:, 0, -1, 1] -= 1.
        self.assertEqual(v[:, 0, 0].mean(), v[:, 0, -1].mean())
        self.assertNotEqual(face_flux(v)['net_outward_m3_s'][0], 0.)


class FilterTests(unittest.TestCase):
    def test_matching_physical_cadence_preserves_filter_order(self):
        import jax
        import jax.numpy as jnp
        from jax_mole_compact_sac_benchmark import (
            build_flow_functions, make_scaled_mole_training_case, make_initial_flow)
        from jax_mole_layouts import canonical_scale_layouts
        self.assertIs(filter_stride_builder(build_flow_functions, 1), build_flow_functions)
        with self.assertRaises(ValueError):
            filter_stride_builder(build_flow_functions, 3)
        case = replace(make_scaled_mole_training_case(canonical_scale_layouts()[3], 1,
              compact_solver='dense_inverse', rhs_assembly='tensor'),
              nx=64, ny=9, nz=32, dt=.1)
        native = build_flow_functions(case, periodic_x=False, include_turbines=False)
        matching = filter_stride_builder(build_flow_functions, 2)(case, periodic_x=False, include_turbines=False)
        unfiltered = build_flow_functions(replace(case, spatial_filter=False), periodic_x=False, include_turbines=False)
        initial = make_initial_flow(case, native, 734)
        yaw = jnp.zeros((1, 3)); weights = jnp.zeros((1,))
        inlet = initial.velocity[:, :, 0]
        for step_index, reference in [(2, native), (3, unfiltered)]:
            state = initial._replace(step_index=jnp.asarray(step_index, jnp.int32))
            a = jax.jit(matching['advance'])(state, yaw, weights, inlet)
            b = jax.jit(reference['advance'])(state, yaw, weights, inlet)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
