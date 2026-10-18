"""Check the profile against the real observation encoder and yaw actuator.

No LES rollout or GPU is needed. The frozen runtime runs these checks on CPU
before admitting the corrected efficiency profiles.
"""
import importlib.util
from types import SimpleNamespace
import unittest


RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None
                        for name in ("jax", "numpy"))


@unittest.skipUnless(RUNTIME_AVAILABLE, "JAX/NumPy runtime required")
class ActualYawObservationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax_mole_continuous_sac import (
            ContinuousControlConfig, apply_control_action, stacked_observation,
        )
        from profile_mole_n81_efficiency import (
            CONTROL_DT_SECONDS, MAX_YAW_DEG, MAX_YAW_SPEED_DEG_PER_S,
            TARGET_YAW_DEG, fixed_yaw_rate_action,
        )
        cls.np, cls.jax = np, jax
        config = ContinuousControlConfig()
        assert config.max_yaw_angle_deg == MAX_YAW_DEG
        assert config.max_yaw_speed_deg_per_s == MAX_YAW_SPEED_DEG_PER_S
        assert config.control_dt_seconds == CONTROL_DT_SECONDS
        case = SimpleNamespace(num_turbines=81)
        frames = jnp.full((2, 1, 77, 81), 6.0, dtype=jnp.float32)
        targets = jnp.where(jnp.arange(81) % 2 == 0,
                            TARGET_YAW_DEG, -TARGET_YAW_DEG)
        cls.targets = np.broadcast_to(np.asarray(targets), (2, 81))

        @jax.jit
        def step(yaw):
            obs = stacked_observation(frames, yaw, case, config)
            action = fixed_yaw_rate_action(obs[..., -81:], targets, clip=jnp.clip)
            return apply_control_action(action, yaw, case, config), action

        @jax.jit
        def legacy_step(yaw):
            obs = stacked_observation(frames, yaw, case, config)
            action = jnp.clip((targets - obs[..., -81:]) / 10.0, -1.0, 1.0)
            return apply_control_action(action, yaw, case, config)

        cls.step, cls.legacy_step = staticmethod(step), staticmethod(legacy_step)

    def test_actual_encoded_observation_and_actuator_reach_and_hold_target(self):
        np = self.np
        for start in (np.zeros((2, 81), dtype=np.float32),
                      -2.0 * self.targets,
                      np.stack((np.linspace(-40, 40, 81),
                                np.linspace(40, -40, 81))).astype(np.float32)):
            yaw = start
            for _ in range(8):
                updated, action = self.step(yaw)
                self.assertLessEqual(float(np.max(np.abs(action))), 1.0)
                self.assertLessEqual(float(np.max(np.abs(np.asarray(updated) - yaw))), 10.00001)
                yaw = np.asarray(updated)
            np.testing.assert_allclose(yaw, self.targets, atol=1e-5)
            held, action = self.step(yaw)
            np.testing.assert_allclose(held, self.targets, atol=1e-5)
            np.testing.assert_allclose(action, 0.0, atol=1e-6)

    def test_original_formula_fails_the_same_physical_contract(self):
        np = self.np
        yaw = np.zeros((2, 81), dtype=np.float32)
        for _ in range(8):
            yaw = np.asarray(self.legacy_step(yaw))
        np.testing.assert_allclose(yaw, 2.0 * self.targets, atol=1e-5)
        self.assertFalse(np.allclose(yaw, self.targets, atol=1e-5))


if __name__ == "__main__":
    if not RUNTIME_AVAILABLE:
        raise SystemExit("Actual-runtime admission requires JAX and NumPy; skipped checks cannot admit GPUs")
    unittest.main()
