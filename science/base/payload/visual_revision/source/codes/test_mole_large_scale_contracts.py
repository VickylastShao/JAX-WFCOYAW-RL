"""CPU-safe active Graph-SAC/checkpoint/accounting contracts; no LES gain claim."""
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_layouts import canonical_scale_layouts
from jax_mole_graph_sac_v2 import build_wake_graph, init_graph_v2_sac, graph_v2_actor_deterministic
from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import ContinuousControlConfig, save_actor_checkpoint, load_actor_checkpoint
from jax_mole_microbatch_distributed import _save_full_checkpoint_atomic, _load_full_checkpoint, FULL_CHECKPOINT_SCHEMA_VERSION
from mole_large_scale_metrics import score_trace, paired_gain
from mole_large_scale_evaluation import scoring_boundary, validate_actor_metadata


def main():
    checks = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for n, edges in ((81, 324), (100, 450)):
            layout = canonical_scale_layouts()[n]
            graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
            sac, _ = init_graph_v2_sac(jax.random.PRNGKey(n), n * 78, graph, 8, 3e-4,
                                     conservative=True, neutral_initial_scale=0.1, initial_alpha=0.1)
            observations = jnp.zeros((1, n * 78), dtype=jnp.float32)
            expected = np.asarray(graph_v2_actor_deterministic(sac.actor, observations, graph))
            case = make_scaled_mole_training_case(layout, 1)
            path = root / f'actor_n{n}_step0.pkl'
            save_actor_checkpoint(path, SimpleNamespace(sac=sac),
                                  {'policy_architecture': 'graph_sac_v2',
                                   'action_parameterization': 'per_turbine',
                                   'actual_global_interactions': 0,
                                   'learner_observation_dim': n * 78,
                                   'compact_solver': case.compact_solver,
                                   'reflection_equivariant_actor': False}, case=case)
            loaded, metadata = load_actor_checkpoint(path, expected_case=case,
                                                      expected_action_parameterization='per_turbine')
            actual = np.asarray(graph_v2_actor_deterministic(loaded, observations, graph))
            config = ContinuousControlConfig(inlet_mode='time_planes')
            validate_actor_metadata(metadata, case, config, 0)
            for field, unsupported in (('reflection_equivariant_actor', True),
                                       ('learner_observation_dim', n * (77 * 2 + 1)),
                                       ('compact_solver', 'cute_fused_41')):
                try:
                    validate_actor_metadata({**metadata, field: unsupported}, case, config, 0)
                except ValueError:
                    checks[f'n{n}_{field}_mismatch_rejected'] = True
                else:
                    checks[f'n{n}_{field}_mismatch_rejected'] = False
            checks[f'n{n}_active_graph_edges'] = len(graph.edge_sources) == edges
            checks[f'n{n}_actor_roundtrip_and_action_count'] = actual.shape == (1, n) and np.array_equal(actual, expected)
            checks[f'n{n}_zero_mean_initial_actor'] = bool(np.all(actual == 0))
            wrong = make_scaled_mole_training_case(canonical_scale_layouts()[100 if n == 81 else 81], 1)
            try:
                load_actor_checkpoint(path, expected_case=wrong)
            except ValueError:
                checks[f'n{n}_wrong_layout_rejected'] = True
            else:
                checks[f'n{n}_wrong_layout_rejected'] = False
        full = root / 'full_rolling0.process0.pkl'
        payload = {'schema_version': FULL_CHECKPOINT_SCHEMA_VERSION,
                   'global_interactions': 2048, 'plane_cursors': [10000, 11000],
                   'actor_states': {'rng': np.array([3, 8], dtype=np.uint32)},
                   'learner': {'replay': np.arange(12, dtype=np.float32)}}
        sha = _save_full_checkpoint_atomic(full, payload)
        restored = _load_full_checkpoint(full)
        checks['atomic_full_state_cursors_and_rng'] = (len(sha) == 64 and restored['plane_cursors'] == payload['plane_cursors']
                                                      and np.array_equal(restored['actor_states']['rng'], payload['actor_states']['rng']))
        # Failure during serialization must leave the previous recoverable file.
        original = full.read_bytes()
        try:
            _save_full_checkpoint_atomic(full, {**payload, 'invalid': lambda: None})
        except (pickle.PicklingError, AttributeError):
            checks['interrupted_write_preserves_previous_checkpoint'] = full.read_bytes() == original
        else:
            checks['interrupted_write_preserves_previous_checkpoint'] = False
        power = np.ones((2, 1, 3), np.float32)
        zero = score_trace({'farm_power_mw': power.sum(-1), 'turbine_power_mw': power,
                            'yaw_deg': np.zeros_like(power)}, np.zeros((1, 3)))
        yaw = np.asarray([[[10., -10., 0.]], [[20., -20., 0.]]], np.float32)
        rl = score_trace({'farm_power_mw': 1.05*power.sum(-1), 'turbine_power_mw': 1.05*power,
                          'yaw_deg': yaw}, np.zeros((1, 3)))
        checks['paired_energy_and_first_move_accounting'] = (abs(paired_gain(rl, zero)['generation_gain_percent'] - 5) < 1e-4
                                                            and rl['yaw_travel_deg_per_turbine'] == [20., 20., 0.]
                                                            and rl['peak_yaw_rate_deg_s'] == 1.)
        bad = dict(rl, controls=1)
        try:
            paired_gain(bad, zero)
        except ValueError:
            checks['unequal_pair_duration_rejected'] = True
        else:
            checks['unequal_pair_duration_rejected'] = False
        common = SimpleNamespace(observation_frames=jnp.ones((1, 1, 2)), observation=jnp.ones((1, 3)),
                                 previous_yaw=jnp.ones((1, 1)), interaction_index=jnp.asarray(150))
        common._replace = lambda **values: SimpleNamespace(**values)
        boundary = scoring_boundary(common)
        checks['explicit_training_compatible_cache_boundary'] = (int(boundary.interaction_index) == 0
                                                                 and not np.asarray(boundary.observation).any())
    assert all(checks.values()), checks
    print(json.dumps({'status': 'pass', 'checks': checks,
                      'scope': 'CPU contracts; no real large-scale turbulent training or gain measured.'}))


if __name__ == '__main__':
    main()
