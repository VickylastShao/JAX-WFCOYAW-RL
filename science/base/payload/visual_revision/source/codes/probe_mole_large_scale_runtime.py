"""Full-grid LES collection and Graph-SAC update smoke; no scientific gain claim."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import (
    ContinuousControlConfig, initialize_continuous_training,
    make_continuous_training_iteration,
)
from jax_mole_graph_sac_v2 import (
    build_wake_graph, graph_v2_actor_sample, graph_v2_sac_update,
    init_graph_v2_sac,
)
from jax_mole_layouts import canonical_scale_layouts


def finite(tree):
    return all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(tree))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--turbines', type=int, choices=(81, 100), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    started = time.monotonic()
    layout = canonical_scale_layouts()[a.turbines]
    case = make_scaled_mole_training_case(layout, batch=1)
    config = ContinuousControlConfig(inlet_mode='time_planes', sample_size=4,
        random_action_transitions=0, minimum_replay_size=1, initial_alpha=0.1)
    inlet = jnp.zeros((1, 3, case.ny, case.nz), jnp.float32).at[:, 0].set(8.0)
    state, _, precursor, farm = initialize_continuous_training(case, config, None,
        seed=20261001 + a.turbines, hidden_dim=256, learning_rate=3e-6,
        replay_capacity=32, initial_inlet=inlet)
    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    sac, optimizers = init_graph_v2_sac(jax.random.PRNGKey(a.turbines),
        state.observation.shape[-1], graph, 256, 3e-6, conservative=True,
        neutral_initial_scale=0.1, initial_alpha=0.1)
    state = state._replace(sac=sac)
    sample = lambda actor, obs, key, bound, mode: graph_v2_actor_sample(
        actor, obs, key, bound, graph)
    collect = jax.jit(make_continuous_training_iteration(case, config, optimizers,
        precursor, farm, updates_per_iteration=0, actor_sample_fn=sample))
    planes = jnp.broadcast_to(inlet[:, None], (1, 50, 3, case.ny, case.nz))
    samples = []
    for index in range(3):
        t = time.monotonic()
        step = state.farm_flow.step_index
        state, metrics = collect(state, planes, step)
        jax.block_until_ready(metrics)
        samples.append(time.monotonic() - t)
        print(json.dumps({'phase': 'collection', 'turbines': a.turbines,
            'interaction': index + 1, 'seconds': samples[-1]}), flush=True)
    def update(sac, replay, key):
        return graph_v2_sac_update(sac, replay, key, optimizers, sample_size=4,
            action_bound=1., gamma=.99, tau=.005, target_entropy=-a.turbines,
            entropy_scale=1. / a.turbines, graph=graph)
    update = jax.jit(update)
    old_actor = jax.tree.map(np.asarray, state.sac.actor)
    learner_started = time.monotonic()
    sac, losses = update(state.sac, state.replay, jax.random.PRNGKey(101))
    jax.block_until_ready(losses)
    learner_seconds = time.monotonic() - learner_started
    state = state._replace(sac=sac)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = a.output.with_suffix('.pkl')
    checkpoint_started = time.monotonic()
    with checkpoint.open('xb') as f:
        pickle.dump(jax.tree.map(np.asarray, state), f, protocol=5)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    with checkpoint.open('rb') as f:
        restored = jax.tree.map(jnp.asarray, pickle.load(f))
    left, lm = collect(state, planes, state.farm_flow.step_index)
    right, rm = collect(restored, planes, restored.farm_flow.step_index)
    jax.block_until_ready((left, right, lm, rm))
    exact_restore = all(np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    actor_changed = any(not np.array_equal(x, np.asarray(y))
        for x, y in zip(jax.tree.leaves(old_actor), jax.tree.leaves(sac.actor)))
    divergence = farm['diagnostic_divergence'](farm['divergence'](left.farm_flow.velocity))
    checks = {'gpu_backend': jax.default_backend() == 'gpu',
        'correct_action_count': left.previous_yaw.shape[-1] == a.turbines,
        'full_grid': list(left.farm_flow.velocity.shape) == [1, 3, case.nx, case.ny, case.nz],
        'finite_flow_learner_and_metrics': finite((left, losses, lm)),
        'actual_actor_updated': actor_changed, 'exact_next_collection_after_restore': exact_restore}
    payload = {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
        'turbines': a.turbines, 'layout_sha256': layout.payload_sha256,
        'jax_version': jax.__version__, 'devices': [str(d) for d in jax.devices()],
        'farm_grid': [case.nx, case.ny, case.nz], 'collection_seconds': samples,
        'learner_compile_and_one_update_seconds': learner_seconds,
        'checkpoint_and_restore_probe_seconds': time.monotonic() - checkpoint_started,
        'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': digest.hexdigest(),
        'full_grid_les_steps_executed': 250, 'learner_updates': int(sac.updates),
        'divergence_rms': float(jnp.sqrt(jnp.mean(divergence ** 2))),
        'memory_stats': jax.devices()[0].memory_stats(),
        'total_seconds': time.monotonic() - started,
        'evidence_boundary': 'Synthetic uniform inlet; real full-grid LES, real Graph-SAC update and full-state restore. Not a learned policy, mature inlet, or control-gain result.'}
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
