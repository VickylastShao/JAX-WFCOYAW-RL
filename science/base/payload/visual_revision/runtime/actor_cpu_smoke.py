#!/usr/bin/env python3
"""Verify trusted, hash-bound Actors on CPU; no flow field, Replay or LES allocation."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'source/codes'))
from resolve_run import source_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frozen_identity = source_checks()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
    from jax_mole_continuous_sac import ContinuousControlConfig, load_actor_checkpoint
    from jax_mole_graph_sac_v2 import build_wake_graph, graph_v2_actor_deterministic
    from jax_mole_layouts import canonical_scale_layouts
    from mole_large_scale_evaluation import validate_actor_metadata
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Actor smoke must use CPU')
    layout = canonical_scale_layouts()[81]
    if layout.payload_sha256 != '67a41a873c604be2b7fce5d3ded7f26edd105d0e7b5c02a75b35876d3f2d841f':
        raise ValueError('layout identity mismatch')
    case = make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse', rhs_assembly='tensor')
    config = ContinuousControlConfig(inlet_mode='time_planes')
    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    width = 81 * (77 * config.frame_stack + 1)
    probe_inputs = [jnp.zeros((1, width), dtype=jnp.float32),
                    jnp.linspace(-1, 1, width, dtype=jnp.float32)[None, :]]
    manifest = json.loads((PACK / 'actors/ACTOR_MANIFEST.json').read_text())
    results = []
    for row in manifest['files']:
        path = PACK / 'actors' / row['path']
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('trusted Actor bytes do not match manifest')
        actor, meta = load_actor_checkpoint(path, expected_case=case,
                                            expected_action_parameterization='per_turbine')
        expected_step = {'actor.pkl': 1000448, 'actor_step0.pkl': 0,
                         'actor_step131072.pkl': 131072, 'actor_step524288.pkl': 524288}[row['path']]
        if expected_step == 1000448:
            validate_actor_metadata(meta, case, config, expected_step)
        else:
            # Frozen milestone exports omit later-added evaluator metadata. Do not
            # alter trusted bytes or fill missing keys and pretend full validation.
            for key, expected in {'policy_architecture':'graph_sac_v2',
                                  'actual_global_interactions':expected_step,
                                  'target_turbines':81, 'action_dim':81,
                                  'action_parameterization':'per_turbine',
                                  'seed':27100181}.items():
                if meta.get(key) != expected:
                    raise ValueError('saved milestone metadata differs: ' + key)
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actor)):
            raise ValueError('nonfinite Actor parameters')
        outputs = [np.asarray(graph_v2_actor_deterministic(actor, obs, graph, config.frame_stack))
                   for obs in probe_inputs]
        if any(out.shape != (1, 81) or not np.isfinite(out).all() or np.max(np.abs(out)) > 1
               for out in outputs):
            raise ValueError('deterministic Actor action contract failed')
        if expected_step == 0 and any(np.any(out != 0) for out in outputs):
            raise ValueError('conservative initial deterministic Actor must be exactly zero')
        results.append({'actor': row['path'], 'sha256': actual_sha,
                        'actual_global_interactions': meta['actual_global_interactions'],
                        'shape': list(outputs[0].shape), 'finite': True,
                        'initial_exact_zero': expected_step == 0,
                        'full_final_evaluator_metadata_validation': expected_step == 1000448,
                        'missing_metadata_fields': [k for k in ('learner_observation_dim','compact_solver')
                                                    if k not in meta],
                        'max_abs_normalized_action': max(float(np.max(np.abs(out))) for out in outputs),
                        'metadata': meta})
    result = {'status': 'actor_cpu_smoke_pass', 'backend': jax.default_backend(),
              'python': platform.python_version(),
              'dependencies': {k: importlib.metadata.version(k) for k in ('jax','jaxlib','flax','optax','numpy','scipy')},
              'frozen_identity': frozen_identity, 'layout_sha256': layout.payload_sha256,
              'observation_width': width, 'actor_results': results,
              'LES_executed': False, 'GPU_executed': False,
              'test_scope': 'Trusted checkpoint loading, exact initial zero and finite deterministic inference on two synthetic CPU observations; no physical trajectories or distributional performance checked.'}
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
