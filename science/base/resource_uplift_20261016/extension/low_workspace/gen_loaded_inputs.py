#!/usr/bin/env python3
"""CPU-only loaded full-grid diagnostic fixture, not qualified turbulence.

Stage this script at TASK root beside source/ and inputs/. The original scaled
case geometry and physical parameters remain unchanged; only the initializer
flag initialize_with_log_profile is enabled via dataclasses.replace. No mature-inflow or policy result is
claimed. Failure of nonuniformity/load assertions stops preparation; no substitute
initialization is silently selected after observing performance.
"""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

SEED = 81731
STEPS = 50
EXPECTED_CORE = 'f65b1a78e8828fe084fb83c57c0968754ac9cf53922b3361c658e691bc2f1066'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    root = args.task_root.resolve()
    source, out = root / 'source', root / 'inputs_loaded'
    out.mkdir(exist_ok=True)
    state_path, inlet_path = out / 'loaded_state.pkl.gz', out / 'loaded_inlet.npy'
    if any(p.exists() for p in [state_path, inlet_path, out / 'GENERATION_PROTOCOL.json', out / 'GENERATION_RESULT.json']):
        raise FileExistsError('Refuse to overwrite a frozen input generation')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('MOLE_JAX_REAL_DTYPE', 'float32')
    if os.environ['JAX_PLATFORMS'] != 'cpu' or os.environ['MOLE_JAX_REAL_DTYPE'] != 'float32':
        raise ValueError('CPU platform and FP32 are required before importing JAX')
    if sha(source / 'jax_mole_compact_sac_benchmark.py') != EXPECTED_CORE:
        raise ValueError('Frozen LES core identity mismatch')
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'case': 'original make_scaled_mole_training_case(rectangular_mole_layout(9,9), batch=1)',
        'initializer': 'original make_initial_flow(replace(case, initialize_with_log_profile=True), original build_flow_functions(case), seed=81731)',
        'only_initializer_override': {'initialize_with_log_profile': True},
        'prescribed_inlet': 'repeat the original initialized x=0 plane exactly 50 times, dt=0.2 s',
        'scope': 'deterministic nonuniform loaded implementation diagnostic; not B_w1 or mature turbulence, not a trained-policy or control-gain evaluation',
        'assertions': 'finite FP32 full state; velocity and streamwise probes nonuniform; no silent fallback',
        'source_files': [{'path': str(f.relative_to(source)), 'sha256': sha(f)} for f in sorted(source.glob('*.py'))],
        'script_sha256': sha(__file__), 'gpu_execution': False,
    }
    save(out / 'GENERATION_PROTOCOL.json', protocol)
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jax_mole_compact_sac_benchmark as core
    from jax_mole_layouts import rectangular_mole_layout
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('CPU-only input generation required')
    layout = rectangular_mole_layout(9, 9)
    case = core.make_scaled_mole_training_case(layout, batch=1)
    if (case.nx, case.ny, case.nz, case.num_turbines) != (496, 41, 483, 81):
        raise ValueError('Expected original 81-turbine grid')
    started = time.monotonic()
    functions = core.build_flow_functions(case, periodic_x=False, include_turbines=True)
    initcase = replace(case, initialize_with_log_profile=True)
    state = core.make_initial_flow(initcase, functions, seed=SEED)
    jax.block_until_ready(state)
    host = jax.tree.map(np.asarray, state)
    if any(not np.isfinite(v).all() for v in jax.tree.leaves(host)):
        raise FloatingPointError('Nonfinite diagnostic initialization')
    if host.velocity.dtype != np.float32:
        raise ValueError('Original FP32 velocity required')
    probes = np.asarray(functions['probes'](state.velocity))
    velocity_spread = float(np.ptp(host.velocity))
    probe_spread = float(np.ptp(probes))
    if velocity_spread <= 1e-8 or probe_spread <= 1e-8:
        raise ValueError('Original initializer did not produce nonuniform velocity/probes; stop without substitute')
    if float(np.mean(probes,dtype=np.float64)) <= 1.:
        raise ValueError('Loaded initializer has insufficient nonzero streamwise speed for the intended diagnostic')
    inlet = np.repeat(host.velocity[0, :, 0][None], STEPS, axis=0)
    with state_path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as stream:
            pickle.dump(host, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with inlet_path.open('xb') as stream:
        np.save(stream, inlet, allow_pickle=False)
    stats = lambda v: {'min': float(np.min(v)), 'max': float(np.max(v)),
                       'mean': float(np.mean(v, dtype=np.float64)),
                       'std': float(np.std(v, dtype=np.float64))}
    value = {
        'status': 'generated_loaded_diagnostic_inputs',
        'purpose': protocol['scope'], 'protocol_sha256': sha(out / 'GENERATION_PROTOCOL.json'),
        'case': asdict(case), 'initializer_case': asdict(initcase), 'seed': SEED, 'jax_version': jax.__version__,
        'numpy_version': np.__version__, 'devices': [str(d) for d in jax.devices()],
        'initialization_seconds': time.monotonic() - started,
        'nominal_configured_hub_speed_mps': case.inflow_hub_speed,
        'resolved_hub_y_m': round(case.hub_height/case.dy)*case.dy,
        'unperturbed_loglaw_hub_speed_mps': float((case.friction_velocity/case.von_karman)*np.log((round(case.hub_height/case.dy)*case.dy+case.roughness_length)/case.roughness_length)),
        'initialization_note': 'Only initialize_with_log_profile is True in the initializer case. The original log law, source-aligned perturbation amplitude, pressure projection and all physical parameters remain unchanged. Its actual resolved hub speed is measured, not forced to the nominal 8 m/s metadata.',
        'velocity_statistics': stats(host.velocity), 'streamwise_statistics': stats(host.velocity[:, 0]),
        'resolved_hub_streamwise_statistics': stats(host.velocity[:, 0, :, round(case.hub_height/case.dy), :]),
        'probe_statistics': stats(probes), 'inlet_statistics': stats(inlet),
        'state': {'path': str(state_path), 'bytes': state_path.stat().st_size, 'sha256': sha(state_path),
                  'type': 'frozen core.FlowState with original seven fields',
                  'velocity_shape': list(host.velocity.shape), 'step_index': int(host.step_index)},
        'inlet': {'path': str(inlet_path), 'bytes': inlet_path.stat().st_size, 'sha256': sha(inlet_path),
                  'shape': list(inlet.shape), 'dtype': str(inlet.dtype), 'steps': STEPS,
                  'physical_seconds': STEPS * case.dt, 'temporal_variation': False},
        'not_established': ['qualified turbulence', 'B/C control performance', 'policy equivalence', 'training acceleration', 'predictive wind-farm accuracy'],
        'gpu_execution': False,
    }
    save(out / 'GENERATION_RESULT.json', value)
    print(json.dumps({'status': value['status'], 'state_bytes': state_path.stat().st_size,
                      'probe_spread': probe_spread, 'initialization_seconds': value['initialization_seconds']}), flush=True)


if __name__ == '__main__':
    main()
