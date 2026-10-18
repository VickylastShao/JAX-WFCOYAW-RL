"""Bounded component and independent numerical diagnostics; no training changes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import statistics
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def face_flux(velocity, lengths=(6174., 500., 5922.)):
    """Geometric quadrature, independent of the solver's mean-outlet correction.

    x/y include endpoints; z is a periodic, endpoint-excluded grid. Periodic
    z faces cancel identically: last grid sample is NOT the opposing face.
    """
    v = np.asarray(velocity)
    if v.ndim != 5 or v.shape[1] != 3:
        raise ValueError('batch/component/x/y/z velocity required')
    _, _, nx, ny, nz = v.shape
    lx, ly, lz = lengths
    wx, wy = np.ones(nx), np.ones(ny)
    wx[[0, -1]] = .5
    wy[[0, -1]] = .5
    dx, dy, dz = lx / (nx - 1), ly / (ny - 1), lz / nz
    qin = np.einsum('byz,y->b', v[:, 0, 0].astype(np.float64), wy) * dy * dz
    qout = np.einsum('byz,y->b', v[:, 0, -1].astype(np.float64), wy) * dy * dz
    qtop = np.einsum('bxz,x->b', v[:, 1, :, -1].astype(np.float64), wx) * dx * dz
    qbottom = np.einsum('bxz,x->b', v[:, 1, :, 0].astype(np.float64), wx) * dx * dz
    net = qout - qin + qtop - qbottom
    return {'inlet_m3_s': qin.tolist(), 'outlet_m3_s': qout.tolist(),
            'top_m3_s': qtop.tolist(), 'bottom_m3_s': qbottom.tolist(),
            'periodic_lateral_net_m3_s': [0.] * v.shape[0],
            'net_outward_m3_s': net.tolist(),
            'relative_net_to_inlet': (net / np.maximum(np.abs(qin), 1e-30)).tolist(),
            'quadrature': 'trapezoid in nonperiodic x/y, periodic rectangular z; no reference subtraction'}


def filter_stride_builder(original, stride):
    """Freeze a diagnostic-only filter cadence; preserve operation ordering."""
    if stride == 1:
        return original
    if stride != 2:
        raise ValueError('only matching .1 s substeps to .2 s filter cadence')
    source = inspect.getsource(original)
    for name in ('mole_farm_all_direction_filter', 'mole_farm_spatial_filter'):
        old = f'velocity = {name}(\n                    velocity, case.filter_coefficient, case.filter_solver\n                )'
        if source.count(old) != 1:
            raise ValueError('frozen filter implementation differs: ' + name)
        new = ('velocity = jax.lax.cond(state.step_index % 2 == 0, '
               f'lambda v: {name}(v, case.filter_coefficient, case.filter_solver), '
               'lambda v: v, velocity)')
        source = source.replace(old, new)
    space = dict(original.__globals__)
    exec(compile(source, '<diagnostic-filter-cadence>', 'exec'), space)
    return space[original.__name__]


def load_common(root, context):
    """Inputs are already audited project states; no untrusted pickle accepted."""
    path = root / 'contexts' / context / 'common_state.pkl'
    result = json.loads((path.parent / 'RESULT.json').read_text())
    expected = result['common_state_sha256']
    if digest(path) != expected:
        raise ValueError('common-state source SHA differs: ' + context)
    with path.open('rb') as f:
        state = pickle.load(f)
    if tuple(state.farm_flow.velocity.shape) != (1, 3, 496, 41, 483):
        raise ValueError('exact executed N81 grid required')
    return state, {'context': context, 'path': str(path), 'sha256': expected}


def cpu_audit(args):
    # Import only after setting JAX_PLATFORMS=cpu in the enclosing container.
    records = []
    for source in 'BC':
        for window in range(1, 5):
            context = f'n81_{source}_w{window}'
            state, identity = load_common(args.evaluation, context)
            v = np.asarray(state.farm_flow.velocity)
            record = {**identity, 'scope': 'zero-yaw warmup endpoint, not controlled flow',
                      'finite': bool(np.isfinite(v).all()), 'flux': face_flux(v),
                      'max_abs_velocity_m_s': float(np.max(np.abs(v))),
                      'hub_plane_y_m': 87.5,
                      'hub_streamwise_mean_by_x_m_s': v[0, 0, :, 7, :].mean(axis=-1).tolist()}
            records.append(record)
            save(args.output / 'PROGRESS.json', {'completed_states': len(records)})
            del state, v
    save(args.output / 'RESULT.json', {'status': 'complete', 'records': records,
         'independent_sources': 2, 'states': 8, 'interpretation':
         'Independent geometric surface-flux diagnostic only. Does not establish compact-operator convergence, controlled-state conservation or physical validation.'})


def microbatch(args):
    import jax
    import jax.numpy as jnp
    from jax_mole_compact_sac_benchmark import build_flow_functions, make_scaled_mole_training_case
    from jax_mole_layouts import canonical_scale_layouts
    if jax.default_backend() != 'gpu':
        raise ValueError('GPU component measurement requires GPU')
    contexts = [f'n81_{s}_w{w}' for s in 'BC' for w in range(1, 5)]
    host, identities = [], []
    for context in contexts:
        state, identity = load_common(args.evaluation, context)
        host.append(state.farm_flow)
        identities.append(identity)
    # All eight true histories remain device-resident for every microbatch.
    flows = [jax.tree.map(jnp.asarray, s) for s in host]
    del host
    state_bytes = sum(x.nbytes for s in flows for x in jax.tree.leaves(s))
    case = make_scaled_mole_training_case(canonical_scale_layouts()[81], args.microbatch,
             compact_solver='dense_inverse', rhs_assembly='tensor')
    functions = build_flow_functions(case, periodic_x=False, include_turbines=True)
    yaw = jnp.zeros((args.microbatch, 81), jnp.float32)
    # Fixed inlet is the saved initial inlet of each context: execution workload,
    # not a replay of B/C benefits or a new physical-control outcome.
    inlets = [s.velocity[:, :, 0] for s in flows]

    def pack(group):
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0) if xs[0].ndim else xs[0], *group)

    def kernel(flow, inlet, yaw):
        # A runtime yaw input keeps the 81-disc weight field in the working
        # graph, rather than capturing multi-GB constant weights at compile.
        weights = functions['disk_weights'](yaw)
        def body(current, _):
            return functions['advance'](current, yaw, weights, inlet)
        return jax.lax.scan(body, flow, None, length=50)

    kernel = jax.jit(kernel)
    packed = pack(flows[:args.microbatch])
    packed_inlet = jnp.concatenate(inlets[:args.microbatch], axis=0)
    before = time.perf_counter()
    executable = kernel.lower(packed, packed_inlet, yaw).compile()
    compile_seconds = time.perf_counter() - before
    memory = executable.memory_analysis()
    del packed, packed_inlet

    def advance_all():
        for start in range(0, 8, args.microbatch):
            state = pack(flows[start:start + args.microbatch])
            inlet = jnp.concatenate(inlets[start:start + args.microbatch], axis=0)
            state, power = executable(state, inlet, yaw)
            jax.block_until_ready((state, power))
            for i in range(args.microbatch):
                flows[start + i] = jax.tree.map(lambda x: x[i:i+1] if x.ndim else x, state)
    before = time.perf_counter()
    advance_all()
    warm_seconds = time.perf_counter() - before
    samples = []
    for i in range(3):
        before = time.perf_counter()
        advance_all()
        samples.append(time.perf_counter() - before)
        save(args.output / 'PROGRESS.json', {'microbatch': args.microbatch, 'retained_passes': i + 1,
             'seconds': samples, 'device_memory': jax.devices()[0].memory_stats()})
    # Retain full final physical histories for CPU equivalence; never download
    # these multi-GB diagnostic states when small comparison receipts suffice.
    for i, state in enumerate(flows):
        with (args.output / f'flow{i}.pkl').open('xb') as f:
            pickle.dump(jax.tree.map(np.asarray, state), f, protocol=5)
    compiled_memory = {k: int(getattr(memory, k)) for k in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes', 'temp_size_in_bytes')}
    save(args.output / 'RESULT.json', {'status': 'complete', 'microbatch': args.microbatch,
         'contexts': identities, 'persistent_contexts': 8, 'persistent_state_bytes': state_bytes,
         'compile_seconds': compile_seconds, 'warm_pass_seconds': warm_seconds,
         'retained_pass_seconds': samples, 'median_seconds': statistics.median(samples),
         'compiled_kernel_memory': compiled_memory, 'device_memory': jax.devices()[0].memory_stats(),
         'substeps_per_context': 200, 'retained_substeps_per_context': 150,
         'scope': 'LES advancement component with fixed inlet/yaw; eight device-resident real histories; no Replay, learner or training-speed claim'})


def compare(args):
    from paper1_n81_collection_ablation_20261013 import compare_numeric_trees
    reports = {m: json.loads((args.output / f'mb{m}/RESULT.json').read_text()) for m in (1, 2, 4)}
    if len({json.dumps(r['contexts'], sort_keys=True) for r in reports.values()}) != 1:
        raise ValueError('microbatch source inputs differ')
    checks = []
    for i in range(8):
        with (args.output / f'mb1/flow{i}.pkl').open('rb') as f:
            reference = pickle.load(f)
        for m in (2, 4):
            with (args.output / f'mb{m}/flow{i}.pkl').open('rb') as f:
                candidate = pickle.load(f)
            checks.append({'context': i, 'microbatch': m,
                           **compare_numeric_trees(reference, candidate)})
    save(args.output / 'COMPARISON.json', {'status': 'complete', 'all_equivalent': all(c['pass'] for c in checks),
         'comparisons': checks, 'reports': reports, 'tolerance_basis':
         'Existing dispatch diagnostic FP32 tolerance rtol=1e-4, atol=1e-5; integers exact; all full physical histories checked.'})


def numeric(args):
    """Same original initialization/inlet, with per-control raw diagnostics."""
    import jax
    import jax.numpy as jnp
    import jax_mole_compact_sac_benchmark as flow_module
    import paper1_n81_timestep_sensitivity_20261013 as base
    import mole_large_scale_evaluation as evaluation
    import sys
    original_builder = flow_module.build_flow_functions
    builder = filter_stride_builder(original_builder, 1 if args.dt == .2 else 2)
    flow_module.build_flow_functions = builder
    evaluation.build_flow_functions = builder
    base.SCORED_CONTROLS = 128
    original_temporal = base.temporal_contract
    def temporal(dt):
        result = original_temporal(dt)
        result.update(native_filter_cadence_seconds=.2,
             scoring_physical_seconds=base.SCORED_CONTROLS * base.CONTROL_SECONDS,
             numerical_scope='diagnostic-only matched physical filter cadence; fresh 150-control warmup, 128 scored controls',
             diagnostic_builder_sha256=hashlib.sha256(inspect.getsource(builder).encode()).hexdigest()
                 if args.dt == .2 else hashlib.sha256(inspect.getsource(original_builder).encode()).hexdigest(),
             wrapper_sha256=digest(__file__))
        return result
    base.temporal_contract = temporal
    original_step = evaluation.make_step
    diag_cache = {}
    def step(case, config, precursor, farm, **kwargs):
        original = original_step(case, config, precursor, farm, **kwargs)
        if 'diag' not in diag_cache:
            @jax.jit
            def diagnostics(v):
                raw = farm['divergence'](v)
                residual = farm['diagnostic_divergence'](raw)
                wy = jnp.ones(case.ny).at[0].set(.5).at[-1].set(.5)
                wx = jnp.ones(case.nx).at[0].set(.5).at[-1].set(.5)
                dy, dz, dx = case.domain_ly/(case.ny-1), case.domain_lz/case.nz, case.domain_lx/(case.nx-1)
                qi = jnp.einsum('byz,y->b', v[:, 0, 0], wy)*dy*dz
                qo = jnp.einsum('byz,y->b', v[:, 0, -1], wy)*dy*dz
                qt = jnp.einsum('bxz,x->b', v[:, 1, :, -1], wx)*dx*dz
                qb = jnp.einsum('bxz,x->b', v[:, 1, :, 0], wx)*dx*dz
                rms = lambda x: jnp.sqrt(jnp.mean(x*x))
                return {'raw_divergence_rms': rms(raw), 'reference_subtracted_rms': rms(residual),
                     'raw_interior_rms': rms(raw[:, 2:-2, 2:-2, :]),
                     'geometric_flux_relative': ((qo-qi+qt-qb)/qi).mean(),
                     'max_velocity': jnp.max(jnp.abs(v)),
                     'hub_u_by_x': v[:, 0, :, 7, :].mean(axis=(0, 2))}
            diag_cache['diag'] = diagnostics
        def wrapped(state, planes, start):
            new, metrics = original(state, planes, start)
            return new, {**metrics, **{'qualification_' + k: v for k, v in diag_cache['diag'](new.farm_flow.velocity).items()}}
        return wrapped
    evaluation.make_step = step
    sys.argv = [base.__file__, '--dt', str(args.dt), '--plane-archive', str(args.archive),
        '--verified-archive', str(args.archive/'VERIFIED.json'), '--actor', str(args.actor),
        '--output-dir', str(args.output), '--block-interactions', '8']
    if args.dt == .1:
        sys.argv += ['--expected-cold-velocity-sha256', '149fb6fafe95f5743f6cc9d6f4a3694a93fdf86e29728d7a447a5ff5f29d97ec']
    base.main()
    value = json.loads((args.output/'RESULT.json').read_text())
    value['boundary'] = 'Fresh matched-filter-cadence 128-control source-A diagnostic; not grid convergence or B/C statistical qualification.'
    save(args.output/'RESULT.json', value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('cpu-audit', 'microbatch', 'compare', 'numeric'))
    p.add_argument('--evaluation', type=Path, default=Path('/evalinputs'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--microbatch', type=int, choices=(1, 2, 4))
    p.add_argument('--dt', type=float, choices=(.2, .1))
    p.add_argument('--archive', type=Path)
    p.add_argument('--actor', type=Path)
    args = p.parse_args()
    if args.mode != 'numeric':
        args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'RESULT.json').exists() and args.mode != 'compare':
        raise FileExistsError('refuse repeat/overwrite of completed diagnostic')
    {'cpu-audit': cpu_audit, 'microbatch': microbatch, 'compare': compare, 'numeric': numeric}[args.mode](args)


if __name__ == '__main__':
    main()
