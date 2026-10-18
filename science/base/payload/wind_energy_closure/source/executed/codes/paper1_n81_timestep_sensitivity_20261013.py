"""Independent N81 physical-time step sensitivity diagnostic (2026-10-13).

The implementation is prepared separately from the scientific training and
general evaluation modules.  GPU execution belongs to the root supervisor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

ARCHIVE_DT = .2
CONTROL_SECONDS = 10.
SOURCE_START = 20000
BURNIN_CONTROLS = 150
SCORED_CONTROLS = 400
INITIALIZER_SEED = 281001
FINAL_ACTOR_SHA256 = 'be68a370939eaecf3408af4bbbc2db84f0e9677fe46fe8f8189f933ff0a54ca5'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_identity(value):
    value = np.asarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            'finite': bool(np.isfinite(value).all())}


def temporal_contract(dt):
    if dt not in (.2, .1):
        raise ValueError('only predeclared .2/.1 second comparison is admitted')
    ratio = int(round(ARCHIVE_DT / dt))
    return {'les_dt_seconds': dt, 'archive_dt_seconds': ARCHIVE_DT,
            'control_dt_seconds': CONTROL_SECONDS,
            'les_substeps_per_control': 50 * ratio,
            'archive_intervals_per_control': 50,
            'plane_start': SOURCE_START,
            'scoring_plane_cursor': SOURCE_START + 50 * BURNIN_CONTROLS,
            'final_plane_cursor_exclusive': SOURCE_START + 50 * (BURNIN_CONTROLS + SCORED_CONTROLS),
            # Every block includes its right endpoint; no clamping/wrapping.
            'required_archive_end_exclusive': SOURCE_START + 50 * (BURNIN_CONTROLS + SCORED_CONTROLS) + 1,
            'burnin_controls': BURNIN_CONTROLS, 'scored_controls': SCORED_CONTROLS,
            'burnin_physical_seconds': 1500., 'scoring_physical_seconds': 4000.,
            'initializer_seed': INITIALIZER_SEED,
            'inlet_function': 'piecewise linear through the frozen .2 s archive knots, sampled at left substep endpoints',
            'native_filter_cadence_seconds': dt,
            'numerical_scope': 'complete frozen LES implementation dt sensitivity; native filtering per substep also changes frequency',
            'not_established': ['pure AB3 temporal convergence', 'grid convergence', 'broad wind generalization']}


def sample_frozen_inlet(knots, dt):
    """Sample ONE physical piecewise-linear function on either time grid.

    Input includes the right endpoint. Coarse samples retain every archived
    knot bitwise; fine midpoint samples interpolate all three components.
    """
    temporal_contract(dt)
    knots = np.asarray(knots)
    if knots.ndim != 4 or knots.shape[0] < 2 or knots.shape[1] != 3 or knots.dtype != np.float32:
        raise ValueError('float32 [intervals+1,3,ny,nz] inlet knots required')
    if not np.isfinite(knots).all():
        raise ValueError('nonfinite inlet knots')
    if dt == .2:
        return np.array(knots[:-1], copy=True)
    values = np.empty((2 * (knots.shape[0] - 1), *knots.shape[1:]), dtype=np.float32)
    values[0::2] = knots[:-1]
    values[1::2] = (knots[:-1] + knots[1:]) * np.float32(.5)
    return values


def physical_control_config(dt, base_class=None):
    """Override the legacy .2 s convenience property without changing core."""
    contract = temporal_contract(dt)
    if base_class is None:
        from jax_mole_continuous_sac import ContinuousControlConfig
        base_class = ContinuousControlConfig

    @dataclass(frozen=True)
    class PhysicalTimeConfig(base_class):
        les_dt_seconds: float = dt

        @property
        def control_dt_seconds(self):
            return self.les_steps_per_interaction * self.les_dt_seconds

    config = PhysicalTimeConfig(inlet_mode='time_planes',
                               les_steps_per_interaction=contract['les_substeps_per_control'])
    if config.control_dt_seconds != CONTROL_SECONDS:
        raise ValueError('physical controller period changed')
    return config


def validate_cold_state(state):
    """Check cold AB startup, never resume .2 s history under .1 s."""
    flow = state.farm_flow
    if int(np.asarray(flow.step_index)) != 0:
        raise ValueError('cold Euler/AB2 startup required; warm history reuse rejected')
    for name in ('filtered_disk_speed', 'bottom_pressure_gradient', 'open_x_pressure_gradient'):
        values = np.asarray(getattr(flow, name))
        if np.any(values != 0):
            raise ValueError('cold initializer history is not zero: ' + name)
    if np.any(np.asarray(state.previous_yaw) != 0) or int(np.asarray(state.interaction_index)) != 0:
        raise ValueError('cold controller initializer required')
    identities = {name: array_identity(getattr(flow, name)) for name in flow._fields}
    if not all(record['finite'] for record in identities.values()):
        raise ValueError('nonfinite cold initial state')
    return identities


def validate_reference(result, *, common_hash, layout_hash, archive_hash,
                       extension_hash, source_hash, actor_hash):
    """A byte-identical reproduced common state binds legacy missing seed/dt.

    Legacy results are not trusted merely because their path says pilot_A.
    Both controllers, archive, numerical choices and warmup must match.
    Trace hashes and file contents are checked separately before reuse.
    """
    if result.get('status') != 'complete' or result.get('turbines') != 81:
        raise ValueError('reference is not a complete N81 evaluation')
    expected = {'layout_sha256': layout_hash, 'archive_manifest_sha256': archive_hash,
                'common_state_sha256': common_hash, 'window_start': SOURCE_START}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError('reference binding mismatch: ' + key)
    if result.get('archive_extension_manifest_sha256', extension_hash) != extension_hash:
        raise ValueError('reference extension hash differs')
    if result.get('source_state', {}).get('sha256') != source_hash:
        raise ValueError('reference precursor source differs')
    if result.get('evaluation_numerics') != {'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor'}:
        raise ValueError('reference numerics differ or are unspecified')
    burnin = result.get('common_burnin', {})
    if (burnin.get('controls') != BURNIN_CONTROLS or burnin.get('duration_seconds') != 1500.
            or burnin.get('initial_plane_cursor') != SOURCE_START
            or burnin.get('final_plane_cursor_exclusive') != SOURCE_START + 7500
            or burnin.get('final_farm_step_index') != 7500):
        raise ValueError('reference common burnin clock differs')
    selected = {}
    for name, record in result.get('controllers', {}).items():
        identity = record.get('actor')
        kind = 'zero' if identity is None and name == 'zero' else 'rl_final'
        if kind == 'rl_final' and (not isinstance(identity, dict) or identity.get('sha256') != actor_hash):
            continue
        score = result.get('scores', {}).get(name, {})
        if score.get('controls', 0) < SCORED_CONTROLS or score.get('duration_seconds') != 10. * score.get('controls', 0):
            raise ValueError('reference scored physical duration differs')
        if (record.get('common_state_sha256') != common_hash
                or record.get('scoring_cursor') != SOURCE_START + 7500
                or record.get('final_cursor_exclusive') != SOURCE_START + 7500 + 50 * score['controls']):
            raise ValueError('reference controller state/cursor differs')
        selected[kind] = record
    if set(selected) != {'zero', 'rl_final'}:
        raise ValueError('reference lacks zero and exact frozen final Actor branches')
    return selected


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, choices=(.2, .1), required=True)
    parser.add_argument('--plane-archive', type=Path, required=True)
    parser.add_argument('--verified-archive', type=Path, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--block-interactions', type=int, default=8)
    parser.add_argument('--reference-result', type=Path)
    parser.add_argument('--reference-result-sha256')
    parser.add_argument('--expected-cold-velocity-sha256')
    args = parser.parse_args()
    if args.block_interactions not in (1, 2, 4, 8):
        raise ValueError('bounded admitted block sizes are 1/2/4/8')
    if args.reference_result is not None and (args.dt != .2 or not args.reference_result_sha256):
        raise ValueError('only coarse dt supports SHA-bound reference reuse')
    if args.dt == .1 and not args.expected_cold_velocity_sha256:
        raise ValueError('fine timestep requires the measured coarse cold velocity SHA')
    if args.output_dir.exists():
        raise FileExistsError('refuse overwrite or silent experiment repetition')
    if sha256_file(args.actor) != FINAL_ACTOR_SHA256:
        raise ValueError('exact frozen final R1 Actor required')
    contract = temporal_contract(args.dt)

    import jax
    import jax.numpy as jnp
    from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
    from jax_mole_continuous_sac import load_actor_checkpoint
    from jax_mole_graph_sac_v2 import build_wake_graph
    from jax_mole_layouts import canonical_scale_layouts
    from mole_large_scale_evaluation import initialize_evaluation, scoring_boundary, make_step, make_scan, validate_actor_metadata
    from mole_large_scale_metrics import score_trace, paired_gain
    from mole_plane_archive import PlaneArchive

    if jax.default_backend() != 'gpu':
        raise ValueError('scientific diagnostic requires GPU runtime; CPU contracts are separate')
    archive = PlaneArchive(args.plane_archive)
    verification = json.loads(args.verified_archive.read_text())
    if (verification.get('manifest_sha256') != archive.manifest_sha256
            or verification.get('extension_manifest_sha256') != archive.extension_manifest_sha256
            or not verification.get('all_chunk_hashes_verified')):
        raise ValueError('current complete archive hash-verification receipt required')
    if float(archive.manifest.get('case', {}).get('dt', -1)) != ARCHIVE_DT:
        raise ValueError('archive time spacing is not the frozen .2 s interval')
    if archive.completed_planes < contract['required_archive_end_exclusive']:
        raise ValueError('archive lacks the required interpolation endpoint')
    layout = canonical_scale_layouts()[81]
    if archive.manifest.get('provenance', {}).get('target_layout_sha256') != layout.payload_sha256:
        raise ValueError('archive belongs to a different layout')
    case = replace(make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse', rhs_assembly='tensor'), dt=args.dt)
    if archive.plane_shape != (3, case.ny, case.nz):
        raise ValueError('archive plane does not match the exact canonical N81 grid')
    config = physical_control_config(args.dt)
    actor, metadata = load_actor_checkpoint(args.actor, expected_case=case, expected_action_parameterization='per_turbine')
    validate_actor_metadata(metadata, case, config, 1000448)
    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    contract.update(layout_sha256=layout.payload_sha256,
                    actor_sha256=FINAL_ACTOR_SHA256,
                    archive_manifest_sha256=archive.manifest_sha256,
                    archive_extension_manifest_sha256=archive.extension_manifest_sha256,
                    source_state=archive.manifest['source_state'],
                    seed=INITIALIZER_SEED, case=asdict(case), frame_stack=config.frame_stack,
                    yaw_rate_limit_deg_per_s=1., yaw_angle_limit_deg=40.,
                    probe_layout_contract=metadata['layout_contract'])
    contract['implementation_sha256'] = {
        name: sha256_file(Path(__file__).with_name(name)) for name in (
            Path(__file__).name, 'jax_mole_compact_sac_benchmark.py',
            'jax_mole_continuous_sac.py', 'jax_mole_graph_sac_v2.py',
            'mole_large_scale_evaluation.py')}
    contract['verified_archive_receipt_sha256'] = sha256_file(args.verified_archive)
    atomic_json(args.output_dir / 'CONTRACT.json', contract)
    common, precursor, farm = initialize_evaluation(case, config, jnp.asarray(archive.read(SOURCE_START, 1)[0][None]), INITIALIZER_SEED)
    cold = validate_cold_state(common)
    atomic_json(args.output_dir / 'COLD_STATE_IDENTITY.json', cold)
    if args.expected_cold_velocity_sha256 and cold['velocity']['sha256'] != args.expected_cold_velocity_sha256:
        raise ValueError('cold velocity differs from coarse initializer; dt-history experiment rejected')
    zero_step = make_step(case, config, precursor, farm)
    scans = {}

    @jax.jit
    def diagnose(velocity):
        raw = farm['divergence'](velocity)
        residual = farm['diagnostic_divergence'](raw)
        inlet = jnp.mean(velocity[:, 0, 0], axis=(-2, -1))
        outlet = jnp.mean(velocity[:, 0, -1], axis=(-2, -1))
        return {'velocity_finite': jnp.all(jnp.isfinite(velocity)),
                'maximum_abs_velocity_m_per_s': jnp.max(jnp.abs(velocity)),
                'raw_divergence_rms_per_s': jnp.sqrt(jnp.mean(raw * raw)),
                'residual_divergence_rms_per_s': jnp.sqrt(jnp.mean(residual * residual)),
                'residual_divergence_max_abs_per_s': jnp.max(jnp.abs(residual)),
                'inlet_mean_u_m_per_s': inlet.mean(), 'outlet_mean_u_m_per_s': outlet.mean(),
                'inlet_outlet_mean_u_difference_m_per_s': (outlet - inlet).mean()}

    def diagnostics(state):
        values = jax.tree.map(np.asarray, diagnose(state.farm_flow.velocity))
        return {key: bool(value) if value.dtype == np.bool_ else float(value) for key, value in values.items()}

    def collect(state, step, cursor, count, label):
        traces, timings, numerical = [], [], []
        for offset in range(0, count, args.block_interactions):
            length = min(args.block_interactions, count - offset)
            key = (label, length)
            if key not in scans:
                scans[key] = make_scan(step, length)
            io = time.monotonic()
            knots = archive.read(cursor, length * 50 + 1)
            inlet_identity = array_identity(knots)
            planes = jnp.asarray(sample_frozen_inlet(knots, args.dt)[None])
            jax.block_until_ready(planes)
            io_seconds = time.monotonic() - io
            compute = time.monotonic()
            state, metrics = scans[key](state, planes, state.farm_flow.step_index)
            jax.block_until_ready(metrics)
            elapsed = time.monotonic() - compute
            host = jax.tree.map(np.asarray, metrics)
            if any(not np.isfinite(value).all() for value in host.values()):
                raise ValueError('nonfinite controller/LES output')
            traces.append(host)
            timings.append({'controls': length, 'input_io_interpolation_seconds': io_seconds,
                            'compute_and_possible_compile_seconds': elapsed,
                            'archive_start': cursor, 'archive_end_exclusive': cursor + length * 50 + 1,
                            'archive_knots': inlet_identity})
            cursor += length * 50
            diagnostic = diagnostics(state)
            if not diagnostic['velocity_finite'] or not all(np.isfinite(v) for v in diagnostic.values()):
                raise ValueError('nonfinite numerical diagnostic')
            numerical.append({'completed_controls': offset + length, **diagnostic})
            atomic_json(args.output_dir / 'PROGRESS.json', {'phase': label, 'completed_controls': offset + length,
                        'les_dt_seconds': args.dt, 'plane_cursor': cursor,
                        'farm_step_index': int(np.asarray(state.farm_flow.step_index)),
                        'elapsed_seconds': time.monotonic() - started})
            print(json.dumps({'phase': label, 'completed_controls': offset + length, 'plane_cursor': cursor}), flush=True)
        trace = {key: np.concatenate([block[key] for block in traces], axis=0) for key in traces[0]}
        return state, trace, cursor, timings, numerical

    common, burnin, scoring_cursor, burnin_timing, burnin_diagnostics = collect(common, zero_step, SOURCE_START, BURNIN_CONTROLS, 'common_burnin')
    with (args.output_dir / 'common_burnin.npz').open('xb') as stream:
        np.savez(stream, **burnin)
    del burnin
    expected_burnin_steps = BURNIN_CONTROLS * config.les_steps_per_interaction
    if int(np.asarray(common.farm_flow.step_index)) != expected_burnin_steps:
        raise ValueError('burnin physical and numerical clocks diverged')
    common = scoring_boundary(common)
    common_path = args.output_dir / 'common_state.pkl'
    with common_path.open('xb') as stream:
        pickle.dump(jax.tree.map(np.asarray, common), stream, protocol=5)
    common_hash = sha256_file(common_path)
    reference = None
    if args.reference_result:
        if sha256_file(args.reference_result) != args.reference_result_sha256:
            raise ValueError('reference RESULT SHA differs')
        reference = json.loads(args.reference_result.read_text())
        reused = validate_reference(reference, common_hash=common_hash, layout_hash=layout.payload_sha256,
                                    archive_hash=archive.manifest_sha256, extension_hash=archive.extension_manifest_sha256,
                                    source_hash=archive.source_state_sha256, actor_hash=FINAL_ACTOR_SHA256)
    scores, records = {}, {}
    for name, parameters in (('zero', None), ('rl_final', actor)):
        if reference is not None:
            old = reused[name]
            # The already mounted reference directory is the only admitted source.
            path = args.reference_result.parent / Path(old['trace']).name
            if sha256_file(path) != old['trace_sha256']:
                raise ValueError('reference trace hash differs')
            with np.load(path, allow_pickle=False) as values:
                trace = {key: values[key][:SCORED_CONTROLS] for key in values.files}
            timings, numerical = [], []
            origin = {'result_sha256': args.reference_result_sha256, 'trace_sha256': old['trace_sha256'],
                      'reused_prefix_controls': SCORED_CONTROLS,
                      'binding': 'reproduced byte-identical common_state; current archive/layout/Actor/NUM and trace hashes verified',
                      'final_numerical_diagnostics': 'not retained in source result; not reconstructed'}
            final_cursor = contract['final_plane_cursor_exclusive']
        else:
            step = zero_step if parameters is None else make_step(case, config, precursor, farm, actor=parameters, graph=graph)
            final, trace, final_cursor, timings, numerical = collect(common, step, scoring_cursor, SCORED_CONTROLS, name)
            if int(np.asarray(final.farm_flow.step_index)) != (BURNIN_CONTROLS + SCORED_CONTROLS) * config.les_steps_per_interaction:
                raise ValueError('scored physical and numerical clocks diverged')
            origin = {'reused': False}
            del final
        if final_cursor != contract['final_plane_cursor_exclusive']:
            raise ValueError('controller inlet cursor mismatch')
        score = score_trace(trace, np.asarray(common.previous_yaw), control_dt=CONTROL_SECONDS)
        if score['controls'] != SCORED_CONTROLS or not score['rate_limit_satisfied'] or not score['angle_limit_satisfied']:
            raise ValueError('scored controller/actuator contract violated')
        trace_path = args.output_dir / (name + '.npz')
        with trace_path.open('xb') as stream:
            np.savez(stream, **trace)
        scores[name] = score
        records[name] = {'trace': str(trace_path), 'trace_sha256': sha256_file(trace_path),
                         'common_state_sha256': common_hash, 'timing': timings,
                         'block_end_numerical_diagnostics': numerical, 'origin': origin,
                         'actor_sha256': FINAL_ACTOR_SHA256 if parameters is not None else None}
        del trace
    result = {'status': 'complete', 'contract': contract, 'cold_state_identity': cold,
              'common_state_sha256': common_hash, 'scores': scores, 'controllers': records,
              'pair': paired_gain(scores['rl_final'], scores['zero']),
              'burnin_timing': burnin_timing, 'burnin_block_end_numerical_diagnostics': burnin_diagnostics,
              'elapsed_wall_seconds': time.monotonic() - started,
              'boundary': 'single A developmental window, 400 controls; numerical sensitivity includes native filter cadence and is not grid convergence'}
    atomic_json(args.output_dir / 'RESULT.json', result)
    print(json.dumps({'status': 'complete', 'dt': args.dt, 'pair': result['pair']}), flush=True)


if __name__ == '__main__':
    main()
