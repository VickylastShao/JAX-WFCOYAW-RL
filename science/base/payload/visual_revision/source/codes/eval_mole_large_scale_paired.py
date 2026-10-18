"""One fixed window: common LES state -> zero and frozen final Graph-SAC Actors."""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import ContinuousControlConfig, load_actor_checkpoint
from jax_mole_graph_sac_v2 import build_wake_graph
from jax_mole_layouts import canonical_scale_layouts
from mole_large_scale_evaluation import (
    initialize_evaluation, scoring_boundary, make_step, make_scan, validate_actor_metadata,
)
from mole_large_scale_metrics import score_trace, paired_gain
from mole_plane_archive import PlaneArchive, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-turbines', type=int, choices=(81, 100), required=True)
    parser.add_argument('--plane-archive', type=Path, required=True)
    parser.add_argument('--verified-archive', type=Path, required=True)
    parser.add_argument('--actor', type=Path, action='append', required=True)
    parser.add_argument('--plane-start', type=int, required=True)
    parser.add_argument('--context-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--burnin', type=int, default=150)
    parser.add_argument('--interactions', type=int, default=2000)
    parser.add_argument('--block-interactions', type=int, default=8)
    parser.add_argument('--seed', type=int, default=281001)
    parser.add_argument('--required-actor-interactions', type=int, default=1000448)
    args = parser.parse_args()
    if args.burnin < 0 or args.interactions < 1 or args.block_interactions < 1 or args.plane_start < 0:
        raise ValueError('invalid evaluation horizon')
    if jax.default_backend() != 'gpu':
        raise ValueError('confirmation requires the actual GPU LES runtime')
    if args.output_dir.exists():
        raise FileExistsError('refuse to overwrite or silently repeat an evaluation context')
    archive = PlaneArchive(args.plane_archive)
    verification = json.loads(args.verified_archive.read_text())
    if verification.get('manifest_sha256') != archive.manifest_sha256 or not verification.get('all_chunk_hashes_verified'):
        raise ValueError('complete archive hash-verification receipt required')
    if verification.get('extension_manifest_sha256') != archive.extension_manifest_sha256:
        raise ValueError('archive extension differs from verification receipt')
    layout = canonical_scale_layouts()[args.target_turbines]
    if archive.manifest.get('provenance', {}).get('target_layout_sha256') != layout.payload_sha256:
        raise ValueError('archive belongs to a different layout')
    case = make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse',
                                          rhs_assembly='tensor')
    if tuple(archive.plane_shape) != (3, case.ny, case.nz):
        raise ValueError('inlet plane shape differs from full LES grid')
    end = args.plane_start + 50 * (args.burnin + args.interactions)
    if end > archive.completed_planes:
        raise ValueError('full burnin/scoring input window exceeds the archive')
    config = ContinuousControlConfig(inlet_mode='time_planes')
    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    controllers = [('zero', None, None)]
    for index, path in enumerate(args.actor, 1):
        actor, metadata = load_actor_checkpoint(path, expected_case=case,
                                                expected_action_parameterization='per_turbine')
        validate_actor_metadata(metadata, case, config, args.required_actor_interactions)
        controllers.append((f'rl_r{index}', actor,
                            {'path': str(path), 'sha256': sha256_file(path), 'metadata': metadata}))
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    initial_inlet = jnp.asarray(archive.read(args.plane_start, 1)[0][None])
    common, precursor, farm = initialize_evaluation(case, config, initial_inlet, args.seed)
    zero_step = make_step(case, config, precursor, farm)
    scans = {}

    def collect(state, step, cursor, count, label):
        traces = []
        timing = []
        for offset in range(0, count, args.block_interactions):
            length = min(args.block_interactions, count - offset)
            key = (label, length)
            if key not in scans:
                scans[key] = make_scan(step, length)
            io_started = time.monotonic()
            planes = jnp.asarray(archive.read(cursor, length * 50)[None])
            jax.block_until_ready(planes)
            io_seconds = time.monotonic() - io_started
            compute_started = time.monotonic()
            state, metrics = scans[key](state, planes, state.farm_flow.step_index)
            jax.block_until_ready(metrics)
            elapsed = time.monotonic() - compute_started
            host = jax.tree.map(np.asarray, metrics)
            if any(not np.isfinite(value).all() for value in jax.tree.leaves(host)):
                raise ValueError('nonfinite scored LES/controller output')
            traces.append(host)
            timing.append({'controls': length, 'input_io_seconds': io_seconds,
                           'compute_and_possible_compile_seconds': elapsed})
            cursor += length * 50
            print(json.dumps({'phase': label, 'completed_controls': offset + length,
                              'plane_cursor': cursor}), flush=True)
        trace = ({key: np.concatenate([block[key] for block in traces], axis=0)
                  for key in traces[0]} if traces else {})
        return state, trace, cursor, timing

    common, burnin_trace, scoring_cursor, burnin_timing = collect(
        common, zero_step, args.plane_start, args.burnin, 'common_burnin')
    burnin_path = args.output_dir / 'common_burnin.npz'
    with burnin_path.open('xb') as stream:
        np.savez(stream, **burnin_trace)
    burnin_record = {'trace': str(burnin_path), 'trace_sha256': sha256_file(burnin_path),
                     'controls': args.burnin,
                     'duration_seconds': args.burnin * config.control_dt_seconds,
                     'initial_plane_cursor': args.plane_start,
                     'final_plane_cursor_exclusive': scoring_cursor,
                     'final_farm_step_index': int(np.asarray(common.farm_flow.step_index)),
                     'excluded_from_scoring': True,
                     'purpose': 'common zero-yaw physical-relaxation diagnostics only'}
    del burnin_trace
    common = scoring_boundary(common)
    common_path = args.output_dir / 'common_state.pkl'
    with common_path.open('xb') as stream:
        pickle.dump(jax.tree.map(np.asarray, common), stream, protocol=5)
    common_hash = sha256_file(common_path)
    scores = {}
    records = {}
    for name, actor, actor_identity in controllers:
        step = zero_step if actor is None else make_step(case, config, precursor, farm, actor=actor, graph=graph)
        final, trace, final_cursor, timings = collect(common, step, scoring_cursor, args.interactions, name)
        score = score_trace(trace, np.asarray(common.previous_yaw))
        if not score['rate_limit_satisfied'] or not score['angle_limit_satisfied']:
            raise ValueError('actual actuator constraint violation')
        path = args.output_dir / f'{name}.npz'
        with path.open('xb') as stream:
            np.savez(stream, **trace)
        scores[name] = score
        records[name] = {'trace': str(path), 'trace_sha256': sha256_file(path),
                         'actor': actor_identity, 'common_state_sha256': common_hash,
                         'scoring_cursor': scoring_cursor, 'final_cursor_exclusive': final_cursor,
                         'timing': timings}
        del final, trace
    result = {'status': 'complete', 'context_id': args.context_id,
              'turbines': args.target_turbines, 'layout_sha256': layout.payload_sha256,
              'source_state': archive.manifest['source_state'],
              'archive_manifest_sha256': archive.manifest_sha256,
              'common_state_sha256': common_hash, 'scores': scores,
              'pairs': {name: paired_gain(score, scores['zero']) for name, score in scores.items() if name != 'zero'},
              'controllers': records, 'burnin_timing': burnin_timing,
              'common_burnin': burnin_record,
              'evaluation_numerics': {'compact_solver': case.compact_solver,
                                      'rhs_assembly': case.rhs_assembly},
              'window_start': args.plane_start, 'window_end_exclusive': end,
              'cache_boundary': 'matches trainer: zero observation/history after common zero-yaw burnin',
              'elapsed_wall_seconds': time.monotonic() - started,
              'independence_unit': 'precursor source, not window or learner seed'}
    temporary = args.output_dir / 'RESULT.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output_dir / 'RESULT.json')
    print(json.dumps({'status': 'complete', 'context_id': args.context_id, 'pairs': result['pairs']}))


if __name__ == '__main__':
    main()
