"""Fixed A-development learning curve; no learner, tuning or B/C gate changes.

Run this only inside the authorized one-GPU budgeted supervisor. The four
predeclared nodes are represented by zero yaw (validated step-0 Actor), two
milestone Actors, and the original final Actor. Every scored branch starts
from one common 150-control zero-yaw state and receives the same 2000 controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np


NODES = (0, 131072, 524288, 1000448)
ACTOR_IDENTITIES = {
    0: ('actor_step0.pkl', '4e6a2c23650237f3c68401e0a37f97efc616b67b86ec368edad649892c9bfe16'),
    131072: ('actor_step131072.pkl', 'f617195d13a9c19d7452623a3ba7022dd4c03e936290c8e54936c528ac3639f1'),
    524288: ('actor_step524288.pkl', '6d709946180fa2e0aac67393811af13e488c81c08e3637168148dc81b19b147c'),
    1000448: ('actor.pkl', 'be68a370939eaecf3408af4bbbc2db84f0e9677fe46fe8f8189f933ff0a54ca5'),
}
SOURCE_STATE_SHA256 = 'd5736766b5c2f0a88f46a856685c703ec4f3ef899bd0e984959a0f9e77f58f95'
ARCHIVE_SHA256 = '347935efc15f856a7a46f2ccfc5ed1caa997715e9b41a9e1af3e68bbbaa47a72'
GRAPH_CODE_SHA256 = '0c6eccc791255c3df326205389f799c749bd45d584ae89098d2cd785d7ac13bb'
OPTIMIZATION_SEED = 27100181
PLANE_START = 20000
BURNIN = 150
INTERACTIONS = 2000
BLOCK_INTERACTIONS = 8
EVALUATION_SEED = 281001
GPU_SECONDS_LIMIT = 5 * 3600


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_formal_report(report, expected_layout):
    """Bind older sparse milestone metadata to the completed formal report."""
    config = report.get('configuration', {})
    expected = {
        'target_turbines': 81, 'training_algorithm': 'graph_sac_v2',
        'action_parameterization': 'per_turbine', 'action_dim': 81,
        'hidden_dim': 256, 'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor',
        'inlet_mode': 'time_planes', 'direction_condition': False,
        'steering_branch_condition': False, 'effective_reflection_equivariant_actor': False,
        'episode_interactions': 500, 'reset_interactions': 150,
        'total_global_transitions_including_compile_warmup': 1000448,
        'source_precursor_sha256': SOURCE_STATE_SHA256,
        'plane_archive_sha256': ARCHIVE_SHA256,
        'initial_actor_checkpoint': None, 'conservative_actor_init': True,
    }
    failures = [name for name, value in expected.items() if config.get(name) != value]
    if report.get('status') != 'pass' or config.get('terminated_early_nonfinite') is not False:
        failures.append('formal_completion')
    if canonical(config.get('layout_contract')) != canonical(expected_layout):
        failures.append('layout_contract')
    checkpoint = report.get('checkpoint', {})
    if Path(checkpoint.get('path_process0_only', '')).name != 'actor.pkl':
        failures.append('final_actor_reference')
    if Path(checkpoint.get('initial', {}).get('actor_process0_only', '')).name != 'actor_step0.pkl':
        failures.append('initial_actor_reference')
    milestones = {record.get('actual_global_interactions'): record
                  for record in checkpoint.get('milestones_process0_only', [])}
    for node in (131072, 524288):
        if Path(milestones.get(node, {}).get('actor_process0_only', '')).name != ACTOR_IDENTITIES[node][0]:
            failures.append(f'milestone_{node}_reference')
    if failures:
        raise ValueError('formal report binding failed: ' + ', '.join(failures))
    return config


def validate_parameter_tree(actor):
    """Require the exact unconditioned FP32 82->256, two-layer Graph Actor."""
    if not isinstance(actor, dict) or set(actor) != {'body', 'head', 'neutral_scale_contract'}:
        raise ValueError('unexpected Actor branch/structure')
    if actor['neutral_scale_contract'] != ():
        raise ValueError('neutral scale marker changed')
    body = actor['body']
    if not isinstance(body, dict) or set(body) != {'encoder', 'layers'}:
        raise ValueError('unexpected graph body')
    layers = body['layers']
    if not isinstance(layers, tuple) or len(layers) != 2:
        raise ValueError('expected exactly two graph message layers')
    shapes = []

    def linear(parameters, input_dim, output_dim, name):
        if not isinstance(parameters, dict) or set(parameters) != {'w', 'b'}:
            raise ValueError('unexpected linear parameter keys: ' + name)
        for key, expected in (('w', (input_dim, output_dim)), ('b', (output_dim,))):
            value = parameters[key]
            if not isinstance(value, np.ndarray) or value.shape != expected or value.dtype != np.dtype('float32'):
                raise ValueError('Actor shape/dtype mismatch: ' + name + '.' + key)
            if not np.isfinite(value).all():
                raise ValueError('nonfinite Actor parameter: ' + name + '.' + key)
            shapes.append({'path': name + '.' + key, 'shape': list(expected), 'dtype': str(value.dtype)})

    linear(body['encoder'], 82, 256, 'body.encoder')
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {'self', 'message'}:
            raise ValueError('unexpected graph layer')
        linear(layer['self'], 256, 256, f'body.layers.{index}.self')
        linear(layer['message'], 259, 256, f'body.layers.{index}.message')
    linear(actor['head'], 256, 2, 'head')
    return shapes


def load_diagnostic_actor(path, expected_sha256, node, expected_layout):
    """Hash trusted input before pickle; never synthesize final-evaluator metadata."""
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError('Actor SHA256 mismatch: ' + str(path))
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('unsupported Actor schema')
    metadata = payload.get('metadata', {})
    expected = {
        'policy_architecture': 'graph_sac_v2', 'training_algorithm': 'graph_sac_v2',
        'actual_global_interactions': node, 'target_turbines': 81,
        'action_dim': 81, 'action_parameterization': 'per_turbine',
        'graph_geometry': 'canonical', 'inlet_mode': 'time_planes',
        'episode_interactions': 500, 'reset_interactions': 150,
        'source_precursor_sha256': SOURCE_STATE_SHA256,
        'plane_archive_sha256': ARCHIVE_SHA256,
        'direction_condition': False, 'steering_branch_condition': False,
        'initial_actor_checkpoint_sha256': None,
    }
    failures = [name for name, value in expected.items() if metadata.get(name) != value]
    if canonical(metadata.get('layout_contract')) != canonical(expected_layout):
        failures.append('layout_contract')
    if metadata.get('reflection_equivariant_actor', False):
        failures.append('reflection_equivariant_actor')
    if node != 1000448 and metadata.get('seed') != OPTIMIZATION_SEED:
        failures.append('optimization_seed')
    for name, value in (('learner_observation_dim', 6318), ('compact_solver', 'dense_inverse')):
        if name in metadata and metadata[name] != value:
            failures.append(name)
    if failures:
        raise ValueError('diagnostic Actor metadata mismatch: ' + ', '.join(failures))
    actor = payload['actor']
    shapes = validate_parameter_tree(actor)
    zero_equivalence = (node == 0 and bool(np.all(actor['head']['w'] == 0))
                        and float(actor['head']['b'][0]) == 0.)
    if node == 0 and not zero_equivalence:
        raise ValueError('initial Actor is not identically zero in deterministic evaluation')
    receipt = {
        'path': str(path), 'sha256': actual_sha256, 'actual_global_interactions': node,
        'checkpoint_metadata': metadata, 'parameter_shapes': shapes,
        'initial_zero_equivalence': zero_equivalence,
        'derived_diagnostic_contract': {
            'learner_observation_dim': 6318, 'compact_solver': 'dense_inverse',
            'rhs_assembly': 'tensor', 'optimization_seed': OPTIMIZATION_SEED,
            'binding': 'formal rank0 report, saved layout, finite FP32 tree and frozen Actor hash',
            'checkpoint_metadata_was_modified': False,
        },
    }
    return actor, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run-root', type=Path, required=True)
    parser.add_argument('--rank0-report', type=Path, required=True)
    parser.add_argument('--rank0-report-sha256', required=True)
    parser.add_argument('--plane-archive', type=Path, required=True)
    parser.add_argument('--verified-archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-seconds-limit', type=int, default=GPU_SECONDS_LIMIT)
    parser.add_argument('--validate-only', action='store_true', help='CPU input contracts only; no LES')
    args = parser.parse_args()
    if not 0 < args.gpu_seconds_limit <= GPU_SECONDS_LIMIT:
        raise ValueError('P2 limit must be positive and no greater than 5 GPUh')
    if args.output_dir.exists():
        raise FileExistsError('refuse overwrite or silent repeat of learning-curve context')
    if sha256_file(args.rank0_report) != args.rank0_report_sha256:
        raise ValueError('formal rank0 report SHA256 mismatch')

    import jax
    import jax.numpy as jnp
    from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case, layout_contract
    from jax_mole_continuous_sac import ContinuousControlConfig
    from jax_mole_graph_sac_v2 import build_wake_graph, graph_v2_actor_deterministic
    from jax_mole_layouts import canonical_scale_layouts
    from mole_large_scale_evaluation import initialize_evaluation, scoring_boundary, make_step, make_scan, validate_actor_metadata
    from mole_large_scale_metrics import score_trace, paired_gain
    from mole_plane_archive import PlaneArchive

    if sha256_file(Path(__file__).with_name('jax_mole_graph_sac_v2.py')) != GRAPH_CODE_SHA256:
        raise ValueError('Graph Actor source differs from formal training stage')
    layout = canonical_scale_layouts()[81]
    case = make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse', rhs_assembly='tensor')
    expected_layout = layout_contract(case)
    report = json.loads(args.rank0_report.read_text())
    config_binding = validate_formal_report(report, expected_layout)
    config = ContinuousControlConfig(inlet_mode='time_planes')
    if config.control_dt_seconds != 10. or config.frame_stack != 1 or config.max_yaw_angle_deg != 40.:
        raise ValueError('original deterministic control contract changed')
    archive = PlaneArchive(args.plane_archive)
    verification = json.loads(args.verified_archive.read_text())
    archive_checks = {
        'source_A': archive.source_state_sha256 == SOURCE_STATE_SHA256,
        'source_seed': archive.manifest['source_state'].get('seed') == 26100181,
        'frozen_manifest': archive.manifest_sha256 == ARCHIVE_SHA256,
        'layout': archive.manifest.get('provenance', {}).get('target_layout_sha256') == layout.payload_sha256,
        'shape': tuple(archive.plane_shape) == (3, case.ny, case.nz),
        'full_verification': verification.get('all_chunk_hashes_verified') is True,
        'receipt_manifest': verification.get('manifest_sha256') == archive.manifest_sha256,
        'receipt_extension': verification.get('extension_manifest_sha256') == archive.extension_manifest_sha256,
        'training_extension': archive.extension_manifest_sha256 == config_binding.get('plane_archive_extension_sha256'),
        'horizon': PLANE_START + 50 * (BURNIN + INTERACTIONS) <= archive.completed_planes,
    }
    if not all(archive_checks.values()):
        raise ValueError('A archive binding failed: ' + repr(archive_checks))
    actors, identities = {}, {}
    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    for node in NODES:
        filename, digest = ACTOR_IDENTITIES[node]
        host_actor, identity = load_diagnostic_actor(args.training_run_root / filename, digest, node, expected_layout)
        actors[node] = jax.tree.map(jnp.asarray, host_actor)
        identities[node] = identity
    validate_actor_metadata(identities[1000448]['checkpoint_metadata'], case, config, 1000448)
    initial_actions = np.asarray(graph_v2_actor_deterministic(
        actors[0], jnp.ones((1, 6318), dtype=jnp.float32), graph, config.frame_stack))
    if initial_actions.shape != (1, 81) or not np.array_equal(initial_actions, np.zeros_like(initial_actions)):
        raise ValueError('initial deterministic zero-action proof failed on active graph code')
    if not args.validate_only and jax.default_backend() != 'gpu':
        raise ValueError('scientific evaluation requires the actual GPU LES runtime')
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / 'INPUTS_VERIFIED.json', {
        'status': 'pass', 'nodes': list(NODES), 'source_id': 'A',
        'rank0_report_sha256': args.rank0_report_sha256, 'archive_checks': archive_checks,
        'actors': identities, 'graph_source_sha256': GRAPH_CODE_SHA256,
        'final_BC_validator_unchanged': True, 'initial_zero_actor_checked': True,
        'scope': 'input and deterministic-policy contracts; no LES gain measured',
    })
    if args.validate_only:
        print(json.dumps({'status': 'validated_cpu_contracts', 'les_started': False}), flush=True)
        return

    started = time.monotonic()
    completed_branches = []
    scans = {}

    def status(phase, **extra):
        elapsed = time.monotonic() - started
        atomic_json(args.output_dir / 'STATUS.json', {
            'phase': phase, 'elapsed_seconds': elapsed, 'gpu_seconds_limit': args.gpu_seconds_limit,
            'completed_branches': completed_branches, 'training_started': False,
            'B_C_used': False, **extra,
        })
        if elapsed >= args.gpu_seconds_limit:
            raise TimeoutError('P2 one-GPU wall-clock budget exhausted; partial results retained')

    def collect(state, step, cursor, count, label):
        blocks, timings = [], []
        for offset in range(0, count, BLOCK_INTERACTIONS):
            status(label, completed_controls=offset)
            length = min(BLOCK_INTERACTIONS, count - offset)
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
            compute_seconds = time.monotonic() - compute_started
            host = jax.tree.map(np.asarray, metrics)
            if any(not np.isfinite(value).all() for value in jax.tree.leaves(host)):
                raise ValueError('nonfinite LES/policy trace')
            blocks.append(host)
            timings.append({'controls': length, 'input_io_seconds': io_seconds,
                            'compute_and_possible_compile_seconds': compute_seconds})
            cursor += length * 50
            print(json.dumps({'phase': label, 'completed_controls': offset + length,
                              'plane_cursor': cursor}), flush=True)
        trace = {name: np.concatenate([block[name] for block in blocks], axis=0) for name in blocks[0]}
        return state, trace, cursor, timings

    def save_trace(name, trace):
        path = args.output_dir / (name + '.npz')
        with path.open('xb') as stream:
            np.savez(stream, **trace)
        return {'trace': str(path), 'trace_sha256': sha256_file(path)}

    try:
        status('initializing')
        inlet = jnp.asarray(archive.read(PLANE_START, 1)[0][None])
        common, precursor, farm = initialize_evaluation(case, config, inlet, EVALUATION_SEED)
        zero_step = make_step(case, config, precursor, farm)
        common, burnin, scoring_cursor, burnin_timing = collect(common, zero_step, PLANE_START, BURNIN, 'common_burnin')
        burnin_record = save_trace('common_burnin', burnin)
        burnin_record.update(controls=BURNIN, excluded_from_scoring=True)
        del burnin
        common = scoring_boundary(common)
        common_path = args.output_dir / 'common_state.pkl'
        with common_path.open('xb') as stream:
            pickle.dump(jax.tree.map(np.asarray, common), stream, protocol=5)
        common_sha256 = sha256_file(common_path)
        scores, controllers = {}, {}
        for node in NODES:
            label = 'zero' if node == 0 else 'rl_step' + str(node)
            step = zero_step if node == 0 else make_step(case, config, precursor, farm, actor=actors[node], graph=graph)
            final_state, trace, final_cursor, timing = collect(common, step, scoring_cursor, INTERACTIONS, label)
            score = score_trace(trace, np.asarray(common.previous_yaw))
            if not score['rate_limit_satisfied'] or not score['angle_limit_satisfied']:
                raise ValueError('actual actuator constraint violation')
            record = save_trace(label, trace)
            record.update(actual_global_interactions=node, actor=identities[node],
                          common_state_sha256=common_sha256, scoring_cursor=scoring_cursor,
                          final_cursor_exclusive=final_cursor, timing=timing,
                          initial_actor_equivalent_zero=(node == 0))
            scores[label], controllers[label] = score, record
            completed_branches.append(label)
            atomic_json(args.output_dir / (label + '.json'), {'score': score, 'controller': record})
            del final_state, trace
        result = {
            'status': 'complete', 'context_id': 'n81_A_fixed_learning_curve_20261013',
            'nodes': list(NODES), 'source_id': 'A', 'source_state': archive.manifest['source_state'],
            'archive_manifest_sha256': archive.manifest_sha256,
            'layout_sha256': layout.payload_sha256, 'common_state_sha256': common_sha256,
            'window_start': PLANE_START, 'window_end_exclusive': scoring_cursor + INTERACTIONS * 50,
            'burnin_controls': BURNIN, 'scored_controls': INTERACTIONS, 'control_dt_seconds': 10.,
            'scores': scores, 'controllers': controllers,
            'pairs': {name: paired_gain(score, scores['zero']) for name, score in scores.items() if name != 'zero'},
            'common_burnin': burnin_record, 'burnin_timing': burnin_timing,
            'evaluation_numerics': {'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor', 'les_dt_seconds': .2},
            'rank0_report_sha256': args.rank0_report_sha256,
            'elapsed_wall_seconds': time.monotonic() - started,
            'boundary': 'fixed training-source development curve; no independent replication, checkpoint selection, learner or B/C tuning',
            'curve_monotonicity_required': False, 'negative_gains_retained': True,
        }
        atomic_json(args.output_dir / 'RESULT.json', result)
        status('complete', result_sha256=sha256_file(args.output_dir / 'RESULT.json'))
        print(json.dumps({'status': 'complete', 'pairs': result['pairs']}), flush=True)
    except BaseException as exc:
        atomic_json(args.output_dir / 'FAILURE.json', {
            'status': 'failed_or_budget_stopped', 'type': type(exc).__name__, 'reason': str(exc),
            'elapsed_seconds': time.monotonic() - started, 'completed_branches': completed_branches,
            'partial_results_retained': True,
        })
        raise


if __name__ == '__main__':
    main()
