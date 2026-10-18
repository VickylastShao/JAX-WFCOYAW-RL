"""CPU admission for N81 dt sensitivity; imports physics, never builds LES."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import load_actor_checkpoint
from jax_mole_graph_sac_v2 import build_wake_graph, graph_v2_actor_deterministic
from jax_mole_layouts import canonical_scale_layouts
from mole_large_scale_evaluation import validate_actor_metadata
from mole_plane_archive import PlaneArchive
from paper1_n81_timestep_sensitivity_20261013 import (
    FINAL_ACTOR_SHA256, SOURCE_START, temporal_contract, physical_control_config,
    sha256_file, array_identity, sample_frozen_inlet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plane-archive', type=Path, required=True)
    parser.add_argument('--verified-archive', type=Path, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refuse to overwrite the existing input-admission receipt')
    started = time.monotonic()
    if jax.default_backend() != 'cpu' or any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('this preflight must expose only CPU devices, never GPU')
    checks = {'actual_physics_import_succeeded': True, 'cpu_only_devices': True}
    actor_hash = sha256_file(args.actor)
    if actor_hash != FINAL_ACTOR_SHA256:
        raise ValueError('Actor differs from the exact frozen final R1 file')
    archive = PlaneArchive(args.plane_archive)
    verification = json.loads(args.verified_archive.read_text())
    layout = canonical_scale_layouts()[81]
    if (verification.get('status') != 'complete_chunk_hash_verification'
            or not verification.get('all_chunk_hashes_verified')
            or verification.get('manifest_sha256') != archive.manifest_sha256
            or verification.get('extension_manifest_sha256') != archive.extension_manifest_sha256
            or verification.get('target_turbines') != 81
            or verification.get('layout_sha256') != layout.payload_sha256
            or verification.get('checked', {}).get('completed_planes') != archive.completed_planes):
        raise ValueError('complete current archive/layout/hash receipt required')
    if (archive.manifest.get('provenance', {}).get('target_layout_sha256') != layout.payload_sha256
            or archive.plane_shape != (3, layout.ny, layout.nz)
            or float(archive.manifest.get('case', {}).get('dt', -1)) != .2
            or archive.manifest.get('dtype') != 'float32'
            or archive.completed_planes < 47501):
        raise ValueError('archive physical-time, N81 geometry, dtype or endpoint contract differs')
    checks['current_verified_archive_receipt_matches'] = True
    checks['archive_has_interpolation_endpoint'] = True

    cases, controls, actor_metadata = {}, {}, None
    coarse_dict = None
    for dt in (.2, .1):
        case = replace(make_scaled_mole_training_case(layout, 1, compact_solver='dense_inverse',
                                                     rhs_assembly='tensor'), dt=dt)
        config = physical_control_config(dt)
        if (config.control_dt_seconds != 10.
                or config.les_steps_per_interaction != (50 if dt == .2 else 100)
                or config.max_yaw_speed_deg_per_s != 1.
                or config.max_yaw_angle_deg != 40.
                or config.frame_stack != 1):
            raise ValueError('real imported control period/action contract differs')
        actor, metadata = load_actor_checkpoint(args.actor, expected_case=case,
                                                expected_action_parameterization='per_turbine')
        validate_actor_metadata(metadata, case, config, 1000448)
        leaves = jax.tree.leaves(actor)
        if not leaves or any(np.asarray(value).dtype != np.float32
                             or not np.isfinite(np.asarray(value)).all() for value in leaves):
            raise ValueError('frozen Actor contains wrong dtype or nonfinite weights')
        payload = asdict(case)
        without_dt = {key: value for key, value in payload.items() if key != 'dt'}
        if coarse_dict is None:
            coarse_dict = without_dt
        elif without_dt != coarse_dict:
            raise ValueError('more than dt changed between the two physical cases')
        cases[str(dt)] = payload
        controls[str(dt)] = {**asdict(config), 'actual_control_dt_seconds': config.control_dt_seconds}
        actor_metadata = metadata
    checks['real_coarse_and_fine_config_control_period_10_seconds'] = True
    checks['physical_cases_differ_only_in_dt'] = True
    checks['frozen_final_actor_layout_and_metadata_accepted_at_both_dt'] = True
    checks['actor_float32_finite'] = True

    graph = build_wake_graph(layout.turbine_positions_m, layout.diameter_m)
    action = np.asarray(graph_v2_actor_deterministic(actor, jnp.zeros((1, 81 * 78), jnp.float32), graph, 1))
    if action.shape != (1, 81) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
        raise ValueError('actual deterministic Graph-SAC inference violates the N81 action contract')
    checks['actual_actor_inference_81_finite_bounded_actions'] = True

    # Only five planes (about 1.2 MB) are read; no full chunks are rehashed here.
    # The large archive integrity statement comes from the current receipt.
    samples = {}
    for cursor in (SOURCE_START, SOURCE_START + 7500, 47500):
        plane = archive.read(cursor, 1)
        if plane.shape != (1, 3, layout.ny, layout.nz) or plane.dtype != np.float32 or not np.isfinite(plane).all():
            raise ValueError('actual sampled inlet plane violates float32/shape/finite contract')
        samples[str(cursor)] = array_identity(plane)
    knots = archive.read(SOURCE_START, 2)
    fine = sample_frozen_inlet(knots, .1)
    coarse = sample_frozen_inlet(knots, .2)
    if fine.shape[0] != 2 or not np.array_equal(fine[::2], coarse):
        raise ValueError('actual inlet interpolation changes the coarse frozen knots')
    checks['actual_sampled_planes_three_components_float32_finite'] = True
    checks['actual_frozen_inlet_knots_preserved_under_fine_sampling'] = True

    module_names = ('check_paper1_n81_timestep_inputs_20261013.py',
                    'paper1_n81_timestep_sensitivity_20261013.py',
                    'jax_mole_compact_sac_benchmark.py', 'jax_mole_continuous_sac.py',
                    'jax_mole_graph_sac_v2.py', 'mole_large_scale_evaluation.py',
                    'mole_plane_archive.py')
    result = {'status': 'inputs_verified', 'checked_at_cst': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'checks': checks, 'cpu_backend': jax.default_backend(),
              'runtime': {'jax': jax.__version__, 'numpy': np.__version__},
              'actor_sha256': actor_hash, 'actor_metadata': actor_metadata,
              'layout_sha256': layout.payload_sha256,
              'archive_manifest_sha256': archive.manifest_sha256,
              'archive_extension_manifest_sha256': archive.extension_manifest_sha256,
              'verified_archive_receipt_sha256': sha256_file(args.verified_archive),
              'completed_archive_planes': archive.completed_planes,
              'required_archive_end_exclusive': 47501,
              'source_state': archive.manifest['source_state'], 'sampled_planes': samples,
              'cases': cases, 'controls': controls,
              'temporal_contracts': {str(dt): temporal_contract(dt) for dt in (.2, .1)},
              'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in module_names},
              'elapsed_seconds': time.monotonic() - started,
              'les_built_or_advanced': False, 'gpu_experiment_started': False,
              'scope': 'CPU admission only; current archive chunk hash receipt reused, five planes read, no full LES, flow maturity or gain certified'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
