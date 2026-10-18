"""Bind a frozen final Actor and three existing sources before held-out use."""
import argparse
import json
from pathlib import Path

from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import ContinuousControlConfig, load_actor_checkpoint
from jax_mole_layouts import canonical_scale_layouts
from mole_large_scale_evaluation import validate_actor_metadata
from mole_plane_archive import PlaneArchive, sha256_file
from run_mole_large_scale_initial_stage import atomic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.batch / 'PLAN.json').read_text())
    actor_path = a.inputs / plan['actor_final']['relative_path']
    if sha256_file(actor_path) != plan['actor_final']['sha256']:
        raise ValueError('final Actor identity changed')
    case = make_scaled_mole_training_case(canonical_scale_layouts()[81], 1,
                                         compact_solver='dense_inverse', rhs_assembly='tensor')
    _, metadata = load_actor_checkpoint(actor_path, expected_case=case,
                                       expected_action_parameterization='per_turbine')
    validate_actor_metadata(metadata, case, ContinuousControlConfig(inlet_mode='time_planes'),
                            required_interactions=1000448)
    records = []
    for source in plan['sources']:
        archive_path = a.inputs / 'archives' / source['archive_name']
        archive = PlaneArchive(archive_path)
        receipt = json.loads((archive_path / 'VERIFIED.json').read_text())
        state = a.inputs / 'qualified_sources' / source['archive_name'] / 'state.npz'
        identity = archive.manifest['source_state']
        checks = {
            'frozen_manifest': archive.manifest_sha256 == source['manifest_sha256'],
            'prior_all_chunk_hash_receipt': receipt.get('all_chunk_hashes_verified') is True,
            'receipt_matches_manifest': receipt.get('manifest_sha256') == archive.manifest_sha256,
            'receipt_matches_extension': receipt.get('extension_manifest_sha256') == archive.extension_manifest_sha256,
            'source_seed': identity['seed'] == source['seed'],
            'source_state_identity': identity['sha256'] == source['source_state_sha256'],
            'current_source_state_hash': sha256_file(state) == source['source_state_sha256'],
            'full_archive_horizon': archive.completed_planes == source['completed_planes'],
            'layout': archive.manifest['provenance']['target_layout_sha256'] == canonical_scale_layouts()[81].payload_sha256,
        }
        if not all(checks.values()):
            raise ValueError('source identity failed: ' + source['source_id'] + ' ' + repr(checks))
        records.append({'source_id': source['source_id'], 'checks': checks,
                        'source_state': identity, 'manifest_sha256': archive.manifest_sha256})
    result = {'status': 'pass', 'actor_sha256': plan['actor_final']['sha256'],
              'actor_metadata': metadata, 'sources': records,
              'boundary': 'Integrity and contract only; does not prove statistical stationarity or positive gain.'}
    atomic(a.batch / 'INPUTS_VERIFIED.json', result)
    print(json.dumps({'status': 'pass', 'actor_sha256': result['actor_sha256'],
                      'sources': [r['source_id'] for r in records]}))


if __name__ == '__main__':
    main()
