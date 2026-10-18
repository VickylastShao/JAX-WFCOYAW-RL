"""Validate one complete large-scale run before starting distributed JAX."""
import argparse
import json
import pickle
from pathlib import Path

from jax_mole_layouts import canonical_scale_layouts
from mole_training_horizon import plane_horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--plane-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-checkpoint', type=Path, action='append')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    contract = plan['training']
    parts = args.run_id.split('_')
    n = int(parts[0][1:])
    if args.run_id not in contract['run_order'] or n not in (81, 100):
        raise ValueError('unknown frozen large-scale training identity')
    layout = canonical_scale_layouts()[n]
    manifest_path = args.plane_archive / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or manifest.get('dtype') != 'float32':
        raise ValueError('complete fp32 inlet archive required')
    if manifest.get('provenance', {}).get('target_layout_sha256') != layout.payload_sha256:
        raise ValueError('inlet archive layout identity mismatch')
    if manifest['plane_shape'] != [3, layout.ny, layout.nz]:
        raise ValueError('inlet plane geometry mismatch')
    members = manifest['member_offsets']
    if len(members) != contract['local_environments']:
        raise ValueError('archive members and rank-local environment count differ')
    ranks = contract['num_processes']
    if args.resume_checkpoint and len(args.resume_checkpoint) != ranks:
        raise ValueError('one full-state resume checkpoint per rank required')
    results = []
    states = []
    for rank in range(ranks):
        initial = [int(v) + plan['input_horizon']['rank_offset_add'][rank] for v in members]
        resume = None
        if args.resume_checkpoint:
            # These are the user's own project checkpoints, not untrusted downloads.
            with args.resume_checkpoint[rank].open('rb') as stream:
                resume = pickle.load(stream)
            if resume['process_id'] != rank or resume['num_processes'] != ranks:
                raise ValueError('checkpoint rank topology differs')
            if resume['configuration']['layout_contract']['num_turbines'] != n:
                raise ValueError('checkpoint turbine count differs')
            cursor = list(map(int, resume['plane_cursors']))
            before = int(resume['global_interactions'])
            remaining = contract['interactions'] - before
            block = contract['global_environments'] * contract['collection_steps']
            if remaining < block or remaining % block:
                raise ValueError('remaining interaction budget is not runnable block cadence')
            iterations = remaining // block
            states.append(before)
        else:
            cursor, before, iterations = initial, None, contract['iterations']
        result = plane_horizon(cursors=cursor, completed_planes=int(manifest['completed_planes']),
                              iterations=iterations, collection_steps=contract['collection_steps'],
                              global_environments=contract['global_environments'],
                              burnin_interactions=contract['burnin_interactions'],
                              episode_interactions=contract['episode_interactions'],
                              reset_interactions=contract['reset_interactions'],
                              resumed_global_interactions=before, warmup_cursors=initial,
                              planned_global_interactions=contract['interactions'])
        results.append({'rank': rank, 'iterations': iterations, 'horizon': result})
        del resume
    if states and len(set(states)) != 1:
        raise ValueError('ranks resume different completed global budgets')
    value = {'status': 'input_horizon_validated', 'run_id': args.run_id, 'turbines': n,
             'ranks': results, 'layout_sha256': layout.payload_sha256,
             'scope': 'Geometry and entire input budget only; physical qualification, all chunk hashes, memory and real learner validation still required.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': value['status'], 'ranks': ranks,
                      'exclusive_end': max(r['horizon']['required_archive_end_exclusive'] for r in results)}))


if __name__ == '__main__':
    main()
