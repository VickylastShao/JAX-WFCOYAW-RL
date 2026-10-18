#!/usr/bin/env python3
"""CPU-only identity/range contract; never imports JAX or advances LES.

This is a separate revision tool. Frozen source manifests and scientific code
are read-only. Extracting a prefix requires the complete original chunk SHA.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_chunks(manifest, start, stop):
    if manifest['dtype'] != 'float32':
        raise ValueError('float32 inlet required')
    if not 0 <= start < stop <= manifest['completed_planes']:
        raise ValueError('invalid half-open interval')
    if manifest['bytes_per_plane'] != 4 * math.prod(manifest['plane_shape']):
        raise ValueError('plane shape/byte count mismatch')
    selected, cursor = [], start
    for chunk in manifest['chunks']:
        if chunk['stop'] <= start or chunk['start'] >= stop:
            continue
        if chunk['start'] > cursor or chunk['stop'] <= cursor or (selected and chunk['start'] != selected[-1]['stop']):
            raise ValueError('gap or overlap in selected original chunks')
        if len(chunk['sha256']) != 64 or any(c not in '0123456789abcdef' for c in chunk['sha256']):
            raise ValueError('missing original SHA binding')
        selected.append(chunk)
        cursor = min(chunk['stop'], stop)
    if cursor != stop:
        raise ValueError('incomplete original chunk coverage')
    return selected


def prefix_plan(manifest_path, result_path, controls):
    if not 1 <= controls <= 32:
        raise ValueError('bounded prefix must contain 1..32 controls')
    manifest = json.loads(Path(manifest_path).read_text())
    result = json.loads(Path(result_path).read_text())
    if sha256(manifest_path) != result['archive_manifest_sha256']:
        raise ValueError('evaluation and original archive manifest differ')
    if manifest['source_state'] != result['source_state']:
        raise ValueError('source identity differs from original evaluation')
    if manifest['plane_shape'] != [3, 41, 483]:
        raise ValueError('81-turbine inlet shape differs')
    records = result['controllers']
    starts = {v['scoring_cursor'] for v in records.values()}
    hashes = {v['common_state_sha256'] for v in records.values()}
    if len(starts) != 1 or hashes != {result['common_state_sha256']}:
        raise ValueError('paired controllers do not share a start')
    burnin = result['common_burnin']
    start = starts.pop()
    if start != burnin['final_plane_cursor_exclusive']:
        raise ValueError('scoring cursor differs from common burnin endpoint')
    if burnin['final_farm_step_index'] != burnin['controls'] * 50:
        raise ValueError('recorded farm step and burnin differ')
    stop = start + 50 * controls
    return {
        'schema_version': 1, 'context_id': result['context_id'],
        'controls': controls, 'simulation_seconds_per_branch': controls * 10,
        'global_plane_start': start, 'global_plane_stop_exclusive': stop,
        'farm_step_index_expected_from_result': burnin['final_farm_step_index'],
        'state_contents_decoded_this_check': False,
        'plane_shape': manifest['plane_shape'], 'dtype': 'float32',
        'payload_bytes': (stop - start) * manifest['bytes_per_plane'],
        'common_state_sha256': result['common_state_sha256'],
        'archive_manifest_sha256': sha256(manifest_path),
        'source_state': manifest['source_state'],
        'original_chunks': select_chunks(manifest, start, stop),
        'physical_replay_executed': False,
        'purpose': 'identity and interface verification, not a new gain experiment',
    }


def extract_verified_chunk(manifest_path, result_path, controls, chunk_path, output):
    import numpy as np
    plan = prefix_plan(manifest_path, result_path, controls)
    if len(plan['original_chunks']) != 1:
        raise ValueError('single-chunk extraction only; shorten prefix or implement reviewed multi-chunk extraction')
    chunk = plan['original_chunks'][0]
    chunk_path, output = Path(chunk_path), Path(output)
    if output.exists() or output.with_suffix('.receipt.json').exists():
        raise FileExistsError('refuse to replace a prefix or receipt')
    if chunk_path.stat().st_size != chunk['bytes'] or sha256(chunk_path) != chunk['sha256']:
        raise ValueError('full original chunk size/SHA mismatch')
    array = np.load(chunk_path, mmap_mode='r', allow_pickle=False)
    if array.dtype != np.dtype('float32') or list(array.shape) != [chunk['stop']-chunk['start'], *plan['plane_shape']]:
        raise ValueError('original array header differs from manifest')
    lo, hi = plan['global_plane_start']-chunk['start'], plan['global_plane_stop_exclusive']-chunk['start']
    prefix = array[lo:hi]
    if not np.isfinite(prefix).all():
        raise ValueError('nonfinite inlet prefix')
    with output.open('xb') as f:
        np.save(f, prefix, allow_pickle=False)
    plan['extracted_prefix'] = {'path': str(output), 'bytes': output.stat().st_size,
                                'sha256': sha256(output), 'shape': list(prefix.shape),
                                'original_full_chunk_sha_verified': True}
    output.with_suffix('.receipt.json').write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--controls', type=int, default=8)
    p.add_argument('--original-chunk', type=Path)
    p.add_argument('--extract-to', type=Path)
    a = p.parse_args()
    if bool(a.original_chunk) != bool(a.extract_to):
        p.error('original-chunk and extract-to must be supplied together')
    result = extract_verified_chunk(a.manifest, a.result, a.controls, a.original_chunk, a.extract_to) if a.extract_to else prefix_plan(a.manifest, a.result, a.controls)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
