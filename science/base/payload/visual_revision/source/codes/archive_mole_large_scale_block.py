"""Generate one bounded archive block and retain temporal inlet diagnostics.

The underlying frozen LES generator is unchanged. Reusing max-chunks=1
allows the supervisor to yield GPUs only after an atomic archive checkpoint.
"""
import json
import sys
from pathlib import Path

import numpy as np

import generate_mole_plane_archive as generator


def temporal_summary(planes, case, subblock_planes=5000):
    hub = round(case['hub_height'] / (case['domain_ly'] / (case['ny'] - 1)))
    u = planes[:, 0, hub, :]
    mean_u = np.mean(u, axis=1, dtype=np.float64)
    spatial_ti = np.std(u, axis=1, dtype=np.float64) / np.maximum(np.abs(mean_u), 1e-9)
    windows = []
    for start in range(0, len(mean_u), subblock_planes):
        stop = min(start + subblock_planes, len(mean_u))
        windows.append({'local_start': start, 'local_stop': stop,
                        'duration_seconds': (stop-start)*case['dt'],
                        'mean_hub_u_m_per_s': float(mean_u[start:stop].mean()),
                        'mean_spatial_ti': float(spatial_ti[start:stop].mean())})
    result = {'duration_seconds': len(mean_u)*case['dt'],
              'mean_hub_u_m_per_s': float(mean_u.mean()),
              'std_plane_mean_hub_u_m_per_s': float(mean_u.std()),
              'mean_spatial_ti': float(spatial_ti.mean()),
              'subblocks': windows,
              'mean_streamwise_profile_m_per_s': np.mean(planes[:, 0], axis=(0, 2), dtype=np.float64).tolist(),
              'precursor_advection_time_seconds': case['domain_lx'] / float(mean_u.mean()),
              'evidence_boundary': 'One source trajectory; subblocks are correlated. Spatial TI is not temporal point TI; no automatic stationarity/CV acceptance gate.'}
    return result, mean_u, spatial_ti


def main():
    args = generator.parse_args()
    if args.max_chunks != 1:
        raise ValueError('This supervisor entry requires --max-chunks 1')
    code = generator.main()
    manifest = json.loads((args.archive_dir / generator.MANIFEST_NAME).read_text())
    chunk = manifest['chunks'][-1]
    planes = np.load(args.archive_dir / chunk['path'], mmap_mode='r', allow_pickle=False)
    report, means, ti = temporal_summary(planes, manifest['case'])
    report.update(start=chunk['start'], stop=chunk['stop'], chunk_sha256=chunk['sha256'],
                  source_sha256=manifest['source_state']['sha256'])
    directory = args.archive_dir / 'temporal'
    directory.mkdir(exist_ok=True)
    stem = 'block_%09d_%09d' % (chunk['start'], chunk['stop'])
    generator._write_json_atomic(directory / (stem + '.json'), report)
    temp = directory / (stem + '.npz.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, plane_mean_hub_u=means, spatial_plane_ti=ti)
    temp.replace(directory / (stem + '.npz'))
    print(json.dumps({'temporal_report': str(directory / (stem + '.json')),
                      'completed_planes': manifest['completed_planes'],
                      'mean_hub_u_m_per_s': report['mean_hub_u_m_per_s'],
                      'mean_spatial_ti': report['mean_spatial_ti']}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
