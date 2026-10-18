"""Write a complete chunk-hash receipt; this does not certify flow maturity."""
import argparse
import json
import os
import time
from pathlib import Path

from jax_mole_layouts import canonical_scale_layouts
from mole_plane_archive import PlaneArchive, sha256_file, EXTENSION_MANIFEST_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--target-turbines', type=int, choices=(81, 100), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('keep the existing verification receipt; use a new output for a new verification')
    started = time.monotonic()
    manifest_hash = sha256_file(args.archive / 'manifest.json')
    extension = args.archive / EXTENSION_MANIFEST_NAME
    extension_hash = sha256_file(extension) if extension.exists() else None
    archive = PlaneArchive(args.archive)
    layout = canonical_scale_layouts()[args.target_turbines]
    if archive.manifest['status'] != 'complete':
        raise ValueError('entire inlet archive must be complete before verification')
    if (archive.manifest.get('provenance', {}).get('target_layout_sha256') != layout.payload_sha256
            or archive.plane_shape != (3, layout.ny, layout.nz)):
        raise ValueError('archive does not match the canonical layout and transverse grid')
    checked = archive.verify(hashes=True)
    if sha256_file(args.archive / 'manifest.json') != manifest_hash:
        raise ValueError('manifest changed while verifying; no receipt written')
    extension_hash_after = sha256_file(extension) if extension.exists() else None
    if extension_hash_after != extension_hash:
        raise ValueError('extension changed while verifying; no receipt written')
    result = {'status': 'complete_chunk_hash_verification', 'archive': str(args.archive.resolve()),
              'target_turbines': args.target_turbines, 'layout_sha256': layout.payload_sha256,
              'manifest_sha256': manifest_hash,
              'extension_manifest_sha256': extension_hash,
              'all_chunk_hashes_verified': True, 'checked': checked,
              'elapsed_seconds': time.monotonic() - started,
              'physical_maturity_qualified': False,
              'scope': 'File geometry and chunk integrity only; flow qualification and entire run horizon are separate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f'.tmp-{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
