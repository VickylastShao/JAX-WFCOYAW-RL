# Simulation starting states

The public-release state set contains **eleven compressed records**, totalling **7,177,998,200 compressed bytes**. They are supplied as release assets separate from the scientific archive.

| Group | Number | Purpose |
|---|---:|---|
| Qualified turbulent source states | 3 | Preserve the accepted physical source starting conditions and their metadata. |
| B/C common-state evaluation starts | 8 | Preserve the paired scoring starts: four windows on each independently seeded source. |

The [browsable state manifest](INPUTS_MANIFEST.json), copied verbatim from the release asset `INPUTS_MANIFEST.json`, is the authority for public asset names, original record names, sizes, SHA-256 identities and source/window mapping. Public names may distinguish otherwise identically named files. Renaming a downloadable asset does not alter its payload; the manifest retains that mapping.

Download the state assets with `INPUTS_MANIFEST.json` and `SHA256SUMS-inputs.txt` from [release v1.0.0](https://github.com/VickylastShao/JAX-WFCOYAW-RL/releases/tag/v1.0.0). The public names are `n81_source_A_state.npz.gz`, `n81_source_B_state.npz.gz`, `n81_source_C_state.npz.gz`, and `n81_{B,C}_w{1,2,3,4}_common_state.pkl.gz`. Verify the compressed states before decompression:

```bash
sha256sum -c SHA256SUMS-inputs.txt
```

The release-wide `SHA256SUMS-portable-20261018.txt` also covers the scientific archive and accompanying release metadata. Where supplied by the manifest, also verify the uncompressed byte count and SHA-256 identity. Retain the accompanying source metadata when reusing a state.

These records preserve physical source and evaluation initial conditions. They are **not full training-resume checkpoints**, and they are not a substitute for the complete subsequent inlet histories. Full-window LES runs need those histories (or their documented regeneration), a compatible solver/runtime and the archived configuration. The preserved Docker image remains private and is not distributed as a public release asset.

The historical 80 s paired replay establishes the reported prefix only. Successful download, byte verification and CPU reanalysis do not demonstrate a complete physical rerun or cross-hardware bitwise reconstruction.
