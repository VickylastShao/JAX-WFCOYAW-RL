# Scientific source mirror

All 210 indexed files are byte-for-byte copies from the current portable scientific archive. `SOURCE_INDEX.json` records each current archive, member, byte size and SHA-256. Preparation metadata, machine paths and package navigation were made portable in the 18 October 2026 packaging amendment; original scientific numerical records are retained. Historical `mole` source identifiers describe provenance and do not rename the current method.

This is a **browseable code mirror**. Training, physical evaluation and figure generation require the complete dependency and numerical-input trees in the Release archive; this smaller Git tree does not provide those data. Copying a script out of its recorded scientific tree is not a full training recipe. Use the frozen archive's execution guide for reproduction. Some original entry points refer to the original execution layout and require resolving the archived runtime recipes and saved states on the target system.

| Purpose | Browseable original source |
|---|---|
| LES compact/staggered operators | `base/payload/visual_revision/source/codes/incompact3d_staggered.py`, `incompact3d_filter.py` |
| 81-turbine distributed SAC collector and learner | `base/payload/visual_revision/source/codes/jax_mole_microbatch_distributed.py` |
| Continuous SAC and compact LES coupling | `base/payload/visual_revision/source/codes/jax_mole_continuous_sac.py`, `jax_mole_compact_sac_benchmark.py` |
| 81-turbine paired source/evaluation workflow | `base/payload/visual_revision/source/codes/run_mole_n81_final_evaluation.py`, `eval_mole_large_scale_paired.py`, `summarize_mole_n81_final_evaluation.py` |
| Training-stage timing and collection ablation | `base/payload/visual_revision/source/codes/profile_mole_n81_efficiency.py`, `paper1_n81_collection_ablation_20261013.py` |
| Three-turbine source-stage bridge | `base/mole_reference/validate_n3_bridge.py`, `base/mole_reference/historical_source/scripts/` |
| Nine-turbine training and seven-controller evaluation | `base/n9_intermediate/source/training/`, `base/n9_intermediate/source/evaluation/`, `base/n9_intermediate/validate_n9_bridge.py` |
| LES time-step and microbatch checks | `base/payload/wind_energy_closure/source/codes/paper1_wind_energy_closure_20261014.py` |
| Low-workspace LES implementation | `base/resource_uplift_20261016/extension/low_workspace/streaming_adm.py`, `benchmark_full_flow.py` |
| Resource-selection analysis | `base/resource_uplift_20261016/extension/log_analysis/` |
| Saved-prefix contract checks | `base/payload/prefix_replay/acceptance/`, `base/payload/prefix_replay/prior_mainline/scripts/` |
| Saved-record CPU acceptance wrapper | `current/run_cpu_acceptance.py` |
| Seven current figure groups | `current/rebuild_figures.py`, `current/redesign/` |
| Other retained scientific figures and validators | `current/figure_sources/` |

Run the lightweight source mirror check from the Git repository root:

```bash
python3 scripts/verify_science_sources.py
```

That check uses only the Python standard library. It verifies SHA and parses Python syntax; it does not run LES, training, Actor inference or numerical reproduction.

For actual saved-record arithmetic acceptance, extract the complete public science Release archive on Linux-native storage and follow its `EXECUTION_GUIDE.en.md`. Use Python 3.12 and the separately pinned NumPy 2.3.5 environment. Its `run_cpu_acceptance.py --workdir /tmp/<new-directory>` verifies the archive and runs 16 bounded saved-record checks. The separate `rebuild_figures.py --final` entry rebuilds seven current figure groups in a writable complete copy; it requires the plotting environment, fonts and input data supplied there.

`requirements-cpu.txt` pins the saved-record acceptance environment. `requirements-plot.txt` describes the separate plotting environment. `requirements-les-observed.txt` preserves the historical JAX/Flax/Optax runtime record and is not a promise that dependency installation alone reconstructs the full original GPU environment. The reference TorchRL dependency record is retained under its original three-turbine historical path.

Third-party notices and license texts are preserved under `science/THIRD_PARTY_NOTICES.md` and `science/LICENSES/`. Their scope remains distinct from the authors' original code and generated scientific data. No dependency binaries, machine-private access configuration, manuscript TeX, full-paper PDF, Docker image or large simulation data are included in this mirror. Original operational snapshots, full input/result trees and provenance remain in the fixed Release archive.
