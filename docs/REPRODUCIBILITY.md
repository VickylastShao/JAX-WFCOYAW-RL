# Reproducing the saved-record results

The browsable Git tree is intended for inspection. The amended public scientific archive retains the scientific layout and numerical data required by the validation wrappers, with portable preparation metadata. Download it from [release v1.0.0](https://github.com/VickylastShao/JAX-WFCOYAW-RL/releases/tag/v1.0.0), together with its `.sha256` sidecar.

## 1. Check the received archive and its payload

Manuscript identifiers inside scientific snapshot metadata refer to the archived scientific edition; current submission files are maintained separately.

Run the following in the directory containing both downloaded files:

```bash
sha256sum -c n81_20261018_portable_science.tar.gz.sha256
tar -xzf n81_20261018_portable_science.tar.gz
cd n81_20261018_portable_science
python3 verify_package.py
```

Extract on Linux-native storage. WSL users should prefer their Linux home directory or `/tmp`: original validation bounds and POSIX path checks are retained. Keep the received archive and one verified extracted copy unchanged.

The archive's `SHA256SUMS.txt` covers every payload member and both manifest files, excluding itself. Its external sidecar binds the compressed archive. These checks establish file identity, not a physical LES rerun.

## 2. Run the 16 saved-record CPU checks

Use Python 3.12 and the archive's CPU requirements, including **NumPy exactly 2.3.5**:

```bash
python3.12 -m venv "$HOME/wind-energy-cpu"
"$HOME/wind-energy-cpu/bin/python" -m pip install -r requirements-cpu.txt
"$HOME/wind-energy-cpu/bin/python" -B run_cpu_acceptance.py --workdir /tmp/wind-energy-v1-check
```

Choose a new work directory outside the immutable extracted delivery. Network may be needed to create the environment, but the checks then operate on saved local records. Successful completion requires `status=complete`, `exit_code=0` and `CPU_command_count=16` in the recorded result. The work directory retains the job handle, logs, numerical outputs and exit record.

This suite checks saved numerical evidence, prefix structure, resource analysis, exporter negative controls, and historical/current navigation consistency. It verifies all 18 current figure groups and their 54 formal exports. It runs no actor inference, training or physical LES integration.

## 3. Rebuild seven figure groups

Make a separate writable copy of the verified extracted archive. In that copy, use a plotting environment separate from the CPU-check environment:

```bash
python3.12 -m venv "$HOME/wind-energy-plots"
"$HOME/wind-energy-plots/bin/python" -m pip install -r requirements-plot.txt
"$HOME/wind-energy-plots/bin/python" -B rebuild_figures.py --final
```

The entry point reconstructs main Figures 2, 4, 5, 6 and 7 and supplementary Figures S9–S10 from saved numerical inputs. Running without `--final` creates colour/grayscale previews and numeric-artist records without replacing formal figure exports. The other eleven groups are retained with their existing source provenance.

The wrapper resolves package paths independently of the caller's working directory. Plotting may change font metrics or rasterization across platforms. Inspect regenerated figures locally; byte-identical PDF/SVG output is not a requirement or promise.

## 4. Inspect physical source and evaluation starts

The eleven compressed state assets are separate from the scientific archive. Their manifest maps public asset names to the original records and gives both compressed and uncompressed identities. See [STATE_FILES.md](STATE_FILES.md).

Starting states alone do not supply complete subsequent inlet histories or a matching LES runtime. The historic paired replay checks 80 s, and the CPU suite checks saved records. Neither establishes full-window physical replay, a complete training restart or cross-hardware bitwise flow reproduction.

The archive's [`EXECUTION_GUIDE.en.md`](ARCHIVE_EXECUTION_GUIDE.en.md) describes wrapper behaviour and time bounds. Preparation-time machine paths and packaging references have been updated; the amendment record identifies these changes. Scientific numerical records and original scientific file names remain available.
