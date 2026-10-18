# Portable scientific reproduction package

This distribution contains the 3-, 9- and 81-turbine scientific records, trained
policies, controller comparisons, implementation sources and figure inputs. The
nine-turbine lookup result, all seven evaluated controllers, failed controls and
documented roster chronology remain included.

This is a new metadata derivative of the previously archived science delivery.
Historical machine locations and execution labels are normalized. Data arrays,
trained-policy bytes, numerical results and scientific numerical assertions are
preserved. Original archives and the detailed original-to-derived identity map
remain privately retained; this package has independently calculated identities.
Pure local dispatch and editorial-history copies are excluded.

`historical-source/` and `historical-runtime/` in retained records are logical
source-location namespaces, not claims that external inputs are included. Actual
supplied files use package-relative manifest paths. The dependency archive is
`artifacts/n81_20261016_portable_science.tar.gz`. Its `MANIFEST.json` lists the
actual derived members. The legacy outer key `immutable_base` is retained only
for validator compatibility; its explicit identity_kind identifies a derivative.

On Linux or WSL, use Python 3.12 with NumPy 2.3.5 for the saved-record CPU suite:

```sh
python -B verify_package.py --root . --output /tmp/package_sha.json
python -B run_cpu_acceptance.py --root . --workdir /tmp/new-science-acceptance
```

The CPU suite executes 16 saved-record checks, including the 959-value numerical
reanalysis, 12 prefix contract tests, nine-turbine independent-source checks and
resource-selection negative controls. It does not run physical LES or training.

For figure reconstruction, use a writable package copy with the separately
pinned plotting environment in requirements-plot.txt:

```sh
python -B rebuild_figures.py --final
```

The package includes 18 figure groups in PDF, SVG and PNG and reconstructs seven
current groups. Full LES evaluation windows require regenerated inlet histories
and a matching runtime. The separately archived 11 simulation starting states
are separate Release assets. The original Docker image is privately archived
and is not distributed here. Full training restoration and cross-hardware
bitwise equivalence are not established by saved-record acceptance.

Author-owned code is licensed under MIT. Author-generated data, trained policies
and figures are licensed under CC BY 4.0. Third-party material retains its own
licence and attribution in THIRD_PARTY_NOTICES.md and LICENSES/.
