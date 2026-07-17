# HPX HADES Module Layout

The script has been split into a package while keeping `HPX_HADES.py` as a compatibility wrapper.

Current structure:

- `hpx_hades/io.py`: BIL/header loading and JSON serialization.
- `hpx_hades/utils.py`: sample ID parsing, path matching, and shape helpers.
- `hpx_hades/preprocessing.py`: dark correction, invalid-pixel repair, smoothing, denoising.
- `hpx_hades/alignment.py`: leaf alignment, ECC affine root refinement, skeleton/landmark mask transforms.
- `hpx_hades/exports.py`: band PNGs, overlays, summary CSVs, optional per-pixel spectra.
- `hpx_hades/master.py`: combined master summary CSV creation from per-mask summary CSVs.
- `hpx_hades/diagnostics.py`: QC spectra and preprocessing metrics/plots.
- `scripts/run_hpx_hades.py`: thin entrypoint with input folders and output switches.
- `scripts/create_master_summary.py`: rebuild a master summary from an existing Analysis folder.
- `scripts/benchmark_alignment_strategies.py`: keep as the experimental alignment benchmark.

Speed-oriented defaults now used in `HPX_HADES.py`:

- Export only the root, visible-root, and leaf bands by default instead of every band PNG.
- Keep preprocessing diagnostic metrics CSVs, but skip diagnostic PNG plotting unless requested.
- Skip per-pixel spectra CSV exports unless requested.
- Write `master_summary.csv` by default after each VNIR analysis folder is processed.
- Write each per-mask `*_summary.csv` in the same metadata-first structure as `master_summary.csv`, so the master file can be rebuilt by concatenating compatible summaries.
- When per-pixel spectra CSV export is enabled, write each `*_pixelspectra.csv` with the same metadata-first structure and create a separate `master_pixelspectra.csv`.

To rebuild the master file from already exported summaries:

```powershell
python scripts\create_master_summary.py --analysis-root "E:\HADES_HPX_04-2026\VNIR2\Analysis"
python scripts\create_master_summary.py --analysis-root "E:\HADES_HPX_04-2026\VNIR2\Analysis" --include-pixel-spectra
```

Suggested next cleanup step:

1. Add small unit tests for alignment transforms and summary export.
2. Replace wildcard imports with explicit imports once behavior is stable.
3. Add a config file for default band indices, output switches, and paths.
