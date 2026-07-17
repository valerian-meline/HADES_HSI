# HADES_HSI

Hyperspectral root-analysis pipeline for HADES HPX experiments.

The project aligns binary root masks to VNIR2 hyperspectral data, applies the same alignment to root-component masks, and exports spectral traits for each mask region.

## What It Does

- Loads VNIR2 hyperspectral `.bil` cubes with ENVI headers.
- Applies dark correction, invalid-pixel repair, spectral smoothing, and spatial denoising.
- Aligns RootCam masks to the VNIR2 image space.
- Uses ECC affine root refinement by default, with grid-IoU fallback.
- Applies the final root alignment to:
  - `main_root_mask`
  - `lateral_root_mask`
  - `node_mask`
  - `tip_mask`
- Treats root masks as skeleton masks.
- Treats node and tip masks as one-pixel landmark objects.
- Exports per-mask summary CSV files and optional per-pixel spectra CSV files.
- Creates master CSV files for downstream pipeline compatibility.

## Repository Layout

```text
hpx_hades/
  alignment.py      Alignment and mask-transform logic
  diagnostics.py    QC metrics and diagnostic exports
  exports.py        Band, overlay, summary, and pixel-spectra exports
  io.py             BIL/ENVI loading helpers
  master.py         Master CSV creation
  pipeline.py       Main VNIR2/root-mask processing pipeline
  preprocessing.py  Hyperspectral preprocessing
  utils.py          Path, ID, and shape helpers

scripts/
  run_hpx_hades.py                  Main command-line runner
  create_master_summary.py          Rebuild master CSVs from existing outputs
  benchmark_alignment_strategies.py Alignment method benchmark
  setup_venv.ps1                    Windows virtual-environment setup helper

HPX_HADES.py                        Backward-compatible wrapper
environment.yml                     Conda environment
requirements.txt                    Pip requirements
```

## Installation

### Option 1: Windows virtual environment

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\python.exe scripts\run_hpx_hades.py --help
```

### Option 2: Conda

```powershell
conda env create -f environment.yml
conda activate hpx-hades
python scripts\run_hpx_hades.py --help
```

## Run The Pipeline

```powershell
python scripts\run_hpx_hades.py `
  --vnir2-root-dir "E:\HADES_HPX_04-2026\VNIR2\Measurement" `
  --root-mask-dir "E:\HADES_HPX_04-2026\ROOT2_analysis"
```

Useful optional flags:

```powershell
--export-all-bands
--export-band-indices 130 180 290
--export-pixel-spectra-csv
--save-preprocessing-plots
--skip-master-summary
```

## Output Files

For each plant folder, the pipeline writes:

- `*_summary.csv`: metadata-first summary table with `mean`, `sum`, and `sd` spectra.
- `*_pixelspectra.csv`: optional metadata-first per-pixel spectra table.
- `*_OverlayQualityCheck.png`: alignment overlay showing main root, lateral roots, nodes, tips, and peri-root zone.
- preprocessing metrics and run-parameter JSON files.

At the VNIR analysis-folder level, the pipeline writes:

- `master_summary.csv`
- `master_pixelspectra.csv` when `--export-pixel-spectra-csv` is enabled

## Master CSV Rebuild

To rebuild the master summary from existing analysis outputs:

```powershell
python scripts\create_master_summary.py --analysis-root "E:\HADES_HPX_04-2026\VNIR2\Analysis"
```

To also rebuild the pixel-spectra master:

```powershell
python scripts\create_master_summary.py `
  --analysis-root "E:\HADES_HPX_04-2026\VNIR2\Analysis" `
  --include-pixel-spectra
```

## Alignment Strategy

Production alignment uses:

1. Leaf-mask alignment for coarse placement.
2. ECC affine root refinement against a root-enhanced VNIR band.
3. Grid-IoU fallback when ECC is unavailable or fails.

Additional methods can be compared with:

```powershell
python scripts\benchmark_alignment_strategies.py --help
```

See [ALIGNMENT_STRATEGIES.md](ALIGNMENT_STRATEGIES.md) for details.


