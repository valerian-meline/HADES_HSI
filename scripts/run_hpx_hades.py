"""Run the HPX HADES hyperspectral root-analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hpx_hades import process_vnir_with_root_masks
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}\n\n"
        "Run this script from the project environment, for example:\n"
        "  .\\scripts\\setup_venv.ps1\n"
        "  .\\.venv\\Scripts\\python.exe scripts\\run_hpx_hades.py --vnir2-root-dir \"E:\\HADES_HPX_04-2026\\VNIR2\\Measurement\" --root-mask-dir \"E:\\HADES_HPX_04-2026\\ROOT2_analysis\"\n\n"
        "Or with conda, if conda is installed:\n"
        "  conda env create -f environment.yml\n"
        "  conda activate hpx-hades\n"
        "  python scripts\\run_hpx_hades.py --vnir2-root-dir \"E:\\HADES_HPX_04-2026\\VNIR2\\Measurement\" --root-mask-dir \"E:\\HADES_HPX_04-2026\\ROOT2_analysis\"\n\n"
        "If an environment already exists, activate it before rerunning."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vnir2-root-dir", required=True, help="VNIR2 Measurement directory")
    parser.add_argument("--root-mask-dir", required=True, help="ROOT2_analysis directory")
    parser.add_argument("--export-all-bands", action="store_true", help="Export every band PNG")
    parser.add_argument("--export-band-indices", nargs="*", type=int, default=None, help="Specific band indices to export")
    parser.add_argument("--export-pixel-spectra-csv", action="store_true", help="Export per-pixel spectra CSVs")
    parser.add_argument("--save-preprocessing-plots", action="store_true", help="Save preprocessing diagnostic PNG plots")
    parser.add_argument("--skip-master-summary", action="store_true", help="Do not write master_summary.csv files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_vnir_with_root_masks(
        args.vnir2_root_dir,
        args.root_mask_dir,
        export_all_bands=args.export_all_bands,
        export_band_indices=args.export_band_indices,
        export_pixel_spectra_csv=args.export_pixel_spectra_csv,
        save_preprocessing_plots=args.save_preprocessing_plots,
        write_master_file=not args.skip_master_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
