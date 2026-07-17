"""Create one master summary CSV from HPX HADES mask summary CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hpx_hades.master import write_master_pixel_spectra, write_master_summary
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}\n\n"
        "Install the project dependencies, then rerun this script. "
        "For example: .\\scripts\\setup_venv.ps1"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        required=True,
        help="Analysis folder to scan recursively, e.g. E:\\HADES_HPX_04-2026\\VNIR2\\Analysis",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <analysis-root>\\master_summary.csv",
    )
    parser.add_argument(
        "--include-pixel-spectra",
        action="store_true",
        help="Also build <analysis-root>\\master_pixelspectra.csv from *_pixelspectra.csv files",
    )
    parser.add_argument(
        "--pixel-output",
        default=None,
        help="Pixel spectra master CSV path. Defaults to <analysis-root>\\master_pixelspectra.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrote_any = False
    output_path = write_master_summary(args.analysis_root, args.output)
    if output_path is not None:
        print(f"Saved master summary to: {output_path}")
        wrote_any = True
    if args.include_pixel_spectra:
        pixel_output_path = write_master_pixel_spectra(args.analysis_root, args.pixel_output)
        if pixel_output_path is not None:
            print(f"Saved master pixel spectra to: {pixel_output_path}")
            wrote_any = True
    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
