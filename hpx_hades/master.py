"""Build master summary tables from HPX HADES mask summary CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MASTER_METADATA_COLUMNS = [
    "Measuring Date",
    "Measuring Time",
    "Experiment ID",
    "Round Order",
    "Tray ID",
    "Tray Info",
    "Plant ID",
    "Position",
    "Plant Name",
    "Plant Info",
    "PID",
    "Camera Position",
]

MASTER_CONTEXT_COLUMNS = [
    "Mask Type",
    "STAT",
]

MASTER_BASE_COLUMNS = MASTER_METADATA_COLUMNS + MASTER_CONTEXT_COLUMNS
PIXEL_SPECTRA_CONTEXT_COLUMNS = ["Mask Type", "y", "x"]
REMOVED_MASTER_COLUMNS = {"AREA_PX", "BIL File", "Source Mask Path", "Source CSV"}
SOURCE_SUMMARY_COLUMNS = {"bil_file", "root_mask_path", "plant_id", "n_pixels", "stat"}
SOURCE_PIXEL_SPECTRA_COLUMNS = SOURCE_SUMMARY_COLUMNS | {"y", "x"}

ANALYSIS_DIR_RE = re.compile(
    r"^(?P<experiment_id>\d+)_(?P<round_order>\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<tray_info>.+)_(?P<pid>VNIR\d+)_Data$"
)

PLANT_RE = re.compile(r"plant_(?P<plant_name>\d+)", re.IGNORECASE)


def _empty_metadata():
    return {column: "" for column in MASTER_METADATA_COLUMNS}


def _parse_analysis_dir(path):
    match = ANALYSIS_DIR_RE.match(path.name)
    if not match:
        return {}

    tray_info = match.group("tray_info")
    tray_id = tray_info.rsplit("_", 1)[-1] if tray_info else ""
    return {
        "Measuring Date": match.group("date"),
        "Measuring Time": match.group("time"),
        "Experiment ID": match.group("experiment_id"),
        "Round Order": match.group("round_order"),
        "Tray ID": tray_id,
        "Tray Info": tray_info,
        "PID": match.group("pid"),
    }


def _find_analysis_dir(path):
    path = Path(path)
    for parent in [path, path.parent, *path.parents]:
        if _parse_analysis_dir(parent):
            return parent
    return None


def _plant_id_from_summary_path(summary_path):
    for part in reversed(summary_path.parts):
        match = PLANT_RE.fullmatch(part)
        if match:
            return part

    match = PLANT_RE.search(summary_path.stem)
    return match.group(0) if match else ""


def _plant_metadata(plant_id):
    metadata = {
        "Plant ID": plant_id,
        "Position": plant_id,
        "Plant Name": "",
        "Plant Info": "",
    }
    match = PLANT_RE.fullmatch(plant_id or "")
    if match:
        metadata["Plant Name"] = match.group("plant_name")
    return metadata


def _strip_summary_suffix(stem):
    for suffix in ("_summary_wide", "_summary"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _mask_type_from_summary_path(summary_path, plant_id):
    base = _strip_summary_suffix(summary_path.stem)
    return mask_type_from_label(base, plant_id)


def metadata_for_plant_summary(plant_folder, plant_id, area_px=None):
    """Return master-format metadata for a per-plant summary output."""
    metadata = _empty_metadata()
    analysis_dir = _find_analysis_dir(plant_folder)
    if analysis_dir:
        metadata.update(_parse_analysis_dir(analysis_dir))
    metadata.update(_plant_metadata(plant_id))
    return metadata


def mask_type_from_label(label, plant_id=None):
    """Return the mask type from labels like plant_1_main_root_mask."""
    base = _strip_summary_suffix(Path(str(label)).stem)
    if plant_id and base.startswith(f"{plant_id}_"):
        return base[len(plant_id) + 1 :]
    if plant_id and base == plant_id:
        return ""

    match = PLANT_RE.search(base)
    if match:
        found_plant_id = match.group(0)
        if base.startswith(f"{found_plant_id}_"):
            return base[len(found_plant_id) + 1 :]
    return base


def _normalize_stat(value):
    value = "" if pd.isna(value) else str(value)
    return "sd" if value.lower() == "std" else value


def _relative_or_absolute(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _summary_files(root):
    candidates = []
    for pattern in ("*_summary.csv", "*_summary_wide.csv"):
        candidates.extend(root.rglob(pattern))

    excluded = {"master_summary.csv", "master_summary_wide.csv"}
    return sorted(
        {
            path
            for path in candidates
            if path.is_file()
            and path.name not in excluded
            and not path.name.endswith("_pixelspectra.csv")
        }
    )


def _pixel_spectra_files(root):
    excluded = {"master_pixelspectra.csv"}
    return sorted(
        {
            path
            for path in root.rglob("*_pixelspectra.csv")
            if path.is_file() and path.name not in excluded
        }
    )


def _trait_columns(columns):
    excluded = SOURCE_SUMMARY_COLUMNS | set(MASTER_BASE_COLUMNS) | REMOVED_MASTER_COLUMNS
    return [column for column in columns if column not in excluded]


def _pixel_spectra_columns(columns):
    excluded = (
        SOURCE_PIXEL_SPECTRA_COLUMNS
        | set(MASTER_METADATA_COLUMNS)
        | set(PIXEL_SPECTRA_CONTEXT_COLUMNS)
        | set(MASTER_CONTEXT_COLUMNS)
        | REMOVED_MASTER_COLUMNS
    )
    return [column for column in columns if column not in excluded]


def _trait_sort_key(column):
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)nm", str(column))
    if match:
        return (0, float(match.group(1)), str(column))
    return (1, str(column))


def master_column_order(trait_columns):
    return MASTER_BASE_COLUMNS + sorted(trait_columns, key=_trait_sort_key)


def pixel_spectra_column_order(trait_columns):
    return MASTER_METADATA_COLUMNS + PIXEL_SPECTRA_CONTEXT_COLUMNS + sorted(trait_columns, key=_trait_sort_key)


def _is_master_formatted(summary_df):
    return all(column in summary_df.columns for column in MASTER_METADATA_COLUMNS) and "STAT" in summary_df.columns


def _is_pixel_spectra_master_formatted(pixel_df):
    required = set(MASTER_METADATA_COLUMNS) | {"Mask Type", "y", "x"}
    return all(column in pixel_df.columns for column in required)


def build_master_summary(analysis_root):
    """Return a master summary DataFrame built from summary CSV files under analysis_root."""
    analysis_root = Path(analysis_root)
    rows = []
    trait_columns = set()
    string_columns = {column: "string" for column in MASTER_BASE_COLUMNS}

    for summary_path in _summary_files(analysis_root):
        summary_df = pd.read_csv(summary_path, dtype=string_columns, keep_default_na=False)
        if _is_master_formatted(summary_df):
            traits = _trait_columns(summary_df.columns)
            trait_columns.update(traits)
            for _, source_row in summary_df.iterrows():
                row = {column: source_row.get(column, "") for column in MASTER_BASE_COLUMNS}
                row["STAT"] = _normalize_stat(row.get("STAT", ""))
                for column in traits:
                    row[column] = source_row[column]
                rows.append(row)
            continue

        plant_id = _plant_id_from_summary_path(summary_path)
        analysis_dir = _find_analysis_dir(summary_path)
        metadata_from_analysis = _parse_analysis_dir(analysis_dir) if analysis_dir else {}
        mask_type = _mask_type_from_summary_path(summary_path, plant_id)
        traits = _trait_columns(summary_df.columns)
        trait_columns.update(traits)

        for _, source_row in summary_df.iterrows():
            row = _empty_metadata()
            row.update(metadata_from_analysis)
            row.update(_plant_metadata(plant_id))
            row.update(
                {
                    "Mask Type": mask_type,
                    "STAT": _normalize_stat(source_row.get("stat", "")),
                }
            )
            for column in traits:
                row[column] = source_row[column]
            rows.append(row)

    return pd.DataFrame(rows, columns=master_column_order(trait_columns))


def _pixel_spectra_header_columns(pixel_spectra_files):
    spectra_columns = set()
    for pixel_path in pixel_spectra_files:
        header_df = pd.read_csv(pixel_path, nrows=0)
        spectra_columns.update(_pixel_spectra_columns(header_df.columns))
    return sorted(spectra_columns, key=_trait_sort_key)


def _convert_pixel_spectra_chunk(chunk, pixel_path, analysis_root, spectra_columns):
    if _is_pixel_spectra_master_formatted(chunk):
        row_data = {
            column: chunk[column] if column in chunk.columns else ""
            for column in MASTER_METADATA_COLUMNS + PIXEL_SPECTRA_CONTEXT_COLUMNS
        }
    else:
        plant_id = _plant_id_from_summary_path(pixel_path)
        analysis_dir = _find_analysis_dir(pixel_path)
        metadata_from_analysis = _parse_analysis_dir(analysis_dir) if analysis_dir else {}
        metadata = _empty_metadata()
        metadata.update(metadata_from_analysis)
        metadata.update(_plant_metadata(plant_id))
        mask_type = mask_type_from_label(pixel_path.stem.replace("_pixelspectra", ""), plant_id)

        row_data = {column: metadata.get(column, "") for column in MASTER_METADATA_COLUMNS}
        row_data.update(
            {
                "Mask Type": mask_type,
                "y": chunk["y"] if "y" in chunk.columns else "",
                "x": chunk["x"] if "x" in chunk.columns else "",
            }
        )

    out_df = pd.DataFrame(row_data)
    for column in spectra_columns:
        out_df[column] = chunk[column] if column in chunk.columns else ""
    return out_df[pixel_spectra_column_order(spectra_columns)]


def write_master_pixel_spectra(analysis_root, output_path=None, chunksize=100_000):
    """Write a master CSV from all per-pixel spectra CSV files under analysis_root."""
    analysis_root = Path(analysis_root)
    pixel_spectra_files = _pixel_spectra_files(analysis_root)
    if not pixel_spectra_files:
        print(f"No pixel spectra CSV files found under: {analysis_root}")
        return None

    output_path = Path(output_path) if output_path is not None else analysis_root / "master_pixelspectra.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    spectra_columns = _pixel_spectra_header_columns(pixel_spectra_files)
    string_columns = {
        column: "string"
        for column in MASTER_METADATA_COLUMNS + PIXEL_SPECTRA_CONTEXT_COLUMNS + list(REMOVED_MASTER_COLUMNS)
    }

    wrote_header = False
    for pixel_path in pixel_spectra_files:
        for chunk in pd.read_csv(pixel_path, dtype=string_columns, keep_default_na=False, chunksize=chunksize):
            out_df = _convert_pixel_spectra_chunk(chunk, pixel_path, analysis_root, spectra_columns)
            out_df.to_csv(output_path, index=False, mode="a", header=not wrote_header)
            wrote_header = True

    return output_path


def write_master_summary(analysis_root, output_path=None):
    """Write a master CSV from all summary CSV files under analysis_root."""
    analysis_root = Path(analysis_root)
    output_path = Path(output_path) if output_path is not None else analysis_root / "master_summary.csv"
    master_df = build_master_summary(analysis_root)
    if master_df.empty:
        print(f"No summary CSV files found under: {analysis_root}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False)
    return output_path
