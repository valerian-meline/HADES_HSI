from .common import *
from .master import (
    mask_type_from_label,
    metadata_for_plant_summary,
    master_column_order,
    pixel_spectra_column_order,
)


def summarize_and_save_pixel_data(bil_path, plant_folder, cube, aligned_mask, wavelengths, plant_name, root_mask_path):
    """
    Save a master-compatible summary CSV for one mask ROI.
    - Columns: metadata/context columns, then 1 column per wavelength
    - Rows: mean, sum, sd for all pixels in the mask
    """

    if not np.any(aligned_mask):
        print(f"⚠️ No pixels found in mask for {plant_name}, skipping summary.")
        return

    # Ensure mask is boolean
    mask_bool = aligned_mask.astype(bool)

    # Use it to select pixels across all bands
    pixel_values = cube[mask_bool, :]  # shape = (n_pixels, n_bands)
    n_pixels = pixel_values.shape[0]

    # Compute statistics
    mean_vals = np.nanmean(pixel_values, axis=0)
    sum_vals = np.nansum(pixel_values, axis=0)
    std_vals = np.nanstd(pixel_values, axis=0)

    out_path = plant_folder / f"{plant_name}_summary.csv"
    plant_id = Path(plant_folder).name
    wavelength_columns = [f"{wl:.1f}nm" for wl in wavelengths]

    base_info = metadata_for_plant_summary(plant_folder, plant_id)
    base_info.update(
        {
            "Mask Type": mask_type_from_label(plant_name, plant_id),
        }
    )

    df = pd.DataFrame([
        {**base_info, **dict(zip(wavelength_columns, mean_vals)), "STAT": "mean"},
        {**base_info, **dict(zip(wavelength_columns, sum_vals)), "STAT": "sum"},
        {**base_info, **dict(zip(wavelength_columns, std_vals)), "STAT": "sd"},
    ])

    df = df[master_column_order(wavelength_columns)]

    df.to_csv(out_path, index=False)
    print(f"📊 Saved summary to: {out_path}")


def _band_export_filename(export_dir, band_index, wavelengths=None):
    if wavelengths is None or band_index >= len(wavelengths):
        return export_dir / f"band_{band_index:03}.png"

    wavelength = float(wavelengths[band_index])
    wavelength_label = f"{wavelength:.1f}".replace("-", "minus")
    filename = export_dir / f"wavelength_{wavelength_label}nm.png"
    if filename.exists():
        filename = export_dir / f"wavelength_{wavelength_label}nm_band_{band_index:03}.png"
    return filename


def export_bands(denoised_vnir2, output_dir, export_all_bands=False, target_band=160, wavelengths=None):
    # Create export directory
    export_dir = Path(output_dir) / "band_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    if export_all_bands:
        band_indices = range(denoised_vnir2.shape[2])
    elif isinstance(target_band, (list, tuple, set)):
        band_indices = sorted({int(idx) for idx in target_band})
    else:
        band_indices = [int(target_band)]

    for band_index in band_indices:
        band = denoised_vnir2[:, :, band_index]
        band_vis = np.clip(band, 0, None)  # visualization only
        vmin, vmax = np.nanmin(band_vis), np.nanmax(band_vis)

        if vmax - vmin == 0:
            band_img = np.zeros_like(band_vis)
        else:
            band_img = np.clip((band_vis - vmin) / (vmax - vmin), 0, 1)

        band_img_8bit = (band_img * 255).astype(np.uint8)
        filename = _band_export_filename(export_dir, band_index, wavelengths)
        skimage.io.imsave(str(filename), band_img_8bit)

    print(f"Exported {len(list(band_indices))} band(s) to: {export_dir}")


def export_pixel_spectra(bil_path, plant_folder, cube, aligned_mask, wavelengths, plant_name, root_mask_path,
                         save_as="npy", max_csv_pixels=None):
    """
    Export per-pixel spectra inside ROI.

    Parameters
    ----------
    bil_path : Path
        Path to original .bil file.
    plant_folder : Path
        Directory to save results.
    cube : np.ndarray
        Hyperspectral cube (H x W x B).
    aligned_mask : np.ndarray
        Boolean mask (H x W) selecting ROI pixels.
    wavelengths : list/array
        List of band wavelengths (length B).
    plant_name : str
        Plant identifier.
    root_mask_path : Path
        Path to root mask file.
    save_as : str
        "npy" (default, compact binary) or "csv" (large, human-readable).
    max_csv_pixels : int or None
        If saving as CSV, optionally limit to N pixels to avoid huge files.
    """
    if not np.any(aligned_mask):
        print(f"⚠️ No pixels found in mask for {plant_name}, skipping pixel spectra export.")
        return

    # Flatten mask & extract spectra
    pixel_coords = np.argwhere(aligned_mask)  # (n_pixels, 2)
    yy, xx = pixel_coords[:, 0], pixel_coords[:, 1]
    spectra = cube[yy, xx, :]
    n_pixels = spectra.shape[0]

    out_base = plant_folder / f"{plant_name}_pixelspectra"
    plant_id = Path(plant_folder).name
    wavelength_columns = [f"{wl:.1f}nm" for wl in wavelengths]
    base_info = metadata_for_plant_summary(plant_folder, plant_id)
    base_info.update({"Mask Type": mask_type_from_label(plant_name, plant_id)})

    if save_as == "npy":
        # Save spectra + metadata in compressed format
        np.savez_compressed(
            out_base.with_suffix(".npz"),
            spectra=spectra.astype(np.float32),
            wavelengths=np.array(wavelengths, dtype=np.float32),
            coords=pixel_coords,
            **base_info
        )
        print(f"💾 Saved per-pixel spectra to: {out_base.with_suffix('.npz')} "
              f"(n_pixels={n_pixels}, n_bands={len(wavelengths)})")

    elif save_as == "csv":
        if max_csv_pixels is not None and spectra.shape[0] > max_csv_pixels:
            idx = np.random.choice(spectra.shape[0], max_csv_pixels, replace=False)
            spectra = spectra[idx]
            pixel_coords = pixel_coords[idx]
            print(f"⚠️ CSV export limited to {max_csv_pixels} pixels to avoid huge file.")

        df = pd.DataFrame(
            spectra,
            columns=wavelength_columns
        )
        df.insert(0, "x", pixel_coords[:, 1])
        df.insert(0, "y", pixel_coords[:, 0])
        df.insert(0, "Mask Type", base_info["Mask Type"])
        for key, val in reversed(list(metadata_for_plant_summary(plant_folder, plant_id).items())):
            df.insert(0, key, val)
        df = df[pixel_spectra_column_order(wavelength_columns)]

        out_path = out_base.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved per-pixel spectra to: {out_path} "
              f"(n_pixels={n_pixels}, n_bands={len(wavelengths)})")

    else:
        raise ValueError(f"Unknown save_as format: {save_as}")


def save_alignment_overlay(base_image, out_path, aligned_root=None, trait_masks=None, dilation_zone=None):
    trait_masks = trait_masks or {}
    band_rgb = np.stack([base_image] * 3, axis=-1)
    if band_rgb.dtype != np.uint8:
        band_rgb = (np.clip(band_rgb, 0, 1) * 255).astype(np.uint8)

    if dilation_zone is not None:
        border = dilation_zone.astype(bool) ^ binary_erosion(dilation_zone.astype(bool), iterations=1)
        band_rgb[border] = [0, 0, 255]

    colors = {
        "main_root_mask": [255, 0, 0],
        "lateral_root_mask": [0, 220, 0],
        "node_mask": [255, 215, 0],
        "tip_mask": [255, 0, 255],
    }
    overlay_radius = {
        "node_mask": 2,
        "tip_mask": 2,
    }
    for mask_name, color in colors.items():
        mask = trait_masks.get(mask_name)
        if mask is not None:
            display_mask = mask.astype(bool)
            if overlay_radius.get(mask_name, 0) > 0:
                display_mask = binary_dilation(
                    display_mask,
                    structure=disk(overlay_radius[mask_name]).astype(bool),
                )
            band_rgb[display_mask] = color

    if not trait_masks and aligned_root is not None:
        band_rgb[aligned_root.astype(bool)] = [255, 0, 0]

    Image.fromarray(band_rgb).save(out_path)
