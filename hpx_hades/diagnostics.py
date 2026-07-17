from .common import *


def quick_qc(cube, label="", row=None, col=None, wavelengths=None, save_path=None):

    print(f"\n📊 QC: {label}")
    print(f"  Shape: {cube.shape}")
    print(f"  Min: {np.min(cube):.4f}, Max: {np.max(cube):.4f}")
    print(f"  NaNs: {np.isnan(cube).sum()}")

    if row is None or col is None:
        # Use full image for averaging instead of a window
        spectrum = np.nanmean(cube, axis=(0, 1))
        print(f"  Averaged spectrum from full image")
    else:
        spectrum = cube[row, col, :]
        print(f"  Spectrum at pixel ({row}, {col}): {spectrum[:10]} ...")

    x_axis = wavelengths if wavelengths is not None else np.arange(spectrum.shape[0])
    x_label = "Wavelength (nm)" if wavelengths is not None else "Wavelength Band Index"

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, spectrum)
    plt.title(f"{'Image-Averaged' if row is None else 'Pixel'} Spectrum - {label}")
    plt.xlabel(x_label)
    plt.ylabel("Intensity / Reflectance")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ---------------------- Helper functions ----------------------------


def visualize_raw_band(cube, band_index, wavelengths=None):
    """
    Display a single band from the raw hyperspectral cube using matplotlib.
    """
    band = cube[:, :, band_index]
    plt.figure(figsize=(6, 6))
    plt.imshow(band, cmap='gray')
    title = f"Raw Band {band_index}"
    if wavelengths:
        title += f" ({wavelengths[band_index]:.1f} nm)"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------------- Preprocessing Functions ----------------------


def qc_plot_spectra(output_dir, wavelengths,
                    raw, white, dark, corrected, repaired, smoothed, denoised,
                    dark_flat, dark_flat_smooth):
    """
    Saves two QC plots:
    - QC Spectra Overview: Raw, White, Dark, Corrected, Repaired, Smoothed, Denoised
    - Dark Processing: Raw dark vs. matched dark mean
    """

    def extract_mean_spectrum(cube):
        if cube.ndim == 1:
            return cube
        return np.nanmean(cube, axis=(0, 1))

    raw_mean = extract_mean_spectrum(raw)
    white_mean = extract_mean_spectrum(white)
    dark_mean = extract_mean_spectrum(dark)
    dark_flat_mean = extract_mean_spectrum(dark_flat)
    dark_flat_smooth_mean = extract_mean_spectrum(dark_flat_smooth)
    corrected_mean = extract_mean_spectrum(corrected)
    repaired_mean = extract_mean_spectrum(repaired)
    smoothed_mean = extract_mean_spectrum(smoothed)
    denoised_mean = extract_mean_spectrum(denoised)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, raw_mean, label='Raw')
    plt.plot(wavelengths, white_mean, label='White Calibration')
    plt.plot(wavelengths, dark_mean, label='Dark Calibration')
    plt.plot(wavelengths, corrected_mean, label='Dark-corrected')
    plt.plot(wavelengths, repaired_mean, label='Invalid-pixel repaired')
    plt.plot(wavelengths, smoothed_mean, label='Spectrally smoothed')
    plt.plot(wavelengths, denoised_mean, label='Spatially denoised')
    plt.title("QC Spectra Overview - Fluorescence Signal")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qc_combined_fluoro.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, dark_mean, label='Original Dark')
    plt.plot(wavelengths, dark_flat_mean, label='Matched Dark Mean')
    plt.plot(wavelengths, dark_flat_smooth_mean, label='Matched Dark Mean (analysis path)')
    plt.title("QC Dark Calibration - Matched Dark Summary")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qc_dark_processing.png"))
    plt.close()


def _safe_spectral_metrics(raw_spec, processed_spec):
    raw_spec = np.asarray(raw_spec, dtype=np.float32)
    processed_spec = np.asarray(processed_spec, dtype=np.float32)
    valid = np.isfinite(raw_spec) & np.isfinite(processed_spec)
    if valid.sum() < 2:
        return {
            'n_bands': int(valid.sum()),
            'mean_diff': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'corr': np.nan,
            'sum_ratio': np.nan,
        }

    raw_v = raw_spec[valid]
    proc_v = processed_spec[valid]
    corr = np.nan
    if np.nanstd(raw_v) > 0 and np.nanstd(proc_v) > 0:
        corr = float(np.corrcoef(raw_v, proc_v)[0, 1])

    raw_sum = np.nansum(raw_v)
    return {
        'n_bands': int(valid.sum()),
        'mean_diff': float(np.nanmean(proc_v - raw_v)),
        'mae': float(np.nanmean(np.abs(proc_v - raw_v))),
        'rmse': float(np.sqrt(np.nanmean((proc_v - raw_v) ** 2))),
        'corr': corr,
        'sum_ratio': float(np.nansum(proc_v) / raw_sum) if abs(raw_sum) > 1e-12 else np.nan,
    }


def _select_representative_pixels(reference_band_img, roi_mask=None):
    valid = np.isfinite(reference_band_img)
    if roi_mask is not None:
        valid &= roi_mask.astype(bool)

    ys, xs = np.where(valid)
    if ys.size == 0:
        return {}

    vals = reference_band_img[ys, xs]
    targets = {
        'bright': np.nanmax(vals),
        'median': np.nanmedian(vals),
        'dim': np.nanpercentile(vals, 10),
    }

    chosen = {}
    used = set()
    for label, target in targets.items():
        order = np.argsort(np.abs(vals - target))
        picked = None
        for idx in order:
            coord = (int(ys[idx]), int(xs[idx]))
            if coord not in used:
                picked = coord
                used.add(coord)
                break
        if picked is None:
            picked = (int(ys[order[0]]), int(xs[order[0]]))
        chosen[label] = picked

    return chosen


def save_preprocessing_diagnostics(output_dir, raw_cube, processed_cube, wavelengths,
                                   reference_band=180, roi_masks=None, prefix='preprocessing',
                                   save_plot_images=False):
    """
    Quantify preprocessing effects by comparing raw and processed spectra on
    representative pixels and ROI-averaged spectra.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_cube.shape != processed_cube.shape:
        raise ValueError(f"Shape mismatch for diagnostics: raw={raw_cube.shape}, processed={processed_cube.shape}")

    ref_idx = int(np.clip(reference_band, 0, processed_cube.shape[2] - 1))
    roi_masks = roi_masks or {}
    roi_masks = {name: np.asarray(mask).astype(bool) for name, mask in roi_masks.items()}

    roi_union = None
    if roi_masks:
        roi_union = np.zeros(processed_cube.shape[:2], dtype=bool)
        for mask in roi_masks.values():
            roi_union |= mask

    pixel_coords = _select_representative_pixels(processed_cube[:, :, ref_idx], roi_union)
    rows = []

    if pixel_coords:
        if save_plot_images:
            fig, axes = plt.subplots(len(pixel_coords), 1, figsize=(10, 3.5 * len(pixel_coords)), squeeze=False)
            axes = axes.ravel()
        else:
            axes = [None] * len(pixel_coords)

        for ax, (label, (yy, xx)) in zip(axes, pixel_coords.items()):
            raw_spec = raw_cube[yy, xx, :]
            proc_spec = processed_cube[yy, xx, :]
            metrics = _safe_spectral_metrics(raw_spec, proc_spec)
            rows.append({
                'target': 'pixel',
                'name': label,
                'y': yy,
                'x': xx,
                **metrics,
            })
            if save_plot_images:
                ax.plot(wavelengths, raw_spec, label='Raw')
                ax.plot(wavelengths, proc_spec, label='Processed')
                ax.set_title(f"Representative pixel: {label} at (y={yy}, x={xx}) | corr={metrics['corr']:.3f} | RMSE={metrics['rmse']:.3f}")
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.grid(True)
                ax.legend()

        if save_plot_images:
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_pixel_diagnostics.png", dpi=300)
            plt.close()

    roi_plot_masks = dict(roi_masks)
    if not roi_plot_masks:
        roi_plot_masks['full_image'] = np.isfinite(processed_cube[:, :, ref_idx])
        h, w = processed_cube.shape[:2]
        cy0, cy1 = int(h * 0.25), int(h * 0.75)
        cx0, cx1 = int(w * 0.25), int(w * 0.75)
        center = np.zeros((h, w), dtype=bool)
        center[cy0:cy1, cx0:cx1] = True
        roi_plot_masks['center_window'] = center & np.isfinite(processed_cube[:, :, ref_idx])

    if save_plot_images:
        fig, axes = plt.subplots(len(roi_plot_masks), 1, figsize=(10, 3.5 * len(roi_plot_masks)), squeeze=False)
        axes = axes.ravel()
    else:
        axes = [None] * len(roi_plot_masks)

    for ax, (roi_name, mask) in zip(axes, roi_plot_masks.items()):
        mask = mask.astype(bool)
        if not np.any(mask):
            continue
        raw_spec = np.nanmean(raw_cube[mask, :], axis=0)
        proc_spec = np.nanmean(processed_cube[mask, :], axis=0)
        metrics = _safe_spectral_metrics(raw_spec, proc_spec)
        rows.append({
            'target': 'roi',
            'name': roi_name,
            'y': np.nan,
            'x': np.nan,
            **metrics,
        })
        if save_plot_images:
            ax.plot(wavelengths, raw_spec, label='Raw mean')
            ax.plot(wavelengths, proc_spec, label='Processed mean')
            ax.set_title(f"ROI mean: {roi_name} | corr={metrics['corr']:.3f} | RMSE={metrics['rmse']:.3f}")
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Mean intensity')
            ax.grid(True)
            ax.legend()

    if save_plot_images:
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_roi_diagnostics.png", dpi=300)
        plt.close()

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_preprocessing_metrics.csv", index=False)

# ---------------------- Main Pipeline ----------------------
