from .common import *


def expand_calibration_to_match(calib_img, target_shape):
    """
    Match a calibration cube (dark or white) to the raw cube shape while preserving
    as much detector structure as possible.

    Expected use case:
    - calib_img is smaller in the row dimension, but has the same width and bands.
    - We average only across the missing row dimension, then repeat across raw rows.

    Parameters
    ----------
    calib_img : np.ndarray
        Calibration cube of shape (h, w, b)
    target_shape : tuple
        Target raw cube shape (H, W, B)

    Returns
    -------
    np.ndarray
        Matched calibration cube of shape (H, W, B)
    """
    calib_img = np.asarray(calib_img, dtype=np.float32)
    H, W, B = target_shape
    h, w, b = calib_img.shape

    if b != B:
        raise ValueError(f"Band mismatch: calibration has {b}, raw has {B}")

    if w != W:
        raise ValueError(
            f"Width mismatch: calibration has {w}, raw has {W}. "
            "This function assumes same width and only fewer rows."
        )

    if h == H:
        return calib_img

    if h < H:
        # Preserve width x band structure; average only over the short row dimension.
        calib_rowavg = np.nanmean(calib_img, axis=0, keepdims=True)   # shape: (1, W, B)
        calib_expanded = np.repeat(calib_rowavg, H, axis=0)           # shape: (H, W, B)
        return calib_expanded.astype(np.float32)

    # If calibration has more rows than raw, crop to match.
    return calib_img[:H, :, :].astype(np.float32)


def repair_invalid_pixels_bandwise(image_cube, neighborhood_size=3):
    """
    Repair invalid pixels (NaN/inf) band-wise using nearest valid neighbors followed
    by local median neighborhood repair. This preserves local spatial structure better
    than replacing invalid values with one global cube mean.
    """
    repaired = np.asarray(image_cube, dtype=np.float32).copy()

    for band_idx in range(repaired.shape[2]):
        band = repaired[:, :, band_idx]
        invalid = ~np.isfinite(band)
        if not invalid.any():
            continue

        valid = np.isfinite(band)
        if not valid.any():
            repaired[:, :, band_idx] = np.zeros_like(band, dtype=np.float32)
            continue

        # Nearest-neighbor fill inside the band.
        nearest_idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        filled = band[tuple(nearest_idx)]

        # Local neighborhood repair for only the invalid locations.
        local_median = median_filter(filled, size=neighborhood_size)
        band[invalid] = local_median[invalid]
        repaired[:, :, band_idx] = band

    return repaired


def remove_bad_pixels(image_cube):
    return repair_invalid_pixels_bandwise(image_cube, neighborhood_size=3)


def apply_baseline_correction(image_cube):
    corrected = np.zeros_like(image_cube)
    for i in range(image_cube.shape[0]):
        for j in range(image_cube.shape[1]):
            spectrum = image_cube[i, j, :]
            baseline, _ = asls(spectrum)
            corrected[i, j, :] = spectrum - baseline
    return corrected


def apply_spectral_smoothing(image_cube, window_length=11, polyorder=2):
    """Apply one Savitzky-Golay pass along the spectral axis."""
    n_bands = image_cube.shape[2]
    max_odd_window = n_bands if n_bands % 2 == 1 else n_bands - 1
    min_valid_window = polyorder + 2
    if min_valid_window % 2 == 0:
        min_valid_window += 1

    wl = min(window_length, max_odd_window)
    if wl < min_valid_window:
        return np.asarray(image_cube, dtype=np.float32).copy()

    return savgol_filter(image_cube, window_length=wl, polyorder=polyorder, axis=2, mode='nearest')


def spatial_denoise_preserve_intensity(cube, filter_size=(3, 3)):
    """
    Apply spatial median filtering to each band without altering intensity magnitude.

    Parameters:
        cube (np.ndarray): Hyperspectral cube (H, W, B)
        filter_size (tuple): Median filter size (default 3x3)

    Returns:
        np.ndarray: Denoised cube with preserved overall intensity
    """
    denoised = np.empty_like(cube)
    for i in range(cube.shape[2]):
        denoised[:, :, i] = median_filter(cube[:, :, i], size=filter_size)
    return denoised


def preprocessing_bil(raw, dark, white=None, mode='VNIR2'):
    """
    Preprocess a hyperspectral cube.

    VNIR2:
        - dark correction only

    VNIR1:
        - white/dark reflectance calibration if white is provided

    Returns
    -------
    denoised, corrected, repaired, smoothed, dark_flat, dark_flat_smooth
    """
    raw = raw.astype(np.float32)

    # Match calibration cubes while preserving detector column structure.
    dark_cube = expand_calibration_to_match(dark, raw.shape)

    if mode == 'VNIR1' and white is not None:
        white_cube = expand_calibration_to_match(white, raw.shape)
        corrected = (raw - dark_cube) / (white_cube - dark_cube + 1e-6)
    else:
        corrected = raw - dark_cube

    # Keep negative values for quantitative analysis; clip only for display/export.
    repaired = remove_bad_pixels(corrected)
    smoothed = apply_spectral_smoothing(repaired)
    denoised = spatial_denoise_preserve_intensity(smoothed)

    # QC summaries (keep return signature compatible with plotting code).
    dark_flat = np.nanmean(dark_cube, axis=(0, 1), keepdims=True)
    dark_flat_smooth = dark_flat.copy()  # no spectral smoothing of dark reference in the analysis path

    return denoised, corrected, repaired, smoothed, dark_flat, dark_flat_smooth
