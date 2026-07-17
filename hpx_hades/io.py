from .common import *


def open_bil_with_temp_envi_header(bil_path, custom_hdr_path):
    """
    Converts a non-ENVI header file to ENVI format and opens the image using a temporary header.

    Parameters:
    - bil_path: Path to the .bil image file
    - custom_hdr_path: Path to the custom-format .hdr file

    Returns:
    - Spectral image object (numpy-compatible) loaded with correct metadata
    """

    # Read the custom header
    with open(custom_hdr_path, 'r') as f:
        lines = f.readlines()

    params = {}
    wavelengths = []
    wavelength_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("WAVELENGTHS"):
            wavelength_section = True
            continue
        if wavelength_section:
            try:
                val = float(line)
                wavelengths.append(val)
            except ValueError:
                wavelength_section = False
                continue
        else:
            if ' ' in line:
                key, value = line.split(maxsplit=1)
                params[key.upper()] = value.strip()

    # Determine ENVI data type
    nbits_to_datatype = {'8': 1, '12': 2, '16': 2, '32': 4, '64': 5}
    envi_datatype = nbits_to_datatype.get(params.get('NBITS', '16'), 1)

    # Compose ENVI header content
    header_lines = [
        "ENVI",
        f"samples = {params['NCOLS']}",
        f"lines   = {params['NROWS']}",
        f"bands   = {params['NBANDS']}",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {envi_datatype}",
        "interleave = bil",
        "byte order = 0",
        "wavelength = {"
    ]
    for i, wl in enumerate(wavelengths):
        sep = "," if i < len(wavelengths) - 1 else ""
        if i % 5 == 0:
            header_lines.append("  ")
        header_lines[-1] += f"{wl:.2f}{sep}"
    header_lines.append("}")

    # Create temporary header file
    with tempfile.NamedTemporaryFile(suffix=".hdr", mode='w+', delete=False) as tmp_hdr:
        tmp_hdr.write("\n".join(header_lines))
        tmp_hdr.flush()
        img = envi.open(tmp_hdr.name, str(bil_path))
        data = img.load().astype(np.float32)

    return data, wavelengths


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_run_parameters(output_dir, params, filename="run_parameters.json"):
    """Save a JSON record of analysis settings for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(_to_jsonable(params), fh, indent=2, sort_keys=True)
    return out_path


def save_band_as_png(band_array, out_path, vmin=0.0, vmax=1.0):
    """
    Save a normalized 2D reflectance band (0–1) as 8-bit grayscale PNG.
    Ensures image is correctly shaped for export (rows x cols).
    """
    if band_array.ndim != 2:
        raise ValueError(f"Expected 2D band for PNG export, got shape: {band_array.shape}")

    band_scaled = np.clip((band_array - vmin) / (vmax - vmin), 0, 1)
    band_8bit = (band_scaled * 255).astype(np.uint8)
    imageio.imwrite(str(out_path), band_8bit)
