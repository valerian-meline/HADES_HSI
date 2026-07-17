from .common import *


def extract_vnir_id(bil_filename):
    """
    Extract an ID like '90_12_Exp32_coumarins_01' from the VNIR filename:
    '90_12_2024-10-07_13-45-48_Exp32_coumarins_01_VNIR2_Data.bil'
    """
    match = re.match(r"(\d+_\d+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.*)_VNIR\d+_Data\.bil", bil_filename)
    if match:
        part1, part2 = match.groups()
        return f"{part1}_{part2}"
    return None


def find_two_closest_root_mask_rounds(sample_id, root_masks):
    """
    For a given VNIR ID, returns two lists of root masks:
    - One with the smallest round number ≥ VNIR round (current round)
    - One with the largest round number < VNIR round (previous round)

    Each list contains all matching root mask paths from that round.
    """
    try:
        exp, round_number_str, sample_suffix = sample_id.split('_', 2)
        vnir_round = int(round_number_str)
    except Exception:
        return [], []  # fallback if parsing fails

    round_to_masks = {}

    for rm in root_masks:
        folder = os.path.basename(os.path.dirname(os.path.dirname(rm)))
        parts = folder.split('_')

        if len(parts) >= 4:
            # Remove the 3rd element (index 2), which is the timestamp
            cleaned_parts = parts[:2] + parts[2:]
            rm_round_str = parts[1]  # e.g., "16"
            rm_suffix = '_'.join(cleaned_parts[2:])  # everything after round, excluding timestamp
            rm_suffix = re.sub(r'_ROOT1_Fish Eye Corrected_A0$', '', rm_suffix)
            rm_suffix = re.sub(r'_ROOT2_Fish Eye Corrected_A0$', '', rm_suffix)
            rm_round = int(rm_round_str)
            if rm_suffix == sample_suffix:
                round_to_masks.setdefault(rm_round, []).append(rm)

    # Split into lower and upper candidates
    lower_rounds = [r for r in round_to_masks if r < vnir_round]
    upper_rounds = [r for r in round_to_masks if r >= vnir_round]

    closest_lower = max(lower_rounds) if lower_rounds else None
    closest_upper = min(upper_rounds) if upper_rounds else None

    matching_lower = round_to_masks[closest_lower] if closest_lower is not None else []
    matching_upper = round_to_masks[closest_upper] if closest_upper is not None else []

    return matching_upper, matching_lower


def pad_cube_to_shape(cube, target_shape=(780, 960)):
    """
    Pads the spatial dimensions of a hyperspectral cube (H, W, B) with np.nan to match the target shape.

    Parameters:
        cube (np.ndarray): Input cube with shape (H, W, B)
        target_shape (tuple): Target (rows, cols)

    Returns:
        np.ndarray: Padded cube with shape (780, 960, B)
    """
    h, w, b = cube.shape
    target_h, target_w = target_shape

    if h > target_h or w > target_w:
        raise ValueError(f"Cube is larger than target shape: got ({h}, {w}), expected max ({target_h}, {target_w})")

    padded = np.full((target_h, target_w, b), np.nan, dtype=cube.dtype)
    padded[:h, :w, :] = cube
    return padded


def find_matching_vnir1(vnir2_path: Path, vnir1_dir: Path) -> Path | None:
    """
    Find the VNIR1 file in vnir1_dir that matches the VNIR2 file,
    ignoring the timestamp but matching both suffix and prefix.
    """
    vnir2_name = vnir2_path.name
    parts = vnir2_name.split('_')

    if len(parts) < 6:
        return None  # Unexpected format

    prefix = '_'.join(parts[:2])  # e.g., '123_17'
    suffix = '_'.join(parts[4:]).replace("VNIR2", "VNIR")  # e.g., 'exp63_col_..._VNIR_Data.bil'

    # Search all .bil files in VNIR1 dir
    for file in vnir1_dir.rglob("*.bil"):
        fname = file.name
        if fname.startswith(prefix) and fname.endswith(suffix):
            return(file)
