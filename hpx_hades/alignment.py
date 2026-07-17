from .common import *


def align_masks(fixed_img, moving_mask):
    """
    Aligns moving_mask to fixed_img using:
    - Resize to W=1353, H=870
    - Top-left corner positioned at (x = -170, y = -32) in fixed_img space
    - Returns a binary mask cropped to fixed_img shape (780, 960)
    """
    import numpy as np
    from skimage.transform import resize

    # 1. Resize the mask to the given Photoshop-exported dimensions
    target_w, target_h = 1353, 870  # Width, Height
    resized_mask = resize(
        moving_mask.astype(float),
        (target_h, target_w),  # (H, W)
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    # 2. Define fixed image size and position offset
    H_fix, W_fix = fixed_img.shape  # expected: (780, 960)
    x_offset, y_offset = -170, -32  # Photoshop top-left corner placement

    # 3. Compute source crop from resized_mask
    sx1 = max(0, -x_offset)
    sy1 = max(0, -y_offset)
    sx2 = min(target_w, W_fix - x_offset)
    sy2 = min(target_h, H_fix - y_offset)

    # 4. Compute destination crop in the output canvas
    dx1 = max(0, x_offset)
    dy1 = max(0, y_offset)
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    # 5. Create output canvas and paste
    canvas = np.zeros_like(fixed_img, dtype=bool)
    canvas[dy1:dy2, dx1:dx2] = resized_mask[sy1:sy2, sx1:sx2]

    return canvas


def local_binary_align(fixed_mask, moving_mask, max_shift=5):
    """
    Find (dy, dx) that best aligns moving_mask to fixed_mask within +/- max_shift.
    Only white pixels count; edges are handled by normalization.
    Returns (dy, dx, score).
    """
    F = (fixed_mask > 0)
    T = (moving_mask > 0)
    H, W = F.shape

    best_score, best_shift = -1.0, (0, 0)

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y1f = max(0, 0 + dy);  y2f = min(H, H + dy)
            x1f = max(0, 0 + dx);  x2f = min(W, W + dx)
            y1t = y1f - dy;        y2t = y2f - dy
            x1t = x1f - dx;        x2t = x2f - dx
            if y2f <= y1f or x2f <= x1f:
                continue

            overlap = np.count_nonzero(F[y1f:y2f, x1f:x2f] & T[y1t:y2t, x1t:x2t])
            support = np.count_nonzero(T[y1t:y2t, x1t:x2t])
            if support == 0:
                continue

            score = overlap / support
            if score > best_score:
                best_score, best_shift = score, (dy, dx)

    return (*best_shift, ), best_score


def threshold_within_dilated_roi(
        fixed_mask,  # 2D array (float/uint)
        coarse,  # 2D binary root mask
        iterations=20,
        clamp_above_root=False,
        method="otsu",  # "otsu" | "midpoint" | "gaussian"
        robust=True
):
    """
    Determine threshold from pixels inside a dilated ROI (no background comparison).
    The dilated ROI should include both root and non-root pixels, enabling
    separation into two classes.

    Returns:
        thr: float threshold
        masks: dict with boolean mask {'roi': dilated ROI}
    """
    # Ensure boolean mask
    coarse = coarse.astype(bool)

    # Dilate the coarse mask to get ROI
    dil = binary_dilation(coarse, iterations=iterations)

    # Optional: clamp above root top
    if clamp_above_root and coarse.any():
        y_top = np.flatnonzero(coarse.any(axis=1))[0]
        dil[:y_top, :] = False

    # Keep only valid pixels
    valid = np.isfinite(fixed_mask)
    roi = dil & valid

    if not roi.any():
        raise ValueError("No valid pixels in dilated ROI.")

    # Extract intensities inside ROI
    vals = fixed_mask[roi]

    # Pick thresholding method
    method = method.lower()
    if method == "otsu":
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(vals)

    elif method == "midpoint":
        # Split intensities by median to get rough two groups, then use their medians
        median_val = np.median(vals)
        group1 = vals[vals <= median_val]
        group2 = vals[vals > median_val]
        if group1.size == 0 or group2.size == 0:
            thr = median_val
        else:
            mu1 = np.median(group1) if robust else group1.mean()
            mu2 = np.median(group2) if robust else group2.mean()
            thr = 0.5 * (mu1 + mu2)

    elif method == "gaussian":
        median_val = np.median(vals)
        group1 = vals[vals <= median_val]
        group2 = vals[vals > median_val]
        if group1.size == 0 or group2.size == 0:
            thr = median_val
        else:
            mu1 = np.median(group1) if robust else group1.mean()
            mu2 = np.median(group2) if robust else group2.mean()
            s1 = (np.percentile(group1, 84) - np.percentile(group1, 16)) / 2.0
            s2 = (np.percentile(group2, 84) - np.percentile(group2, 16)) / 2.0
            s1 = max(s1, 1e-9)
            s2 = max(s2, 1e-9)

            A = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
            B = -mu1 / (s1 ** 2) + mu2 / (s2 ** 2)
            C = (mu1 ** 2) / (2 * s1 ** 2) - (mu2 ** 2) / (2 * s2 ** 2) + np.log(s1 / s2)
            if abs(A) < 1e-12:
                thr = 0.5 * (mu1 + mu2)
            else:
                disc = B * B - 4 * A * C
                if disc < 0:
                    thr = 0.5 * (mu1 + mu2)
                else:
                    x1 = (-B + np.sqrt(disc)) / (2 * A)
                    x2 = (-B - np.sqrt(disc)) / (2 * A)
                    lo, hi = sorted([mu1, mu2])
                    candidates = [x for x in (x1, x2) if lo <= x <= hi]
                    thr = candidates[0] if candidates else 0.5 * (mu1 + mu2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(thr), {"roi": roi}


def _paste_with_offset(src, dst_shape, offset_yx):
    H, W = dst_shape
    oy, ox = offset_yx
    canvas = np.zeros((H, W), dtype=src.dtype)
    sy1 = max(0, -oy);
    sx1 = max(0, -ox)
    sy2 = min(src.shape[0], H - oy);
    sx2 = min(src.shape[1], W - ox)
    if sy2 <= sy1 or sx2 <= sx1:
        return canvas
    dy1 = max(0, oy);
    dx1 = max(0, ox)
    dy2 = dy1 + (sy2 - sy1);
    dx2 = dx1 + (sx2 - sx1)
    canvas[dy1:dy2, dx1:dx2] = src[sy1:sy2, sx1:sx2]
    return canvas


def _shift_binary(img, dy, dx):
    H, W = img.shape
    out = np.zeros_like(img)
    sy1 = max(0, -dy);
    sx1 = max(0, -dx)
    sy2 = min(H, H - dy);
    sx2 = min(W, W - dx)
    if sy2 <= sy1 or sx2 <= sx1:
        return out
    dy1 = max(0, dy);
    dx1 = max(0, dx)
    dy2 = dy1 + (sy2 - sy1);
    dx2 = dx1 + (sx2 - sx1)
    out[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]
    return out


def _pick_medoid_from_plateau(S, rng):
    maxv = S.max()
    plateau = np.isclose(S, maxv)
    conn = generate_binary_structure(2, 2)
    comp_map, n = label(plateau, structure=conn)
    if n == 0:
        return 0, 0
    best_comp, best_L1 = None, np.inf
    for c in range(1, n + 1):
        ys, xs = np.where(comp_map == c)
        dy_c, dx_c = rng[ys], rng[xs]
        L1_min = np.min(np.abs(dy_c) + np.abs(dx_c))
        if L1_min < best_L1:
            best_L1, best_comp = L1_min, c
    ys, xs = np.where(comp_map == best_comp)
    dy_c, dx_c = rng[ys], rng[xs]
    D = (np.abs(dy_c[:, None] - dy_c[None, :]) +
         np.abs(dx_c[:, None] - dx_c[None, :]))
    k = np.argmin(D.sum(axis=1))
    return int(dy_c[k]), int(dx_c[k])


def _rotate_and_center_crop(mask, angle_deg, target_shape):
    """Rotate with resize=True to prevent clipping, then center-crop/pad back to target_shape."""
    rh, rw = target_shape
    rot = rotate(mask.astype(float), angle_deg, order=0, preserve_range=True, resize=True)
    rot = (rot > 0.5).astype(np.uint8)
    H, W = rot.shape

    # pad if needed
    py0 = max(0, (rh - H) // 2); py1 = max(0, rh - H - py0)
    px0 = max(0, (rw - W) // 2); px1 = max(0, rw - W - px0)
    if py0 or py1 or px0 or px1:
        rot = np.pad(rot, ((py0, py1), (px0, px1)), mode="constant")

    # center-crop
    H2, W2 = rot.shape
    y0 = max(0, (H2 - rh) // 2); y1 = y0 + rh
    x0 = max(0, (W2 - rw) // 2); x1 = x0 + rw
    return rot[y0:y1, x0:x1].astype(np.uint8)


def _normalize_registration_image(img):
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr[~finite] = 0
    return arr.astype(np.float32)


def _warp_affine_binary(mask, affine_matrix, output_shape):
    if cv2 is None:
        raise ImportError("OpenCV is required for ECC affine warping.")
    H, W = output_shape
    warped = cv2.warpAffine(
        (mask > 0).astype(np.uint8),
        np.asarray(affine_matrix, dtype=np.float32),
        (W, H),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0).astype(np.uint8)


def estimate_ecc_affine_alignment(
    fixed_img,
    moving_mask,
    iterations=300,
    gaussian_kernel=5,
    termination_eps=1e-6,
):
    """
    Refine a coarse aligned binary root mask with OpenCV ECC affine registration.

    `moving_mask` is expected to already be roughly in fixed image coordinates,
    for example after the leaf-mask coarse alignment.
    """
    if cv2 is None:
        raise ImportError("OpenCV is not installed; cannot run ECC affine alignment.")

    if not np.any(moving_mask > 0):
        raise ValueError("Cannot run ECC affine alignment on an empty moving mask.")

    fixed = _normalize_registration_image(fixed_img)
    moving = _normalize_registration_image((moving_mask > 0).astype(np.float32))

    if gaussian_kernel and gaussian_kernel > 1:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        fixed = cv2.GaussianBlur(fixed, (gaussian_kernel, gaussian_kernel), 0)
        moving = cv2.GaussianBlur(moving, (gaussian_kernel, gaussian_kernel), 0)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, termination_eps)
    ecc_score, warp_matrix = cv2.findTransformECC(
        fixed,
        moving,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        gaussian_kernel if gaussian_kernel and gaussian_kernel > 1 else 1,
    )

    aligned_mask = _warp_affine_binary(moving_mask, warp_matrix, fixed.shape)
    params = {
        "method": "ecc_affine",
        "motion_model": "affine",
        "fixed_shape": tuple(fixed.shape),
        "affine_matrix": warp_matrix.tolist(),
        "ecc_score": float(ecc_score),
        "iterations": int(iterations),
        "termination_eps": float(termination_eps),
        "gaussian_kernel": int(gaussian_kernel),
        "transform_chain": [
            ("ecc_affine", {"matrix": warp_matrix.tolist(), "score": float(ecc_score)}),
        ],
    }
    return params, aligned_mask


def estimate_root_alignment_default(
    fixed_root_img,
    fixed_root_binary,
    coarse_root_mask,
    fallback_heatmap_path=None,
    fallback_search_radius=50,
    fallback_fine_max_shift=50,
    fallback_fine_score="iou",
):
    """
    Default root alignment: ECC affine refinement, with grid IoU fallback.
    """
    try:
        params, aligned = estimate_ecc_affine_alignment(
            fixed_root_img,
            coarse_root_mask,
            iterations=300,
            gaussian_kernel=5,
            termination_eps=1e-6,
        )
        params["fallback_used"] = False
        return params, aligned
    except Exception as exc:
        warnings.warn(f"ECC affine root alignment failed; falling back to grid IoU alignment. Reason: {exc}")
        params, _, _, _ = estimate_alignment_from_reference(
            fixed_root_binary,
            coarse_root_mask,
            initial_offset_yx=(0, 0),
            target_size_yx=fixed_root_binary.shape,
            search_radius=fallback_search_radius,
            fine_max_shift=fallback_fine_max_shift,
            fine_score=fallback_fine_score,
            return_heatmap=True,
            save_heatmap_path=fallback_heatmap_path,
            heatmap_title="Fallback grid IoU shift score heatmap",
        )
        params["fixed_shape"] = fixed_root_binary.shape
        params["method"] = "grid_iou_fallback"
        params["fallback_used"] = True
        params["fallback_reason"] = str(exc)
        aligned = apply_alignment_to_mask(coarse_root_mask, params)
        return params, aligned


def estimate_alignment_from_reference(
    fixed_mask,
    moving_mask,
    initial_offset_yx=(-32, -134),
    rotation_deg=0,                 # central angle (from Photoshop)
    target_size_yx=(869, 1322),
    search_radius=50,
    fine_max_shift=50,
    fine_score="intersection",
    return_heatmap=True,
    save_heatmap_path=None,
    heatmap_title="Shift score heatmap",
    angle_candidates_deg = np.linspace(0.5, -0.5, 10),         # NEW: list/iterable of angles to try
):
    # default: sweep ±2° around rotation_deg in 0.5° steps
    if angle_candidates_deg is None:
        angle_candidates_deg = [rotation_deg + a for a in np.arange(-2.0, 2.0 + 1e-9, 0.5)]

    # === resize moving mask ===
    target_h, target_w = target_size_yx
    base = resize(moving_mask.astype(float),
                  (target_h, target_w),
                  order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    H_fix, W_fix = fixed_mask.shape
    rng = np.arange(-fine_max_shift, fine_max_shift + 1)

    # Track global best
    best = dict(score=-1.0, angle=None, dy=0, dx=0, S=None)

    for ang in angle_candidates_deg:
        # rotate (keeps final size via center-crop)
        rot = _rotate_and_center_crop(base, ang, (target_h, target_w))

        # paste into fixed canvas
        coarse = _paste_with_offset(rot, (H_fix, W_fix), initial_offset_yx)

        # restrict to local region
        local = binary_dilation(coarse, iterations=search_radius).astype(np.uint8)
        fixed_local = fixed_mask.copy()
        fixed_local[local == 0] = 0

        # crop to bbox
        ys, xs = np.where(coarse > 0)
        if ys.size == 0:
            continue
        pad = fine_max_shift + 2
        y0 = max(0, ys.min() - pad); y1 = min(fixed_local.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad); x1 = min(fixed_local.shape[1], xs.max() + pad + 1)
        fl = fixed_local[y0:y1, x0:x1]
        ml = coarse[y0:y1, x0:x1]

        # heatmap over (dy, dx)
        S = np.zeros((len(rng), len(rng)), dtype=float)
        dy_best = dx_best = 0
        best_score = -1.0
        best_cost = 1e9

        for i, dy in enumerate(rng):
            for j, dx in enumerate(rng):
                m = _shift_binary(ml, dy, dx)
                inter = np.logical_and(fl, m).sum()
                s = inter if fine_score == "intersection" else inter / (np.logical_or(fl, m).sum() + 1e-9)
                S[i, j] = s
                cost = abs(dy) + abs(dx)  # tie-break on smaller shift
                if (s > best_score) or (s == best_score and cost < best_cost):
                    best_score, best_cost = s, cost
                    dy_best, dx_best = dy, dx

        # Optional plateau medoid
        # max_score = float(S.max())
        # if (S == max_score).sum() > 1:
        #     dy_best, dx_best = _pick_medoid_from_plateau(S, rng)

        # update global best
        if (best_score > best["score"]) or (best_score == best["score"] and best_cost < (abs(best["dy"]) + abs(best["dx"]))):
            best.update(dict(score=best_score, angle=ang, dy=dy_best, dx=dx_best, S=S))

    # If nothing worked
    if best["angle"] is None:
        params = {
            "resize_shape": (target_h, target_w),
            "resize_order": 0,
            "initial_offset_yx": tuple(initial_offset_yx),
            "fine_shift_yx": (0, 0),
            "rotation_deg": rotation_deg,
            "transform_chain": [
                ("resize", {"shape": (target_h, target_w), "order": 0}),
                ("rotate", {"angle_deg": rotation_deg}),
                ("translate", {"offset_yx": tuple(initial_offset_yx)}),
            ],
        }
        return params, np.zeros_like(fixed_mask), (0, 0), None

    # Rebuild aligned mask using the best angle and shift
    rot_best = _rotate_and_center_crop(base, best["angle"], (target_h, target_w))
    coarse_best = _paste_with_offset(rot_best, (H_fix, W_fix), initial_offset_yx)
    aligned_mask = _shift_binary(coarse_best, best["dy"], best["dx"])

    # Optional heatmap save for the best angle
    if save_heatmap_path:
        S = best["S"]
        plt.figure(figsize=(6, 5))
        plt.imshow(S, origin="lower",
                   extent=[rng[0]-0.5, rng[-1]+0.5, rng[0]-0.5, rng[-1]+0.5],
                   aspect="auto")
        plt.xlabel("dx"); plt.ylabel("dy")
        plt.title(f"{heatmap_title} (angle={best['angle']:.2f}°)")
        plt.colorbar(label="score")
        plt.scatter([best["dx"]], [best["dy"]], marker="x")
        plt.text(best["dx"], best["dy"], f"({best['dy']},{best['dx']})", va="bottom")
        plt.tight_layout(); plt.savefig(save_heatmap_path, dpi=300); plt.close()

    # Return
    params = {
        "resize_shape": (target_h, target_w),
        "resize_order": 0,
        "initial_offset_yx": tuple(initial_offset_yx),
        "fine_shift_yx": (best["dy"], best["dx"]),
        "rotation_deg": float(best["angle"]),
        "transform_chain": [
            ("resize", {"shape": (target_h, target_w), "order": 0}),
            ("rotate", {"angle_deg": float(best["angle"])}),
            ("translate", {"offset_yx": tuple(initial_offset_yx)}),
            ("translate", {"offset_yx": (best["dy"], best["dx"])}),
        ],
    }
    heatmap = ({"S": best["S"], "dy_vals": rng, "dx_vals": rng,
                "angle_deg": float(best["angle"]),
                "max_score": float(best["S"].max()),
                "count_max": int((best["S"] == best["S"].max()).sum())}
               if return_heatmap else None)

    return params, aligned_mask, (best["dy"], best["dx"]), heatmap


def apply_alignment_to_mask(mask, params):
    """
    Apply the previously estimated alignment (resize -> rotate -> coarse offset -> fine shift)
    to any binary mask.

    Expects in `params`:
        - resize_shape: (H, W)
        - resize_order: int (default 0)
        - rotation_deg: float (may be 0)
        - initial_offset_yx: (dy, dx) coarse paste offset
        - fine_shift_yx: (dy, dx) fine shift
        - fixed_shape or paste_shape: (H_fix, W_fix) destination canvas size
    """
    # --- Resize ---
    target_h, target_w = params["resize_shape"]
    mask = resize(
        mask.astype(float),
        (target_h, target_w),
        order=params.get("resize_order", 0),
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    # --- Rotate (centered, keep size via center-crop) ---
    rot_deg = float(params.get("rotation_deg", 0.0))
    if abs(rot_deg) > 1e-9:
        mask = _rotate_and_center_crop(mask, rot_deg, (target_h, target_w))

    # --- Coarse paste into fixed canvas ---
    if "fixed_shape" in params:
        H_fix, W_fix = params["fixed_shape"]
    elif "paste_shape" in params:  # fallback name if you used this elsewhere
        H_fix, W_fix = params["paste_shape"]
    else:
        raise KeyError(
            "apply_alignment_to_mask: `fixed_shape` (or `paste_shape`) missing in params. "
            "Save the destination canvas shape when estimating alignment."
        )

    mask = _paste_with_offset(mask, (H_fix, W_fix), params["initial_offset_yx"])

    # --- Fine shift ---
    dy, dx = params.get("fine_shift_yx", (0, 0))
    mask = _shift_binary(mask, dy, dx)

    return mask


def load_binary_mask(mask_path):
    mask = skimage.io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def apply_root_refinement_to_aligned_mask(aligned_mask, root_params):
    """
    Apply the already estimated root-stage refinement to another mask that has
    already gone through the leaf/coarse alignment.
    """
    if "affine_matrix" in root_params:
        fixed_shape = tuple(root_params.get("fixed_shape", aligned_mask.shape))
        return _warp_affine_binary(aligned_mask, root_params["affine_matrix"], fixed_shape)

    if root_params.get("method") == "grid_iou_fallback" or "resize_shape" in root_params:
        return apply_alignment_to_mask(aligned_mask, root_params)

    raise ValueError(f"Unsupported root alignment parameters: {root_params.get('method', 'unknown')}")


def _component_centroids_yx(mask):
    labels, n_labels = label(mask.astype(bool), structure=generate_binary_structure(2, 2))
    centroids = []
    for component_id in range(1, n_labels + 1):
        ys, xs = np.where(labels == component_id)
        if ys.size == 0:
            continue
        centroids.append((float(np.mean(ys)), float(np.mean(xs))))
    return np.asarray(centroids, dtype=np.float32)


def _foreground_points_yx(mask):
    return np.argwhere(mask.astype(bool)).astype(np.float32)


def _transform_points_with_alignment(points_yx, source_shape, params):
    if points_yx.size == 0:
        return points_yx.astype(np.float32)

    points = np.asarray(points_yx, dtype=np.float32).copy()
    src_h, src_w = source_shape
    target_h, target_w = params["resize_shape"]

    points[:, 0] = (points[:, 0] + 0.5) * (target_h / src_h) - 0.5
    points[:, 1] = (points[:, 1] + 0.5) * (target_w / src_w) - 0.5

    rot_deg = float(params.get("rotation_deg", 0.0))
    if abs(rot_deg) > 1e-9:
        theta = np.deg2rad(rot_deg)
        cy = (target_h - 1) / 2.0
        cx = (target_w - 1) / 2.0
        y = points[:, 0] - cy
        x = points[:, 1] - cx
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        points[:, 1] = cos_t * x - sin_t * y + cx
        points[:, 0] = sin_t * x + cos_t * y + cy

    oy, ox = params["initial_offset_yx"]
    dy, dx = params.get("fine_shift_yx", (0, 0))
    points[:, 0] += oy + dy
    points[:, 1] += ox + dx
    return points


def _transform_points_with_root_refinement(points_yx, root_params):
    if points_yx.size == 0:
        return points_yx.astype(np.float32)

    points = np.asarray(points_yx, dtype=np.float32).copy()
    if "affine_matrix" in root_params:
        matrix = np.asarray(root_params["affine_matrix"], dtype=np.float64)
        matrix3 = np.eye(3, dtype=np.float64)
        matrix3[:2, :] = matrix
        inv_matrix = np.linalg.inv(matrix3)
        xy1 = np.column_stack([points[:, 1], points[:, 0], np.ones(points.shape[0])])
        transformed = (inv_matrix @ xy1.T).T
        return np.column_stack([transformed[:, 1], transformed[:, 0]]).astype(np.float32)

    if root_params.get("method") == "grid_iou_fallback" or "resize_shape" in root_params:
        return _transform_points_with_alignment(points, tuple(root_params["fixed_shape"]), root_params)

    raise ValueError(f"Unsupported root alignment parameters: {root_params.get('method', 'unknown')}")


def _draw_skeleton_points(points_yx, output_shape):
    out = np.zeros(output_shape, dtype=np.uint8)
    if points_yx.size == 0:
        return out
    H, W = output_shape
    yy = np.clip(np.rint(points_yx[:, 0]).astype(int), 0, H - 1)
    xx = np.clip(np.rint(points_yx[:, 1]).astype(int), 0, W - 1)
    out[yy, xx] = 1
    return skeletonize(out.astype(bool)).astype(np.uint8)


def _draw_one_pixel_objects(points_yx, output_shape):
    out = np.zeros(output_shape, dtype=np.uint8)
    if points_yx.size == 0:
        return out

    H, W = output_shape
    occupied = set()
    for y_float, x_float in points_yx:
        y0 = int(np.clip(np.rint(y_float), 0, H - 1))
        x0 = int(np.clip(np.rint(x_float), 0, W - 1))
        chosen = None
        for radius in range(0, max(H, W) + 1):
            candidates = []
            for yy in range(max(0, y0 - radius), min(H, y0 + radius + 1)):
                for xx in range(max(0, x0 - radius), min(W, x0 + radius + 1)):
                    if (yy, xx) not in occupied:
                        candidates.append((abs(yy - y0) + abs(xx - x0), yy, xx))
            if candidates:
                _, yy, xx = min(candidates)
                chosen = (yy, xx)
                break
        if chosen is None:
            chosen = (y0, x0)
        occupied.add(chosen)
        out[chosen] = 1
    return out


def align_skeleton_mask(mask, leaf_params, root_params, output_shape):
    mask = np.asarray(mask).astype(bool)
    points = _foreground_points_yx(mask)
    leaf_points = _transform_points_with_alignment(points, mask.shape, leaf_params)
    final_points = _transform_points_with_root_refinement(leaf_points, root_params)
    return _draw_skeleton_points(final_points, output_shape)


def align_skeleton_mask_to_leaf(mask, leaf_params):
    mask = np.asarray(mask).astype(bool)
    points = _foreground_points_yx(mask)
    leaf_points = _transform_points_with_alignment(points, mask.shape, leaf_params)
    return _draw_skeleton_points(leaf_points, tuple(leaf_params["fixed_shape"]))


def align_landmark_mask(mask_path, leaf_params, root_params, output_shape):
    mask = load_binary_mask(mask_path)
    points = _component_centroids_yx(mask)
    leaf_points = _transform_points_with_alignment(points, mask.shape, leaf_params)
    final_points = _transform_points_with_root_refinement(leaf_points, root_params)
    return _draw_one_pixel_objects(final_points, output_shape), int(points.shape[0])


def align_related_root_masks(mask_paths, leaf_params, root_params):
    aligned_masks = {}
    object_counts = {}
    for mask_name, mask_path in mask_paths.items():
        if not mask_path.exists():
            warnings.warn(f"Related mask not found and will be skipped: {mask_path}")
            continue
        if mask_name in {"node_mask", "tip_mask"}:
            fixed_shape = tuple(root_params.get("fixed_shape", leaf_params["fixed_shape"]))
            aligned, object_count = align_landmark_mask(mask_path, leaf_params, root_params, fixed_shape)
            object_counts[mask_name] = object_count
        else:
            fixed_shape = tuple(root_params.get("fixed_shape", leaf_params["fixed_shape"]))
            aligned = align_skeleton_mask(load_binary_mask(mask_path), leaf_params, root_params, fixed_shape)
        aligned_masks[mask_name] = (aligned > 0).astype(np.uint8)
    return aligned_masks, object_counts


def count_mask_objects(mask_path):
    if not mask_path.exists():
        return 0
    labels, n_labels = label(load_binary_mask(mask_path).astype(bool), structure=generate_binary_structure(2, 2))
    return int(n_labels)
