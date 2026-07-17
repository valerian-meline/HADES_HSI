"""
Benchmark registration strategies for RootCam masks against VNIR root bands.

This is an experimental harness. It does not replace the production ROI export
in HPX_HADES.py; it compares candidate refinements and writes ranked IoU/Dice
scores plus overlay images so the best strategy can be promoted deliberately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from skimage import exposure, feature, filters, measure, morphology, transform, util
from skimage.registration import phase_cross_correlation

try:
    import cv2
except Exception:  # pragma: no cover - depends on local environment
    cv2 = None


@dataclass
class RegistrationResult:
    plant: str
    method: str
    transform_model: str
    score_name: str
    score: float
    iou: float
    dice: float
    precision: float
    recall: float
    notes: str
    aligned_mask: np.ndarray | None = None


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_mask(path: Path, threshold: float = 0.5) -> np.ndarray:
    return read_gray(path) > threshold


def save_mask_overlay(fixed_image: np.ndarray, aligned_mask: np.ndarray, out_path: Path) -> None:
    img = normalize01(fixed_image)
    rgb = np.dstack([img, img, img])
    border = aligned_mask ^ morphology.binary_erosion(aligned_mask)
    dil = morphology.binary_dilation(aligned_mask, morphology.disk(4)) & ~aligned_mask
    rgb[dil] = [0.0, 0.0, 1.0]
    rgb[border] = [1.0, 0.0, 0.0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out_path)


def normalize01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def fixed_leaf_mask_from_band(band: np.ndarray) -> np.ndarray:
    band = normalize01(band)
    return band > filters.threshold_otsu(band)


def root_enhanced_band(band: np.ndarray, radius: int = 15) -> np.ndarray:
    band = normalize01(band)
    enhanced = morphology.white_tophat(band, morphology.disk(radius))
    return normalize01(enhanced)


def fixed_root_mask_from_band(band: np.ndarray, radius: int = 15) -> np.ndarray:
    enhanced = root_enhanced_band(band, radius=radius)
    return enhanced > filters.threshold_otsu(enhanced)


def resize_nearest(mask: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    return transform.resize(
        mask.astype(np.float32),
        shape_yx,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5


def paste_with_offset(src: np.ndarray, dst_shape: tuple[int, int], offset_yx: tuple[int, int]) -> np.ndarray:
    h, w = dst_shape
    oy, ox = offset_yx
    canvas = np.zeros((h, w), dtype=bool)
    sy1 = max(0, -oy)
    sx1 = max(0, -ox)
    sy2 = min(src.shape[0], h - oy)
    sx2 = min(src.shape[1], w - ox)
    if sy2 <= sy1 or sx2 <= sx1:
        return canvas
    dy1 = max(0, oy)
    dx1 = max(0, ox)
    dy2 = dy1 + (sy2 - sy1)
    dx2 = dx1 + (sx2 - sx1)
    canvas[dy1:dy2, dx1:dx2] = src[sy1:sy2, sx1:sx2]
    return canvas


def shift_binary(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros_like(img, dtype=bool)
    sy1 = max(0, -dy)
    sx1 = max(0, -dx)
    sy2 = min(h, h - dy)
    sx2 = min(w, w - dx)
    if sy2 <= sy1 or sx2 <= sx1:
        return out
    dy1 = max(0, dy)
    dx1 = max(0, dx)
    dy2 = dy1 + (sy2 - sy1)
    dx2 = dx1 + (sx2 - sx1)
    out[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]
    return out


def rotate_and_center_crop(mask: np.ndarray, angle_deg: float, target_shape: tuple[int, int]) -> np.ndarray:
    rotated = transform.rotate(
        mask.astype(np.float32),
        angle_deg,
        order=0,
        preserve_range=True,
        resize=True,
    ) > 0.5
    rh, rw = target_shape
    h, w = rotated.shape
    py0 = max(0, (rh - h) // 2)
    py1 = max(0, rh - h - py0)
    px0 = max(0, (rw - w) // 2)
    px1 = max(0, rw - w - px0)
    if py0 or py1 or px0 or px1:
        rotated = np.pad(rotated, ((py0, py1), (px0, px1)), mode="constant")
    h2, w2 = rotated.shape
    y0 = max(0, (h2 - rh) // 2)
    x0 = max(0, (w2 - rw) // 2)
    return rotated[y0 : y0 + rh, x0 : x0 + rw]


def apply_hpx_alignment(mask: np.ndarray, params: dict) -> np.ndarray:
    target_h, target_w = params["resize_shape"]
    out = resize_nearest(mask, (int(target_h), int(target_w)))
    angle = float(params.get("rotation_deg", 0.0))
    if abs(angle) > 1e-9:
        out = rotate_and_center_crop(out, angle, (int(target_h), int(target_w)))
    fixed_shape = tuple(int(v) for v in params["fixed_shape"])
    out = paste_with_offset(out, fixed_shape, tuple(int(v) for v in params["initial_offset_yx"]))
    dy, dx = tuple(int(v) for v in params.get("fine_shift_yx", (0, 0)))
    return shift_binary(out, dy, dx)


def load_current_root_alignment(root_mask: np.ndarray, run_params_path: Path) -> np.ndarray:
    params = json.loads(run_params_path.read_text())
    leaf_params = params["leaf_alignment"]["estimated_transform"]
    root_params = params["root_alignment"]["estimated_transform"]
    first = apply_hpx_alignment(root_mask, leaf_params)
    return apply_hpx_alignment(first, root_params)


def metrics(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, float, float, float]:
    fixed = fixed.astype(bool)
    moving = moving.astype(bool)
    inter = int(np.count_nonzero(fixed & moving))
    union = int(np.count_nonzero(fixed | moving))
    moving_sum = int(np.count_nonzero(moving))
    fixed_sum = int(np.count_nonzero(fixed))
    iou = inter / union if union else 0.0
    dice = (2 * inter) / (fixed_sum + moving_sum) if fixed_sum + moving_sum else 0.0
    precision = inter / moving_sum if moving_sum else 0.0
    recall = inter / fixed_sum if fixed_sum else 0.0
    return iou, dice, precision, recall


def sparse_translation_search(
    fixed: np.ndarray,
    moving: np.ndarray,
    max_shift: int,
    step: int = 1,
    score_name: str = "iou",
) -> tuple[np.ndarray, tuple[int, int], float]:
    fixed = fixed.astype(bool)
    coords = np.argwhere(moving)
    if coords.size == 0:
        return moving.copy(), (0, 0), 0.0

    fixed_sum = int(np.count_nonzero(fixed))
    h, w = fixed.shape
    best = (-math.inf, 0, 0)

    for dy in range(-max_shift, max_shift + 1, step):
        yy = coords[:, 0] + dy
        y_ok = (yy >= 0) & (yy < h)
        if not np.any(y_ok):
            continue
        base_x = coords[y_ok, 1]
        base_y = yy[y_ok]
        for dx in range(-max_shift, max_shift + 1, step):
            xx = base_x + dx
            ok = (xx >= 0) & (xx < w)
            support = int(np.count_nonzero(ok))
            if support == 0:
                continue
            inter = int(np.count_nonzero(fixed[base_y[ok], xx[ok]]))
            if score_name == "intersection":
                score = float(inter)
            elif score_name == "dice":
                score = (2 * inter) / (fixed_sum + support) if fixed_sum + support else 0.0
            else:
                union = fixed_sum + support - inter
                score = inter / union if union else 0.0
            cost = abs(dy) + abs(dx)
            if score > best[0] or (score == best[0] and cost < abs(best[1]) + abs(best[2])):
                best = (score, dy, dx)

    aligned = shift_binary(moving, best[1], best[2])
    return aligned, (best[1], best[2]), float(best[0])


def register_translation_grid(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    max_shift: int,
    step: int,
    score_name: str,
) -> RegistrationResult:
    aligned, (dy, dx), score = sparse_translation_search(
        fixed_mask,
        moving_mask,
        max_shift=max_shift,
        step=step,
        score_name=score_name,
    )
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method=f"grid_{score_name}",
        transform_model="translation",
        score_name=score_name,
        score=score,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=f"dy={dy}, dx={dx}",
        aligned_mask=aligned,
    )


def register_euclidean_grid(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    angle_degrees: Iterable[float],
    max_shift: int,
    step: int,
    score_name: str,
) -> RegistrationResult:
    best: RegistrationResult | None = None
    best_angle = 0.0
    for angle in angle_degrees:
        rotated = transform.rotate(
            moving_mask.astype(np.float32),
            angle,
            resize=False,
            order=0,
            preserve_range=True,
        ) > 0.5
        result = register_translation_grid(fixed_mask, rotated, max_shift, step, score_name)
        if best is None or result.score > best.score:
            best = result
            best_angle = float(angle)

    assert best is not None
    best.method = f"grid_{score_name}"
    best.transform_model = "euclidean"
    best.notes = f"angle_deg={best_angle:.4f}, {best.notes}"
    return best


def affine_transform_from_params(params: np.ndarray) -> transform.AffineTransform:
    angle_deg, scale_x, scale_y, shear_deg, tx, ty = params
    return transform.AffineTransform(
        scale=(scale_x, scale_y),
        rotation=np.deg2rad(angle_deg),
        shear=np.deg2rad(shear_deg),
        translation=(tx, ty),
    )


def warp_mask_with_transform(
    moving_mask: np.ndarray,
    tform: transform.GeometricTransform,
    output_shape: tuple[int, int],
) -> np.ndarray:
    return (
        transform.warp(
            moving_mask.astype(np.float32),
            inverse_map=tform.inverse,
            output_shape=output_shape,
            order=0,
            preserve_range=True,
        )
        > 0.5
    )


def translation_transform(dx: float, dy: float) -> transform.AffineTransform:
    return transform.AffineTransform(translation=(dx, dy))


def centered_similarity_transform(
    shape_yx: tuple[int, int],
    angle_deg: float = 0.0,
    scale: float = 1.0,
) -> transform.AffineTransform:
    h, w = shape_yx
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle) * scale
    sin_a = np.sin(angle) * scale
    matrix = np.array(
        [
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
            [sin_a, cos_a, cy - sin_a * cx - cos_a * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform.AffineTransform(matrix=matrix)


def transformed_mask_from_matrix(
    moving_mask: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    return warp_mask_with_transform(moving_mask, transform.AffineTransform(matrix=matrix), output_shape)


def mask_points_xy(mask: np.ndarray, max_points: int | None = None, seed: int = 0) -> np.ndarray:
    yx = np.argwhere(mask)
    if yx.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = yx[:, ::-1].astype(np.float64)
    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def estimate_euclidean_matrix(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    src_centroid = src_xy.mean(axis=0)
    dst_centroid = dst_xy.mean(axis=0)
    src0 = src_xy - src_centroid
    dst0 = dst_xy - dst_centroid
    h = src0.T @ dst0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - src_centroid @ r.T
    matrix = np.eye(3)
    matrix[:2, :2] = r
    matrix[:2, 2] = t
    return matrix


def estimate_affine_matrix(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    design = np.column_stack([src_xy, np.ones(src_xy.shape[0])])
    coeffs_x, *_ = np.linalg.lstsq(design, dst_xy[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(design, dst_xy[:, 1], rcond=None)
    matrix = np.eye(3)
    matrix[0, :] = coeffs_x
    matrix[1, :] = coeffs_y
    return matrix


def register_affine_iou(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    max_shift: int,
    maxiter: int,
) -> RegistrationResult:
    def objective(params: np.ndarray) -> float:
        warped = warp_mask_with_transform(moving_mask, affine_transform_from_params(params), fixed_mask.shape)
        return -metrics(fixed_mask, warped)[0]

    bounds = [
        (-3.0, 3.0),
        (0.95, 1.05),
        (0.95, 1.05),
        (-2.0, 2.0),
        (-max_shift, max_shift),
        (-max_shift, max_shift),
    ]
    opt = differential_evolution(objective, bounds=bounds, maxiter=maxiter, polish=True, seed=7)
    aligned = warp_mask_with_transform(moving_mask, affine_transform_from_params(opt.x), fixed_mask.shape)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="differential_evolution_iou",
        transform_model="affine",
        score_name="iou",
        score=iou,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=(
            f"angle={opt.x[0]:.4f}, sx={opt.x[1]:.4f}, sy={opt.x[2]:.4f}, "
            f"shear={opt.x[3]:.4f}, tx={opt.x[4]:.2f}, ty={opt.x[5]:.2f}"
        ),
        aligned_mask=aligned,
    )


def register_phase_correlation(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    upsample_factor: int,
) -> RegistrationResult:
    fixed = normalize01(fixed_image)
    moving = soft_mask(moving_mask, sigma=2.0)
    shift_yx, error, phasediff = phase_cross_correlation(
        fixed,
        moving,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    dy, dx = float(shift_yx[0]), float(shift_yx[1])
    aligned = warp_mask_with_transform(moving_mask, translation_transform(dx, dy), fixed_mask.shape)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="phase_correlation",
        transform_model="translation",
        score_name="phase_error",
        score=float(error),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=f"dy={dy:.3f}, dx={dx:.3f}, phasediff={float(phasediff):.6f}",
        aligned_mask=aligned,
    )


def register_logpolar_phase_correlation(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    max_shift: int,
    step: int,
    upsample_factor: int,
) -> RegistrationResult:
    fixed = normalize01(fixed_image)
    moving = soft_mask(moving_mask, sigma=2.0)
    radius = max(8, min(fixed.shape) // 2)
    rows = 360
    cols = radius
    fixed_lp = transform.warp_polar(
        fixed,
        radius=radius,
        scaling="log",
        output_shape=(rows, cols),
    )
    moving_lp = transform.warp_polar(
        moving,
        radius=radius,
        scaling="log",
        output_shape=(rows, cols),
    )
    shift_yx, error, _ = phase_cross_correlation(
        fixed_lp,
        moving_lp,
        upsample_factor=upsample_factor,
        normalization=None,
    )

    angle = float(shift_yx[0] * 360.0 / rows)
    log_base = np.exp(np.log(radius) / max(cols - 1, 1))
    scale = float(log_base ** shift_yx[1])
    scale = float(np.clip(scale, 0.80, 1.25))

    candidates: list[tuple[float, float]] = []
    for angle_candidate in (angle, -angle):
        for scale_candidate in (scale, 1.0 / scale if scale != 0 else 1.0):
            candidates.append((angle_candidate, scale_candidate))
    candidates.append((0.0, 1.0))

    best: RegistrationResult | None = None
    for angle_candidate, scale_candidate in candidates:
        tform = centered_similarity_transform(fixed_mask.shape, angle_candidate, scale_candidate)
        warped = warp_mask_with_transform(moving_mask, tform, fixed_mask.shape)
        refined = register_translation_grid(
            fixed_mask,
            warped,
            max_shift=max_shift,
            step=step,
            score_name="iou",
        )
        refined.method = "logpolar_phase_correlation"
        refined.transform_model = "similarity"
        refined.score_name = "iou_after_phase"
        refined.notes = (
            f"angle_deg={angle_candidate:.4f}, scale={scale_candidate:.5f}, "
            f"phase_error={float(error):.6f}, {refined.notes}"
        )
        if best is None or refined.iou > best.iou:
            best = refined

    assert best is not None
    return best


def chamfer_score(fixed_distance: np.ndarray, moving_mask: np.ndarray, cap_distance: float) -> float:
    coords = np.argwhere(moving_mask)
    if coords.size == 0:
        return -cap_distance
    distances = fixed_distance[coords[:, 0], coords[:, 1]]
    return -float(np.mean(np.minimum(distances, cap_distance)))


def register_chamfer_translation_grid(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    max_shift: int,
    step: int,
    cap_distance: float,
) -> RegistrationResult:
    fixed_distance = ndi.distance_transform_edt(~fixed_mask.astype(bool))
    coords = np.argwhere(moving_mask)
    if coords.size == 0:
        raise RuntimeError("moving mask is empty")

    h, w = fixed_mask.shape
    best = (-math.inf, 0, 0)
    for dy in range(-max_shift, max_shift + 1, step):
        yy = coords[:, 0] + dy
        y_ok = (yy >= 0) & (yy < h)
        if not np.any(y_ok):
            continue
        base_y = yy[y_ok]
        base_x = coords[y_ok, 1]
        for dx in range(-max_shift, max_shift + 1, step):
            xx = base_x + dx
            ok = (xx >= 0) & (xx < w)
            if not np.any(ok):
                continue
            distances = fixed_distance[base_y[ok], xx[ok]]
            score = -float(np.mean(np.minimum(distances, cap_distance)))
            if score > best[0] or (score == best[0] and abs(dy) + abs(dx) < abs(best[1]) + abs(best[2])):
                best = (score, dy, dx)

    aligned = shift_binary(moving_mask, best[1], best[2])
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="chamfer",
        transform_model="translation",
        score_name="negative_mean_distance",
        score=float(best[0]),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=f"mean_distance={-best[0]:.4f}, dy={best[1]}, dx={best[2]}",
        aligned_mask=aligned,
    )


def register_chamfer_euclidean_grid(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    angle_degrees: Iterable[float],
    max_shift: int,
    step: int,
    cap_distance: float,
) -> RegistrationResult:
    best: RegistrationResult | None = None
    best_angle = 0.0
    for angle in angle_degrees:
        rotated = transform.rotate(
            moving_mask.astype(np.float32),
            angle,
            resize=False,
            order=0,
            preserve_range=True,
        ) > 0.5
        result = register_chamfer_translation_grid(fixed_mask, rotated, max_shift, step, cap_distance)
        if best is None or result.score > best.score:
            best = result
            best_angle = float(angle)

    assert best is not None
    best.method = "chamfer"
    best.transform_model = "euclidean"
    best.notes = f"angle_deg={best_angle:.4f}, {best.notes}"
    return best


def register_chamfer_affine(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    max_shift: int,
    cap_distance: float,
    maxiter: int,
) -> RegistrationResult:
    fixed_distance = ndi.distance_transform_edt(~fixed_mask.astype(bool))

    def objective(params: np.ndarray) -> float:
        warped = warp_mask_with_transform(moving_mask, affine_transform_from_params(params), fixed_mask.shape)
        return -chamfer_score(fixed_distance, warped, cap_distance)

    bounds = [
        (-3.0, 3.0),
        (0.95, 1.05),
        (0.95, 1.05),
        (-2.0, 2.0),
        (-max_shift, max_shift),
        (-max_shift, max_shift),
    ]
    opt = differential_evolution(objective, bounds=bounds, maxiter=maxiter, polish=True, seed=17)
    aligned = warp_mask_with_transform(moving_mask, affine_transform_from_params(opt.x), fixed_mask.shape)
    score = chamfer_score(fixed_distance, aligned, cap_distance)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="chamfer",
        transform_model="affine",
        score_name="negative_mean_distance",
        score=score,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=(
            f"mean_distance={-score:.4f}, angle={opt.x[0]:.4f}, sx={opt.x[1]:.4f}, "
            f"sy={opt.x[2]:.4f}, shear={opt.x[3]:.4f}, tx={opt.x[4]:.2f}, ty={opt.x[5]:.2f}"
        ),
        aligned_mask=aligned,
    )


def register_skeleton_icp(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    model: str,
    iterations: int,
    max_points: int,
    max_correspondence: float,
) -> RegistrationResult:
    fixed_skel = morphology.skeletonize(fixed_mask.astype(bool))
    moving_skel = morphology.skeletonize(moving_mask.astype(bool))
    fixed_pts = mask_points_xy(fixed_skel, max_points=max_points, seed=23)
    moving_pts = mask_points_xy(moving_skel, max_points=max_points, seed=29)
    if fixed_pts.shape[0] < 8 or moving_pts.shape[0] < 8:
        raise RuntimeError("not enough skeleton points for ICP")

    tree = cKDTree(fixed_pts)
    total = np.eye(3)
    last_error = math.inf
    used = 0
    mean_error = math.inf

    for _ in range(iterations):
        moving_h = np.column_stack([moving_pts, np.ones(moving_pts.shape[0])])
        transformed = (total @ moving_h.T).T[:, :2]
        distances, indices = tree.query(transformed, k=1)
        keep = distances <= max_correspondence
        used = int(np.count_nonzero(keep))
        if used < 8:
            keep = np.argsort(distances)[: max(8, min(200, distances.shape[0]))]
            src = transformed[keep]
            dst = fixed_pts[indices[keep]]
            mean_error = float(np.mean(distances[keep]))
        else:
            src = transformed[keep]
            dst = fixed_pts[indices[keep]]
            mean_error = float(np.mean(distances[keep]))

        if model == "euclidean":
            delta = estimate_euclidean_matrix(src, dst)
        elif model == "affine":
            delta = estimate_affine_matrix(src, dst)
        else:
            raise ValueError(f"Unsupported ICP model: {model}")

        total = delta @ total
        if abs(last_error - mean_error) < 1e-3:
            break
        last_error = mean_error

    aligned = transformed_mask_from_matrix(moving_mask, total, fixed_mask.shape)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="skeleton_icp",
        transform_model=model,
        score_name="mean_nearest_distance",
        score=-float(mean_error),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=f"mean_distance={mean_error:.4f}, correspondences={used}, matrix={np.array2string(total, precision=4, separator=',')}",
        aligned_mask=aligned,
    )


def nmi_score(fixed_image: np.ndarray, moving_image: np.ndarray, bins: int = 32) -> float:
    hist, _, _ = np.histogram2d(
        fixed_image.ravel(),
        moving_image.ravel(),
        bins=bins,
        range=[[0, 1], [0, 1]],
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    eps = 1e-12
    hx = -np.sum(px[px > 0] * np.log(px[px > 0] + eps))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0] + eps))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0] + eps))
    return (hx + hy) / hxy if hxy > 0 else 0.0


def soft_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    blurred = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return normalize01(blurred)


def downsample_pair(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    if factor <= 1:
        return fixed_image, moving_image
    shape = (fixed_image.shape[0] // factor, fixed_image.shape[1] // factor)
    fixed_small = transform.resize(fixed_image, shape, anti_aliasing=True, preserve_range=True)
    moving_small = transform.resize(moving_image, shape, anti_aliasing=True, preserve_range=True)
    return normalize01(fixed_small), normalize01(moving_small)


def register_mi(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    model: str,
    max_shift: int,
    downsample: int,
    maxiter: int,
) -> RegistrationResult:
    moving_soft = soft_mask(moving_mask)
    fixed_small, moving_small = downsample_pair(normalize01(fixed_image), moving_soft, downsample)
    shift_small = max(2, int(round(max_shift / max(downsample, 1))))

    if model == "translation":
        bounds = [(-shift_small, shift_small), (-shift_small, shift_small)]

        def build(params: np.ndarray) -> transform.AffineTransform:
            tx, ty = params
            return transform.AffineTransform(translation=(tx, ty))

        def upscale(params: np.ndarray) -> np.ndarray:
            tx, ty = params
            return np.array([tx * downsample, ty * downsample])

    elif model == "euclidean":
        bounds = [(-3.0, 3.0), (-shift_small, shift_small), (-shift_small, shift_small)]

        def build(params: np.ndarray) -> transform.EuclideanTransform:
            angle, tx, ty = params
            return transform.EuclideanTransform(rotation=np.deg2rad(angle), translation=(tx, ty))

        def upscale(params: np.ndarray) -> np.ndarray:
            angle, tx, ty = params
            return np.array([angle, tx * downsample, ty * downsample])

    elif model == "affine":
        bounds = [
            (-3.0, 3.0),
            (0.95, 1.05),
            (0.95, 1.05),
            (-2.0, 2.0),
            (-shift_small, shift_small),
            (-shift_small, shift_small),
        ]

        def build(params: np.ndarray) -> transform.AffineTransform:
            return affine_transform_from_params(params)

        def upscale(params: np.ndarray) -> np.ndarray:
            out = params.copy()
            out[4] *= downsample
            out[5] *= downsample
            return out

    else:
        raise ValueError(f"Unsupported MI model: {model}")

    def objective(params: np.ndarray) -> float:
        warped = transform.warp(
            moving_small,
            inverse_map=build(params).inverse,
            output_shape=fixed_small.shape,
            order=1,
            preserve_range=True,
        )
        return -nmi_score(fixed_small, normalize01(warped))

    opt = differential_evolution(objective, bounds=bounds, maxiter=maxiter, polish=False, seed=11)
    refined = minimize(objective, opt.x, method="Powell", options={"maxiter": maxiter * 20})
    params = upscale(refined.x if refined.success else opt.x)
    if model == "translation":
        final_tform = transform.AffineTransform(translation=(params[0], params[1]))
    elif model == "euclidean":
        final_tform = transform.EuclideanTransform(rotation=np.deg2rad(params[0]), translation=(params[1], params[2]))
    else:
        final_tform = affine_transform_from_params(params)

    aligned = warp_mask_with_transform(moving_mask, final_tform, fixed_mask.shape)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="mutual_information",
        transform_model=model,
        score_name="nmi",
        score=nmi_score(
            normalize01(fixed_image),
            normalize01(
                transform.warp(
                    moving_soft,
                    inverse_map=final_tform.inverse,
                    output_shape=fixed_mask.shape,
                    order=1,
                    preserve_range=True,
                )
            ),
        ),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes="params=" + np.array2string(params, precision=4, separator=","),
        aligned_mask=aligned,
    )


def register_orb(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    model: str,
    min_matches: int,
) -> RegistrationResult:
    fixed = exposure.equalize_adapthist(normalize01(fixed_image))
    moving = exposure.equalize_adapthist(soft_mask(moving_mask, sigma=1.5))
    orb_fixed = feature.ORB(n_keypoints=800, fast_threshold=0.05)
    orb_moving = feature.ORB(n_keypoints=800, fast_threshold=0.05)
    orb_fixed.detect_and_extract(fixed)
    orb_moving.detect_and_extract(moving)
    matches = feature.match_descriptors(
        orb_moving.descriptors,
        orb_fixed.descriptors,
        cross_check=True,
        max_ratio=0.8,
    )
    if matches.shape[0] < min_matches:
        raise RuntimeError(f"ORB found only {matches.shape[0]} matches")

    src = orb_moving.keypoints[matches[:, 0]][:, ::-1]
    dst = orb_fixed.keypoints[matches[:, 1]][:, ::-1]
    if model == "euclidean":
        model_cls = transform.EuclideanTransform
    elif model == "affine":
        model_cls = transform.AffineTransform
    elif model == "homography":
        model_cls = transform.ProjectiveTransform
    else:
        raise ValueError(f"Unsupported ORB model: {model}")

    tform, inliers = measure.ransac(
        (src, dst),
        model_cls,
        min_samples=4 if model == "homography" else 3,
        residual_threshold=4,
        max_trials=500,
    )
    if tform is None or inliers is None or int(np.count_nonzero(inliers)) < min_matches:
        raise RuntimeError("ORB/RANSAC did not produce a stable transform")

    aligned = warp_mask_with_transform(moving_mask, tform, fixed_mask.shape)
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="orb_ransac",
        transform_model=model,
        score_name="inliers",
        score=float(np.count_nonzero(inliers)),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes=f"matches={matches.shape[0]}, inliers={int(np.count_nonzero(inliers))}",
        aligned_mask=aligned,
    )


def register_ecc(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    model: str,
    iterations: int,
) -> RegistrationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; ECC skipped")

    fixed = normalize01(fixed_image).astype(np.float32)
    moving = soft_mask(moving_mask, sigma=2.0).astype(np.float32)
    fixed = cv2.GaussianBlur(fixed, (5, 5), 0)
    moving = cv2.GaussianBlur(moving, (5, 5), 0)

    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }
    if model not in motion_map:
        raise ValueError(f"Unsupported ECC model: {model}")

    warp = np.eye(3, 3, dtype=np.float32) if model == "homography" else np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)
    cc, warp = cv2.findTransformECC(fixed, moving, warp, motion_map[model], criteria, None, 5)

    h, w = fixed_mask.shape
    if model == "homography":
        warped = cv2.warpPerspective(
            moving_mask.astype(np.uint8),
            warp,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpAffine(
            moving_mask.astype(np.uint8),
            warp,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    aligned = warped > 0
    iou, dice, precision, recall = metrics(fixed_mask, aligned)
    return RegistrationResult(
        plant="",
        method="ecc",
        transform_model=model,
        score_name="ecc",
        score=float(cc),
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        notes="warp=" + np.array2string(warp, precision=4, separator=","),
        aligned_mask=aligned,
    )


def result_row(result: RegistrationResult) -> dict[str, str | float]:
    return {
        "plant": result.plant,
        "method": result.method,
        "transform_model": result.transform_model,
        "score_name": result.score_name,
        "score": result.score,
        "iou": result.iou,
        "dice": result.dice,
        "precision": result.precision,
        "recall": result.recall,
        "notes": result.notes,
    }


def try_method(label: str, fn: Callable[[], RegistrationResult]) -> RegistrationResult:
    try:
        return fn()
    except Exception as exc:
        return RegistrationResult(
            plant="",
            method=label,
            transform_model="failed",
            score_name="",
            score=0.0,
            iou=0.0,
            dice=0.0,
            precision=0.0,
            recall=0.0,
            notes=f"{type(exc).__name__}: {exc}",
            aligned_mask=None,
        )


def plant_names(root_analysis_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(p.name for p in root_analysis_dir.glob("plant_*") if p.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="VNIR analysis directory containing band_export/")
    parser.add_argument("--root-analysis-dir", type=Path, required=True, help="RootCam analysis directory containing plant_N masks")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: analysis-dir/alignment_benchmark")
    parser.add_argument("--plants", nargs="*", default=None, help="Plant names to process, e.g. plant_1 plant_2")
    parser.add_argument("--root-band", type=int, default=130)
    parser.add_argument("--leaf-band", type=int, default=290)
    parser.add_argument("--root-tophat-radius", type=int, default=15)
    parser.add_argument("--max-shift", type=int, default=120)
    parser.add_argument("--grid-step", type=int, default=2)
    parser.add_argument("--affine-maxiter", type=int, default=20)
    parser.add_argument("--mi-maxiter", type=int, default=12)
    parser.add_argument("--mi-downsample", type=int, default=3)
    parser.add_argument("--ecc-iterations", type=int, default=300)
    parser.add_argument("--min-orb-matches", type=int, default=12)
    parser.add_argument("--phase-upsampling", type=int, default=10)
    parser.add_argument("--chamfer-cap-distance", type=float, default=30.0)
    parser.add_argument("--chamfer-maxiter", type=int, default=18)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-max-points", type=int, default=2500)
    parser.add_argument("--icp-max-correspondence", type=float, default=35.0)
    parser.add_argument("--skip-slow", action="store_true", help="Skip affine-IoU, chamfer-affine, and MI optimizers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir
    root_analysis_dir = args.root_analysis_dir
    out_dir = args.out_dir or analysis_dir / "alignment_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    root_band = read_gray(analysis_dir / "band_export" / f"band_{args.root_band:03}.png")
    leaf_band = read_gray(analysis_dir / "band_export" / f"band_{args.leaf_band:03}.png")
    fixed_root_image = root_enhanced_band(root_band, radius=args.root_tophat_radius)
    fixed_root_mask = fixed_root_mask_from_band(root_band, radius=args.root_tophat_radius)
    fixed_leaf_mask = fixed_leaf_mask_from_band(leaf_band)

    Image.fromarray((fixed_root_mask.astype(np.uint8) * 255)).save(out_dir / "fixed_root_mask.png")
    Image.fromarray((fixed_leaf_mask.astype(np.uint8) * 255)).save(out_dir / "fixed_leaf_mask.png")

    rows: list[dict[str, str | float]] = []
    plants = plant_names(root_analysis_dir, args.plants)
    angle_candidates = np.linspace(-2.0, 2.0, 17)

    for plant in plants:
        plant_dir = root_analysis_dir / plant
        run_params_path = analysis_dir / plant / "run_parameters.json"
        if not run_params_path.exists():
            warnings.warn(f"Skipping {plant}: missing {run_params_path}")
            continue

        root_mask = read_mask(plant_dir / "root_mask.png")
        current = load_current_root_alignment(root_mask, run_params_path)
        iou, dice, precision, recall = metrics(fixed_root_mask, current)
        baseline = RegistrationResult(
            plant=plant,
            method="current_hpx_saved",
            transform_model="resize_rotate_translate",
            score_name="saved",
            score=iou,
            iou=iou,
            dice=dice,
            precision=precision,
            recall=recall,
            notes="Rebuilt from plant run_parameters.json",
            aligned_mask=current,
        )

        candidates: list[RegistrationResult] = [baseline]
        candidates.append(
            register_translation_grid(
                fixed_root_mask,
                current,
                max_shift=args.max_shift,
                step=args.grid_step,
                score_name="iou",
            )
        )
        candidates.append(
            register_translation_grid(
                fixed_root_mask,
                current,
                max_shift=args.max_shift,
                step=args.grid_step,
                score_name="dice",
            )
        )
        candidates.append(
            register_euclidean_grid(
                fixed_root_mask,
                current,
                angle_candidates,
                max_shift=args.max_shift,
                step=args.grid_step,
                score_name="iou",
            )
        )
        candidates.append(
            try_method(
                "phase_correlation",
                lambda: register_phase_correlation(
                    fixed_root_image,
                    fixed_root_mask,
                    current,
                    upsample_factor=args.phase_upsampling,
                ),
            )
        )
        candidates.append(
            try_method(
                "logpolar_phase_correlation",
                lambda: register_logpolar_phase_correlation(
                    fixed_root_image,
                    fixed_root_mask,
                    current,
                    max_shift=args.max_shift,
                    step=args.grid_step,
                    upsample_factor=args.phase_upsampling,
                ),
            )
        )
        candidates.append(
            try_method(
                "chamfer_translation",
                lambda: register_chamfer_translation_grid(
                    fixed_root_mask,
                    current,
                    max_shift=args.max_shift,
                    step=args.grid_step,
                    cap_distance=args.chamfer_cap_distance,
                ),
            )
        )
        candidates.append(
            try_method(
                "chamfer_euclidean",
                lambda: register_chamfer_euclidean_grid(
                    fixed_root_mask,
                    current,
                    angle_candidates,
                    max_shift=args.max_shift,
                    step=args.grid_step,
                    cap_distance=args.chamfer_cap_distance,
                ),
            )
        )
        for model in ("euclidean", "affine"):
            candidates.append(
                try_method(
                    f"skeleton_icp_{model}",
                    lambda model=model: register_skeleton_icp(
                        fixed_root_mask,
                        current,
                        model=model,
                        iterations=args.icp_iterations,
                        max_points=args.icp_max_points,
                        max_correspondence=args.icp_max_correspondence,
                    ),
                )
            )

        if not args.skip_slow:
            candidates.append(
                try_method(
                    "affine_iou",
                    lambda: register_affine_iou(
                        fixed_root_mask,
                        current,
                        max_shift=args.max_shift,
                        maxiter=args.affine_maxiter,
                    ),
                )
            )
            candidates.append(
                try_method(
                    "chamfer_affine",
                    lambda: register_chamfer_affine(
                        fixed_root_mask,
                        current,
                        max_shift=args.max_shift,
                        cap_distance=args.chamfer_cap_distance,
                        maxiter=args.chamfer_maxiter,
                    ),
                )
            )
            for model in ("translation", "euclidean", "affine"):
                candidates.append(
                    try_method(
                        f"mi_{model}",
                        lambda model=model: register_mi(
                            fixed_root_image,
                            fixed_root_mask,
                            current,
                            model=model,
                            max_shift=args.max_shift,
                            downsample=args.mi_downsample,
                            maxiter=args.mi_maxiter,
                        ),
                    )
                )

        for model in ("translation", "euclidean", "affine", "homography"):
            candidates.append(
                try_method(
                    f"ecc_{model}",
                    lambda model=model: register_ecc(
                        fixed_root_image,
                        fixed_root_mask,
                        current,
                        model=model,
                        iterations=args.ecc_iterations,
                    ),
                )
            )

        for model in ("euclidean", "affine", "homography"):
            candidates.append(
                try_method(
                    f"orb_{model}",
                    lambda model=model: register_orb(
                        fixed_root_image,
                        fixed_root_mask,
                        current,
                        model=model,
                        min_matches=args.min_orb_matches,
                    ),
                )
            )

        plant_out = out_dir / plant
        plant_out.mkdir(parents=True, exist_ok=True)
        for result in candidates:
            result.plant = plant
            rows.append(result_row(result))
            if result.aligned_mask is not None:
                overlay_name = f"{result.method}_{result.transform_model}.png".replace("/", "_")
                save_mask_overlay(fixed_root_image, result.aligned_mask, plant_out / overlay_name)

        best = max((c for c in candidates if c.aligned_mask is not None), key=lambda c: c.iou)
        save_mask_overlay(fixed_root_image, best.aligned_mask, plant_out / "best_overlay.png")
        print(f"{plant}: baseline IoU={baseline.iou:.4f}; best {best.method}/{best.transform_model} IoU={best.iou:.4f}")

    csv_path = out_dir / "alignment_benchmark.csv"
    fieldnames = ["plant", "method", "transform_model", "score_name", "score", "iou", "dice", "precision", "recall", "notes"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
