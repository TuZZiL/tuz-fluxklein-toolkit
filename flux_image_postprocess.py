"""
TUZ Klein Edit Composite.

Postprocess node for compositing a generated edit back onto the original image.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

try:  # pragma: no cover - package vs direct import
    from .edit_composite_reporting import build_debug_gallery, build_report_lines
except ImportError:  # pragma: no cover
    from edit_composite_reporting import build_debug_gallery, build_report_lines

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "TUZ Klein Edit Composite requires opencv-python-headless. "
            "Install it and restart ComfyUI."
        )
    return cv2


def _to_numpy_image(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,3] or [H,W,3], got {arr.shape!r}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _to_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().float().numpy()
    else:
        arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected MASK tensor with shape [B,H,W] or [H,W], got {arr.shape!r}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _image_size(image: np.ndarray) -> Tuple[int, int]:
    return int(image.shape[0]), int(image.shape[1])


def _diag(h: int, w: int) -> float:
    return math.sqrt(float(h) * float(h) + float(w) * float(w))


def _pct_to_px(pct: float, diag: float) -> int:
    return max(0, int(round(abs(pct) * diag / 100.0)))


def _kernel_for_radius(radius_px: int) -> Tuple[int, int]:
    k = max(3, radius_px * 2 + 1)
    if k % 2 == 0:
        k += 1
    return (k, k)


def _resize_rgb(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    cv = _require_cv2()
    h, w = size
    if image.shape[:2] == (h, w):
        return image
    return cv.resize(image, (w, h), interpolation=cv.INTER_LANCZOS4)


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    cv = _require_cv2()
    h, w = size
    if mask.shape[:2] == (h, w):
        return mask
    return cv.resize(mask, (w, h), interpolation=cv.INTER_LINEAR)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    lab = cv.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    lab_u8 = lab.copy().astype(np.float32)
    lab_u8[..., 0] = lab_u8[..., 0] * 255.0 / 100.0
    lab_u8[..., 1] += 128.0
    lab_u8[..., 2] += 128.0
    rgb = cv.cvtColor(np.clip(lab_u8, 0.0, 255.0).astype(np.uint8), cv.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _create_color_wheel():
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry

    colorwheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col:col + yg, 1] = 255
    col += yg

    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc

    colorwheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(0, cb) / cb)
    colorwheel[col:col + cb, 2] = 255
    col += cb

    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm

    colorwheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(0, mr) / mr)
    colorwheel[col:col + mr, 0] = 255
    return colorwheel


COLORWHEEL = _create_color_wheel()


def _flow_to_color(flow, max_flow=None):
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi

    if max_flow is None:
        max_flow = np.percentile(mag, 99)

    if max_flow > 0:
        mag = np.clip(mag * 8 / max_flow, 0, 8)

    angle = (angle + 1) / 2
    fk = (angle * (COLORWHEEL.shape[0] - 1) + 0.5).astype(np.int32)
    fk = np.clip(fk, 0, COLORWHEEL.shape[0] - 1)
    color = COLORWHEEL[fk]
    mag = np.clip(mag, 0, 1)
    color = (1 - mag[..., np.newaxis]) * 255 + mag[..., np.newaxis] * color
    return color.astype(np.uint8)


def _apply_heatmap(img_float, mask_float, colormap=None):
    cv = _require_cv2()
    cmap = cv.COLORMAP_JET if colormap is None else colormap
    mask_u8 = np.clip(mask_float * 255, 0, 255).astype(np.uint8)
    heatmap = cv.applyColorMap(mask_u8, cmap)
    heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = 0.6
    return np.clip((1 - alpha) * img_float + alpha * heatmap, 0.0, 1.0)


def _stack_images(images, target_h=384):
    cv = _require_cv2()
    prepared = []
    for img in images:
        if img is None:
            continue
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            img = cv.resize(img, (int(img.shape[1] * scale), target_h))
        prepared.append(img)

    if not prepared:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)

    while len(prepared) < 3:
        h, w = prepared[0].shape[:2]
        prepared.append(np.full((h, w, 3), 64, dtype=np.uint8))

    rows = []
    for i in range(0, len(prepared), 3):
        row = prepared[i:i + 3]
        if len(row) < 3:
            h, w = row[0].shape[:2]
            while len(row) < 3:
                row.append(np.full((h, w, 3), 64, dtype=np.uint8))
        rows.append(np.hstack(row))

    if len(rows) == 1:
        return rows[0]

    max_width = max(row.shape[1] for row in rows)
    padded_rows = [
        np.hstack([row, np.full((row.shape[0], max_width - row.shape[1], 3), 64, dtype=np.uint8)])
        if row.shape[1] < max_width else row
        for row in rows
    ]
    return np.vstack(padded_rows)


def _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, matches, inlier_mask=None):
    cv = _require_cv2()
    if len(kp1) == 0 or len(kp2) == 0 or len(matches) == 0:
        h = max(gray_orig.shape[0], gray_gen.shape[0])
        w = gray_orig.shape[1] + gray_gen.shape[1]
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[:gray_orig.shape[0], :gray_orig.shape[1]] = gray_orig
        canvas[:gray_gen.shape[0], gray_orig.shape[1]:] = gray_gen
        return cv.cvtColor(canvas, cv.COLOR_GRAY2RGB)

    h1, w1 = gray_orig.shape[:2]
    h2, w2 = gray_gen.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = cv.cvtColor(gray_orig, cv.COLOR_GRAY2RGB)
    canvas[:h2, w1:w1 + w2, :] = cv.cvtColor(gray_gen, cv.COLOR_GRAY2RGB)

    kp2_shifted = [cv.KeyPoint(p.pt[0] + w1, p.pt[1], p.size) for p in kp2]
    inlier_set = set()
    if inlier_mask is not None:
        inlier_set = set(np.where(inlier_mask.ravel())[0])

    for i, m in enumerate(matches):
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2_shifted[m.trainIdx].pt[0]), int(kp2_shifted[m.trainIdx].pt[1]))
        color = (0, 255, 0) if (inlier_mask is not None and i in inlier_set) else (0, 0, 255) if inlier_mask is not None else (128, 128, 128)
        cv.line(canvas, pt1, pt2, color, 1, cv.LINE_AA)
        cv.circle(canvas, pt1, 3, color, -1, cv.LINE_AA)
        cv.circle(canvas, pt2, 3, color, -1, cv.LINE_AA)

    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        cv.putText(canvas, f"Matches: {len(matches)} | Inliers: {n_inliers}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return canvas.astype(np.float32) / 255.0


def _compute_diff_map(orig_np: np.ndarray, gen_np: np.ndarray, blur_kernel: Tuple[int, int]) -> np.ndarray:
    cv = _require_cv2()
    o_blur = cv.GaussianBlur(orig_np, blur_kernel, 0)
    g_blur = cv.GaussianBlur(gen_np, blur_kernel, 0)

    o_lab = _rgb_to_lab(o_blur)
    g_lab = _rgb_to_lab(g_blur)
    diff_lab = o_lab - g_lab
    diff_lab[..., 0] *= 0.5
    diff_lab[..., 1] *= 1.2
    diff_lab[..., 2] *= 1.2
    color_diff = np.sqrt(np.sum(diff_lab**2, axis=-1))

    o_gray = cv.cvtColor((o_blur * 255).astype(np.uint8), cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g_gray = cv.cvtColor((g_blur * 255).astype(np.uint8), cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    o_gx = cv.Sobel(o_gray, cv.CV_32F, 1, 0, ksize=3)
    o_gy = cv.Sobel(o_gray, cv.CV_32F, 0, 1, ksize=3)
    g_gx = cv.Sobel(g_gray, cv.CV_32F, 1, 0, ksize=3)
    g_gy = cv.Sobel(g_gray, cv.CV_32F, 0, 1, ksize=3)
    o_mag = np.sqrt(o_gx**2 + o_gy**2)
    g_mag = np.sqrt(g_gx**2 + g_gy**2)
    struct_diff = np.abs(o_mag - g_mag) * 40.0
    return color_diff + struct_diff


def _auto_threshold_mad(diff_map: np.ndarray, valid_mask: np.ndarray = None, k: float = 6.0, min_t: float = 3.0, max_t: float = 60.0) -> float:
    if valid_mask is not None and valid_mask.sum() > 100:
        sample = diff_map[valid_mask > 0.5]
    else:
        sample = diff_map.flatten()
    if len(sample) > 50000:
        sample = np.random.choice(sample, 50000, replace=False)
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    mad = max(mad, 0.5)
    threshold = med + k * mad
    return float(np.clip(threshold, min_t, max_t))


def _open_by_reconstruction(mask: np.ndarray, radius_px: int) -> np.ndarray:
    cv = _require_cv2()
    if radius_px <= 0:
        return mask
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    marker = cv.erode(mask.astype(np.uint8), k).astype(np.float32)
    orig = mask.astype(np.float32)
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    while True:
        expanded = np.minimum(cv.dilate(marker, k3), orig)
        if np.array_equal(expanded, marker):
            break
        marker = expanded
    return marker


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    binary = (mask > 0.5).astype(np.uint8)
    h, w = binary.shape
    n, labeled = cv.connectedComponents(binary, connectivity=8)
    result = binary.copy()
    for island_id in range(1, n):
        island = (labeled == island_id).astype(np.uint8)
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:h + 1, 1:w + 1] = island
        inv = 1 - padded
        cv.floodFill(inv, None, (0, 0), 0)
        interior = inv[1:h + 1, 1:w + 1]
        result = np.clip(result + interior, 0, 1).astype(np.uint8)
    return result.astype(np.float32)


def _keep_largest_islands(mask: np.ndarray, max_islands: int) -> np.ndarray:
    cv = _require_cv2()
    if max_islands <= 0:
        return mask
    binary = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = [(label, stats[label, cv.CC_STAT_AREA]) for label in range(1, num_labels)]
    areas.sort(key=lambda item: item[1], reverse=True)
    keep = {label for label, _ in areas[:max_islands]}
    return np.where(np.isin(labels, list(keep)), 1.0, 0.0).astype(np.float32)


def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    cv = _require_cv2()
    if grow_px == 0:
        return mask
    radius = abs(grow_px)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    binary = (mask > 0.5).astype(np.uint8) * 255
    if grow_px > 0:
        out = cv.dilate(binary, kernel)
    else:
        out = cv.erode(binary, kernel)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def _bleed_mask(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    invalid = (valid_mask < 0.5).astype(np.uint8)
    if invalid.max() == 0:
        return mask
    dist = cv.distanceTransform(invalid, cv.DIST_L2, 3)
    max_depth = int(np.ceil(np.max(dist)))
    if max_depth == 0:
        return mask
    bled_mask = mask.copy()
    remaining = max_depth
    while remaining > 0:
        step = min(remaining, 60)
        k_size = step * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        bled_mask = cv.dilate(bled_mask, kernel)
        remaining -= step
    return np.where(valid_mask > 0.5, mask, bled_mask).astype(np.float32)


def _finalize_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    grow_px: int,
    close_px: int,
    noise_removal_px: int,
    max_islands: int,
    fill_holes: bool,
    fill_borders: bool,
    feather_px: float,
) -> np.ndarray:
    cv = _require_cv2()
    current = np.clip(mask.astype(np.float32), 0.0, 1.0)

    if noise_removal_px > 0:
        current = _open_by_reconstruction(current, noise_removal_px)

    if close_px > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, _kernel_for_radius(close_px))
        current = cv.morphologyEx(current.astype(np.float32), cv.MORPH_CLOSE, kernel).astype(np.float32)

    if fill_holes:
        current = _fill_holes(current)

    if max_islands > 0:
        current = _keep_largest_islands(current, max_islands)

    if grow_px != 0:
        current = _grow_mask(current, grow_px)

    if fill_borders:
        current = _bleed_mask(current, valid_mask)
    else:
        current = current * valid_mask

    if feather_px > 0:
        kernel = _kernel_for_radius(max(1, int(round(feather_px))))
        current = cv.GaussianBlur(current.astype(np.float32), kernel, 0)

    return np.clip(current, 0.0, 1.0).astype(np.float32)


def _apply_color_match(
    orig_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    composite_mask: np.ndarray,
    valid_mask: np.ndarray,
    blend_strength: float,
) -> Tuple[np.ndarray, bool]:
    if blend_strength <= 0.0:
        return gen_rgb, False

    bg_mask = (composite_mask < 0.05) & (valid_mask > 0.5)
    if bg_mask.sum() < 100:
        return gen_rgb, False

    orig_lab = _rgb_to_lab(orig_rgb)
    gen_lab = _rgb_to_lab(gen_rgb)
    orig_mean = orig_lab[bg_mask].mean(axis=0)
    orig_std = orig_lab[bg_mask].std(axis=0) + 1e-5
    gen_mean = gen_lab[bg_mask].mean(axis=0)
    gen_std = gen_lab[bg_mask].std(axis=0) + 1e-5
    matched_lab = ((gen_lab - gen_mean) / gen_std) * orig_std + orig_mean
    matched_rgb = _lab_to_rgb(matched_lab)
    blended_rgb = (matched_rgb * blend_strength) + (gen_rgb * (1.0 - blend_strength))
    return np.clip(blended_rgb, 0.0, 1.0).astype(np.float32), True


def _seamless_blend(orig_float: np.ndarray, gen_float: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    orig_u8 = (np.clip(orig_float, 0, 1) * 255).astype(np.uint8)
    gen_u8 = (np.clip(gen_float, 0, 1) * 255).astype(np.uint8)
    binary_mask = (mask_float > 0.1).astype(np.uint8) * 255
    binary_mask[0, :] = 0
    binary_mask[-1, :] = 0
    binary_mask[:, 0] = 0
    binary_mask[:, -1] = 0

    x, y, w, h = cv.boundingRect(binary_mask)
    m3 = mask_float[..., np.newaxis]
    if w == 0 or h == 0:
        return np.clip(orig_float * (1.0 - m3) + gen_float * m3, 0, 1)

    center = (x + w // 2, y + h // 2)
    try:
        cloned_u8 = cv.seamlessClone(gen_u8, orig_u8, binary_mask, center, cv.NORMAL_CLONE)
        cloned_float = cloned_u8.astype(np.float32) / 255.0
        return np.clip(orig_float * (1.0 - m3) + cloned_float * m3, 0, 1)
    except Exception:
        return np.clip(orig_float * (1.0 - m3) + gen_float * m3, 0, 1)


def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    cv = _require_cv2()
    flow = cv.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)
    flow[..., 0] = cv.medianBlur(flow[..., 0], 5)
    flow[..., 1] = cv.medianBlur(flow[..., 1], 5)
    return flow


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv.remap(image, map_x, map_y, cv.INTER_LINEAR, cv.BORDER_REFLECT)


def _fwd_bwd_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    h, w = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bwd_x = cv.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1], cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
    bwd_y = cv.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1], cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
    return np.sqrt((flow_fwd[..., 0] + bwd_x) ** 2 + (flow_fwd[..., 1] + bwd_y) ** 2)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    return (_fwd_bwd_error(flow_fwd, flow_bwd) > threshold).astype(np.float32)


def _detect_and_align(orig_rgb: np.ndarray, gen_rgb: np.ndarray, orig_mask: Optional[np.ndarray] = None, debug: bool = False):
    cv = _require_cv2()
    h, w = orig_rgb.shape[:2]
    debug_dict = {} if debug else None

    gray_orig = cv.cvtColor((np.clip(orig_rgb, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
    gray_gen = cv.cvtColor((np.clip(gen_rgb, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_orig = clahe.apply(gray_orig)
    eq_gen = clahe.apply(gray_gen)

    fallback_valid = np.ones((h, w), dtype=np.float32)
    detector = None
    norm = None
    detector_name = "none"

    if hasattr(cv, "SIFT_create"):
        try:
            detector = cv.SIFT_create(nfeatures=8000, contrastThreshold=0.03)
            norm = cv.NORM_L2
            detector_name = "sift"
        except Exception:
            detector = None

    if detector is None:
        detector = cv.ORB_create(nfeatures=8000)
        norm = cv.NORM_HAMMING
        detector_name = "orb"

    kp1, des1 = detector.detectAndCompute(eq_orig, mask=orig_mask)
    kp2, des2 = detector.detectAndCompute(eq_gen, mask=None)

    if debug:
        debug_dict["orig_kp"] = len(kp1) if kp1 else 0
        debug_dict["gen_kp"] = len(kp2) if kp2 else 0
        debug_dict["detector"] = detector_name

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        if debug:
            debug_dict["match_viz"] = _draw_sift_matches(eq_orig, eq_gen, kp1 or [], kp2 or [], [])
        return gen_rgb.copy(), None, 0, fallback_valid, debug_dict

    matcher = cv.BFMatcher(norm)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    ratio = 0.75 if norm == cv.NORM_L2 else 0.8
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if debug:
        debug_dict["matches_before_ransac"] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, None)

    if len(good) < 8:
        if debug:
            debug_dict["match_viz"] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, None)
        return gen_rgb.copy(), None, 0, fallback_valid, debug_dict

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_gen = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ransac_method = getattr(cv, "USAC_MAGSAC", cv.RANSAC)
    H_mat, inlier_mask = cv.findHomography(pts_gen, pts_orig, ransac_method, 4.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if debug:
        debug_dict["match_viz"] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, inlier_mask)
        debug_dict["n_matches"] = len(good)
        debug_dict["n_inliers"] = n_inliers

    if H_mat is None:
        return gen_rgb.copy(), None, 0, fallback_valid, debug_dict

    det = H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0]
    if not (0.2 < det < 5.0):
        if debug:
            debug_dict["failure_reason"] = f"Bad determinant: {det:.2f} (needs 0.2-5.0)"
        return gen_rgb.copy(), None, 0, fallback_valid, debug_dict

    aligned = cv.warpPerspective(gen_rgb, H_mat, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    valid_mask = cv.warpPerspective(np.ones((h, w), dtype=np.float32), H_mat, (w, h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    if debug:
        debug_dict["homography"] = H_mat
        debug_dict["determinant"] = det

    return aligned, H_mat, n_inliers, valid_mask, debug_dict


def _merge_custom_mask(base_mask: np.ndarray, custom_mask: Optional[np.ndarray], mode: str) -> np.ndarray:
    if custom_mask is None:
        return base_mask
    if mode == "replace":
        return custom_mask
    if mode == "add":
        return np.maximum(base_mask, custom_mask)
    if mode == "subtract":
        return np.clip(base_mask * (1.0 - custom_mask), 0.0, 1.0)
    return base_mask


def _composite_with_custom_mask(
    original_np: np.ndarray,
    generated_np: np.ndarray,
    *,
    h: int,
    w: int,
    diag: float,
    gray_orig: np.ndarray,
    flow_preset: int,
    occlusion_threshold: float,
    feather_px: float,
    color_match_blend: float,
    use_occlusion: bool,
    fill_borders: bool,
    custom_mask: np.ndarray,
    custom_mask_mode: str,
    poisson_blend_edges: bool,
    debug: bool,
    debug_images,
):
    cv = _require_cv2()
    auto_report = {}
    gen_pre, _, inliers, valid, debug_sift = _detect_and_align(original_np, generated_np, orig_mask=None, debug=debug)
    if debug:
        debug_images["pass1_sift_matches"] = debug_sift.get("match_viz", np.zeros((h, w, 3), dtype=np.float32))
        debug_images["pass1_validity_mask"] = valid.copy()

    sharp_mask = np.clip(custom_mask, 0.0, 1.0) * valid
    if fill_borders:
        sharp_mask = _bleed_mask(sharp_mask, valid)

    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist = cv.distanceTransform(inv_mask, cv.DIST_L2, 5)
            fade_dist = feather_px * 2.5
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    if not fill_borders:
        composite_mask *= valid

    gen_pre, color_matched = _apply_color_match(original_np, gen_pre, composite_mask, valid, color_match_blend)
    if color_matched:
        auto_report["color_match_applied"] = True

    if poisson_blend_edges:
        result = _seamless_blend(original_np, gen_pre, composite_mask)
    else:
        m3 = composite_mask[..., np.newaxis]
        result = np.clip(original_np * (1.0 - m3) + gen_pre * m3, 0, 1)

    flow_fwd_final = _dis_flow(
        gray_orig,
        cv.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY),
        flow_preset,
    )
    if use_occlusion:
        flow_bwd_final = _dis_flow(
            cv.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY),
            gray_orig,
            flow_preset,
        )
        occ_thresh = (
            occlusion_threshold
            if occlusion_threshold >= 0
            else _auto_threshold_mad(_fwd_bwd_error(flow_fwd_final, flow_bwd_final), valid, k=5.0, min_t=1.0, max_t=30.0)
        )
        if occlusion_threshold < 0:
            auto_report["auto_occlusion"] = occ_thresh
    else:
        flow_bwd_final = None
        occ_thresh = 0

    flow_mag = np.sqrt((flow_fwd_final**2).sum(axis=2))
    stats = {
        "changed_pct": 100 * float((sharp_mask > 0.5).sum()) / (h * w),
        "occluded_px": int((_fwd_bwd_error(flow_fwd_final, flow_bwd_final) > occ_thresh).sum()) if use_occlusion else 0,
        "flow_mean_px": float(flow_mag.mean()),
        "flow_p99_px": float(np.percentile(flow_mag, 99)),
        "median_de": 0.0,
        "resolution": f"{w}x{h}",
        "diagonal_px": round(diag),
        "pass1_inliers": inliers,
        "pass2_used": False,
        "custom_mask": True,
        "poisson_used": poisson_blend_edges,
        "custom_mask_mode": custom_mask_mode,
    }
    stats.update(auto_report)

    if debug:
        debug_images["final_flow"] = _flow_to_color(flow_fwd_final)
        debug_images["final_alignment"] = (
            np.hstack([original_np, gen_pre]) if original_np.shape == gen_pre.shape else _stack_images([original_np, gen_pre])
        )
        debug_images["composite_breakdown"] = np.hstack([
            original_np, gen_pre, result, np.stack([composite_mask] * 3, axis=-1),
        ])

    return result, composite_mask, stats, debug_images


def _composite_with_auto_mask(
    original_np: np.ndarray,
    generated_np: np.ndarray,
    *,
    h: int,
    w: int,
    diag: float,
    gray_orig: np.ndarray,
    blur_kernel: Tuple[int, int],
    sk: int,
    delta_e_threshold: float,
    flow_preset: int,
    occlusion_threshold: float,
    grow_px: int,
    close_radius: int,
    feather_px: float,
    color_match_blend: float,
    noise_removal_px: int,
    max_islands: int,
    fill_holes: bool,
    use_occlusion: bool,
    fill_borders: bool,
    custom_mask: np.ndarray,
    custom_mask_mode: str,
    poisson_blend_edges: bool,
    debug: bool,
    debug_images,
):
    cv = _require_cv2()
    auto_report = {}
    gen_pre_1, _, inliers_1, valid_1, debug_sift1 = _detect_and_align(original_np, generated_np, orig_mask=None, debug=debug)

    if debug:
        debug_images["pass1_sift_matches"] = debug_sift1.get("match_viz", np.zeros((h, w, 3), dtype=np.float32))
        debug_images["pass1_validity_mask"] = valid_1.copy()
        valid_overlay = original_np.copy()
        valid_overlay[valid_1 < 0.5] = [1, 0, 0]
        debug_images["pass1_validity_overlay"] = valid_overlay

    gray_gen_pre_1 = cv.cvtColor((np.clip(gen_pre_1, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
    flow_fwd_1 = _dis_flow(gray_orig, gray_gen_pre_1, flow_preset)
    flow_bwd_1 = _dis_flow(gray_gen_pre_1, gray_orig, flow_preset) if use_occlusion else None
    warped_gen_1 = _warp(gen_pre_1, flow_fwd_1)

    diff_1_direct = _compute_diff_map(original_np, gen_pre_1, blur_kernel)
    diff_1_warped = _compute_diff_map(original_np, warped_gen_1, blur_kernel)
    de_thresh = delta_e_threshold if delta_e_threshold >= 0 else _auto_threshold_mad(diff_1_direct, valid_1)

    blend_w_1 = np.clip(diff_1_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_1_raw = blend_w_1 * diff_1_direct + (1.0 - blend_w_1) * diff_1_warped
    delta_e_1 = cv.GaussianBlur(delta_e_1_raw, (sk, sk), 0)
    local_thresh_1 = _auto_threshold_mad(delta_e_1, valid_1, k=5.0, min_t=3.0, max_t=60.0)
    thresh_map_1 = np.minimum(de_thresh, local_thresh_1)
    coarse_mask = (delta_e_1 > thresh_map_1).astype(np.float32)

    if use_occlusion:
        occ_thresh = (
            occlusion_threshold
            if occlusion_threshold >= 0
            else _auto_threshold_mad(_fwd_bwd_error(flow_fwd_1, flow_bwd_1), valid_1, k=5.0, min_t=1.0, max_t=30.0)
        )
        coarse_mask = np.maximum(coarse_mask, _occlusion_mask(flow_fwd_1, flow_bwd_1, occ_thresh))
    coarse_mask *= valid_1

    if debug:
        de_normalized = np.clip(delta_e_1 / (de_thresh * 1.5 + 1e-6), 0, 1)
        debug_images["pass1_delta_e"] = _apply_heatmap(original_np, de_normalized)
        debug_images["pass1_coarse_mask"] = coarse_mask.copy()
        debug_images["pass1_flow"] = _flow_to_color(flow_fwd_1)
        debug_images["pass1_alignment"] = (
            np.hstack([original_np, warped_gen_1]) if original_np.shape == warped_gen_1.shape else _stack_images([original_np, warped_gen_1])
        )

    bg_mask_u8 = (coarse_mask < 0.1).astype(np.uint8) * 255
    safe_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    bg_mask_safe = cv.erode(bg_mask_u8, safe_k)

    pass2_used = False
    if (bg_mask_safe > 0).sum() > (h * w * 0.05):
        final_aligned_gen, H_sift2, inliers_2, valid_2, debug_sift2 = _detect_and_align(
            original_np, generated_np, orig_mask=bg_mask_safe, debug=debug
        )
        if H_sift2 is not None:
            pass2_used = True
            auto_report["pass2_inliers"] = inliers_2
            if debug:
                debug_images["pass2_sift_matches"] = debug_sift2.get("match_viz", np.zeros((h, w, 3), dtype=np.float32))
                debug_images["pass2_validity_mask"] = valid_2.copy()
        else:
            final_aligned_gen, valid_2 = gen_pre_1, valid_1
            auto_report["pass2_inliers"] = 0
    else:
        final_aligned_gen, valid_2 = gen_pre_1, valid_1
        auto_report["pass2_inliers"] = 0
        if debug:
            debug_images["pass2_skip_reason"] = f"Background too small ({(bg_mask_safe > 0).sum()} px)"

    gray_gen_pre_2 = cv.cvtColor((np.clip(final_aligned_gen, 0, 1) * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
    flow_fwd_2 = _dis_flow(gray_orig, gray_gen_pre_2, flow_preset)
    flow_bwd_2 = _dis_flow(gray_gen_pre_2, gray_orig, flow_preset) if use_occlusion else None
    warped_gen_final = _warp(final_aligned_gen, flow_fwd_2)

    diff_2_direct = _compute_diff_map(original_np, final_aligned_gen, blur_kernel)
    diff_2_warped = _compute_diff_map(original_np, warped_gen_final, blur_kernel)
    blend_w_2 = np.clip(diff_2_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_2_raw = blend_w_2 * diff_2_direct + (1.0 - blend_w_2) * diff_2_warped
    delta_e_2 = cv.GaussianBlur(delta_e_2_raw, (sk, sk), 0)
    local_thresh_2 = _auto_threshold_mad(delta_e_2, valid_2, k=5.0, min_t=3.0, max_t=60.0)
    thresh_map_2 = np.minimum(de_thresh, local_thresh_2)
    coarse_mask_2 = (delta_e_2 > thresh_map_2).astype(np.float32)

    if use_occlusion:
        occ_thresh = (
            occlusion_threshold
            if occlusion_threshold >= 0
            else _auto_threshold_mad(_fwd_bwd_error(flow_fwd_2, flow_bwd_2), valid_2, k=5.0, min_t=1.0, max_t=30.0)
        )
        coarse_mask_2 = np.maximum(coarse_mask_2, _occlusion_mask(flow_fwd_2, flow_bwd_2, occ_thresh))
        if occlusion_threshold < 0:
            auto_report["auto_occlusion"] = occ_thresh
    else:
        occ_thresh = 0

    coarse_mask_2 *= valid_2
    composite_mask = _finalize_mask(
        coarse_mask_2,
        valid_2,
        grow_px=grow_px,
        close_px=close_radius,
        noise_removal_px=noise_removal_px,
        max_islands=max_islands,
        fill_holes=fill_holes,
        fill_borders=fill_borders,
        feather_px=feather_px,
    )

    if custom_mask is not None:
        custom_mask_resized = _resize_mask(custom_mask, (h, w))
        composite_mask = _merge_custom_mask(composite_mask, custom_mask_resized, custom_mask_mode)
        composite_mask = np.clip(composite_mask, 0.0, 1.0).astype(np.float32)
        composite_mask = _finalize_mask(
            composite_mask,
            valid_2,
            grow_px=0,
            close_px=0,
            noise_removal_px=0,
            max_islands=max_islands,
            fill_holes=False,
            fill_borders=fill_borders,
            feather_px=feather_px,
        )

    gen_matched, color_matched = _apply_color_match(original_np, final_aligned_gen, composite_mask, valid_2, color_match_blend)
    if color_matched:
        auto_report["color_match_applied"] = True

    if poisson_blend_edges:
        result = _seamless_blend(original_np, gen_matched, composite_mask)
    else:
        m3 = composite_mask[..., np.newaxis]
        result = np.clip(original_np * (1.0 - m3) + gen_matched * m3, 0, 1)

    flow_mag = np.sqrt((flow_fwd_2**2).sum(axis=2))
    stats = {
        "changed_pct": 100 * float((composite_mask > 0.5).sum()) / (h * w),
        "occluded_px": int((_fwd_bwd_error(flow_fwd_2, flow_bwd_2) > occ_thresh).sum()) if use_occlusion else 0,
        "flow_mean_px": float(flow_mag.mean()),
        "flow_p99_px": float(np.percentile(flow_mag, 99)),
        "median_de": float(np.median(delta_e_2)),
        "resolution": f"{w}x{h}",
        "diagonal_px": round(diag),
        "pass1_inliers": inliers_1,
        "pass2_inliers": auto_report.get("pass2_inliers", 0),
        "pass2_used": pass2_used,
        "poisson_used": poisson_blend_edges,
    }
    stats.update(auto_report)

    if custom_mask is not None:
        stats["custom_mask"] = True
        stats["custom_mask_mode"] = custom_mask_mode

    if debug:
        debug_images["final_alignment"] = (
            np.hstack([original_np, final_aligned_gen])
            if original_np.shape == final_aligned_gen.shape
            else _stack_images([original_np, final_aligned_gen])
        )
        debug_images["final_delta_e"] = _apply_heatmap(original_np, np.clip(delta_e_2 / (de_thresh * 1.5 + 1e-6), 0, 1))
        debug_images["final_flow"] = _flow_to_color(flow_fwd_2)
        debug_images["mask_overlay"] = _apply_heatmap(original_np, composite_mask)
        debug_images["composite_breakdown"] = np.hstack([
            original_np,
            final_aligned_gen,
            result,
            np.stack([composite_mask] * 3, axis=-1),
        ])

    return result, composite_mask, stats, debug_images


def _composite(
    original_np: np.ndarray,
    generated_np: np.ndarray,
    delta_e_threshold: float,
    flow_preset: int,
    occlusion_threshold: float,
    grow_px: int,
    close_radius: int,
    feather_px: float,
    color_match_blend: float,
    noise_removal_px: int = 0,
    max_islands: int = 0,
    fill_holes: bool = False,
    use_occlusion: bool = False,
    fill_borders: bool = True,
    custom_mask: np.ndarray = None,
    custom_mask_mode: str = "replace",
    poisson_blend_edges: bool = False,
    debug: bool = False,
) -> tuple:
    cv = _require_cv2()
    h, w = original_np.shape[:2]
    diag = _diag(h, w)
    debug_images = {} if debug else None

    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8 = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv.cvtColor(orig_u8, cv.COLOR_RGB2GRAY)
    gray_gen = cv.cvtColor(gen_u8, cv.COLOR_RGB2GRAY)
    blur_kernel = _kernel_for_radius(max(1, int(round(diag * 0.008))))
    sk = max(blur_kernel[0], 5)
    if sk % 2 == 0:
        sk += 1

    auto_report = {}

    if custom_mask is not None and custom_mask_mode == "replace":
        return _composite_with_custom_mask(
            original_np,
            generated_np,
            h=h,
            w=w,
            diag=diag,
            gray_orig=gray_orig,
            flow_preset=flow_preset,
            occlusion_threshold=occlusion_threshold,
            feather_px=feather_px,
            color_match_blend=color_match_blend,
            use_occlusion=use_occlusion,
            fill_borders=fill_borders,
            custom_mask=custom_mask,
            custom_mask_mode=custom_mask_mode,
            poisson_blend_edges=poisson_blend_edges,
            debug=debug,
            debug_images=debug_images,
        )

    return _composite_with_auto_mask(
        original_np,
        generated_np,
        h=h,
        w=w,
        diag=diag,
        gray_orig=gray_orig,
        blur_kernel=blur_kernel,
        sk=sk,
        delta_e_threshold=delta_e_threshold,
        flow_preset=flow_preset,
        occlusion_threshold=occlusion_threshold,
        grow_px=grow_px,
        close_radius=close_radius,
        feather_px=feather_px,
        color_match_blend=color_match_blend,
        noise_removal_px=noise_removal_px,
        max_islands=max_islands,
        fill_holes=fill_holes,
        use_occlusion=use_occlusion,
        fill_borders=fill_borders,
        custom_mask=custom_mask,
        custom_mask_mode=custom_mask_mode,
        poisson_blend_edges=poisson_blend_edges,
        debug=debug,
        debug_images=debug_images,
    )


class TuzKleinEditComposite:
    CATEGORY = "image/TUZ"
    TITLE = "TUZ Klein Edit Composite"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "delta_e_threshold": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Change sensitivity. Set -1 for auto-threshold using MAD.",
                }),
                "flow_quality": (["medium", "fast", "ultrafast"], {
                    "default": "medium",
                    "tooltip": "Precision of optical flow used during alignment and diff refinement.",
                }),
                "use_occlusion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add forward-backward flow consistency to the change mask.",
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Occlusion sensitivity. Set -1 for auto-threshold using MAD.",
                }),
                "noise_removal_pct": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Remove speckle noise from the mask as a % of image diagonal.",
                }),
                "close_radius_pct": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Morphological close radius as a % of image diagonal.",
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Fill enclosed holes inside the change mask.",
                }),
                "fill_borders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bleed the mask into invalid warped borders.",
                }),
                "max_islands": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Keep only the N largest connected regions. 0 disables pruning.",
                }),
                "grow_mask_pct": ("FLOAT", {
                    "default": 0.0,
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Grow or shrink the final mask as a % of image diagonal.",
                }),
                "feather_pct": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.25,
                    "tooltip": "Soft blend width at the final mask edge.",
                }),
                "color_match_blend": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Reinhard-like color matching blend against original background.",
                }),
                "poisson_blend_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Poisson/seamless clone for border blending when possible.",
                }),
            },
            "optional": {
                "custom_mask": ("MASK", {
                    "tooltip": "Optional override/merge mask. Use with custom_mask_mode below.",
                }),
                "custom_mask_mode": (["replace", "add", "subtract"], {
                    "default": "replace",
                    "tooltip": "How to combine custom_mask with auto-detected mask.",
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Emit a debug gallery and richer report text.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("composited_image", "change_mask", "report", "debug_gallery")
    FUNCTION = "run"

    def run(
        self,
        generated_image,
        original_image,
        delta_e_threshold=-1.0,
        flow_quality="medium",
        use_occlusion=False,
        occlusion_threshold=-1.0,
        noise_removal_pct=0.0,
        close_radius_pct=0.5,
        fill_holes=False,
        fill_borders=True,
        max_islands=0,
        grow_mask_pct=0.0,
        feather_pct=2.0,
        color_match_blend=1.0,
        poisson_blend_edges=False,
        custom_mask=None,
        custom_mask_mode="replace",
        enable_debug=False,
    ):
        cv = _require_cv2()

        orig_np = _to_numpy_image(original_image)
        gen_np = _to_numpy_image(generated_image)
        if orig_np.shape != gen_np.shape:
            orig_np = _resize_rgb(orig_np, gen_np.shape[:2])

        h, w = gen_np.shape[:2]
        diag = _diag(h, w)
        grow_px = int(round(grow_mask_pct * diag / 100.0))
        feather_px = abs(feather_pct) * diag / 100.0
        close_px = _pct_to_px(close_radius_pct, diag)
        noise_removal_px = _pct_to_px(noise_removal_pct, diag)

        custom_mask_np = None
        if custom_mask is not None:
            custom_mask_np = _to_numpy_mask(custom_mask)
            if custom_mask_np.shape != (h, w):
                custom_mask_np = _resize_mask(custom_mask_np, (h, w))

        flow_preset = {
            "ultrafast": cv.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }[flow_quality]

        result, change_mask, stats, debug_images = _composite(
            orig_np,
            gen_np,
            delta_e_threshold=delta_e_threshold,
            flow_preset=flow_preset,
            occlusion_threshold=occlusion_threshold,
            grow_px=grow_px,
            close_radius=close_px,
            noise_removal_px=noise_removal_px,
            max_islands=max_islands,
            fill_holes=fill_holes,
            use_occlusion=use_occlusion,
            fill_borders=fill_borders,
            feather_px=feather_px,
            color_match_blend=color_match_blend,
            custom_mask=custom_mask_np,
            custom_mask_mode=custom_mask_mode,
            poisson_blend_edges=poisson_blend_edges,
            debug=enable_debug,
        )

        report_lines = build_report_lines(
            stats,
            delta_e_threshold=delta_e_threshold,
            occlusion_threshold=occlusion_threshold,
            grow_mask_pct=grow_mask_pct,
            grow_px=grow_px,
            noise_removal_pct=noise_removal_pct,
            noise_removal_px=noise_removal_px,
            max_islands=max_islands,
            fill_holes=fill_holes,
            fill_borders=fill_borders,
            use_occlusion=use_occlusion,
            feather_pct=feather_pct,
            feather_px=feather_px,
            color_match_blend=color_match_blend,
            flow_quality=flow_quality,
        )

        debug_gallery = build_debug_gallery(enable_debug, debug_images, cv, _stack_images)

        return (
            torch.from_numpy(result.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(change_mask.astype(np.float32)).unsqueeze(0),
            "\n".join(report_lines),
            debug_gallery,
        )


NODE_CLASS_MAPPINGS = {
    "TuzKleinEditComposite": TuzKleinEditComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TuzKleinEditComposite": "TUZ Klein Edit Composite",
}

