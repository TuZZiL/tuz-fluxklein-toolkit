"""
Reporting helpers for TUZ Klein Edit Composite.
"""

from __future__ import annotations

import numpy as np
import torch


def build_report_lines(stats, *, delta_e_threshold, occlusion_threshold, grow_mask_pct, grow_px, noise_removal_pct, noise_removal_px, max_islands, fill_holes, fill_borders, use_occlusion, feather_pct, feather_px, color_match_blend, flow_quality):
    report_lines = [
        "=== TUZ Klein Edit Composite ===",
        f"Resolution:       {stats['resolution']} (diag {stats['diagonal_px']}px)",
        f"Poisson Blending: {'ENABLED' if stats.get('poisson_used') else 'Disabled'}",
    ]

    if stats.get("custom_mask"):
        report_lines.append(f"Mask source:      CUSTOM {stats.get('custom_mask_mode', 'replace').upper()}")
    elif stats.get("pass2_inliers", 0) > 0:
        report_lines.append("Alignment:        Two-pass success")
        report_lines.append(f"Pass 1 Inliers:   {stats['pass1_inliers']}")
        report_lines.append(f"Pass 2 Inliers:   {stats['pass2_inliers']}")
    else:
        report_lines.append(f"Alignment:        {'Pass 2 executed' if stats.get('pass2_used') else 'Pass 2 skipped'}")

    if "auto_delta_e" in stats:
        report_lines.append(f"Diff Threshold:   AUTO (MAD) -> {stats['auto_delta_e']:.1f}")
    else:
        report_lines.append(f"Diff Threshold:   {delta_e_threshold:.1f}")

    if "auto_occlusion" in stats:
        report_lines.append(f"Occlusion Thresh: AUTO (MAD) -> {stats['auto_occlusion']:.1f}")
    else:
        report_lines.append(f"Occlusion Thresh: {occlusion_threshold:.1f}")

    report_lines.extend([
        f"Grow mask:        {grow_mask_pct:+.1f}% -> {grow_px:+d}px",
        f"Noise removal:    {noise_removal_pct:.2f}% -> {noise_removal_px}px",
        f"Max islands:      {max_islands if max_islands > 0 else 'disabled'}",
        f"Fill holes:       {'yes' if fill_holes else 'no'}",
        f"Fill borders:     {'yes' if fill_borders else 'no'}",
        f"Occlusion:        {'enabled' if use_occlusion else 'disabled'}",
        f"Feather:          {feather_pct:.1f}% -> {feather_px:.0f}px",
        f"Color Match:      {color_match_blend * 100:.0f}% {'(Applied)' if stats.get('color_match_applied') else '(Skipped)'}",
        f"Flow quality:     {flow_quality}",
        f"Changed region:   {stats['changed_pct']:.1f}%",
        f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
        f"Median Diff:      {stats['median_de']:.2f}",
    ])
    return report_lines


def build_debug_gallery(enable_debug, debug_images, cv, stack_images_fn):
    debug_gallery = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    if not enable_debug:
        return debug_gallery

    if not debug_images:
        warning = np.zeros((512, 512, 3), dtype=np.uint8)
        cv.putText(warning, "DEBUG ENABLED BUT NO DATA", (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(warning, "Check that enable_debug=True", (50, 300), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return torch.from_numpy(warning.astype(np.float32) / 255.0).unsqueeze(0)

    gallery = stack_images_fn([
        debug_images.get("pass1_sift_matches"),
        debug_images.get("pass1_alignment"),
        debug_images.get("pass1_delta_e"),
        debug_images.get("final_alignment"),
        debug_images.get("final_delta_e"),
        debug_images.get("final_flow"),
        debug_images.get("mask_overlay"),
        debug_images.get("composite_breakdown"),
    ])
    return torch.from_numpy(gallery.astype(np.float32) / 255.0).unsqueeze(0)

