"""
Pure advisory heuristics for FLUX LoRA preflight analysis.

This module stays free of ComfyUI/runtime imports so it can be unit-tested in
isolation and reused by the advisory nodes.
"""

from __future__ import annotations

import json

try:  # pragma: no cover - import style depends on package context
    from .edit_presets import auto_select_preset
    from .flux_constants import N_DOUBLE, N_SINGLE, TOTAL_COMPONENTS
except ImportError:  # pragma: no cover
    from edit_presets import auto_select_preset
    from flux_constants import N_DOUBLE, N_SINGLE, TOTAL_COMPONENTS



def _clamp(value, low, high):
    return max(low, min(high, value))


def _mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compat_summary(compat_report):
    compat_report = compat_report or {}
    total = int(compat_report.get("total_modules", 0) or 0)
    matched = int(compat_report.get("matched_modules", 0) or 0)
    incomplete = int(compat_report.get("incomplete_modules", 0) or 0)
    skipped = int(compat_report.get("skipped_modules", 0) or 0)
    if matched <= 0:
        status = "failed"
    elif matched < total or incomplete > 0:
        status = "partial"
    else:
        status = "ok"
    return {
        "status": status,
        "matched": matched,
        "total": total,
        "skipped": skipped,
        "incomplete": incomplete,
        "sample_skipped": list(compat_report.get("sample_skipped", ()) or ()),
        "sample_incomplete": list(compat_report.get("sample_incomplete", ()) or ()),
    }


def summarize_analysis(analysis):
    analysis = analysis or {}
    db = analysis.get("db", {}) or {}
    sb = analysis.get("sb", {}) or {}

    db_img = []
    db_txt = []
    sb_early = []
    sb_mid = []
    sb_late = []
    all_norms = []

    for i in range(N_DOUBLE):
        item = db.get(i, {}) or {}
        img = item.get("img")
        txt = item.get("txt")
        if img is not None:
            db_img.append(float(img))
            all_norms.append(float(img))
        if txt is not None:
            db_txt.append(float(txt))
            all_norms.append(float(txt))

    for i in range(N_SINGLE):
        value = sb.get(i)
        if value is None:
            continue
        value = float(value)
        all_norms.append(value)
        if i < 8:
            sb_early.append(value)
        elif i < 16:
            sb_mid.append(value)
        else:
            sb_late.append(value)

    mean_all = _mean(all_norms)
    max_all = max(all_norms) if all_norms else 0.0
    mean_db_img = _mean(db_img)
    mean_db_txt = _mean(db_txt)
    mean_early = _mean(sb_early)
    mean_mid = _mean(sb_mid)
    mean_late = _mean(sb_late)

    return {
        "mean_norm": mean_all,
        "max_norm": max_all,
        "db_img_mean": mean_db_img,
        "db_txt_mean": mean_db_txt,
        "early_mean": mean_early,
        "mid_mean": mean_mid,
        "late_mean": mean_late,
        "late_ratio": (mean_late / mean_all) if mean_all > 1e-8 else 0.0,
        "mid_ratio": (mean_mid / mean_all) if mean_all > 1e-8 else 0.0,
        "db_ratio": ((_mean(db_img + db_txt)) / mean_all) if mean_all > 1e-8 else 0.0,
        "img_txt_ratio": (mean_db_img / mean_db_txt) if mean_db_txt > 1e-8 else 1.0,
        "coverage_ratio": len(all_norms) / float(TOTAL_COMPONENTS),
        "active_component_count": len(all_norms),
        "rank": int(analysis.get("rank", 0) or 0),
        "alpha": analysis.get("alpha", None),
    }


def classify_profile(summary):
    tags = []
    if summary["coverage_ratio"] < 0.28 and summary["max_norm"] < 1.15:
        tags.append("sparse")
    if summary["coverage_ratio"] >= 0.75 and summary["max_norm"] < 1.18:
        tags.append("broad")
    if summary["late_ratio"] >= 1.35:
        tags.append("late-heavy")
    elif summary["late_ratio"] >= 1.15:
        tags.append("single-heavy")
    if summary["img_txt_ratio"] >= 1.2 and summary["late_ratio"] < 1.05:
        tags.append("style-heavy")
    if summary["max_norm"] <= 1.10:
        tags.append("uniform")
    return tags


def recommend_edit_mode_balance(analysis, use_case="Edit"):
    use_case = use_case if use_case in ("Edit", "Generate") else "Edit"
    preset, balance = auto_select_preset(analysis or {}, use_case=use_case)
    if preset == "None":
        balance = 1.0
    return preset, round(float(balance), 2)


def recommend_strength(summary, compat, edit_mode, use_case="Edit"):
    total = max(int(compat["total"]), 0)
    matched = int(compat["matched"])
    matched_ratio = (matched / total) if total > 0 else 0.0
    strength = 0.95

    if compat["status"] == "failed":
        strength = 0.50
    elif matched_ratio < 0.45:
        strength = 0.60
    elif matched_ratio < 0.75:
        strength = 0.80

    if summary["late_ratio"] >= 1.35:
        strength -= 0.20
    elif summary["late_ratio"] >= 1.15:
        strength -= 0.10

    if summary["mid_ratio"] >= 1.08:
        strength -= 0.05

    if summary["img_txt_ratio"] >= 1.2 and summary["late_ratio"] < 1.05:
        strength += 0.08

    if summary["coverage_ratio"] < 0.25 and summary["max_norm"] < 1.15 and matched_ratio > 0.60:
        strength += 0.10
    elif summary["coverage_ratio"] > 0.80 and summary["max_norm"] < 1.12:
        strength += 0.05

    if compat["incomplete"] > 0:
        strength -= 0.05

    if edit_mode == "Preserve Body":
        strength -= 0.05
    elif edit_mode == "Preserve Face":
        strength -= 0.03
    elif edit_mode == "Style Only":
        strength += 0.05
    elif edit_mode == "Boost Prompt":
        strength -= 0.03

    if use_case == "Generate" and edit_mode in ("None", "Style Only"):
        strength += 0.03

    return round(_clamp(strength, 0.35, 1.25), 2)


def _risk_level(summary, compat):
    score = 0
    if compat["status"] == "failed":
        score += 3
    elif compat["status"] == "partial":
        score += 1

    if compat["incomplete"] > 0:
        score += 1

    if summary["late_ratio"] >= 1.35:
        score += 2
    elif summary["late_ratio"] >= 1.15:
        score += 1

    if summary["max_norm"] >= 1.35 and summary["coverage_ratio"] > 0.65:
        score += 1

    if summary["coverage_ratio"] > 0.90 and summary["max_norm"] < 1.12 and compat["status"] == "ok":
        score -= 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _single_warnings(summary, compat, risk_level):
    warnings = []
    if compat["status"] == "failed":
        warnings.append("No model modules matched this LoRA cleanly.")
    elif compat["status"] == "partial":
        warnings.append(f"Compatibility is partial: {compat['matched']}/{compat['total']} modules matched.")
    if compat["incomplete"] > 0:
        warnings.append(f"{compat['incomplete']} incomplete A/B module pairs were found.")
    if summary["late_ratio"] >= 1.35:
        warnings.append("Late single blocks dominate, so structural drift risk is high.")
    elif summary["late_ratio"] >= 1.15:
        warnings.append("Late single blocks are stronger than average; keep the first strength conservative.")
    if summary["img_txt_ratio"] >= 1.2 and summary["late_ratio"] < 1.05:
        warnings.append("Image stream is stronger than text stream; this looks style-biased rather than structural.")
    if summary["coverage_ratio"] < 0.25 and summary["max_norm"] < 1.15:
        warnings.append("The signal is sparse and weak; you may need a slightly higher starter strength.")
    if risk_level == "high" and not warnings:
        warnings.append("High overall risk detected from the compatibility and layer profile.")
    return warnings


def build_single_advice(analysis, compat_report, use_case="Edit", source_name=None):
    summary = summarize_analysis(analysis)
    compat = _compat_summary(compat_report)
    tags = classify_profile(summary)
    edit_mode, balance = recommend_edit_mode_balance(analysis, use_case=use_case)
    strength = recommend_strength(summary, compat, edit_mode, use_case=use_case)
    risk_level = _risk_level(summary, compat)
    warnings = _single_warnings(summary, compat, risk_level)

    lines = []
    title = f"Preflight Advisor: {source_name}" if source_name else "Preflight Advisor"
    lines.append(title)
    lines.append(f"Use case: {use_case}")
    lines.append(f"Compatibility: {compat['matched']}/{compat['total']} ({compat['status']})")
    lines.append(f"Risk: {risk_level}")
    lines.append(
        "Profile: "
        + ", ".join(tags) if tags else "Profile: balanced"
    )
    if summary["rank"]:
        lines.append(f"Rank: {summary['rank']}")
    if summary["alpha"] is not None:
        lines.append(f"Alpha: {summary['alpha']}")
    lines.append("Recommendation:")
    lines.append(f"  edit_mode: {edit_mode}")
    lines.append(f"  balance: {balance:.2f}")
    lines.append(f"  strength: {strength:.2f}")
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    return {
        "report": "\n".join(lines),
        "compat_status": compat["status"],
        "matched_modules": compat["matched"],
        "total_modules": compat["total"],
        "skipped_modules": compat["skipped"],
        "incomplete_modules": compat["incomplete"],
        "recommended_edit_mode": edit_mode,
        "recommended_balance": balance,
        "recommended_strength": strength,
        "risk_level": risk_level,
        "warnings": warnings,
        "profile_tags": tags,
        "analysis_summary": summary,
        "compat_summary": compat,
    }


def _active_slot(slot):
    if not isinstance(slot, dict):
        return False
    enabled = slot.get("enabled", True)
    lora_name = slot.get("lora", "None")
    strength = slot.get("strength", 1.0)
    return bool(enabled) and lora_name != "None" and abs(float(strength)) > 1e-8


def _multi_overlap_scale(entries):
    active = [entry for entry in entries if entry.get("active")]
    if len(active) <= 1:
        return 1.0

    late_heavy = sum(1 for entry in active if "late-heavy" in entry["advice"]["profile_tags"])
    partial = sum(1 for entry in active if entry["advice"]["compat_status"] == "partial")
    failed = sum(1 for entry in active if entry["advice"]["compat_status"] == "failed")
    pressure = 0.05 * (len(active) - 1)
    pressure += 0.10 * max(0, late_heavy - 1)
    pressure += 0.08 * partial
    pressure += 0.12 * failed
    return round(_clamp(1.0 - pressure, 0.55, 1.0), 2)


def build_multi_advice(entries, use_case="Edit", source_name=None):
    entries = list(entries or [])
    active = [entry for entry in entries if entry.get("active")]
    scale = _multi_overlap_scale(entries)

    adjusted_entries = []
    warnings = []
    risk_level = "low"

    if len(active) > 1:
        warnings.append(f"{len(active)} active slots are stacked; strengths were scaled by {scale:.2f}.")

    late_heavy_count = 0
    partial_count = 0
    failed_count = 0

    for entry in entries:
        slot = dict(entry.get("slot", {}))
        advice = dict(entry.get("advice", {}))
        if entry.get("active") and advice:
            if "late-heavy" in advice.get("profile_tags", []):
                late_heavy_count += 1
            if advice.get("compat_status") == "partial":
                partial_count += 1
            if advice.get("compat_status") == "failed":
                failed_count += 1
            advice["recommended_strength"] = round(
                _clamp(float(advice["recommended_strength"]) * scale, 0.35, 1.25), 2
            )
            slot["strength"] = advice["recommended_strength"]
            slot["edit_mode"] = advice["recommended_edit_mode"]
            slot["balance"] = advice["recommended_balance"]
        adjusted_entries.append({
            "index": entry.get("index", 0),
            "slot": slot,
            "advice": advice,
            "active": bool(entry.get("active")),
        })

    if failed_count > 0:
        risk_level = "high"
        warnings.append(f"{failed_count} slot(s) have failed compatibility.")
    elif late_heavy_count >= 2:
        risk_level = "high"
        warnings.append("Multiple late-heavy slots target the same structural layers.")
    elif late_heavy_count == 1 or partial_count > 0:
        risk_level = "medium"
        if late_heavy_count == 1:
            warnings.append("One late-heavy slot is present; watch for identity drift.")
        if partial_count > 0:
            warnings.append(f"{partial_count} slot(s) have partial compatibility.")

    recommended_slots = [entry["slot"] for entry in adjusted_entries]

    lines = []
    title = f"Preflight Advisor: {source_name}" if source_name else "Preflight Advisor"
    lines.append(title)
    lines.append(f"Use case: {use_case}")
    lines.append(f"Active slots: {len(active)}")
    lines.append(f"Overlap scale: {scale:.2f}")
    lines.append(f"Risk: {risk_level}")
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    lines.append("Per-slot recommendations:")
    for entry in adjusted_entries:
        slot = entry["slot"]
        advice = entry["advice"]
        label = slot.get("lora", "None")
        if not entry["active"]:
            lines.append(f"  - Slot {entry['index'] + 1}: inactive / unchanged")
            continue
        lines.append(
            f"  - Slot {entry['index'] + 1}: {label} -> "
            f"{advice.get('recommended_edit_mode', 'None')} / "
            f"balance {advice.get('recommended_balance', 1.0):.2f} / "
            f"strength {advice.get('recommended_strength', slot.get('strength', 1.0)):.2f}"
        )

    return {
        "report": "\n".join(lines),
        "recommended_slots": recommended_slots,
        "recommended_slot_data_json": json.dumps(recommended_slots, sort_keys=True),
        "active_slot_count": len(active),
        "risk_level": risk_level,
        "overlap_scale": scale,
        "warnings": warnings,
        "slot_entries": adjusted_entries,
    }
