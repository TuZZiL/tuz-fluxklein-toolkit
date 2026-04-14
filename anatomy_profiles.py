"""
Intent-based anatomy shielding profiles for FLUX LoRA editing.
"""

from __future__ import annotations

import json

try:  # pragma: no cover - package vs direct import
    from .flux_constants import N_DOUBLE, N_SINGLE
except ImportError:  # pragma: no cover
    from flux_constants import N_DOUBLE, N_SINGLE


ANATOMY_PROFILE_NAMES = [
    "None",
    "Balanced Identity",
    "Undress Safe",
    "Undress Body Lock",
    "Cloth Swap Flexible",
    "Robot Frame Lock",
    "Armor Hard Surface",
    "Anime Stylized Lock",
    "Texture Only",
    "Prompt Freedom",
    "Custom",
]


ANATOMY_PROFILES = {
    "None": None,
    "Balanced Identity": {
        "db_img": 0.80,
        "db_txt": 0.85,
        "sb_bands": [0.70, 0.72, 0.78, 0.86, 0.92, 0.95],
        "strict_zero": {"db": [], "sb": []},
    },
    "Undress Safe": {
        "db_img": 0.55,
        "db_txt": 0.62,
        "sb_bands": [0.35, 0.40, 0.50, 0.65, 0.78, 0.86],
        "strict_zero": {"db": [], "sb": []},
    },
    "Undress Body Lock": {
        "db_img": 0.40,
        "db_txt": 0.48,
        "sb_bands": [0.20, 0.26, 0.35, 0.52, 0.68, 0.78],
        "strict_zero": {"db": [0, 1, 2, 3], "sb": [0, 1, 2, 3]},
    },
    "Cloth Swap Flexible": {
        "db_img": 0.72,
        "db_txt": 0.80,
        "sb_bands": [0.62, 0.68, 0.76, 0.84, 0.90, 0.93],
        "strict_zero": {"db": [], "sb": []},
    },
    "Robot Frame Lock": {
        "db_img": 0.38,
        "db_txt": 0.45,
        "sb_bands": [0.18, 0.24, 0.32, 0.48, 0.64, 0.75],
        "strict_zero": {"db": [0, 1, 2, 3, 4, 5], "sb": [0, 1, 2, 3, 4, 5]},
    },
    "Armor Hard Surface": {
        "db_img": 0.46,
        "db_txt": 0.54,
        "sb_bands": [0.30, 0.36, 0.45, 0.58, 0.72, 0.82],
        "strict_zero": {"db": [0, 1, 2, 3], "sb": []},
    },
    "Anime Stylized Lock": {
        "db_img": 0.50,
        "db_txt": 0.58,
        "sb_bands": [0.28, 0.34, 0.44, 0.58, 0.72, 0.84],
        "strict_zero": {"db": [], "sb": []},
    },
    "Texture Only": {
        "db_img": 0.35,
        "db_txt": 0.90,
        "sb_bands": [0.25, 0.32, 0.44, 0.62, 0.80, 0.92],
        "strict_zero": {"db": [], "sb": []},
    },
    "Prompt Freedom": {
        "db_img": 0.92,
        "db_txt": 0.95,
        "sb_bands": [0.88, 0.90, 0.93, 0.96, 0.98, 1.00],
        "strict_zero": {"db": [], "sb": []},
    },
}


def _neutral_profile():
    return {
        "db": {str(i): {"img": 1.0, "txt": 1.0} for i in range(N_DOUBLE)},
        "sb": {str(i): 1.0 for i in range(N_SINGLE)},
        "strict_zero": {"db": [], "sb": []},
    }


def _coerce_strict_zero(raw):
    if not isinstance(raw, dict):
        return {"db": [], "sb": []}
    out = {"db": [], "sb": []}
    for key in ("db", "sb"):
        values = raw.get(key, [])
        if isinstance(values, (list, tuple)):
            seen = []
            for value in values:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if idx >= 0 and idx not in seen:
                    seen.append(idx)
            out[key] = seen
    return out


def expand_profile(profile):
    if not profile:
        return _neutral_profile()

    sb_bands = list(profile.get("sb_bands", []))
    if len(sb_bands) != 6:
        raise ValueError("anatomy profile requires exactly 6 sb_bands values")

    expanded = {
        "db": {
            str(i): {
                "img": float(profile.get("db_img", 1.0)),
                "txt": float(profile.get("db_txt", 1.0)),
            }
            for i in range(N_DOUBLE)
        },
        "sb": {},
        "strict_zero": _coerce_strict_zero(profile.get("strict_zero")),
    }

    for i in range(N_SINGLE):
        band = min(i // 4, len(sb_bands) - 1)
        expanded["sb"][str(i)] = float(sb_bands[band])
    return expanded


def interpolate_profile(expanded_profile, strength):
    strength = max(0.0, min(1.0, float(strength)))
    neutral = _neutral_profile()
    result = {"db": {}, "sb": {}, "strict_zero": expanded_profile.get("strict_zero", {"db": [], "sb": []})}

    for idx, cfg in expanded_profile.get("db", {}).items():
        neutral_cfg = neutral["db"][str(idx)]
        result["db"][str(idx)] = {
            "img": neutral_cfg["img"] - (neutral_cfg["img"] - float(cfg.get("img", 1.0))) * strength,
            "txt": neutral_cfg["txt"] - (neutral_cfg["txt"] - float(cfg.get("txt", 1.0))) * strength,
        }

    for idx, value in expanded_profile.get("sb", {}).items():
        result["sb"][str(idx)] = 1.0 - (1.0 - float(value)) * strength
    return result


def parse_custom_profile(raw_value):
    if not raw_value:
        return None
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("custom anatomy profile must be a JSON object")
    return parsed


def resolve_profile(profile_name, strength=0.65, custom_json=""):
    if profile_name in (None, "", "None"):
        return None
    if profile_name == "Custom":
        profile = parse_custom_profile(custom_json)
    else:
        profile = ANATOMY_PROFILES.get(profile_name)
    if not profile:
        return None
    return interpolate_profile(expand_profile(profile), strength)
