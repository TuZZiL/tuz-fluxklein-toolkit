"""
Edit-mode presets for FLUX.2 Klein 9B LoRA loading.

Each preset defines per-layer strength multipliers that control how strongly
the LoRA affects different parts of the model.  Based on community research
(comfyUI-Realtime-Lora / flux_klein_debiaser) showing that:

  - Double blocks (0-7): img and txt streams are ISOLATED.
    img_attn/img_mlp handle reference+noisy together.
    txt_attn/txt_mlp handle text independently.
    → Cannot cause text-driven image corruption on their own.

  - Single blocks (0-23): JOINT cross-modal processing (img+txt concatenated).
    → This is where the text prompt overwrites the reference image.
    → Late single blocks (12-23) are the most aggressive.

Preset format matches the existing layer_strengths JSON:
  {"db": {"0": {"img": float, "txt": float}, ...}, "sb": {"0": float, ...}}

Values are multipliers relative to global_strength:
  1.0 = full LoRA effect, 0.3 = 30% effect, 1.15 = 115% (boosted)
"""

N_DOUBLE = 8
N_SINGLE = 24
USE_CASE_NAMES = ["Edit", "Generate"]

EDIT_PRESETS = {
    "None": None,

    "Preserve Face": {
        # Preserves face/identity during editing.
        # Double blocks: keep img stream, slightly reduce txt influence.
        # Single blocks: gradient from 1.0 (early) down to 0.30 (late).
        # Body proportions may still change — use Preserve Body for full protection.
        "db": {
            "0": {"img": 1.0, "txt": 0.90}, "1": {"img": 1.0, "txt": 0.90},
            "2": {"img": 1.0, "txt": 0.90}, "3": {"img": 1.0, "txt": 0.90},
            "4": {"img": 1.0, "txt": 0.85}, "5": {"img": 1.0, "txt": 0.85},
            "6": {"img": 1.0, "txt": 0.85}, "7": {"img": 1.0, "txt": 0.85},
        },
        "sb": {
            "0": 1.0,  "1": 1.0,  "2": 1.0,  "3": 1.0,
            "4": 1.0,  "5": 1.0,  "6": 0.95, "7": 0.95,
            "8": 0.90, "9": 0.85, "10": 0.80, "11": 0.75,
            "12": 0.65, "13": 0.60, "14": 0.55, "15": 0.50,
            "16": 0.45, "17": 0.40, "18": 0.38, "19": 0.35,
            "20": 0.33, "21": 0.32, "22": 0.30, "23": 0.30,
        },
    },

    "Preserve Body": {
        # Aggressive identity+body preservation during editing.
        # Protects face, body proportions (breast size, waist, figure).
        # Double blocks: keep img stream strong, reduce txt more.
        # Single blocks: dampened from sb4 onward (wider protection than Preserve Face).
        # Trade-off: LoRA editing effect will be weaker — use higher balance if needed.
        "db": {
            "0": {"img": 1.0, "txt": 0.85}, "1": {"img": 1.0, "txt": 0.85},
            "2": {"img": 1.0, "txt": 0.85}, "3": {"img": 1.0, "txt": 0.85},
            "4": {"img": 1.0, "txt": 0.80}, "5": {"img": 1.0, "txt": 0.80},
            "6": {"img": 1.0, "txt": 0.80}, "7": {"img": 1.0, "txt": 0.80},
        },
        "sb": {
            "0": 1.0,  "1": 1.0,  "2": 1.0,  "3": 0.95,
            "4": 0.75, "5": 0.72, "6": 0.70, "7": 0.68,
            "8": 0.65, "9": 0.62, "10": 0.60, "11": 0.55,
            "12": 0.50, "13": 0.47, "14": 0.44, "15": 0.40,
            "16": 0.38, "17": 0.35, "18": 0.33, "19": 0.32,
            "20": 0.30, "21": 0.30, "22": 0.30, "23": 0.30,
        },
    },

    "Style Only": {
        # Applies only stylistic changes, minimal structural impact.
        # Double blocks: reduce img stream (less structural change), keep txt.
        # Single blocks: early kept, late strongly reduced.
        "db": {
            "0": {"img": 0.40, "txt": 1.0}, "1": {"img": 0.40, "txt": 1.0},
            "2": {"img": 0.45, "txt": 1.0}, "3": {"img": 0.45, "txt": 1.0},
            "4": {"img": 0.50, "txt": 1.0}, "5": {"img": 0.50, "txt": 1.0},
            "6": {"img": 0.55, "txt": 1.0}, "7": {"img": 0.55, "txt": 1.0},
        },
        "sb": {
            "0": 1.0,  "1": 1.0,  "2": 1.0,  "3": 0.95,
            "4": 0.90, "5": 0.85, "6": 0.80, "7": 0.75,
            "8": 0.70, "9": 0.65, "10": 0.60, "11": 0.55,
            "12": 0.50, "13": 0.45, "14": 0.40, "15": 0.38,
            "16": 0.35, "17": 0.33, "18": 0.30, "19": 0.30,
            "20": 0.30, "21": 0.30, "22": 0.30, "23": 0.30,
        },
    },

    "Edit Subject": {
        # Edit clothing/objects while preserving identity.
        # Compromise between Preserve Face and full LoRA.
        # Double blocks: slightly boost txt for prompt compliance.
        # Single blocks: moderate protection on late blocks.
        "db": {
            "0": {"img": 1.0, "txt": 1.05}, "1": {"img": 1.0, "txt": 1.05},
            "2": {"img": 1.0, "txt": 1.05}, "3": {"img": 1.0, "txt": 1.05},
            "4": {"img": 0.95, "txt": 1.0}, "5": {"img": 0.95, "txt": 1.0},
            "6": {"img": 0.95, "txt": 1.0}, "7": {"img": 0.95, "txt": 1.0},
        },
        "sb": {
            "0": 1.0,  "1": 1.0,  "2": 1.0,  "3": 1.0,
            "4": 1.0,  "5": 1.0,  "6": 1.0,  "7": 0.95,
            "8": 0.90, "9": 0.85, "10": 0.80, "11": 0.75,
            "12": 0.65, "13": 0.60, "14": 0.55, "15": 0.50,
            "16": 0.50, "17": 0.50, "18": 0.50, "19": 0.50,
            "20": 0.50, "21": 0.50, "22": 0.50, "23": 0.50,
        },
    },

    "Boost Prompt": {
        # Strengthens prompt compliance (opposite of Preserve Face).
        # Double blocks: boost txt stream, slightly reduce img.
        # Single blocks: boost mid blocks where cross-modal mixing peaks.
        "db": {
            "0": {"img": 0.90, "txt": 1.15}, "1": {"img": 0.90, "txt": 1.15},
            "2": {"img": 0.90, "txt": 1.15}, "3": {"img": 0.90, "txt": 1.15},
            "4": {"img": 0.85, "txt": 1.10}, "5": {"img": 0.85, "txt": 1.10},
            "6": {"img": 0.85, "txt": 1.10}, "7": {"img": 0.85, "txt": 1.10},
        },
        "sb": {
            "0": 1.0,  "1": 1.0,  "2": 1.0,  "3": 1.0,
            "4": 1.05, "5": 1.05, "6": 1.05, "7": 1.05,
            "8": 1.10, "9": 1.10, "10": 1.10, "11": 1.10,
            "12": 1.10, "13": 1.10, "14": 1.10, "15": 1.10,
            "16": 1.05, "17": 1.05, "18": 1.05, "19": 1.05,
            "20": 1.0,  "21": 1.0,  "22": 1.0,  "23": 1.0,
        },
    },
}

PRESET_NAMES = list(EDIT_PRESETS.keys()) + ["Auto"]


def interpolate_preset(preset_cfg, balance):
    """
    Interpolate between a preset and neutral (all 1.0).

    balance = 0.0  →  full preset (maximum protection/effect)
    balance = 1.0  →  all weights = 1.0 (standard LoRA, no edit-mode effect)

    Formula:  final = preset_value + (1.0 - preset_value) * balance
    """
    if preset_cfg is None:
        return {}

    result = {"db": {}, "sb": {}}

    for idx, cfg in preset_cfg["db"].items():
        result["db"][idx] = {
            "img": cfg["img"] + (1.0 - cfg["img"]) * balance,
            "txt": cfg["txt"] + (1.0 - cfg["txt"]) * balance,
        }

    for idx, val in preset_cfg["sb"].items():
        result["sb"][idx] = val + (1.0 - val) * balance

    return result


def merge_preset_over(base_cfg, preset_cfg):
    """
    Multiply preset weights on top of an existing layer_cfg.
    This allows combining AutoStrength (ΔW-based) with semantic edit presets.

    For each layer: final = base_value * preset_value
    """
    if not preset_cfg:
        return base_cfg
    if not base_cfg:
        return preset_cfg

    merged = {"db": {}, "sb": {}}

    # Merge double blocks
    all_db_keys = set(base_cfg.get("db", {}).keys()) | set(preset_cfg.get("db", {}).keys())
    for idx in all_db_keys:
        base_db = base_cfg.get("db", {}).get(idx, {"img": 1.0, "txt": 1.0})
        preset_db = preset_cfg.get("db", {}).get(idx, {"img": 1.0, "txt": 1.0})
        if isinstance(base_db, dict):
            base_img = base_db.get("img", 1.0)
            base_txt = base_db.get("txt", 1.0)
        else:
            base_img = base_txt = float(base_db)
        merged["db"][idx] = {
            "img": base_img * preset_db.get("img", 1.0),
            "txt": base_txt * preset_db.get("txt", 1.0),
        }

    # Merge single blocks
    all_sb_keys = set(base_cfg.get("sb", {}).keys()) | set(preset_cfg.get("sb", {}).keys())
    for idx in all_sb_keys:
        base_val = float(base_cfg.get("sb", {}).get(idx, 1.0))
        preset_val = float(preset_cfg.get("sb", {}).get(idx, 1.0))
        merged["sb"][idx] = base_val * preset_val

    return merged


def auto_select_preset(analysis, use_case="Edit"):
    """
    Analyze ΔW norms from lora_meta.analyse_for_node() and pick the best
    preset + balance automatically.

    Decision logic depends on the intended workflow:
      Edit:
        - strong late single-block dominance           → Preserve Body
        - moderate late/mid single-block dominance     → Preserve Face
        - image-heavy double blocks with calm singles  → Style Only
        - full-coverage uniform LoRA                   → Preserve Face
        - very soft / sparse structural LoRA           → None
        - otherwise                                    → Preserve Face

      Generate:
        - strong late single-block dominance           → Preserve Face
        - image-heavy double blocks with calm singles  → Style Only
        - full-coverage / uniform LoRA                 → None
        - otherwise                                    → None

    Balance is computed from max/mean ratio:
      - Higher concentration = lower balance (more protection needed)

    Returns: (preset_name: str, balance: float)
    """
    def mean_or_zero(values):
        return (sum(values) / len(values)) if values else 0.0

    db_norms = []
    db_img = []
    db_txt = []
    sb_early, sb_mid, sb_late = [], [], []
    active_components = 0

    for i in range(N_DOUBLE):
        db = analysis.get("db", {}).get(i, {})
        if isinstance(db, dict):
            if db.get("img") is not None:
                db_norms.append(db["img"])
                db_img.append(db["img"])
                active_components += 1
            if db.get("txt") is not None:
                db_norms.append(db["txt"])
                db_txt.append(db["txt"])
                active_components += 1

    for i in range(N_SINGLE):
        v = analysis.get("sb", {}).get(i)
        if v is not None:
            active_components += 1
            if i < 8:
                sb_early.append(v)
            elif i < 16:
                sb_mid.append(v)
            else:
                sb_late.append(v)

    all_norms = db_norms + sb_early + sb_mid + sb_late
    if use_case not in USE_CASE_NAMES:
        use_case = "Edit"

    if not all_norms:
        return ("Preserve Face", 0.40)

    mean_all = sum(all_norms) / len(all_norms)
    if mean_all < 1e-8:
        return ("Preserve Face", 0.40)

    max_all = max(all_norms)
    late_mean = mean_or_zero(sb_late)
    mid_mean = mean_or_zero(sb_mid)
    db_mean = mean_or_zero(db_norms)
    img_mean = mean_or_zero(db_img)
    txt_mean = mean_or_zero(db_txt)

    late_ratio = late_mean / mean_all
    mid_ratio = mid_mean / mean_all
    db_ratio = db_mean / mean_all
    max_ratio = max_all / mean_all
    img_txt_ratio = (img_mean / txt_mean) if txt_mean > 1e-8 else 1.0
    coverage_ratio = active_components / float(N_DOUBLE * 2 + N_SINGLE)

    # Pick preset
    if use_case == "Generate":
        if img_txt_ratio > 1.22 and late_ratio < 1.02 and mid_ratio < 1.05:
            preset = "Style Only"
        elif late_ratio > 1.35 or (late_ratio > 1.22 and max_ratio > 1.40):
            preset = "Preserve Face"
        elif coverage_ratio > 0.85 or (db_ratio > 0.82 and late_ratio < 1.05):
            preset = "None"
        else:
            preset = "None"
    else:
        if late_ratio > 1.30 or (late_ratio > 1.20 and max_ratio > 1.35):
            preset = "Preserve Body"
        elif late_ratio > 1.08 or mid_ratio > 1.05:
            preset = "Preserve Face"
        elif img_txt_ratio > 1.18 and late_ratio < 1.00 and mid_ratio < 1.02:
            preset = "Style Only"
        elif coverage_ratio > 0.85:
            preset = "Preserve Face"
        elif db_ratio > 0.85 and late_ratio < 0.95 and max_ratio < 1.12:
            preset = "None"
        else:
            preset = "Preserve Face"

    # Pick balance: higher concentration → lower balance (more protection)
    # max_ratio 1.0–1.2 → balance ~0.55 (mild)
    # max_ratio 1.5–2.0 → balance ~0.25 (strong protection)
    balance = max(0.20, min(0.60, 0.70 - 0.25 * max_ratio))
    if use_case == "Generate" and preset == "None":
        balance = max(balance, 0.50)
    elif preset == "None":
        balance = max(balance, 0.55)
    elif preset == "Style Only":
        balance = max(balance, 0.35)
    balance = round(balance / 0.05) * 0.05  # snap to 0.05 grid

    return (preset, balance)


def resolve_preset_selection(edit_mode, balance, analysis=None, use_case="Edit"):
    """
    Resolve a user-facing edit_mode into the preset name + balance to apply.

    Manual modes ignore use_case; only Auto is use-case aware.
    """
    if edit_mode == "Auto":
        return auto_select_preset(analysis or {}, use_case=use_case)
    return (edit_mode, balance)
