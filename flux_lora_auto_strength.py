"""
flux_lora_auto_strength.py

Uses lora_meta.analyse_for_node() — the same forensic analysis from the
standalone script — to compute per-layer strengths. One source of truth.

FluxLoraAutoStrength  — outputs layer_strengths JSON + lora_name.
                        Wire to FluxLoraLoader. Bars auto-populate.
                        One knob: global_strength.

FluxLoraAutoLoader    — fully self-contained. model in → patched model out.
                        One knob: global_strength.
"""

import json
import math
import logging
import os
import sys
from pathlib import Path

import numpy as np
import folder_paths
import comfy.utils
import comfy.lora

# Import the forensic analysis directly from lora_meta.py
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from lora_meta import analyse_for_node

logger = logging.getLogger(__name__)

N_DOUBLE = 8
N_SINGLE = 24
_FLOOR   = 0.30
_CEILING = 1.50


# ─────────────────────────────────────────────────────────────────────────────
# STRENGTH COMPUTATION  (inverse-proportional to ΔW norm)
# ─────────────────────────────────────────────────────────────────────────────

def _all_norms(analysis):
    out = []
    for i in range(N_DOUBLE):
        db = analysis["db"].get(i, {})
        if db.get("img") is not None: out.append(db["img"])
        if db.get("txt") is not None: out.append(db["txt"])
    for i in range(N_SINGLE):
        v = analysis["sb"].get(i)
        if v is not None: out.append(v)
    return out


def compute_strengths(analysis, global_strength):
    all_norms = _all_norms(analysis)
    if not all_norms:
        return {
            "db": {str(i): {"img": global_strength, "txt": global_strength} for i in range(N_DOUBLE)},
            "sb": {str(i): global_strength for i in range(N_SINGLE)},
        }

    mean_norm = float(np.mean(all_norms))

    def clamp(v):
        return max(_FLOOR, min(_CEILING, v))

    def map_norm(norm):
        if norm is None or norm < 1e-8:
            return global_strength
        # Inverse-proportional: high ΔW → lower strength, low ΔW → higher strength
        # Targets the mean layer at exactly global_strength
        return clamp(global_strength * (mean_norm / norm))

    return {
        "db": {
            str(i): {
                "img": round(map_norm(analysis["db"][i].get("img")), 4),
                "txt": round(map_norm(analysis["db"][i].get("txt")), 4),
            }
            for i in range(N_DOUBLE)
        },
        "sb": {
            str(i): round(map_norm(analysis["sb"].get(i)), 4)
            for i in range(N_SINGLE)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_report(lora_name, analysis, strengths, global_strength):
    all_norms = _all_norms(analysis)
    median    = float(np.median(all_norms)) if all_norms else 1.0
    alpha_str = f"{analysis['alpha']:.4f}" if analysis['alpha'] is not None else "1.0 (not embedded)"

    lines = [
        "═══ FLUX LoRA Auto Strength ════════════════════════",
        f"  LoRA   : {lora_name}",
        f"  Rank   : {analysis['rank']}   Alpha: {alpha_str}",
        f"  Global : {global_strength}",
    ]
    if all_norms:
        lines.append(
            f"  ΔW     : mean={np.mean(all_norms):.3f}  "
            f"median={median:.3f}  max={np.max(all_norms):.3f}"
        )
    lines += ["", "  DOUBLE BLOCKS", "  " + "─" * 50]
    for i in range(N_DOUBLE):
        db_n = analysis["db"].get(i, {})
        db_s = strengths["db"].get(str(i), {})
        img_n = db_n.get("img") or 0.0
        txt_n = db_n.get("txt") or 0.0
        lines.append(
            f"  [{i}] img ΔW={img_n:.3f}→{db_s.get('img', 0):.4f}   "
            f"txt ΔW={txt_n:.3f}→{db_s.get('txt', 0):.4f}"
        )
    lines += ["", "  SINGLE BLOCKS", "  " + "─" * 50]
    for i in range(N_SINGLE):
        v  = analysis["sb"].get(i)
        s  = strengths["sb"].get(str(i), global_strength)
        vv = v if v is not None else 0.0
        hot = " ★" if (v and v > median * 1.5) else ""
        lines.append(f"  [{i:>2}] ΔW={vv:.3f}→{s:.4f}{hot}")
    lines.append("═════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED APPLY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def apply_layer_strengths(lora_sd, layer_cfg, global_strength):
    if not layer_cfg or abs(global_strength) < 1e-8:
        return lora_sd

    db_cfg = {str(k): v for k, v in layer_cfg.get("db", {}).items()}
    sb_cfg = {str(k): v for k, v in layer_cfg.get("sb", {}).items()}
    scaled = {}

    for key, tensor in lora_sd.items():
        if not (key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight")):
            scaled[key] = tensor
            continue
        parts  = key.split(".")
        target = None
        for i, p in enumerate(parts):
            if p == "double_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in db_cfg:
                    cfg    = db_cfg[idx]
                    is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                    side   = "txt" if is_txt else "img"
                    target = cfg.get(side, global_strength) if isinstance(cfg, dict) else float(cfg)
                break
            if p == "single_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in sb_cfg:
                    target = float(sb_cfg[idx])
                break
        scaled[key] = tensor * (target / global_strength) if target is not None else tensor

    return scaled


def build_key_map(model):
    key_map = {}
    for model_key in model.model.state_dict().keys():
        if not model_key.endswith(".weight"):
            continue
        base = model_key[: -len(".weight")]
        bare = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
        for pfx in ("diffusion_model.", "transformer.", ""):
            key_map[f"{pfx}{bare}"] = model_key
        key_map["lora_unet_" + bare.replace(".", "_")] = model_key
    return key_map


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — FluxLoraAutoStrength
# ─────────────────────────────────────────────────────────────────────────────

class FluxLoraAutoStrength:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name":       (folder_paths.get_filename_list("loras"),),
            "global_strength": ("FLOAT", {
                "default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01,
                "tooltip": "Master strength. All per-layer values are auto-computed from ΔW forensics.",
            }),
        }}

    RETURN_TYPES  = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES  = ("layer_strengths", "analysis_report", "global_strength", "lora_name")
    FUNCTION      = "run"
    CATEGORY      = "loaders/FLUX"
    TITLE         = "FLUX LoRA Auto Strength"

    def run(self, lora_name, global_strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        analysis  = analyse_for_node(lora_path)
        strengths = compute_strengths(analysis, global_strength)
        report    = build_report(lora_name, analysis, strengths, global_strength)
        logger.info(f"[AutoStrength] {lora_name} analysed via lora_meta. rank={analysis['rank']} alpha={analysis['alpha']}")
        return (json.dumps(strengths), report, global_strength, lora_name)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — FluxLoraAutoLoader
# ─────────────────────────────────────────────────────────────────────────────

class FluxLoraAutoLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":           ("MODEL",),
            "lora_name":       (folder_paths.get_filename_list("loras"),),
            "global_strength": ("FLOAT", {
                "default": 0.75, "min": -2.0, "max": 2.0, "step": 0.01,
                "tooltip": "Master strength. Everything else is computed automatically from the LoRA weights.",
            }),
        }}

    RETURN_TYPES  = ("MODEL", "STRING")
    RETURN_NAMES  = ("model", "analysis_report")
    FUNCTION      = "run"
    CATEGORY      = "loaders/FLUX"
    TITLE         = "FLUX LoRA Auto Loader"

    def run(self, model, lora_name, global_strength):
        if global_strength == 0:
            return (model, "Skipped — global_strength is 0.")

        lora_path  = folder_paths.get_full_path("loras", lora_name)

        # Use lora_meta forensic analysis — same as the standalone script
        analysis   = analyse_for_node(lora_path)
        strengths  = compute_strengths(analysis, global_strength)
        report     = build_report(lora_name, analysis, strengths, global_strength)

        # Load via comfy for the torch tensors, bake strengths in, apply
        sd         = comfy.utils.load_torch_file(lora_path, safe_load=True)
        sd_scaled  = apply_layer_strengths(sd, strengths, global_strength)
        key_map    = build_key_map(model)
        patch_dict = comfy.lora.load_lora(sd_scaled, key_map, log_missing=False)

        model_out  = model.clone()
        model_out.add_patches(patch_dict, strength_patch=1.0, strength_model=1.0)

        logger.info(f"[AutoLoader] {lora_name} — {len(patch_dict)} patches @ global={global_strength}")
        return (model_out, report)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FluxLoraAutoStrength": FluxLoraAutoStrength,
    "FluxLoraAutoLoader":   FluxLoraAutoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraAutoStrength": "FLUX LoRA Auto Strength",
    "FluxLoraAutoLoader":   "FLUX LoRA Auto Loader",
}
