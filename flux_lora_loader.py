"""
FLUX LoRA Loader — Consolidated node pack (3 nodes).

Nodes:
  FluxLoraLoader    — Single LoRA with interactive graph widget + auto-strength toggle
  FluxLoraMulti     — Dynamic multi-slot loader (rgthree-style "+ Add LoRA")
  FluxLoraScheduled — Per-step temporal scheduling, absorbs SetCondHooks

Architecture-aware LoRA loading for FLUX DiT (double_blocks / single_blocks).

Problem it solves:
  Training tools using HuggingFace diffusers FLUX naming export LoRAs with
  SEPARATE to_q / to_k / to_v projections. ComfyUI's native FLUX model stores
  these as a single fused QKV matrix, so the standard loader silently drops them.

What it does:

  Double blocks  (8 layers):
    attn.to_q + to_k + to_v           → img_attn.qkv   (block-diag fused)
    attn.add_q_proj + add_k + add_v   → txt_attn.qkv   (block-diag fused)
    attn.to_out.0                     → img_attn.proj
    attn.to_add_out                   → txt_attn.proj
    ff.net.0.proj                     → img_mlp.0
    ff.net.2                          → img_mlp.2
    ff_context.net.0.proj             → txt_mlp.0
    ff_context.net.2                  → txt_mlp.2

  Single blocks  (24 layers):
    attn.to_q + to_k + to_v + proj_mlp → linear1   (block-diag fused)
    proj_out                           → linear2

  Fusion math (block-diagonal LoRA):
    For weight W_fused = [W_q; W_k; W_v] and separate LoRAs B_i @ A_i:
      ΔW_fused = [B_q@A_q ; B_k@A_k ; B_v@A_v]

    Represented as a rank-(3r) LoRA:
      A_fused = cat([A_q, A_k, A_v], dim=0)             [3r × in]
      B_fused = block_diag(B_q, B_k, B_v)               [3·out × 3r]

    alpha/rank scaling is pre-baked into B_fused so no extra alpha needed.

  Per-layer strength is written by the JS graph widget into `layer_strengths`.
"""

import json
import re
import torch
import numpy as np
import comfy.utils
import comfy.lora
import folder_paths
import logging

from .edit_presets import EDIT_PRESETS, PRESET_NAMES, interpolate_preset, auto_select_preset
from .schedules import SCHEDULE_NAMES, build_keyframes

logger = logging.getLogger(__name__)

N_DOUBLE = 8
N_SINGLE = 24
_AUTO_FLOOR   = 0.30
_AUTO_CEILING = 1.50


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS  (module-level, used by all 3 nodes)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Format detection ──────────────────────────────────────────────────────────

def _is_diffusers_format(lora_sd):
    markers = (
        ".to_q.",               # separate Q proj
        ".to_k.",               # separate K proj
        ".to_v.",               # separate V proj
        ".add_q_proj.",         # FLUX txt Q
        ".add_k_proj.",         # FLUX txt K
        ".add_v_proj.",         # FLUX txt V
        "transformer_blocks.",  # diffusers double-block prefix
        "single_transformer_blocks.",  # diffusers single-block prefix
        ".ff.net.",             # diffusers FFN naming
        ".ff_context.",         # FLUX txt FFN
        ".to_add_out.",         # FLUX txt out proj
        ".proj_mlp.",           # FLUX single block mlp gate
        ".to_qkv_mlp_proj.",    # Musubi Tuner fused single block
        ".ff.linear_in.",       # Musubi Tuner FFN naming
        ".ff.linear_out.",      # Musubi Tuner FFN naming
    )
    return any(m in k for k in lora_sd for m in markers)


# ── Key normalization ─────────────────────────────────────────────────────────

def _normalize_keys(lora_sd):
    """Strip prefix variations and remap diffusers layer names to native."""
    out = {}
    for key, val in lora_sd.items():
        k = key
        for pfx in ("transformer.", "diffusion_model.", "unet."):
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        k = re.sub(r'^transformer_blocks\.',        'double_blocks.', k)
        k = re.sub(r'^single_transformer_blocks\.', 'single_blocks.', k)
        k = k.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
        k = k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        k = k.replace(".ff.linear_in.", ".ff.net.0.proj.")
        k = k.replace(".ff.linear_out.", ".ff.net.2.")
        k = k.replace(".ff_context.linear_in.", ".ff_context.net.0.proj.")
        k = k.replace(".ff_context.linear_out.", ".ff_context.net.2.")
        out[k] = val
    return out


def _alpha_scale(norm, base):
    """Return alpha/rank for one LoRA component (defaults to 1.0 if absent)."""
    down_key  = f"{base}.lora_A.weight"
    alpha_key = f"{base}.alpha"
    if alpha_key in norm and down_key in norm:
        rank = norm[down_key].shape[0]
        if rank > 0:
            return float(norm[alpha_key]) / rank
    return 1.0


# ── QKV / linear1 fusion ─────────────────────────────────────────────────────

def _fuse_qkv(norm, out, done, q, k, v, dst):
    """
    Fuse three separate LoRAs (Q, K, V) into one block-diagonal LoRA.

    ΔW_qkv = [B_q@A_q ; B_k@A_k ; B_v@A_v]

    Represented as:
      A_fused = [A_q ; A_k ; A_v]              shape [3r, in]
      B_fused = block_diag(B_q, B_k, B_v)      shape [3·out, 3r]
    """
    keys_A = [f"{b}.lora_A.weight" for b in (q, k, v)]
    keys_B = [f"{b}.lora_B.weight" for b in (q, k, v)]

    if not all(kk in norm for kk in keys_A + keys_B):
        return

    A_q, A_k, A_v = [norm[kk] for kk in keys_A]
    B_q, B_k, B_v = (norm[kk] * _alpha_scale(norm, b)
                      for kk, b in zip(keys_B, (q, k, v)))

    r_q, r_k, r_v   = A_q.shape[0], A_k.shape[0], A_v.shape[0]
    o_q, o_k, o_v   = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    r_total = r_q + r_k + r_v
    o_total = o_q + o_k + o_v

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    B_fused = torch.zeros(o_total, r_total, dtype=B_q.dtype, device=B_q.device)
    B_fused[0         : o_q,          0         : r_q         ] = B_q
    B_fused[o_q       : o_q + o_k,   r_q       : r_q + r_k   ] = B_k
    B_fused[o_q + o_k : o_total,     r_q + r_k : r_total     ] = B_v

    out[f"{dst}.lora_A.weight"] = A_fused
    out[f"{dst}.lora_B.weight"] = B_fused
    out[f"{dst}.alpha"] = torch.tensor(float(r_total))

    for kk in keys_A + keys_B:
        done.add(kk)
    for b in (q, k, v):
        done.add(f"{b}.alpha")


def _fuse_linear1(norm, out, done, sb_base):
    """
    Fuse single-block components into linear1 (same block-diag logic as QKV).

    linear1 shape: [36864, 4096]  =  [q+k+v=12288  +  mlp_gate_up=24576,  in=4096]
    """
    components = [
        (f"{sb_base}.attn.to_q",  "q"),
        (f"{sb_base}.attn.to_k",  "k"),
        (f"{sb_base}.attn.to_v",  "v"),
        (f"{sb_base}.proj_mlp",   "mlp"),
    ]
    dst = f"diffusion_model.{sb_base}.linear1"

    present = [
        (base, label)
        for base, label in components
        if f"{base}.lora_A.weight" in norm and f"{base}.lora_B.weight" in norm
    ]
    if not present:
        return

    A_list, B_scaled = [], []
    for base, _ in present:
        A = norm[f"{base}.lora_A.weight"]
        B = norm[f"{base}.lora_B.weight"] * _alpha_scale(norm, base)
        A_list.append(A)
        B_scaled.append(B)
        done.update([f"{base}.lora_A.weight", f"{base}.lora_B.weight", f"{base}.alpha"])

    ranks  = [A.shape[0] for A in A_list]
    outs   = [B.shape[0] for B in B_scaled]
    r_total, o_total = sum(ranks), sum(outs)

    A_fused = torch.cat(A_list, dim=0)
    B_fused = torch.zeros(o_total, r_total,
                           dtype=B_scaled[0].dtype, device=B_scaled[0].device)
    r_off, o_off = 0, 0
    for A, B in zip(A_list, B_scaled):
        r, o = A.shape[0], B.shape[0]
        B_fused[o_off : o_off + o, r_off : r_off + r] = B
        r_off += r
        o_off += o

    out[f"{dst}.lora_A.weight"] = A_fused
    out[f"{dst}.lora_B.weight"] = B_fused
    out[f"{dst}.alpha"] = torch.tensor(float(r_total))


def _remap(norm, out, done, src_base, dst_base):
    """Copy a LoRA pair (lora_A / lora_B / alpha) from src to dst key name."""
    key_A = f"{src_base}.lora_A.weight"
    key_B = f"{src_base}.lora_B.weight"
    if key_A not in norm or key_B not in norm:
        return
    out[f"{dst_base}.lora_A.weight"] = norm[key_A]
    out[f"{dst_base}.lora_B.weight"] = norm[key_B]
    done.update([key_A, key_B])
    alpha_key = f"{src_base}.alpha"
    if alpha_key in norm:
        out[f"{dst_base}.alpha"] = norm[alpha_key]
        done.add(alpha_key)


# ── Full conversion pipeline ─────────────────────────────────────────────────

def _convert_to_native(lora_sd):
    """Convert diffusers/Musubi LoRA keys to ComfyUI-native format."""
    norm = _normalize_keys(lora_sd)
    out = {}
    done = set()

    for i in range(N_DOUBLE):
        db = f"double_blocks.{i}"

        _fuse_qkv(
            norm, out, done,
            q=f"{db}.attn.to_q",
            k=f"{db}.attn.to_k",
            v=f"{db}.attn.to_v",
            dst=f"diffusion_model.{db}.img_attn.qkv",
        )

        _fuse_qkv(
            norm, out, done,
            q=f"{db}.attn.add_q_proj",
            k=f"{db}.attn.add_k_proj",
            v=f"{db}.attn.add_v_proj",
            dst=f"diffusion_model.{db}.txt_attn.qkv",
        )

        for src, dst in [
            (f"{db}.attn.to_out.0",        f"diffusion_model.{db}.img_attn.proj"),
            (f"{db}.attn.to_add_out",       f"diffusion_model.{db}.txt_attn.proj"),
            (f"{db}.ff.net.0.proj",         f"diffusion_model.{db}.img_mlp.0"),
            (f"{db}.ff.net.2",              f"diffusion_model.{db}.img_mlp.2"),
            (f"{db}.ff_context.net.0.proj", f"diffusion_model.{db}.txt_mlp.0"),
            (f"{db}.ff_context.net.2",      f"diffusion_model.{db}.txt_mlp.2"),
        ]:
            _remap(norm, out, done, src, dst)

    for i in range(N_SINGLE):
        sb = f"single_blocks.{i}"
        _remap(norm, out, done, f"{sb}.attn.to_qkv_mlp_proj", f"diffusion_model.{sb}.linear1")
        _remap(norm, out, done, f"{sb}.attn.to_out", f"diffusion_model.{sb}.linear2")
        _fuse_linear1(norm, out, done, sb)
        _remap(norm, out, done, f"{sb}.proj_out", f"diffusion_model.{sb}.linear2")

    for key, val in norm.items():
        if key not in done:
            out[key] = val

    n_converted = sum(1 for k in done if k in norm)
    logger.info(f"[FLUX LoRA] Converted {n_converted} diffusers keys → {len(out)} native keys")
    return out


# ── Per-layer strength scaling ────────────────────────────────────────────────

def _apply_layer_strengths(lora_sd, layer_cfg, global_strength):
    """
    Scale lora_B tensors per-layer before patching.

    layer_cfg format (from the JS graph widget):
      {
        "db": { "0": {"img": 1.2, "txt": 0.8}, "1": {...}, ... },
        "sb": { "0": 0.9, "1": 1.1, ... }
      }
    """
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


# ── Edit-mode multipliers ─────────────────────────────────────────────────────

def _apply_edit_multipliers(lora_sd, preset_cfg):
    """
    Scale lora_B tensors by edit-mode preset multipliers.
    Applies multipliers directly: tensor * multiplier.
    """
    db_cfg = {str(k): v for k, v in preset_cfg.get("db", {}).items()}
    sb_cfg = {str(k): v for k, v in preset_cfg.get("sb", {}).items()}
    scaled = {}

    for key, tensor in lora_sd.items():
        if not key.endswith(".lora_B.weight"):
            scaled[key] = tensor
            continue
        parts = key.split(".")
        multiplier = None
        for i, p in enumerate(parts):
            if p == "double_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in db_cfg:
                    cfg = db_cfg[idx]
                    is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                    side = "txt" if is_txt else "img"
                    multiplier = cfg.get(side, 1.0) if isinstance(cfg, dict) else float(cfg)
                break
            if p == "single_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in sb_cfg:
                    multiplier = float(sb_cfg[idx])
                break
        if multiplier is not None and abs(multiplier - 1.0) > 1e-6:
            scaled[key] = tensor * multiplier
        else:
            scaled[key] = tensor

    return scaled


# ── Key map ───────────────────────────────────────────────────────────────────

def _build_key_map(model):
    """Build lora_key_base → model_state_dict_key mapping."""
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


# ── Edit-mode resolution ─────────────────────────────────────────────────────

def _resolve_edit_mode(edit_mode, balance, lora_path, node_label="FLUX LoRA"):
    """Resolve edit_mode (including Auto) into a preset config or None."""
    if edit_mode == "None":
        return None
    if edit_mode == "Auto":
        from .lora_meta import analyse_for_node
        analysis = analyse_for_node(lora_path)
        auto_preset, auto_balance = auto_select_preset(analysis)
        if auto_preset == "None":
            logger.info(f"[{node_label}] Auto → None (LoRA is safe, no preset needed)")
            return None
        preset_raw = EDIT_PRESETS.get(auto_preset)
        if preset_raw is not None:
            cfg = interpolate_preset(preset_raw, auto_balance)
            logger.info(f"[{node_label}] Auto → {auto_preset} (balance={auto_balance:.2f})")
            return cfg
        return None
    preset_raw = EDIT_PRESETS.get(edit_mode)
    if preset_raw is not None:
        cfg = interpolate_preset(preset_raw, balance)
        logger.info(f"[{node_label}] Edit mode '{edit_mode}' applied (balance={balance:.2f})")
        return cfg
    return None


# ── Auto-strength computation (ported from flux_lora_auto_strength.py) ────────

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


def _compute_strengths(analysis, global_strength):
    """Inverse-proportional auto-strength: high ΔW → lower strength."""
    all_norms = _all_norms(analysis)
    if not all_norms:
        return {
            "db": {str(i): {"img": global_strength, "txt": global_strength} for i in range(N_DOUBLE)},
            "sb": {str(i): global_strength for i in range(N_SINGLE)},
        }

    mean_norm = float(np.mean(all_norms))

    def clamp(v):
        return max(_AUTO_FLOOR, min(_AUTO_CEILING, v))

    def map_norm(norm):
        if norm is None or norm < 1e-8:
            return global_strength
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


# ── Unified load-and-patch helper ─────────────────────────────────────────────

def _load_and_patch(model, lora_name, strength, auto_convert, edit_mode, balance,
                    layer_cfg=None, auto_strength=False, node_label="FLUX LoRA"):
    """
    Shared pipeline: load → convert → apply edits → apply layer strengths → patch.
    Returns patched model clone.
    """
    if strength == 0:
        return model

    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    logger.info(f"[{node_label}] Loading: {lora_name}  ({len(lora_sd)} keys)")

    # Auto-strength: compute layer strengths from ΔW analysis
    if auto_strength and not layer_cfg:
        from .lora_meta import analyse_for_node
        analysis = analyse_for_node(lora_path)
        layer_cfg = _compute_strengths(analysis, abs(strength))
        logger.info(f"[{node_label}] Auto-strength computed for {lora_name}")

    # Resolve edit-mode preset
    edit_preset_cfg = _resolve_edit_mode(edit_mode, balance, lora_path, node_label)

    # Convert format if needed
    if auto_convert and _is_diffusers_format(lora_sd):
        logger.info(f"[{node_label}] Detected diffusers format — converting")
        lora_sd = _convert_to_native(lora_sd)

    # Apply per-layer strengths (from graph widget or auto-strength)
    if layer_cfg:
        lora_sd = _apply_layer_strengths(lora_sd, layer_cfg, strength)

    # Apply edit-mode multipliers
    if edit_preset_cfg:
        lora_sd = _apply_edit_multipliers(lora_sd, edit_preset_cfg)

    # Build patches and apply
    key_map = _build_key_map(model)
    patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
    logger.info(f"[{node_label}] Applied {len(patch_dict)} patches")

    model_out = model.clone()
    effective_strength = 1.0 if layer_cfg else strength
    model_out.add_patches(patch_dict, strength_patch=effective_strength, strength_model=1.0)
    return model_out


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — FluxLoraLoader  (single LoRA, graph widget, auto-strength toggle)
# ═══════════════════════════════════════════════════════════════════════════════

class FluxLoraLoader:
    """
    Loads a single FLUX LoRA with interactive per-layer strength graph widget.
    Supports auto-strength toggle that computes optimal per-layer strengths
    from ΔW forensic analysis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                }),
            },
            "optional": {
                "auto_strength": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Auto (ΔW analysis)",
                    "label_off": "Manual (graph bars)",
                    "tooltip": "When ON: auto-compute per-layer strengths from LoRA weight analysis. Bars auto-populate.",
                }),
                "edit_mode": (PRESET_NAMES, {
                    "default": "None",
                    "tooltip": "Semantic edit preset for Klein 9B. Controls which layers are dampened to preserve identity, style, etc.",
                }),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.0 = full preset effect (max protection), 1.0 = standard LoRA (no protection).",
                }),
                # Written by the JS graph widget — never shown as a text box
                "layer_strengths": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/FLUX"
    TITLE = "FLUX LoRA Loader"

    def load_lora(self, model, lora_name, strength,
                  auto_convert=True, auto_strength=False, layer_strengths="{}",
                  edit_mode="None", balance=0.5):
        if strength == 0:
            return (model,)

        # Parse per-layer strengths from graph widget
        layer_cfg = {}
        try:
            raw = json.loads(layer_strengths)
            if isinstance(raw, dict) and ("db" in raw or "sb" in raw):
                layer_cfg = raw
        except Exception:
            pass

        model_out = _load_and_patch(
            model, lora_name, strength, auto_convert, edit_mode, balance,
            layer_cfg=layer_cfg, auto_strength=auto_strength,
            node_label="FLUX LoRA Loader",
        )
        return (model_out,)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — FluxLoraMulti  (dynamic multi-slot, rgthree-style)
# ═══════════════════════════════════════════════════════════════════════════════

class FluxLoraMulti:
    """
    Dynamic multi-LoRA loader with per-slot control.
    Slots are managed by JS widget (+ Add LoRA / ✕ Remove).
    Each slot has: enabled, lora, strength, edit_mode, balance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                }),
                # Hidden — populated by JS widget, JSON array of slot configs
                "slot_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "loaders/FLUX"
    TITLE = "FLUX LoRA Multi"

    def load_loras(self, model, auto_convert=True, slot_data="[]"):
        try:
            slots = json.loads(slot_data)
        except Exception:
            logger.warning("[FLUX LoRA Multi] Invalid slot_data JSON, skipping")
            return (model,)

        if not isinstance(slots, list):
            return (model,)

        current = model
        for i, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue

            enabled   = slot.get("enabled", True)
            lora_name = slot.get("lora", "None")
            strength  = slot.get("strength", 1.0)
            edit_mode = slot.get("edit_mode", "None")
            balance   = slot.get("balance", 0.5)

            if not enabled or lora_name == "None" or strength == 0:
                continue

            current = _load_and_patch(
                current, lora_name, strength, auto_convert, edit_mode, balance,
                node_label=f"FLUX LoRA Multi slot {i+1}",
            )

        return (current,)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — FluxLoraScheduled  (temporal scheduling, absorbs SetCondHooks)
# ═══════════════════════════════════════════════════════════════════════════════

class FluxLoraScheduled:
    """
    Loads a FLUX LoRA with per-step strength scheduling via ComfyUI's Hook system.

    Instead of constant LoRA strength across all sampling steps, this node
    applies a strength curve (e.g., Fade Out: strong at start, weak at end).
    Combined with edit-mode presets, this gives two dimensions of control:
      - edit_mode: WHICH layers are affected (spatial)
      - schedule:  WHEN the LoRA is active (temporal)

    Absorbs FluxSetCondHooks: takes conditioning as input, returns modified
    conditioning with hooks attached directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Base LoRA strength. The schedule multiplies this value.",
                }),
                "schedule": (SCHEDULE_NAMES, {
                    "default": "Fade Out",
                    "tooltip": "Strength curve over sampling steps.",
                }),
            },
            "optional": {
                "edit_mode": (PRESET_NAMES, {
                    "default": "Auto",
                    "tooltip": "Semantic edit preset (per-layer control). Auto analyzes the LoRA automatically.",
                }),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.0 = full preset effect, 1.0 = standard LoRA.",
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                }),
                "keyframes": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 10,
                    "tooltip": "Number of keyframes for the schedule. More = smoother.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/FLUX"
    TITLE = "FLUX LoRA Scheduled"

    def load_lora(self, model, conditioning, lora_name, strength, schedule="Fade Out",
                  edit_mode="Auto", balance=0.5, auto_convert=True, keyframes=5):
        import comfy.hooks

        if strength == 0:
            return (model, conditioning)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        logger.info(f"[FLUX LoRA Scheduled] Loading: {lora_name}  ({len(lora_sd)} keys)")

        if auto_convert and _is_diffusers_format(lora_sd):
            logger.info("[FLUX LoRA Scheduled] Detected diffusers format — converting")
            lora_sd = _convert_to_native(lora_sd)

        # Apply edit-mode multipliers (per-layer control)
        edit_preset_cfg = _resolve_edit_mode(edit_mode, balance, lora_path, "FLUX LoRA Scheduled")
        if edit_preset_cfg:
            lora_sd = _apply_edit_multipliers(lora_sd, edit_preset_cfg)

        # Build patches
        key_map = _build_key_map(model)
        patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
        logger.info(f"[FLUX LoRA Scheduled] {len(patch_dict)} patches, schedule='{schedule}', keyframes={keyframes}")

        # Create hook with scheduling
        hook_group = comfy.hooks.HookGroup()
        hook = comfy.hooks.WeightHook(strength_model=strength, strength_clip=0.0)
        hook_group.add(hook)

        # Build keyframe schedule
        kf_group = build_keyframes(schedule, num_keyframes=keyframes)
        hook.hook_keyframe = kf_group

        for kf in kf_group.keyframes:
            logger.info(f"[FLUX LoRA Scheduled]   {kf.start_percent:.0%} → strength×{kf.strength:.2f}")

        # Register as hook patches (not regular patches)
        model_out = model.clone()
        model_out.add_hook_patches(hook=hook, patches=patch_dict, strength_patch=1.0)

        # Absorb SetCondHooks: attach hooks to conditioning directly
        cond_out = comfy.hooks.set_hooks_for_conditioning(conditioning, hook_group)

        return (model_out, cond_out)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "FluxLoraLoader":    FluxLoraLoader,
    "FluxLoraMulti":     FluxLoraMulti,
    "FluxLoraScheduled": FluxLoraScheduled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraLoader":    "FLUX LoRA Loader",
    "FluxLoraMulti":     "FLUX LoRA Multi",
    "FluxLoraScheduled": "FLUX LoRA Scheduled",
}
