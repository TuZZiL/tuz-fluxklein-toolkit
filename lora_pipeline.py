"""
Shared FLUX LoRA loader pipeline helpers.

This module keeps the loader nodes thinner by isolating format conversion,
compatibility checks, patch preparation, and patch application.
"""

from __future__ import annotations

import json
import logging

import comfy.lora
import comfy.utils
import folder_paths
import numpy as np
import torch

try:  # pragma: no cover - package vs direct import
    from .anatomy_profiles import resolve_profile as resolve_anatomy_profile
    from .edit_presets import EDIT_PRESETS, interpolate_preset, resolve_preset_selection
    from .flux_constants import (
        AUTO_STRENGTH_CEILING,
        AUTO_STRENGTH_FLOOR,
        N_DOUBLE,
        N_SINGLE,
    )
    from .lora_compat import build_compatibility_report, build_key_map, normalize_lora_keys
except ImportError:  # pragma: no cover
    from anatomy_profiles import resolve_profile as resolve_anatomy_profile
    from edit_presets import EDIT_PRESETS, interpolate_preset, resolve_preset_selection
    from flux_constants import AUTO_STRENGTH_CEILING, AUTO_STRENGTH_FLOOR, N_DOUBLE, N_SINGLE
    from lora_compat import build_compatibility_report, build_key_map, normalize_lora_keys

logger = logging.getLogger(__name__)


def is_diffusers_format(lora_sd):
    markers = (
        ".to_q.",
        ".to_k.",
        ".to_v.",
        ".add_q_proj.",
        ".add_k_proj.",
        ".add_v_proj.",
        "transformer_blocks.",
        "single_transformer_blocks.",
        ".ff.net.",
        ".ff_context.",
        ".to_add_out.",
        ".proj_mlp.",
        ".to_qkv_mlp_proj.",
        ".ff.linear_in.",
        ".ff.linear_out.",
    )
    return any(marker in key for key in lora_sd for marker in markers)


def _normalize_keys(lora_sd):
    return normalize_lora_keys(lora_sd)


def _alpha_scale(norm, base):
    down_key = f"{base}.lora_A.weight"
    alpha_key = f"{base}.alpha"
    if alpha_key in norm and down_key in norm:
        rank = norm[down_key].shape[0]
        if rank > 0:
            alpha = norm[alpha_key]
            if hasattr(alpha, "item"):
                alpha = alpha.item()
            return float(alpha) / rank
    return 1.0


def _fuse_qkv(norm, out, done, q, k, v, dst):
    keys_A = [f"{base}.lora_A.weight" for base in (q, k, v)]
    keys_B = [f"{base}.lora_B.weight" for base in (q, k, v)]

    if not all(key in norm for key in keys_A + keys_B):
        return

    A_q, A_k, A_v = [norm[key] for key in keys_A]
    B_q, B_k, B_v = [
        norm[key] * _alpha_scale(norm, base)
        for key, base in zip(keys_B, (q, k, v))
    ]

    r_q, r_k, r_v = A_q.shape[0], A_k.shape[0], A_v.shape[0]
    o_q, o_k, o_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    r_total = r_q + r_k + r_v
    o_total = o_q + o_k + o_v

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)
    B_fused = torch.zeros(o_total, r_total, dtype=B_q.dtype, device=B_q.device)
    B_fused[0:o_q, 0:r_q] = B_q
    B_fused[o_q:o_q + o_k, r_q:r_q + r_k] = B_k
    B_fused[o_q + o_k:o_total, r_q + r_k:r_total] = B_v

    out[f"{dst}.lora_A.weight"] = A_fused
    out[f"{dst}.lora_B.weight"] = B_fused
    out[f"{dst}.alpha"] = torch.tensor(float(r_total))

    for key in keys_A + keys_B:
        done.add(key)
    for base in (q, k, v):
        done.add(f"{base}.alpha")


def _fuse_linear1(norm, out, done, sb_base):
    components = [
        (f"{sb_base}.attn.to_q", "q"),
        (f"{sb_base}.attn.to_k", "k"),
        (f"{sb_base}.attn.to_v", "v"),
        (f"{sb_base}.proj_mlp", "mlp"),
    ]
    dst = f"diffusion_model.{sb_base}.linear1"

    present = [
        (base, label)
        for base, label in components
        if f"{base}.lora_A.weight" in norm and f"{base}.lora_B.weight" in norm
    ]
    if not present:
        return

    A_list = []
    B_scaled = []
    for base, _ in present:
        A = norm[f"{base}.lora_A.weight"]
        B = norm[f"{base}.lora_B.weight"] * _alpha_scale(norm, base)
        A_list.append(A)
        B_scaled.append(B)
        done.update([f"{base}.lora_A.weight", f"{base}.lora_B.weight", f"{base}.alpha"])

    ranks = [A.shape[0] for A in A_list]
    outs = [B.shape[0] for B in B_scaled]
    r_total, o_total = sum(ranks), sum(outs)

    A_fused = torch.cat(A_list, dim=0)
    B_fused = torch.zeros(o_total, r_total, dtype=B_scaled[0].dtype, device=B_scaled[0].device)
    r_off = 0
    o_off = 0
    for A, B in zip(A_list, B_scaled):
        r, o = A.shape[0], B.shape[0]
        B_fused[o_off:o_off + o, r_off:r_off + r] = B
        r_off += r
        o_off += o

    out[f"{dst}.lora_A.weight"] = A_fused
    out[f"{dst}.lora_B.weight"] = B_fused
    out[f"{dst}.alpha"] = torch.tensor(float(r_total))


def _remap(norm, out, done, src_base, dst_base):
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


def convert_to_native(lora_sd):
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
            (f"{db}.attn.to_out.0", f"diffusion_model.{db}.img_attn.proj"),
            (f"{db}.attn.to_add_out", f"diffusion_model.{db}.txt_attn.proj"),
            (f"{db}.ff.net.0.proj", f"diffusion_model.{db}.img_mlp.0"),
            (f"{db}.ff.net.2", f"diffusion_model.{db}.img_mlp.2"),
            (f"{db}.ff_context.net.0.proj", f"diffusion_model.{db}.txt_mlp.0"),
            (f"{db}.ff_context.net.2", f"diffusion_model.{db}.txt_mlp.2"),
        ]:
            _remap(norm, out, done, src, dst)

    for i in range(N_SINGLE):
        sb = f"single_blocks.{i}"
        _remap(norm, out, done, f"{sb}.attn.to_qkv_mlp_proj", f"diffusion_model.{sb}.linear1")
        _remap(norm, out, done, f"{sb}.attn.to_out", f"diffusion_model.{sb}.linear2")
        _fuse_linear1(norm, out, done, sb)
        _remap(norm, out, done, f"{sb}.proj_out", f"diffusion_model.{sb}.linear2")

    for key, value in norm.items():
        if key not in done:
            out[key] = value

    n_converted = sum(1 for key in done if key in norm)
    logger.info(f"[FLUX LoRA] Converted {n_converted} diffusers keys -> {len(out)} native keys")
    return out


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
        parts = key.split(".")
        target = None
        for i, part in enumerate(parts):
            if part == "double_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in db_cfg:
                    cfg = db_cfg[idx]
                    is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                    side = "txt" if is_txt else "img"
                    target = cfg.get(side, global_strength) if isinstance(cfg, dict) else float(cfg)
                break
            if part == "single_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in sb_cfg:
                    target = float(sb_cfg[idx])
                break
        scaled[key] = tensor * (target / global_strength) if target is not None else tensor

    return scaled


def apply_edit_multipliers(lora_sd, preset_cfg):
    db_cfg = {str(k): v for k, v in preset_cfg.get("db", {}).items()}
    sb_cfg = {str(k): v for k, v in preset_cfg.get("sb", {}).items()}
    scaled = {}

    for key, tensor in lora_sd.items():
        if not (key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight")):
            scaled[key] = tensor
            continue
        parts = key.split(".")
        multiplier = None
        for i, part in enumerate(parts):
            if part == "double_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in db_cfg:
                    cfg = db_cfg[idx]
                    is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                    side = "txt" if is_txt else "img"
                    multiplier = cfg.get(side, 1.0) if isinstance(cfg, dict) else float(cfg)
                break
            if part == "single_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in sb_cfg:
                    multiplier = float(sb_cfg[idx])
                break
        if multiplier is not None and abs(multiplier - 1.0) > 1e-6:
            scaled[key] = tensor * multiplier
        else:
            scaled[key] = tensor

    return scaled


def apply_anatomy_profile(lora_sd, profile_cfg, strict_zero=False):
    if not profile_cfg:
        return lora_sd

    db_cfg = {str(k): v for k, v in profile_cfg.get("db", {}).items()}
    sb_cfg = {str(k): v for k, v in profile_cfg.get("sb", {}).items()}
    zero_cfg = profile_cfg.get("strict_zero", {}) if strict_zero else {}
    zero_db = {str(i) for i in zero_cfg.get("db", [])}
    zero_sb = {str(i) for i in zero_cfg.get("sb", [])}
    scaled = {}

    for key, tensor in lora_sd.items():
        if not (key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight")):
            scaled[key] = tensor
            continue
        parts = key.split(".")
        multiplier = None
        for i, part in enumerate(parts):
            if part == "double_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in zero_db:
                    multiplier = 0.0
                elif idx in db_cfg:
                    cfg = db_cfg[idx]
                    is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                    side = "txt" if is_txt else "img"
                    multiplier = cfg.get(side, 1.0) if isinstance(cfg, dict) else float(cfg)
                break
            if part == "single_blocks" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in zero_sb:
                    multiplier = 0.0
                elif idx in sb_cfg:
                    multiplier = float(sb_cfg[idx])
                break
        if multiplier is not None and abs(multiplier - 1.0) > 1e-6:
            scaled[key] = tensor * multiplier
        else:
            scaled[key] = tensor
    return scaled


def collect_compatibility_report(lora_sd, key_map):
    return build_compatibility_report(_normalize_keys(lora_sd).keys(), key_map)


def resolved_compatibility_counts(report, applied_modules=None):
    total = report.get("total_modules", 0)
    estimated = report.get("matched_modules", 0)
    matched = estimated if applied_modules is None else min(estimated, applied_modules, total)
    skipped = max(total - matched, 0)
    incomplete = report.get("incomplete_modules", 0)
    return matched, total, skipped, incomplete


def compatibility_status(matched, total, incomplete):
    if matched <= 0:
        return "failed"
    if matched < total or incomplete > 0:
        return "partial"
    return "ok"


def log_compatibility_report(node_label, report, applied_modules=None):
    matched, total, skipped, incomplete = resolved_compatibility_counts(report, applied_modules)
    logger.info(
        f"[{node_label}] Compatibility {matched}/{total} matched"
        f" | skipped={skipped} | incomplete={incomplete}"
    )
    if applied_modules is not None and report.get("matched_modules", 0) != applied_modules:
        logger.warning(
            f"[{node_label}] Compatibility estimate mismatch:"
            f" expected {report.get('matched_modules', 0)} matched, applied {applied_modules}"
        )
    for base in report.get("sample_skipped", ()):
        logger.warning(f"[{node_label}] Skipped module: {base}")
    for base in report.get("sample_incomplete", ()):
        logger.warning(f"[{node_label}] Incomplete module pair: {base}")


def send_compatibility_report(node_id, report, applied_modules=None):
    if node_id is None:
        return
    matched, total, skipped, incomplete = resolved_compatibility_counts(report, applied_modules)
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "flux_lora.compat_report",
            {
                "node": str(node_id),
                "status": compatibility_status(matched, total, incomplete),
                "matched_modules": int(matched),
                "total_modules": int(total),
                "skipped_modules": int(skipped),
            },
        )
    except Exception:
        logger.exception("[FLUX LoRA] Failed to send compatibility report to UI")


def resolve_edit_mode(edit_mode, balance, lora_path, node_label="FLUX LoRA", use_case="Edit"):
    if edit_mode == "None":
        return None
    if edit_mode == "Auto":
        try:
            from .lora_meta import analyse_for_node
        except ImportError:  # pragma: no cover
            from lora_meta import analyse_for_node
        analysis = analyse_for_node(lora_path)
        auto_preset, auto_balance = resolve_preset_selection(
            edit_mode, balance, analysis=analysis, use_case=use_case
        )
        if auto_preset == "None":
            logger.info(f"[{node_label}] Auto({use_case}) -> None (raw LoRA is enough)")
            return None
        preset_raw = EDIT_PRESETS.get(auto_preset)
        if preset_raw is not None:
            cfg = interpolate_preset(preset_raw, auto_balance)
            logger.info(f"[{node_label}] Auto({use_case}) -> {auto_preset} (protection={auto_balance:.2f})")
            return cfg
        return None
    preset_name, resolved_balance = resolve_preset_selection(edit_mode, balance, use_case=use_case)
    preset_raw = EDIT_PRESETS.get(preset_name)
    if preset_raw is not None:
        cfg = interpolate_preset(preset_raw, resolved_balance)
        logger.info(f"[{node_label}] Edit mode '{preset_name}' applied (protection={resolved_balance:.2f})")
        return cfg
    return None


def _all_norms(analysis):
    out = []
    for i in range(N_DOUBLE):
        db = analysis["db"].get(i, {})
        if db.get("img") is not None:
            out.append(db["img"])
        if db.get("txt") is not None:
            out.append(db["txt"])
    for i in range(N_SINGLE):
        value = analysis["sb"].get(i)
        if value is not None:
            out.append(value)
    return out


def compute_strengths(analysis, global_strength):
    all_norms = _all_norms(analysis)
    if not all_norms:
        return {
            "db": {str(i): {"img": global_strength, "txt": global_strength} for i in range(N_DOUBLE)},
            "sb": {str(i): global_strength for i in range(N_SINGLE)},
        }

    mean_norm = float(np.mean(all_norms))

    def clamp(value):
        return max(AUTO_STRENGTH_FLOOR, min(AUTO_STRENGTH_CEILING, value))

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


def prepare_patch_data(
    model,
    lora_name,
    strength,
    auto_convert,
    edit_mode,
    balance,
    anatomy_profile="None",
    anatomy_strength=0.65,
    anatomy_strict_zero=False,
    anatomy_custom_json="",
    use_case="Edit",
    layer_cfg=None,
    auto_strength=False,
    node_label="FLUX LoRA",
    node_id=None,
):
    if strength == 0:
        return None

    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    logger.info(f"[{node_label}] Loading: {lora_name}  ({len(lora_sd)} keys)")

    if auto_strength and not layer_cfg:
        try:
            from .lora_meta import analyse_for_node
        except ImportError:  # pragma: no cover
            from lora_meta import analyse_for_node
        analysis = analyse_for_node(lora_path)
        layer_cfg = compute_strengths(analysis, abs(strength))
        logger.info(f"[{node_label}] Auto-strength computed for {lora_name}")
        if node_id is not None:
            try:
                from server import PromptServer
                PromptServer.instance.send_sync(
                    "flux_lora.auto_strength",
                    {
                        "node": str(node_id),
                        "layer_strengths": json.dumps(layer_cfg, sort_keys=True),
                    },
                )
            except Exception:
                logger.exception(f"[{node_label}] Failed to send auto-strength update to UI")

    edit_preset_cfg = resolve_edit_mode(edit_mode, balance, lora_path, node_label, use_case=use_case)
    try:
        anatomy_profile_cfg = resolve_anatomy_profile(
            anatomy_profile,
            strength=anatomy_strength,
            custom_json=anatomy_custom_json,
        )
    except Exception as exc:
        logger.warning(f"[{node_label}] Invalid anatomy profile settings ignored: {exc}")
        anatomy_profile_cfg = None

    if auto_convert and is_diffusers_format(lora_sd):
        logger.info(f"[{node_label}] Detected diffusers format -> converting")
        lora_sd = convert_to_native(lora_sd)

    if layer_cfg:
        lora_sd = apply_layer_strengths(lora_sd, layer_cfg, strength)

    if edit_preset_cfg:
        lora_sd = apply_edit_multipliers(lora_sd, edit_preset_cfg)
    if anatomy_profile_cfg:
        lora_sd = apply_anatomy_profile(lora_sd, anatomy_profile_cfg, strict_zero=anatomy_strict_zero)
        logger.info(
            f"[{node_label}] Anatomy profile '{anatomy_profile}' applied"
            f" (strength={float(anatomy_strength):.2f}, strict_zero={bool(anatomy_strict_zero)})"
        )

    key_map = build_key_map(model)
    compat_report = collect_compatibility_report(lora_sd, key_map)
    patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
    log_compatibility_report(node_label, compat_report, applied_modules=len(patch_dict))
    send_compatibility_report(node_id, compat_report, applied_modules=len(patch_dict))
    logger.info(f"[{node_label}] Prepared {len(patch_dict)} patches")
    return {
        "patch_dict": patch_dict,
        "compat_report": compat_report,
        "lora_path": lora_path,
        "strength": strength,
        "layer_cfg": layer_cfg,
    }


def load_and_patch(
    model,
    lora_name,
    strength,
    auto_convert,
    edit_mode,
    balance,
    anatomy_profile="None",
    anatomy_strength=0.65,
    anatomy_strict_zero=False,
    anatomy_custom_json="",
    use_case="Edit",
    layer_cfg=None,
    auto_strength=False,
    node_label="FLUX LoRA",
    node_id=None,
):
    prepared = prepare_patch_data(
        model,
        lora_name,
        strength,
        auto_convert,
        edit_mode,
        balance,
        anatomy_profile,
        anatomy_strength,
        anatomy_strict_zero,
        anatomy_custom_json,
        use_case,
        layer_cfg=layer_cfg,
        auto_strength=auto_strength,
        node_label=node_label,
        node_id=node_id,
    )
    if prepared is None:
        return model

    model_out = model.clone()
    effective_strength = 1.0 if layer_cfg else strength
    model_out.add_patches(prepared["patch_dict"], strength_patch=effective_strength, strength_model=1.0)
    return model_out
