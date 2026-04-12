"""
Helpers for LoRA key normalization and compatibility reporting.

This module stays free of ComfyUI/runtime dependencies so it can be reused by
the loader, the forensic analyser, and lightweight unit tests.
"""

from __future__ import annotations

import re


def normalize_lora_key(key: str) -> str:
    """Normalize common LoRA key variants into a canonical FLUX-friendly form."""
    k = key
    for pfx in ("transformer.", "diffusion_model.", "unet."):
        if k.startswith(pfx):
            k = k[len(pfx):]
            break
    k = re.sub(r"^transformer_blocks\.", "double_blocks.", k)
    k = re.sub(r"^single_transformer_blocks\.", "single_blocks.", k)
    k = k.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
    k = k.replace(".lora.down.", ".lora_A.").replace(".lora.up.", ".lora_B.")
    k = k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
    k = k.replace(".ff.linear_in.", ".ff.net.0.proj.")
    k = k.replace(".ff.linear_out.", ".ff.net.2.")
    k = k.replace(".ff_context.linear_in.", ".ff_context.net.0.proj.")
    k = k.replace(".ff_context.linear_out.", ".ff_context.net.2.")
    return k


def normalize_lora_keys(lora_sd):
    """Return a copy of a state dict with normalized keys."""
    return {normalize_lora_key(key): val for key, val in lora_sd.items()}


def build_key_map(model):
    """Build lora_key_base -> model_state_dict_key mapping."""
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


def parse_lora_key(key: str):
    """
    Returns (base_key, role) where role is one of:
      lora_down / lora_up / alpha / dora_scale / bias / other
    """
    k = key
    suffix_map = {
        "lora_down.weight": "lora_down",
        "lora_up.weight": "lora_up",
        "lora.down.weight": "lora_down",
        "lora.up.weight": "lora_up",
        "lora_A.weight": "lora_down",
        "lora_B.weight": "lora_up",
        "lora_A.default.weight": "lora_down",
        "lora_B.default.weight": "lora_up",
        "alpha": "alpha",
        "dora_scale": "dora_scale",
        "bias": "bias",
    }
    for suffix, role in suffix_map.items():
        if k.endswith("." + suffix) or k.endswith("_" + suffix):
            base = k[: -(len(suffix) + 1)]
            return base, role
    return k, "other"


def build_module_inventory(keys):
    """
    Group LoRA keys into complete and incomplete modules.

    A module counts only if it has at least one LoRA up/down tensor role.
    Alpha-only or metadata-only entries are ignored.
    """
    modules = {}
    for key in keys:
        base, role = parse_lora_key(key)
        entry = modules.setdefault(base, {"roles": set(), "keys": []})
        entry["roles"].add(role)
        entry["keys"].append(key)

    complete = []
    incomplete = []
    for base, info in modules.items():
        has_down = "lora_down" in info["roles"]
        has_up = "lora_up" in info["roles"]
        if has_down and has_up:
            complete.append(base)
        elif has_down or has_up:
            incomplete.append(base)

    return {
        "modules": modules,
        "complete_modules": sorted(complete),
        "incomplete_modules": sorted(incomplete),
    }


def build_compatibility_report(keys, key_map, sample_limit=3):
    """
    Compare normalized LoRA module bases against a model key map.

    Returns counts based on complete A/B pairs and tracks incomplete pairs
    separately for diagnostics.
    """
    inventory = build_module_inventory(keys)
    complete = inventory["complete_modules"]
    incomplete = inventory["incomplete_modules"]
    matched = [base for base in complete if base in key_map]
    skipped = [base for base in complete if base not in key_map]

    return {
        "total_modules": len(complete),
        "matched_modules": len(matched),
        "skipped_modules": len(skipped),
        "incomplete_modules": len(incomplete),
        "sample_skipped": skipped[:sample_limit],
        "sample_incomplete": incomplete[:sample_limit],
        "matched_module_bases": matched,
        "skipped_module_bases": skipped,
        "complete_module_bases": complete,
    }
