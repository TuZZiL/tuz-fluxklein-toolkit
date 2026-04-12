"""
FLUX LoRA Loader — Consolidated node pack (4 nodes).

Nodes:
  FluxLoraLoader    — Single LoRA with interactive graph widget + auto-strength toggle
  FluxLoraMulti     — Dynamic multi-slot loader (rgthree-style "+ Add LoRA")
  FluxLoraScheduled — Per-step temporal scheduling, absorbs SetCondHooks
  FluxLoraComposer  — Compact role-based multi-LoRA composer

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
import comfy.utils
import folder_paths
import logging

from .edit_presets import (
    PRESET_NAMES,
    USE_CASE_NAMES,
    build_graph_presets,
)
from .lora_compat import build_key_map
from .schedules import SCHEDULE_NAMES, build_keyframes
from .composer_policy import (
    GOAL_NAMES,
    SAFETY_NAMES,
    compose_slot_policies,
    summarize_policies,
)
from .lora_pipeline import (
    apply_edit_multipliers as _apply_edit_multipliers,
    collect_compatibility_report as _collect_compatibility_report,
    compute_strengths as _compute_strengths,
    convert_to_native as _convert_to_native,
    is_diffusers_format as _is_diffusers_format,
    load_and_patch as _load_and_patch,
    log_compatibility_report as _log_compatibility_report,
    prepare_patch_data as _prepare_patch_data,
    resolve_edit_mode as _resolve_edit_mode,
)
from .node_json_contracts import (
    parse_layer_strengths_json as _parse_layer_strengths_json,
    parse_slot_data_json as _parse_slot_data_json,
)

logger = logging.getLogger(__name__)

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
                    "tooltip": "Overall LoRA strength. Lower it first if the edit is too aggressive; raise it if the LoRA feels too weak.",
                }),
                "use_case": (USE_CASE_NAMES, {
                    "default": "Edit",
                    "tooltip": "Tells Auto what you are trying to do. Edit keeps the reference person/object steadier. Generate gives the LoRA more freedom for text-to-image or loose restyling.",
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                    "tooltip": "Leave ON for most downloaded FLUX LoRAs. Turn OFF only if you know the file is already native and want raw passthrough.",
                }),
            },
            "optional": {
                "auto_strength": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Auto (analyze layers)",
                    "label_off": "Manual (graph bars)",
                    "tooltip": "Analyze LoRA layer strength automatically. Useful when some layers hit too hard and others feel too weak.",
                }),
                "edit_mode": (PRESET_NAMES, {
                    "default": "None",
                    "tooltip": "How protective the loader should be. Auto is the safest starting point. 'None' here means Raw / No Protection, not an unselected value.",
                }),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of the preset to keep. 0.0 = strongest preset effect, 1.0 = raw LoRA behavior with no extra protection/boost.",
                }),
                # Canonical graph button masks from edit_presets.py — hidden by JS
                "graph_presets": ("STRING", {"default": json.dumps(build_graph_presets(), sort_keys=True)}),
                # Written by the JS graph widget — never shown as a text box
                "layer_strengths": ("STRING", {"default": "{}"}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/FLUX"
    TITLE = "TUZ FLUX LoRA Loader"

    def load_lora(self, model, lora_name, strength, use_case="Edit",
                  auto_convert=True, auto_strength=False, layer_strengths="{}",
                  edit_mode="None", balance=0.5, graph_presets=None, node_id=None):
        if strength == 0:
            return (model,)

        # Parse per-layer strengths from graph widget
        layer_cfg = _parse_layer_strengths_json(layer_strengths, "FLUX LoRA Loader")

        model_out = _load_and_patch(
            model, lora_name, strength, auto_convert, edit_mode, balance, use_case,
            layer_cfg=layer_cfg, auto_strength=auto_strength,
            node_label="FLUX LoRA Loader", node_id=node_id,
        )
        return (model_out,)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — FluxLoraMulti  (dynamic multi-slot, rgthree-style)
# ═══════════════════════════════════════════════════════════════════════════════

class FluxLoraMulti:
    """
    Dynamic multi-LoRA loader with per-slot control.
    Slots are managed by JS widget (+ Add LoRA / ✕ Remove).
    Each slot has: enabled, lora, strength, use_case, edit_mode, balance.
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
                    "tooltip": "Leave ON for most downloaded FLUX LoRAs. Turn OFF only when you are sure the file already matches Klein-native naming.",
                }),
                # Hidden — populated by JS widget, JSON array of slot configs
                "slot_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": "Hidden JSON contract used by the Multi node UI. If the UI fails, this field must still contain a JSON list of slot objects.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "loaders/FLUX"
    TITLE = "TUZ FLUX LoRA Multi"

    def load_loras(self, model, auto_convert=True, slot_data="[]"):
        slots = _parse_slot_data_json(slot_data, "FLUX LoRA Multi")
        if slots is None:
            return (model,)

        current = model
        for i, slot in enumerate(slots):
            if not isinstance(slot, dict):
                logger.warning(f"[FLUX LoRA Multi] Ignoring non-object slot at index {i}")
                continue

            enabled   = slot.get("enabled", True)
            lora_name = slot.get("lora", "None")
            strength  = slot.get("strength", 1.0)
            use_case  = slot.get("use_case", "Edit")
            edit_mode = slot.get("edit_mode", "None")
            balance   = slot.get("balance", 0.5)

            if not enabled or lora_name == "None" or strength == 0:
                continue

            current = _load_and_patch(
                current, lora_name, strength, auto_convert, edit_mode, balance, use_case,
                node_label=f"FLUX LoRA Multi slot {i+1}",
            )

        return (current,)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — FluxLoraComposer  (compact role-based composition)
# ═══════════════════════════════════════════════════════════════════════════════

class FluxLoraComposer:
    """
    Compact role-based multi-LoRA composer.
    Users pick LoRAs, strengths, and semantic roles; the node handles the
    layer policy, edit-mode, and normalization internally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "goal": (GOAL_NAMES, {
                    "default": "Edit",
                    "tooltip": "Edit protects structure most, Restyle opens up style changes, Generate allows the freest LoRA composition.",
                }),
                "safety": (SAFETY_NAMES, {
                    "default": "Balanced",
                    "tooltip": "How strictly Composer should prevent LoRAs from fighting over the same FLUX regions. Safe = stricter conflict control, Strong = hotter mix.",
                }),
            },
            "optional": {
                "auto_normalize": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-normalize overlap",
                    "label_off": "Raw slot stacking",
                    "tooltip": "Automatically tones down overlapping LoRAs so they do not overload the same FLUX block groups.",
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                    "tooltip": "Leave ON for most downloaded FLUX LoRAs.",
                }),
                "slot_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": "Hidden JSON contract used by the Composer UI. If the UI fails, this field must still contain a JSON list of slot objects.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "compose_loras"
    CATEGORY = "loaders/FLUX"
    TITLE = "TUZ FLUX LoRA Composer"

    def compose_loras(self, model, goal="Edit", safety="Balanced",
                      auto_normalize=True, auto_convert=True, slot_data="[]"):
        slots = _parse_slot_data_json(slot_data, "FLUX LoRA Composer")
        if slots is None:
            return (model,)

        policies = compose_slot_policies(
            slots, goal=goal, safety=safety, auto_normalize=auto_normalize
        )
        summary = summarize_policies(policies)
        logger.info(
            f"[FLUX LoRA Composer] goal={goal} safety={safety}"
            f" | active={summary['active_count']} | main={summary['main_lora']}"
            f" | support={summary['support_count']} | normalized={summary['normalized']}"
        )

        current = model
        for entry in policies:
            slot = entry["slot"]
            if not slot["enabled"] or slot["lora"] == "None" or slot["strength"] == 0:
                continue

            if entry["normalized"]:
                logger.info(
                    f"[FLUX LoRA Composer] Normalized overlap for '{slot['lora']}'"
                    f" role={slot['role']} groups={','.join(sorted(set(entry['conflicts'])))}"
                )
            else:
                logger.info(
                    f"[FLUX LoRA Composer] Applying '{slot['lora']}'"
                    f" role={slot['role']} edit_mode={entry['edit_mode']}"
                )

            prepared = _prepare_patch_data(
                current,
                slot["lora"],
                1.0,
                auto_convert,
                entry["edit_mode"],
                entry["balance"],
                use_case="Edit" if goal == "Edit" else "Generate",
                layer_cfg=entry["layer_cfg"],
                auto_strength=False,
                node_label=f"FLUX LoRA Composer slot {entry['index'] + 1}",
            )
            if prepared is None:
                continue

            current = current.clone()
            current.add_patches(
                prepared["patch_dict"],
                strength_patch=slot["strength"],
                strength_model=1.0,
            )

        return (current,)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — FluxLoraScheduled  (temporal scheduling, absorbs SetCondHooks)
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
                    "tooltip": "Base LoRA strength before the schedule curve is applied. Lower it if the scheduled effect still feels too strong overall.",
                }),
                "use_case": (USE_CASE_NAMES, {
                    "default": "Edit",
                    "tooltip": "Tells Auto what you are trying to do. Edit protects the input image more. Generate gives the LoRA more freedom.",
                }),
                "schedule": (SCHEDULE_NAMES, {
                    "default": "Fade Out",
                    "tooltip": "When during sampling the LoRA should be strongest. Fade Out is usually the safest starting point for image editing.",
                }),
            },
            "optional": {
                "edit_mode": (PRESET_NAMES, {
                    "default": "Auto",
                    "tooltip": "How protective the loader should be across Klein layers. Auto analyzes the LoRA and picks a starting mode for you.",
                }),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How strongly to apply the chosen mode. Lower = safer / more preserving. Higher = closer to raw LoRA.",
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                    "tooltip": "Leave ON for most downloaded FLUX LoRAs. Turn OFF only if you know the file is already native.",
                }),
                "keyframes": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 10,
                    "tooltip": "How finely the schedule curve is sampled. More keyframes gives smoother transitions, but 5 is enough for most edits.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/FLUX"
    TITLE = "TUZ FLUX LoRA Scheduled"

    def load_lora(self, model, conditioning, lora_name, strength, use_case="Edit", schedule="Fade Out",
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
        edit_preset_cfg = _resolve_edit_mode(
            edit_mode, balance, lora_path, "FLUX LoRA Scheduled", use_case=use_case
        )
        if edit_preset_cfg:
            lora_sd = _apply_edit_multipliers(lora_sd, edit_preset_cfg)

        # Build patches
        key_map = build_key_map(model)
        compat_report = _collect_compatibility_report(lora_sd, key_map)
        patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
        _log_compatibility_report("FLUX LoRA Scheduled", compat_report, applied_modules=len(patch_dict))
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
    "FluxLoraComposer":  FluxLoraComposer,
    "FluxLoraScheduled": FluxLoraScheduled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraLoader":    "TUZ FLUX LoRA Loader",
    "FluxLoraMulti":     "TUZ FLUX LoRA Multi",
    "FluxLoraComposer":  "TUZ FLUX LoRA Composer",
    "FluxLoraScheduled": "TUZ FLUX LoRA Scheduled",
}
