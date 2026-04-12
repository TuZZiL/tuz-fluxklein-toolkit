"""
Preflight advisory nodes for FLUX LoRA workflows.

These nodes analyze a LoRA file and model key compatibility, then return
recommendations without mutating the model or requiring schedule logic.
"""

from __future__ import annotations

import json
import logging

import comfy.utils
import folder_paths

try:  # pragma: no cover - import style depends on package context
    from .lora_compat import build_compatibility_report, build_key_map, normalize_lora_keys
    from .lora_meta import analyse_for_node
    from .preflight_policy import build_multi_advice, build_single_advice, _active_slot
except ImportError:  # pragma: no cover
    from lora_compat import build_compatibility_report, build_key_map, normalize_lora_keys
    from lora_meta import analyse_for_node
    from preflight_policy import build_multi_advice, build_single_advice, _active_slot

logger = logging.getLogger(__name__)


def _load_lora(lora_name):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        raise FileNotFoundError(f"LoRA file not found: {lora_name}")
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    return lora_path, lora_sd


def _parse_slot_data(slot_data, use_case="Edit"):
    try:
        slots = json.loads(slot_data)
    except Exception as exc:
        return None, _failure_report(f"Invalid slot_data JSON: {exc}", use_case=use_case)
    if not isinstance(slots, list):
        return None, _failure_report("slot_data must be a JSON list", use_case=use_case)
    return slots, None


def _failure_report(reason, use_case="Edit"):
    return {
        "report": f"Preflight Advisor\nUse case: {use_case}\nError: {reason}",
        "compat_status": "failed",
        "matched_modules": 0,
        "total_modules": 0,
        "skipped_modules": 0,
        "incomplete_modules": 0,
        "recommended_edit_mode": "None",
        "recommended_balance": 1.0,
        "recommended_strength": 0.5,
        "risk_level": "high",
        "warnings": [str(reason)],
        "profile_tags": [],
        "analysis_summary": {},
        "compat_summary": {
            "status": "failed",
            "matched": 0,
            "total": 0,
            "skipped": 0,
            "incomplete": 0,
            "sample_skipped": [],
            "sample_incomplete": [],
        },
    }


class FluxLoraPreflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "use_case": (["Edit", "Generate"], {
                    "default": "Edit",
                    "tooltip": "Edit assumes you want to preserve identity/structure more. Generate assumes freer restyling or text-to-image behavior.",
                }),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "STRING",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "report",
        "recommended_edit_mode",
        "recommended_balance",
        "recommended_strength",
        "compat_status",
        "matched_modules",
        "total_modules",
    )
    FUNCTION = "analyze"
    CATEGORY = "analysis/FLUX"
    TITLE = "TUZ FLUX Preflight Advisor"

    def analyze(self, model, lora_name, use_case="Edit"):
        try:
            lora_path, lora_sd = _load_lora(lora_name)
            analysis = analyse_for_node(lora_path)
            key_map = build_key_map(model)
            compat_report = build_compatibility_report(normalize_lora_keys(lora_sd).keys(), key_map)
            advice = build_single_advice(analysis, compat_report, use_case=use_case, source_name=lora_name)
            return (
                advice["report"],
                advice["recommended_edit_mode"],
                float(advice["recommended_balance"]),
                float(advice["recommended_strength"]),
                advice["compat_status"],
                int(advice["matched_modules"]),
                int(advice["total_modules"]),
            )
        except Exception as exc:  # pragma: no cover - exercised when file/runtime load fails
            logger.exception("[FLUX Preflight] Failed to analyze LoRA")
            advice = _failure_report(exc, use_case=use_case)
            return (
                advice["report"],
                advice["recommended_edit_mode"],
                float(advice["recommended_balance"]),
                float(advice["recommended_strength"]),
                advice["compat_status"],
                int(advice["matched_modules"]),
                int(advice["total_modules"]),
            )


class FluxLoraMultiPreflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "slot_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": "JSON slot list using the same shape as TUZ FLUX LoRA Multi. Invalid JSON returns a failure report instead of a silent noop.",
                }),
            },
            "optional": {
                "use_case": (["Edit", "Generate"], {
                    "default": "Edit",
                    "tooltip": "Edit assumes more identity preservation. Generate assumes freer composition.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("report", "recommended_slot_data_json", "active_slot_count", "risk_level")
    FUNCTION = "analyze"
    CATEGORY = "analysis/FLUX"
    TITLE = "TUZ FLUX Multi Preflight Advisor"

    def analyze(self, model, slot_data="[]", use_case="Edit"):
        slots, advice = _parse_slot_data(slot_data, use_case=use_case)
        if advice is not None:
            return advice["report"], "[]", 0, advice["risk_level"]

        entries = []
        report_lines = []
        for index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            normalized = dict(slot)
            normalized.setdefault("enabled", True)
            normalized.setdefault("lora", "None")
            normalized.setdefault("strength", 1.0)
            normalized.setdefault("use_case", use_case)
            normalized.setdefault("edit_mode", "None")
            normalized.setdefault("balance", 0.5)
            active = _active_slot(normalized)
            if not active:
                entries.append({
                    "index": index,
                    "slot": normalized,
                    "advice": {},
                    "active": False,
                })
                continue

            try:
                lora_path, lora_sd = _load_lora(normalized["lora"])
                analysis = analyse_for_node(lora_path)
                key_map = build_key_map(model)
                compat_report = build_compatibility_report(normalize_lora_keys(lora_sd).keys(), key_map)
                advice = build_single_advice(
                    analysis,
                    compat_report,
                    use_case=normalized.get("use_case", use_case),
                    source_name=normalized["lora"],
                )
                entries.append({
                    "index": index,
                    "slot": normalized,
                    "advice": advice,
                    "active": True,
                })
            except Exception as exc:  # pragma: no cover - runtime load failures
                logger.exception("[FLUX Multi Preflight] Failed to analyze slot %s", index + 1)
                advice = _failure_report(exc, use_case=normalized.get("use_case", use_case))
                entries.append({
                    "index": index,
                    "slot": normalized,
                    "advice": advice,
                    "active": True,
                })

        multi = build_multi_advice(entries, use_case=use_case, source_name="slot_data")
        report_lines.append(multi["report"])
        return (
            "\n".join(report_lines).strip(),
            multi["recommended_slot_data_json"],
            int(multi["active_slot_count"]),
            multi["risk_level"],
        )


NODE_CLASS_MAPPINGS = {
    "FluxLoraPreflight": FluxLoraPreflight,
    "FluxLoraMultiPreflight": FluxLoraMultiPreflight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraPreflight": "TUZ FLUX Preflight Advisor",
    "FluxLoraMultiPreflight": "TUZ FLUX Multi Preflight Advisor",
}
