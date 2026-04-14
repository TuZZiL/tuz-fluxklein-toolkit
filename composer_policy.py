"""
Policy helpers for FLUX LoRA Composer.

This module stays runtime-light so it can be unit-tested without ComfyUI.
It converts a simple role-based slot list into concrete layer policies.
"""

from __future__ import annotations

from copy import deepcopy

try:  # pragma: no cover - package vs direct import
    from .flux_constants import N_DOUBLE, N_SINGLE
except ImportError:  # pragma: no cover
    from flux_constants import N_DOUBLE, N_SINGLE

GOAL_NAMES = ["Edit", "Restyle", "Generate"]
SAFETY_NAMES = ["Safe", "Balanced", "Strong"]
COMPOSER_ROLE_NAMES = ["Main Edit", "Style", "Detail", "Identity", "Prompt Boost"]

GROUP_NAMES = ("db_img", "db_txt", "sb_early", "sb_mid", "sb_late")

ROLE_POLICIES = {
    "Main Edit": {
        "edit_mode": "None",
        "balance": 0.50,
        "priority": 1.35,
        "groups": {"db_img": 1.00, "db_txt": 1.00, "sb_early": 1.00, "sb_mid": 1.00, "sb_late": 1.00},
    },
    "Style": {
        "edit_mode": "Style Only",
        "balance": 0.70,
        "priority": 0.95,
        "groups": {"db_img": 0.92, "db_txt": 1.00, "sb_early": 0.98, "sb_mid": 1.00, "sb_late": 0.95},
    },
    "Detail": {
        "edit_mode": "None",
        "balance": 0.75,
        "priority": 0.80,
        "groups": {"db_img": 0.95, "db_txt": 0.95, "sb_early": 0.92, "sb_mid": 0.96, "sb_late": 1.00},
    },
    "Identity": {
        "edit_mode": "Preserve Face",
        "balance": 0.65,
        "priority": 1.20,
        "groups": {"db_img": 1.00, "db_txt": 0.95, "sb_early": 0.98, "sb_mid": 0.94, "sb_late": 0.88},
    },
    "Prompt Boost": {
        "edit_mode": "Boost Prompt",
        "balance": 0.75,
        "priority": 0.90,
        "groups": {"db_img": 0.96, "db_txt": 1.06, "sb_early": 0.98, "sb_mid": 1.02, "sb_late": 0.98},
    },
}

GOAL_MODIFIERS = {
    "Edit": {"db_img": 1.00, "db_txt": 0.98, "sb_early": 1.00, "sb_mid": 0.96, "sb_late": 0.92},
    "Restyle": {"db_img": 0.97, "db_txt": 1.00, "sb_early": 0.98, "sb_mid": 1.00, "sb_late": 0.97},
    "Generate": {"db_img": 1.00, "db_txt": 1.02, "sb_early": 1.00, "sb_mid": 1.02, "sb_late": 1.00},
}

SAFETY_BUDGETS = {
    "Safe": {"db_img": 2.20, "db_txt": 2.10, "sb_early": 2.10, "sb_mid": 1.80, "sb_late": 1.40},
    "Balanced": {"db_img": 2.80, "db_txt": 2.70, "sb_early": 2.60, "sb_mid": 2.20, "sb_late": 1.80},
    "Strong": {"db_img": 3.50, "db_txt": 3.30, "sb_early": 3.20, "sb_mid": 2.90, "sb_late": 2.50},
}

SAFETY_BALANCE_OFFSETS = {"Safe": -0.10, "Balanced": 0.0, "Strong": 0.10}


def _scale_budgets(base_budgets, n_active):
    """Reduce safety budgets for 3+ active LoRAs to prevent cross-interference.

    More LoRAs pulling weights in different directions cause destructive
    interference even when total magnitude stays within budget.  Scaling down
    by ~10 % per extra LoRA beyond 2 keeps the combined perturbation safe.

    n=2 → 1.00,  n=3 → 0.90,  n=4 → 0.80,  n=5 → 0.70  (floor 0.50)
    """
    if n_active <= 2:
        return base_budgets
    factor = max(0.50, 1.0 - 0.10 * (n_active - 2))
    return {g: round(v * factor, 2) for g, v in base_budgets.items()}


def _clamp(value, low, high):
    return max(low, min(high, value))


def normalize_slot(slot):
    role = slot.get("role", "Main Edit")
    if role not in COMPOSER_ROLE_NAMES:
        role = "Main Edit"
    return {
        "enabled": bool(slot.get("enabled", True)),
        "lora": slot.get("lora", "None"),
        "strength": float(slot.get("strength", 1.0)),
        "role": role,
        "collapsed": bool(slot.get("collapsed", True)),
        "anatomy_profile": slot.get("anatomy_profile", "None"),
        "anatomy_strength": float(slot.get("anatomy_strength", 0.65)),
        "anatomy_strict_zero": bool(slot.get("anatomy_strict_zero", False)),
        "anatomy_custom_json": slot.get("anatomy_custom_json", ""),
    }


def normalize_slots(slots):
    return [normalize_slot(slot) for slot in slots if isinstance(slot, dict)]


def assign_main_edit(slots):
    slots = [deepcopy(slot) for slot in normalize_slots(slots)]
    active = [
        (index, slot)
        for index, slot in enumerate(slots)
        if slot["enabled"] and slot["lora"] != "None" and abs(slot["strength"]) > 1e-8
    ]
    if not active:
        return slots
    if any(slot["role"] == "Main Edit" for _, slot in active):
        return slots
    strongest_index, _ = max(active, key=lambda item: abs(item[1]["strength"]))
    slots[strongest_index]["role"] = "Main Edit"
    return slots


def role_edit_profile(role, safety="Balanced"):
    policy = ROLE_POLICIES.get(role, ROLE_POLICIES["Main Edit"])
    profile = {
        "edit_mode": policy["edit_mode"],
        "balance": _clamp(policy["balance"] + SAFETY_BALANCE_OFFSETS.get(safety, 0.0), 0.0, 1.0),
        # Keep anatomy shielding opt-in for Composer to avoid silent behavior
        # changes in existing workflows that relied on prior defaults.
        "anatomy_profile": "None",
        "anatomy_strength": 0.65,
        "anatomy_strict_zero": False,
    }
    return profile


def build_group_profile(role, goal="Edit"):
    role_policy = ROLE_POLICIES.get(role, ROLE_POLICIES["Main Edit"])
    goal_policy = GOAL_MODIFIERS.get(goal, GOAL_MODIFIERS["Edit"])
    return {
        group: round(role_policy["groups"][group] * goal_policy[group], 4)
        for group in GROUP_NAMES
    }


def build_layer_cfg(group_profile):
    group_profile = dict(group_profile)
    return {
        "db": {
            str(i): {
                "img": float(group_profile.get("db_img", 1.0)),
                "txt": float(group_profile.get("db_txt", 1.0)),
            }
            for i in range(N_DOUBLE)
        },
        "sb": {
            str(i): float(group_profile.get(
                "sb_early" if i < 8 else "sb_mid" if i < 16 else "sb_late", 1.0
            ))
            for i in range(N_SINGLE)
        },
    }


def compose_slot_policies(slots, goal="Edit", safety="Balanced", auto_normalize=True):
    slots = assign_main_edit(slots)
    base_budgets = SAFETY_BUDGETS.get(safety, SAFETY_BUDGETS["Balanced"])
    prepared = []
    active_indices = []

    for index, slot in enumerate(slots):
        normalized = normalize_slot(slot)
        profile = build_group_profile(normalized["role"], goal)
        edit_profile = role_edit_profile(normalized["role"], safety)
        priority = ROLE_POLICIES.get(normalized["role"], ROLE_POLICIES["Main Edit"])["priority"]
        demand = {group: abs(normalized["strength"]) * profile[group] for group in GROUP_NAMES}
        entry = {
            "index": index,
            "slot": normalized,
            "priority": priority,
            "base_groups": profile,
            "demand": demand,
            "group_factors": {group: 1.0 for group in GROUP_NAMES},
            "conflicts": [],
            **edit_profile,
        }
        if normalized["anatomy_profile"] not in (None, "", "None"):
            entry["anatomy_profile"] = normalized["anatomy_profile"]
            entry["anatomy_strength"] = _clamp(normalized["anatomy_strength"], 0.0, 1.0)
            entry["anatomy_strict_zero"] = normalized["anatomy_strict_zero"]
            entry["anatomy_custom_json"] = normalized["anatomy_custom_json"]
        prepared.append(entry)
        if normalized["enabled"] and normalized["lora"] != "None" and abs(normalized["strength"]) > 1e-8:
            active_indices.append(index)

    single_active_mode = len(active_indices) <= 1
    budgets = _scale_budgets(base_budgets, len(active_indices))

    if auto_normalize and len(active_indices) > 1:
        for group in GROUP_NAMES:
            total_demand = sum(prepared[i]["demand"][group] for i in active_indices)
            budget = budgets[group]
            if total_demand <= budget + 1e-8:
                continue
            weighted_total = sum(prepared[i]["demand"][group] * prepared[i]["priority"] for i in active_indices)
            if weighted_total <= 1e-8:
                continue
            for i in active_indices:
                factor = min(1.0, budget * prepared[i]["priority"] / weighted_total)
                prepared[i]["group_factors"][group] = round(factor, 4)
                if factor < 0.999:
                    prepared[i]["conflicts"].append(group)

    for entry in prepared:
        if single_active_mode:
            final_groups = {group: 1.0 for group in GROUP_NAMES}
        else:
            final_groups = {
                group: round(entry["base_groups"][group] * entry["group_factors"][group], 4)
                for group in GROUP_NAMES
            }
        entry["final_groups"] = final_groups
        entry["layer_cfg"] = {} if single_active_mode else build_layer_cfg(final_groups)
        entry["normalized"] = any(factor < 0.999 for factor in entry["group_factors"].values())

    return prepared


def summarize_policies(policies):
    active = [
        entry for entry in policies
        if entry["slot"]["enabled"] and entry["slot"]["lora"] != "None" and abs(entry["slot"]["strength"]) > 1e-8
    ]
    main = next((entry["slot"]["lora"] for entry in active if entry["slot"]["role"] == "Main Edit"), None)
    support_count = sum(1 for entry in active if entry["slot"]["role"] != "Main Edit")
    normalized = any(entry["normalized"] for entry in active)
    return {
        "active_count": len(active),
        "main_lora": main or "None",
        "support_count": support_count,
        "normalized": normalized,
    }
