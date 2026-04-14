import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from composer_policy import (  # noqa: E402
    assign_main_edit,
    build_group_profile,
    build_layer_cfg,
    compose_slot_policies,
    role_edit_profile,
    _scale_budgets,
)


class ComposerPolicyTests(unittest.TestCase):
    def test_assign_main_edit_promotes_strongest_active_slot(self):
        slots = [
            {"enabled": True, "lora": "style.safetensors", "strength": 0.6, "role": "Style"},
            {"enabled": True, "lora": "edit.safetensors", "strength": 1.2, "role": "Identity"},
        ]
        assigned = assign_main_edit(slots)
        self.assertEqual(assigned[1]["role"], "Main Edit")

    def test_role_edit_profile_adjusts_balance_by_safety(self):
        safe = role_edit_profile("Style", safety="Safe")
        strong = role_edit_profile("Style", safety="Strong")
        self.assertEqual(safe["edit_mode"], "Style Only")
        self.assertLess(safe["balance"], strong["balance"])
        self.assertEqual(safe["anatomy_profile"], "None")

    def test_build_group_profile_reflects_goal_modifier(self):
        edit = build_group_profile("Style", goal="Edit")
        generate = build_group_profile("Style", goal="Generate")
        self.assertLess(edit["sb_late"], generate["sb_late"])
        self.assertGreater(generate["db_txt"], edit["db_txt"])

    def test_build_layer_cfg_maps_sb_groups_to_early_mid_late_ranges(self):
        cfg = build_layer_cfg({
            "db_img": 0.8,
            "db_txt": 1.1,
            "sb_early": 0.7,
            "sb_mid": 0.9,
            "sb_late": 0.5,
        })
        self.assertEqual(cfg["db"]["0"]["img"], 0.8)
        self.assertEqual(cfg["db"]["7"]["txt"], 1.1)
        self.assertEqual(cfg["sb"]["0"], 0.7)
        self.assertEqual(cfg["sb"]["8"], 0.9)
        self.assertEqual(cfg["sb"]["23"], 0.5)

    def test_compose_slot_policies_normalizes_overlap_more_in_safe_mode(self):
        slots = [
            {"enabled": True, "lora": "main.safetensors", "strength": 1.8, "role": "Main Edit"},
            {"enabled": True, "lora": "style.safetensors", "strength": 1.4, "role": "Style"},
            {"enabled": True, "lora": "detail.safetensors", "strength": 1.2, "role": "Detail"},
        ]
        safe = compose_slot_policies(slots, goal="Edit", safety="Safe", auto_normalize=True)
        strong = compose_slot_policies(slots, goal="Edit", safety="Strong", auto_normalize=True)

        safe_mid = safe[0]["final_groups"]["sb_mid"]
        strong_mid = strong[0]["final_groups"]["sb_mid"]

        self.assertLess(safe_mid, strong_mid)
        self.assertTrue(any(entry["normalized"] for entry in safe))

    def test_compose_slot_policies_keeps_main_edit_stronger_than_style_when_normalized(self):
        slots = [
            {"enabled": True, "lora": "edit.safetensors", "strength": 1.5, "role": "Main Edit"},
            {"enabled": True, "lora": "style.safetensors", "strength": 1.5, "role": "Style"},
        ]
        policies = compose_slot_policies(slots, goal="Edit", safety="Balanced", auto_normalize=True)
        self.assertGreater(
            policies[0]["final_groups"]["sb_mid"],
            policies[1]["final_groups"]["sb_mid"],
        )

    def test_single_active_slot_bypasses_layer_policy(self):
        slots = [
            {"enabled": True, "lora": "edit.safetensors", "strength": 1.0, "role": "Main Edit"},
            {"enabled": False, "lora": "style.safetensors", "strength": 1.0, "role": "Style"},
        ]
        policies = compose_slot_policies(slots, goal="Edit", safety="Balanced", auto_normalize=True)
        self.assertEqual(policies[0]["layer_cfg"], {})
        self.assertEqual(policies[0]["edit_mode"], "None")
        self.assertEqual(policies[0]["anatomy_profile"], "None")

    def test_slot_level_anatomy_override_beats_role_default(self):
        slots = [
            {
                "enabled": True,
                "lora": "edit.safetensors",
                "strength": 1.0,
                "role": "Main Edit",
                "anatomy_profile": "Robot Frame Lock",
                "anatomy_strength": 0.9,
                "anatomy_strict_zero": True,
                "anatomy_custom_json": '{"db_img":0.5,"db_txt":0.5,"sb_bands":[0.5,0.5,0.5,0.5,0.5,0.5]}',
            },
        ]
        policies = compose_slot_policies(slots, goal="Edit", safety="Balanced", auto_normalize=True)
        self.assertEqual(policies[0]["anatomy_profile"], "Robot Frame Lock")
        self.assertEqual(policies[0]["anatomy_strength"], 0.9)
        self.assertTrue(policies[0]["anatomy_strict_zero"])

    def test_scale_budgets_unchanged_for_two_loras(self):
        base = {"db_img": 2.80, "sb_late": 1.80}
        self.assertEqual(_scale_budgets(base, 2), base)
        self.assertEqual(_scale_budgets(base, 1), base)

    def test_scale_budgets_reduces_for_four_loras(self):
        base = {"db_img": 2.80, "sb_late": 1.80}
        scaled = _scale_budgets(base, 4)
        self.assertAlmostEqual(scaled["db_img"], 2.24)
        self.assertAlmostEqual(scaled["sb_late"], 1.44)

    def test_scale_budgets_floors_at_half(self):
        base = {"db_img": 2.80}
        scaled = _scale_budgets(base, 10)
        self.assertAlmostEqual(scaled["db_img"], 2.80 * 0.50)

    def test_four_loras_get_tighter_budgets_than_two(self):
        slots_2 = [
            {"enabled": True, "lora": "a.safetensors", "strength": 1.0, "role": "Main Edit"},
            {"enabled": True, "lora": "b.safetensors", "strength": 1.0, "role": "Style"},
        ]
        slots_4 = slots_2 + [
            {"enabled": True, "lora": "c.safetensors", "strength": 1.0, "role": "Detail"},
            {"enabled": True, "lora": "d.safetensors", "strength": 1.0, "role": "Identity"},
        ]
        p2 = compose_slot_policies(slots_2, goal="Edit", safety="Balanced")
        p4 = compose_slot_policies(slots_4, goal="Edit", safety="Balanced")
        # With 4 LoRAs the Main Edit slot should be normalized more aggressively
        active_2 = [e for e in p2 if e["slot"]["role"] == "Main Edit"][0]
        active_4 = [e for e in p4 if e["slot"]["role"] == "Main Edit"][0]
        self.assertLessEqual(
            active_4["final_groups"]["sb_late"],
            active_2["final_groups"]["sb_late"],
        )


if __name__ == "__main__":
    unittest.main()
