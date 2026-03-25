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


if __name__ == "__main__":
    unittest.main()
