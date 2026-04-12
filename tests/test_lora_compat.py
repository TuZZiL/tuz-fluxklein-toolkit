import json
import os
import shutil
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lora_meta
from edit_presets import auto_select_preset, interpolate_preset, resolve_preset_selection
from lora_compat import build_compatibility_report, build_key_map, normalize_lora_key, parse_lora_key


def write_minimal_safetensors(path: Path):
    header = json.dumps({"__metadata__": {}}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def make_test_dir(name: str) -> Path:
    path = ROOT / ".test_tmp" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class LoraCompatTests(unittest.TestCase):
    def make_analysis(self, db_img=1.0, db_txt=1.0, sb_values=None):
        sb_values = sb_values or [1.0] * 24
        return {
            "db": {i: {"img": db_img, "txt": db_txt} for i in range(8)},
            "sb": {i: sb_values[i] for i in range(24)},
            "rank": 16,
            "alpha": None,
            "layer_stats": [],
        }

    def test_normalize_alias_dot_keys(self):
        key = "transformer.transformer_blocks.0.attn.to_q.lora.down.weight"
        self.assertEqual(
            normalize_lora_key(key),
            "double_blocks.0.attn.to_q.lora_A.weight",
        )

    def test_parse_alias_dot_roles(self):
        base, role = parse_lora_key("single_blocks.0.linear1.lora.up.weight")
        self.assertEqual(base, "single_blocks.0.linear1")
        self.assertEqual(role, "lora_up")

    def test_build_key_map_includes_expected_prefix_variants(self):
        class FakeInnerModel:
            def state_dict(self):
                return {
                    "diffusion_model.double_blocks.0.img_attn.qkv.weight": object(),
                    "diffusion_model.single_blocks.0.linear1.weight": object(),
                }

        class FakeModel:
            model = FakeInnerModel()

        key_map = build_key_map(FakeModel())
        self.assertIn("diffusion_model.double_blocks.0.img_attn.qkv", key_map)
        self.assertIn("transformer.double_blocks.0.img_attn.qkv", key_map)
        self.assertIn("double_blocks.0.img_attn.qkv", key_map)
        self.assertIn("lora_unet_double_blocks_0_img_attn_qkv", key_map)

    def test_compatibility_report_counts_full_partial_and_incomplete(self):
        keys = [
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "diffusion_model.double_blocks.1.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.1.img_attn.qkv.lora_B.weight",
            "diffusion_model.single_blocks.0.linear1.lora_A.weight",
        ]
        key_map = {
            "diffusion_model.double_blocks.0.img_attn.qkv": "model.qkv.0.weight",
        }

        report = build_compatibility_report(keys, key_map)

        self.assertEqual(report["total_modules"], 2)
        self.assertEqual(report["matched_modules"], 1)
        self.assertEqual(report["skipped_modules"], 1)
        self.assertEqual(report["incomplete_modules"], 1)
        self.assertIn("diffusion_model.double_blocks.1.img_attn.qkv", report["sample_skipped"])
        self.assertIn("diffusion_model.single_blocks.0.linear1", report["sample_incomplete"])

    def test_analyse_for_node_uses_and_invalidates_cache(self):
        lora_meta._ANALYSIS_CACHE.clear()

        tmpdir = make_test_dir("lora_meta_cache")
        try:
            path = tmpdir / "mini.safetensors"
            write_minimal_safetensors(path)

            with mock.patch.object(lora_meta, "read_header", wraps=lora_meta.read_header) as wrapped:
                first = lora_meta.analyse_for_node(path)
                second = lora_meta.analyse_for_node(path)

                self.assertEqual(wrapped.call_count, 1)
                self.assertEqual(first, second)

                new_time = path.stat().st_mtime + 2
                os.utime(path, (new_time, new_time))
                third = lora_meta.analyse_for_node(path)

                self.assertEqual(wrapped.call_count, 2)
                self.assertEqual(first, third)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_read_header_rejects_impossible_header_length(self):
        tmpdir = make_test_dir("lora_meta_bad_header")
        try:
            path = tmpdir / "bad_header.safetensors"
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", 999999))
                f.write(b"{}")

            with self.assertRaises(ValueError):
                lora_meta.read_header(path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_analysis_cache_is_bounded(self):
        lora_meta._ANALYSIS_CACHE.clear()
        original_max = lora_meta._ANALYSIS_CACHE_MAXSIZE
        lora_meta._ANALYSIS_CACHE_MAXSIZE = 2
        try:
            tmpdir = make_test_dir("lora_meta_cache_bound")
            paths = []
            for i in range(3):
                path = tmpdir / f"mini_{i}.safetensors"
                write_minimal_safetensors(path)
                paths.append(path)

            for path in paths:
                lora_meta.analyse_for_node(path)

            self.assertEqual(len(lora_meta._ANALYSIS_CACHE), 2)
            oldest_key = (
                str(paths[0].resolve()),
                paths[0].stat().st_size,
                paths[0].stat().st_mtime_ns,
            )
            self.assertNotIn(oldest_key, lora_meta._ANALYSIS_CACHE)
        finally:
            lora_meta._ANALYSIS_CACHE_MAXSIZE = original_max
            shutil.rmtree(ROOT / ".test_tmp", ignore_errors=True)

    def test_auto_select_preset_prefers_preserve_face_for_uniform_full_coverage(self):
        preset, balance = auto_select_preset(self.make_analysis(), use_case="Edit")
        self.assertEqual(preset, "Preserve Face")
        self.assertGreaterEqual(balance, 0.5)

    def test_interpolate_preset_uses_intuitive_protection_scale(self):
        preset_cfg = {
            "db": {0: {"img": 0.4, "txt": 0.7}},
            "sb": {0: 0.25},
        }

        raw = interpolate_preset(preset_cfg, 0.0)
        protected = interpolate_preset(preset_cfg, 1.0)

        self.assertEqual(raw["db"][0]["img"], 1.0)
        self.assertEqual(raw["db"][0]["txt"], 1.0)
        self.assertEqual(raw["sb"][0], 1.0)
        self.assertEqual(protected["db"][0]["img"], 0.4)
        self.assertEqual(protected["db"][0]["txt"], 0.7)
        self.assertEqual(protected["sb"][0], 0.25)

    def test_auto_select_preset_picks_style_only_for_image_heavy_profile(self):
        sb = [0.92] * 24
        preset, _ = auto_select_preset(
            self.make_analysis(db_img=1.35, db_txt=0.95, sb_values=sb),
            use_case="Edit",
        )
        self.assertEqual(preset, "Style Only")

    def test_auto_select_preset_picks_preserve_body_for_late_heavy_profile(self):
        sb = [0.75] * 8 + [0.95] * 8 + [1.75] * 8
        preset, _ = auto_select_preset(
            self.make_analysis(db_img=0.95, db_txt=0.95, sb_values=sb),
            use_case="Edit",
        )
        self.assertEqual(preset, "Preserve Body")

    def test_auto_select_preset_keeps_none_for_sparse_soft_structural_profile(self):
        analysis = {
            "db": {
                0: {"img": 1.0, "txt": 0.95},
                1: {"img": 0.98, "txt": 0.94},
            },
            "sb": {0: 0.85, 1: 0.82, 2: 0.80},
            "rank": 4,
            "alpha": None,
            "layer_stats": [],
        }
        preset, balance = auto_select_preset(analysis, use_case="Edit")
        self.assertEqual(preset, "None")
        self.assertLessEqual(balance, 0.45)

    def test_auto_select_preset_generate_prefers_none_for_uniform_full_coverage(self):
        preset, balance = auto_select_preset(self.make_analysis(), use_case="Generate")
        self.assertEqual(preset, "None")
        self.assertGreaterEqual(balance, 0.50)

    def test_auto_select_preset_generate_softens_late_heavy_to_preserve_face(self):
        sb = [0.75] * 8 + [0.95] * 8 + [1.75] * 8
        preset, _ = auto_select_preset(
            self.make_analysis(db_img=0.95, db_txt=0.95, sb_values=sb),
            use_case="Generate",
        )
        self.assertEqual(preset, "Preserve Face")

    def test_manual_preset_resolution_ignores_use_case(self):
        preset_edit = resolve_preset_selection("Style Only", 0.35, use_case="Edit")
        preset_generate = resolve_preset_selection("Style Only", 0.35, use_case="Generate")
        self.assertEqual(preset_edit, ("Style Only", 0.35))
        self.assertEqual(preset_generate, ("Style Only", 0.35))


if __name__ == "__main__":
    unittest.main()
