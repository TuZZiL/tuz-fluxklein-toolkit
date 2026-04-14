import sys
import types
import unittest
from unittest import mock
from pathlib import Path
import importlib.util
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from anatomy_profiles import expand_profile, interpolate_profile, resolve_profile  # noqa: E402


def _import_apply_anatomy_profile():
    """
    Import lora_pipeline in an isolated module namespace with lightweight stubs.
    This avoids polluting global sys.modules with fake numpy/torch/comfy packages.
    """
    comfy = types.ModuleType("comfy")
    comfy_lora = types.ModuleType("comfy.lora")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy.lora = comfy_lora
    comfy.utils = comfy_utils

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda *_args, **_kwargs: "dummy.safetensors"

    numpy = types.ModuleType("numpy")
    numpy.mean = lambda values: sum(values) / len(values) if values else 0.0

    torch = types.ModuleType("torch")

    stubs = {
        "comfy": comfy,
        "comfy.lora": comfy_lora,
        "comfy.utils": comfy_utils,
        "folder_paths": folder_paths,
        "numpy": numpy,
        "torch": torch,
    }

    module_name = f"_test_lora_pipeline_{uuid.uuid4().hex}"
    module_path = ROOT / "lora_pipeline.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs, clear=False):
        spec.loader.exec_module(module)
    return module.apply_anatomy_profile


class AnatomyProfileTests(unittest.TestCase):
    def test_expand_profile_maps_six_sb_bands_to_all_single_blocks(self):
        expanded = expand_profile({
            "db_img": 0.5,
            "db_txt": 0.7,
            "sb_bands": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "strict_zero": {"db": [0], "sb": [1]},
        })
        self.assertEqual(expanded["db"]["0"]["img"], 0.5)
        self.assertEqual(expanded["db"]["7"]["txt"], 0.7)
        self.assertEqual(expanded["sb"]["0"], 0.1)
        self.assertEqual(expanded["sb"]["4"], 0.2)
        self.assertEqual(expanded["sb"]["8"], 0.3)
        self.assertEqual(expanded["sb"]["12"], 0.4)
        self.assertEqual(expanded["sb"]["16"], 0.5)
        self.assertEqual(expanded["sb"]["23"], 0.6)
        self.assertEqual(expanded["strict_zero"]["db"], [0])
        self.assertEqual(expanded["strict_zero"]["sb"], [1])

    def test_interpolate_profile_uses_neutral_to_profile_strength_scale(self):
        expanded = expand_profile({
            "db_img": 0.4,
            "db_txt": 0.8,
            "sb_bands": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "strict_zero": {"db": [], "sb": []},
        })
        raw = interpolate_profile(expanded, 0.0)
        protected = interpolate_profile(expanded, 1.0)
        self.assertEqual(raw["db"]["0"]["img"], 1.0)
        self.assertEqual(raw["db"]["0"]["txt"], 1.0)
        self.assertEqual(raw["sb"]["0"], 1.0)
        self.assertEqual(protected["db"]["0"]["img"], 0.4)
        self.assertEqual(protected["db"]["0"]["txt"], 0.8)
        self.assertAlmostEqual(protected["sb"]["0"], 0.2)

    def test_resolve_profile_accepts_custom_json(self):
        resolved = resolve_profile(
            "Custom",
            strength=1.0,
            custom_json='{"db_img":0.6,"db_txt":0.7,"sb_bands":[0.5,0.5,0.5,0.5,0.5,0.5],"strict_zero":{"db":[1],"sb":[2]}}',
        )
        self.assertEqual(resolved["db"]["0"]["img"], 0.6)
        self.assertEqual(resolved["strict_zero"]["db"], [1])
        self.assertEqual(resolved["strict_zero"]["sb"], [2])

    def test_apply_anatomy_profile_scales_and_strict_zeros_targeted_layers(self):
        apply_anatomy_profile = _import_apply_anatomy_profile()
        lora_sd = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": 10.0,
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight": 10.0,
            "diffusion_model.single_blocks.0.linear1.lora_B.weight": 10.0,
            "diffusion_model.single_blocks.6.linear1.lora_B.weight": 10.0,
            "diffusion_model.single_blocks.12.linear1.lora_B.weight": 10.0,
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": 5.0,
        }
        cfg = {
            "db": {"0": {"img": 0.5, "txt": 0.8}},
            "sb": {"0": 0.3, "6": 0.6, "12": 0.9},
            "strict_zero": {"db": [0], "sb": [0]},
        }
        out = apply_anatomy_profile(lora_sd, cfg, strict_zero=True)
        self.assertEqual(out["diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight"], 0.0)
        self.assertEqual(out["diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight"], 0.0)
        self.assertEqual(out["diffusion_model.single_blocks.0.linear1.lora_B.weight"], 0.0)
        self.assertEqual(out["diffusion_model.single_blocks.6.linear1.lora_B.weight"], 6.0)
        self.assertEqual(out["diffusion_model.single_blocks.12.linear1.lora_B.weight"], 9.0)
        self.assertEqual(out["diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"], 5.0)


if __name__ == "__main__":
    unittest.main()
