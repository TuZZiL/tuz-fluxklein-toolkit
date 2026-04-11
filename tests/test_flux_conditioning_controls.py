import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux_conditioning_controls import (  # noqa: E402
    Flux2KleinColorAnchor,
    Flux2KleinMaskRefController,
    Flux2KleinRefLatentController,
    Flux2KleinTextRefBalance,
    _apply_mask_to_reference_latent,
    _reference_token_span,
)


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.attn_patch = None

    def clone(self):
        return copy.deepcopy(self)

    def set_model_attn1_patch(self, fn):
        self.attn_patch = fn


class FluxConditioningControlsTests(unittest.TestCase):
    def test_reference_token_span_maps_selected_reference(self):
        span = _reference_token_span({"reference_image_num_tokens": [2, 3, 4]}, 1)
        self.assertEqual(span["seq_start"], -7)
        self.assertEqual(span["seq_end"], -4)
        self.assertEqual(span["num_ref_tokens"], 3)

    def test_mask_helper_only_affects_selected_channel_band(self):
        ref = torch.ones(1, 128, 2, 2)
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        modified = _apply_mask_to_reference_latent(
            ref,
            mask,
            strength=1.0,
            invert_mask=False,
            feather=0,
            channel_mode="low",
        )

        self.assertTrue(torch.allclose(modified[:, 64:, :, :], ref[:, 64:, :, :]))
        self.assertEqual(float(modified[:, :64, 0, 1].sum().item()), 0.0)

    def test_mask_controller_updates_reference_latents(self):
        node = Flux2KleinMaskRefController()
        conditioning = [
            (
                torch.zeros(1, 4, 8),
                {"reference_latents": [torch.ones(1, 128, 2, 2)]},
            )
        ]
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        (out,) = node.apply_mask(
            conditioning,
            mask,
            strength=1.0,
            invert_mask=False,
            feather=0,
            channel_mode="high",
        )

        updated = out[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(updated[:, :64, :, :], torch.ones(1, 64, 2, 2)))
        self.assertEqual(float(updated[:, 64:, 0, 1].sum().item()), 0.0)

    def test_ref_latent_controller_scales_only_selected_reference_tokens(self):
        node = Flux2KleinRefLatentController()
        model = FakeModel()
        conditioning = [
            (
                torch.zeros(1, 8, 4),
                {"reference_latents": [torch.ones(1, 128, 2, 2)]},
            )
        ]

        model_out, _ = node.control(
            model,
            conditioning,
            strength=2.0,
            reference_index=1,
        )

        patch = model_out.attn_patch
        self.assertIsNotNone(patch)
        q = torch.zeros(1, 1, 5, 4)
        k = torch.ones(1, 1, 5, 4)
        v = torch.ones(1, 1, 5, 4)
        patched = patch(q, k, v, extra_options={"reference_image_num_tokens": [2, 3], "block_index": 0})
        self.assertTrue(torch.allclose(patched["k"][:, :, :2, :], torch.ones(1, 1, 2, 4)))
        self.assertTrue(torch.allclose(patched["k"][:, :, 2:, :], torch.full((1, 1, 3, 4), 2.0)))

    def test_text_ref_balance_scales_text_and_reference_regions(self):
        node = Flux2KleinTextRefBalance()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.ones(1, 128, 2, 2)]},
        )]

        model_out, _ = node.balance_streams(model, conditioning, balance=0.25)
        patch = model_out.attn_patch
        q = torch.zeros(1, 1, 4, 4)
        k = torch.ones(1, 1, 4, 4)
        v = torch.ones(1, 1, 4, 4)
        patched = patch(q, k, v, extra_options={"img_slice": [2, 4], "reference_image_num_tokens": [2], "block_index": 1})
        self.assertTrue(torch.allclose(patched["k"][:, :, :2, :], torch.full((1, 1, 2, 4), 0.5)))
        self.assertTrue(torch.allclose(patched["k"][:, :, 2:, :], torch.ones(1, 1, 2, 4)))

    def test_color_anchor_adds_sampler_hook(self):
        node = Flux2KleinColorAnchor()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.full((1, 128, 2, 2), 3.0)]},
        )]

        model_out = node.apply(model, conditioning, strength=0.5, ramp_curve=1.0)[0]
        hooks = model_out.model_options.get("sampler_post_cfg_function", [])
        self.assertEqual(len(hooks), 1)

        denoised = torch.zeros(1, 128, 2, 2)
        result = hooks[0]({"denoised": denoised, "sigma": torch.tensor(1.0)})
        self.assertGreater(float(result.mean().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
