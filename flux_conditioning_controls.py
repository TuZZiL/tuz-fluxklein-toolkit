"""
Companion conditioning nodes for FLUX.2 Klein.

These nodes operate on conditioning and reference-latent metadata instead of
LoRA patches. They are intentionally small, composable, and ComfyUI-friendly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _iter_conditioning_meta(conditioning):
    for item in conditioning or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        yield item[0], item[1] if isinstance(item[1], dict) else {}


def _extract_reference_latents(meta):
    ref_latents = meta.get("reference_latents", None)
    if ref_latents:
        return ref_latents

    model_conds = meta.get("model_conds", {})
    ref_latents = model_conds.get("ref_latents", None)
    if ref_latents is None:
        return None
    if hasattr(ref_latents, "cond"):
        return ref_latents.cond
    return ref_latents


def _find_reference_latent(conditioning, reference_index=0):
    for _, meta in _iter_conditioning_meta(conditioning):
        ref_latents = _extract_reference_latents(meta)
        if ref_latents is not None and reference_index < len(ref_latents):
            return ref_latents[reference_index]
    return None


def _reference_token_span(extra_options, reference_index):
    ref_tokens = extra_options.get("reference_image_num_tokens", [])
    if not ref_tokens or reference_index >= len(ref_tokens):
        return None

    total_ref = sum(ref_tokens)
    tok_start = sum(ref_tokens[:reference_index])
    tok_end = tok_start + ref_tokens[reference_index]
    seq_start = -total_ref + tok_start
    seq_end = -total_ref + tok_end
    seq_end_idx = None if seq_end == 0 else seq_end
    return {
        "total_ref": total_ref,
        "tok_start": tok_start,
        "tok_end": tok_end,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "seq_end_idx": seq_end_idx,
        "num_ref_tokens": ref_tokens[reference_index],
    }


def _spatial_fade_weights(num_tokens, ref_latent, mode, fade_strength, device):
    if mode == "none" or ref_latent is None:
        return None

    _, _, height, width = ref_latent.shape
    patch_size = 2
    h_p = (height + patch_size // 2) // patch_size
    w_p = (width + patch_size // 2) // patch_size

    y = torch.linspace(0.0, 1.0, h_p, device=device)
    x = torch.linspace(0.0, 1.0, w_p, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    if mode == "center_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = 1.0 - dist * fade_strength
    elif mode == "edges_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = (1.0 - fade_strength) + dist * fade_strength
    elif mode == "top_down":
        weights = 1.0 - yy * fade_strength
    elif mode == "left_right":
        weights = 1.0 - xx * fade_strength
    else:
        return None

    weights = weights.clamp(0.0, 5.0).flatten()
    n = weights.shape[0]
    if n > num_tokens:
        weights = weights[:num_tokens]
    elif n < num_tokens:
        weights = torch.cat([weights, torch.ones(num_tokens - n, device=device)])

    return weights


def _resize_mask_to_latent(mask, lat_h, lat_w):
    m = mask[0:1].unsqueeze(1).float()
    return F.interpolate(m, size=(lat_h, lat_w), mode="bilinear", align_corners=False)


def _feather_mask(mask, radius):
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    sigma = radius / 3.0
    axis = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - radius
    gauss_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(mask, kernel, padding=radius)
    return blurred.clamp(0.0, 1.0)


def _apply_mask_to_reference_latent(ref_latent, mask, strength, invert_mask=False, feather=0, channel_mode="all"):
    if ref_latent is None:
        return None

    ref = ref_latent.float().clone()
    _, num_ch, lat_h, lat_w = ref.shape

    spatial_mask = _resize_mask_to_latent(mask, lat_h, lat_w)
    if invert_mask:
        spatial_mask = 1.0 - spatial_mask
    if feather > 0:
        spatial_mask = _feather_mask(spatial_mask, feather)

    multiplier = 1.0 - strength * (1.0 - spatial_mask)
    if channel_mode == "low":
        ch_start, ch_end = 0, num_ch // 2
    elif channel_mode == "high":
        ch_start, ch_end = num_ch // 2, num_ch
    else:
        ch_start, ch_end = 0, num_ch

    modified = ref.clone()
    expanded = multiplier.expand(-1, ch_end - ch_start, -1, -1).to(ref.device)
    modified[:, ch_start:ch_end, :, :] = ref[:, ch_start:ch_end, :, :] * expanded
    return modified


class Flux2KleinRefLatentController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 7}),
            },
            "optional": {
                "spatial_fade": (["none", "center_out", "edges_out", "top_down", "left_right"], {"default": "none"}),
                "spatial_fade_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def control(self, model, conditioning, strength=1.0, reference_index=0, spatial_fade="none", spatial_fade_strength=0.5, debug=False):
        if strength == 0:
            return (model, conditioning)

        ref_latent = None
        if conditioning and spatial_fade != "none":
            ref_latent = _find_reference_latent(conditioning, reference_index)

        current = model.clone()

        def ref_weight_patch(q, k, v, extra_options={}, **kwargs):
            span = _reference_token_span(extra_options, reference_index)
            if span is None:
                return {}

            if spatial_fade != "none" and ref_latent is not None:
                token_w = _spatial_fade_weights(
                    span["num_ref_tokens"], ref_latent, spatial_fade, spatial_fade_strength, k.device
                )
                if token_w is not None:
                    scale = (strength * token_w).view(1, 1, -1, 1).to(k.dtype)
                else:
                    scale = strength
            else:
                scale = strength

            k = k.clone()
            v = v.clone()
            k[:, :, span["seq_start"] : span["seq_end_idx"], :] *= scale
            v[:, :, span["seq_start"] : span["seq_end_idx"], :] *= scale

            if debug:
                block_idx = extra_options.get("block_index", "?")
                print(
                    f"[RefLatentController] block={block_idx} ref_index={reference_index} "
                    f"tokens=[{span['seq_start']}:{span['seq_end']}] strength={strength:.3f}"
                )
            return {"q": q, "k": k, "v": v}

        current.set_model_attn1_patch(ref_weight_patch)
        return (current, conditioning)


class Flux2KleinTextRefBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": ("FLOAT", {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance_streams"
    CATEGORY = "conditioning/flux2klein"

    def balance_streams(self, model, conditioning, balance=0.5, debug=False):
        current = model.clone()

        if balance <= 0.5:
            text_scale = balance * 2.0
            ref_scale = 1.0
        else:
            text_scale = 1.0
            ref_scale = (1.0 - balance) * 2.0

        def balance_patch(q, k, v, extra_options={}, **kwargs):
            img_slice = extra_options.get("img_slice", None)
            ref_tokens = extra_options.get("reference_image_num_tokens", [])

            if img_slice is None and not ref_tokens:
                return {}

            k = k.clone()
            v = v.clone()

            if img_slice is not None and text_scale != 1.0:
                txt_end = img_slice[0]
                k[:, :, :txt_end, :] *= text_scale
                v[:, :, :txt_end, :] *= text_scale

            if ref_tokens and ref_scale != 1.0:
                total_ref = sum(ref_tokens)
                k[:, :, -total_ref:, :] *= ref_scale
                v[:, :, -total_ref:, :] *= ref_scale

            if debug:
                block_idx = extra_options.get("block_index", "?")
                print(
                    f"[TextRefBalance] block={block_idx} text_scale={text_scale:.3f} ref_scale={ref_scale:.3f}"
                )
            return {"q": q, "k": k, "v": v}

        current.set_model_attn1_patch(balance_patch)
        return (current, conditioning)


class Flux2KleinMaskRefController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "channel_mode": (["all", "low", "high"], {"default": "all"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/flux2klein"

    def apply_mask(self, conditioning, mask, strength=1.0, invert_mask=False, feather=0, channel_mode="all", debug=False):
        if not conditioning or strength == 0.0:
            return (conditioning,)

        output = []

        for idx, (cond_tensor, meta) in enumerate(_iter_conditioning_meta(conditioning)):
            new_meta = meta.copy()
            ref_latents = _extract_reference_latents(meta)

            if ref_latents is None or len(ref_latents) == 0:
                if debug:
                    print(f"[MaskRefController] Item {idx}: no reference_latents found")
                output.append((cond_tensor, new_meta))
                continue

            ref = ref_latents[0]
            modified = _apply_mask_to_reference_latent(
                ref,
                mask,
                strength=strength,
                invert_mask=invert_mask,
                feather=feather,
                channel_mode=channel_mode,
            )

            if modified is None:
                output.append((cond_tensor, new_meta))
                continue

            original_dtype = ref.dtype
            new_meta["reference_latents"] = [modified.to(original_dtype)]
            output.append((cond_tensor, new_meta))

            if debug:
                _, _, lat_h, lat_w = ref.shape
                print(
                    f"[MaskRefController] Item {idx} ref={ref.shape} mask={tuple(mask.shape)} "
                    f"latent={lat_h}x{lat_w} invert={invert_mask} feather={feather} channel_mode={channel_mode}"
                )

        return (output,)


class Flux2KleinColorAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "ramp_curve": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1}),
                "ref_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "channel_weights": (["uniform", "by_variance"], {"default": "uniform"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, conditioning, strength=0.5, ramp_curve=1.5, ref_index=0, channel_weights="uniform", debug=False):
        if strength == 0.0:
            return (model,)

        ref_latent = _find_reference_latent(conditioning, ref_index)
        if ref_latent is None:
            if debug:
                print("[ColorAnchor] No reference latent found in conditioning; node inactive.")
            return (model,)

        ref_means = ref_latent.float().mean(dim=(-2, -1), keepdim=True)
        ch_trust = None
        if channel_weights == "by_variance":
            spatial_var = ref_latent.float().var(dim=(-2, -1), keepdim=True)
            ch_trust = 1.0 / (1.0 + spatial_var)
            ch_trust = ch_trust / ch_trust.max().clamp(min=1e-8)

        state = {
            "sigma_max": None,
            "last_sigma_logged": None,
            "step": 0,
        }
        curve = max(ramp_curve, 1e-3)

        def color_anchor_fn(args):
            denoised = args["denoised"]
            sigma = args["sigma"]

            try:
                s = sigma.max().item()
            except Exception:
                s = float(sigma)

            if state["sigma_max"] is None or s > state["sigma_max"]:
                state["sigma_max"] = s
                state["step"] = 0

            sigma_max = state["sigma_max"]
            sigma_progress = max(0.0, min(1.0, (sigma_max - s) / sigma_max if sigma_max > 1e-6 else 0.0))

            state["step"] += 1
            step_progress = 1.0 - 0.5 ** state["step"]
            progress = max(sigma_progress, step_progress)
            effective = strength * (progress ** (1.0 / curve))

            if effective < 1e-5:
                return denoised

            ref = ref_means.to(denoised.device, dtype=denoised.dtype)
            cur = denoised.mean(dim=(-2, -1), keepdim=True)
            correction = ref - cur

            if ch_trust is not None:
                correction = correction * ch_trust.to(denoised.device, dtype=denoised.dtype)

            corrected = denoised + correction * effective

            if debug and s != state["last_sigma_logged"]:
                state["last_sigma_logged"] = s
                mean_drift = (ref - cur).abs().mean().item()
                applied = (correction * effective).abs().mean().item()
                print(
                    f"[ColorAnchor] step={state['step']} sigma={s:.4f} sigma_prog={sigma_progress:.3f} "
                    f"step_prog={step_progress:.3f} progress={progress:.3f} effective={effective:.3f} "
                    f"mean_drift={mean_drift:.5f} applied={applied:.5f}"
                )

            return corrected

        current = model.clone()
        current.model_options.setdefault("sampler_post_cfg_function", []).append(color_anchor_fn)
        return (current,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinRefLatentController": Flux2KleinRefLatentController,
    "Flux2KleinTextRefBalance": Flux2KleinTextRefBalance,
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinRefLatentController": "FLUX.2 Klein Ref Latent Controller",
    "Flux2KleinTextRefBalance": "FLUX.2 Klein Text/Ref Balance",
    "Flux2KleinMaskRefController": "FLUX.2 Klein Mask Ref Controller",
    "Flux2KleinColorAnchor": "FLUX.2 Klein Color Anchor",
}
