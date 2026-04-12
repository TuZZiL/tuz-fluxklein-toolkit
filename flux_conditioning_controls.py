"""
Companion conditioning nodes for FLUX.2 Klein.

These nodes operate on conditioning and reference-latent metadata instead of
LoRA patches. They are intentionally small, composable, and ComfyUI-friendly.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - package vs direct import
    from .conditioning_common import (
        apply_preserve_blend,
        clone_meta,
        dampen_toward_neutral,
        get_reference_latents,
        reference_indices,
        reference_token_spans,
        set_reference_latents,
    )
    from .conditioning_reference import (
        apply_mask_to_reference_latent as _apply_mask_to_reference_latent_impl,
        apply_masked_reference_mix,
        mix_reference_latent,
        rebalance_reference_appearance,
    )
except ImportError:  # pragma: no cover
    from conditioning_common import (
        apply_preserve_blend,
        clone_meta,
        dampen_toward_neutral,
        get_reference_latents,
        reference_indices,
        reference_token_spans,
        set_reference_latents,
    )
    from conditioning_reference import (
        apply_mask_to_reference_latent as _apply_mask_to_reference_latent_impl,
        apply_masked_reference_mix,
        mix_reference_latent,
        rebalance_reference_appearance,
    )


def _iter_conditioning_meta(conditioning):
    for item in conditioning or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        yield item[0], item[1] if isinstance(item[1], dict) else {}


def _extract_reference_latents(meta):
    return get_reference_latents(meta)


def _find_reference_latent(conditioning, reference_index=0):
    for _, meta in _iter_conditioning_meta(conditioning):
        ref_latents = _extract_reference_latents(meta)
        if not ref_latents:
            continue
        indices = reference_indices(len(ref_latents), reference_index)
        if indices:
            return ref_latents[indices[0]]
    return None


def _reference_token_span(extra_options, reference_index):
    spans = reference_token_spans(extra_options, reference_index)
    if not spans:
        return None
    return spans[0]


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


def _apply_mask_to_reference_latent(
    ref_latent,
    mask,
    strength,
    invert_mask=False,
    feather=0,
    channel_mode="all",
    channel_mask_start=0,
    channel_mask_end=0,
):
    return _apply_mask_to_reference_latent_impl(
        ref_latent,
        mask,
        strength=strength,
        invert_mask=invert_mask,
        feather=feather,
        channel_mode=channel_mode,
        channel_start=int(channel_mask_start) if channel_mask_end > channel_mask_start else None,
        channel_end=int(channel_mask_end) if channel_mask_end > channel_mask_start else None,
    )


class Flux2KleinRefLatentController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05,
                    "tooltip": "How strongly to preserve the selected reference tokens inside attention. Higher values keep the reference more dominant.",
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": -1, "max": 63,
                    "tooltip": "Which reference to target. Use -1 to target all references instead of only one.",
                }),
            },
            "optional": {
                "spatial_fade": (["none", "center_out", "edges_out", "top_down", "left_right"], {
                    "default": "none",
                    "tooltip": "Optional spatial bias for the preserved reference signal. Useful when only some image regions should stay stronger.",
                }),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly the spatial fade should bias the preserved reference regions.",
                }),
                "appearance_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Scale the coarse, low-frequency appearance signal before attention-path preservation.",
                }),
                "detail_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Scale the fine-detail component before attention-path preservation. Lower values loosen rigid texture/detail lock.",
                }),
                "blur_radius": ("INT", {
                    "default": 2, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Blur radius used to split coarse appearance from fine detail.",
                }),
                "channel_mask_start": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Optional channel range start for appearance/detail rebalance. Leave 0/0 to use the full latent channel range.",
                }),
                "channel_mask_end": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Optional channel range end for appearance/detail rebalance. Leave 0/0 to use the full latent channel range.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def control(
        self,
        model,
        conditioning,
        strength=1.0,
        reference_index=0,
        spatial_fade="none",
        spatial_fade_strength=0.5,
        appearance_scale=1.0,
        detail_scale=1.0,
        blur_radius=2,
        channel_mask_start=0,
        channel_mask_end=0,
        debug=False,
    ):
        if strength == 0:
            return (model, conditioning)

        updated_conditioning = []
        rebalance_needed = appearance_scale != 1.0 or detail_scale != 1.0
        for cond_tensor, meta in _iter_conditioning_meta(conditioning):
            new_meta = clone_meta(meta)
            refs = get_reference_latents(new_meta)
            if refs and rebalance_needed:
                indices = reference_indices(len(refs), reference_index)
                if indices:
                    new_refs = []
                    for ref_idx, ref in enumerate(refs):
                        if ref_idx in indices:
                            new_refs.append(
                                rebalance_reference_appearance(
                                    ref,
                                    appearance_scale=float(appearance_scale),
                                    detail_scale=float(detail_scale),
                                    blur_radius=int(blur_radius),
                                    channel_start=int(channel_mask_start),
                                    channel_end=int(channel_mask_end) if channel_mask_end > channel_mask_start else ref.shape[1],
                                )
                            )
                        else:
                            new_refs.append(ref.clone())
                    new_meta = set_reference_latents(new_meta, new_refs)
            updated_conditioning.append((cond_tensor, new_meta))
        conditioning = updated_conditioning

        ref_latent = None
        if conditioning and spatial_fade != "none":
            ref_latent = _find_reference_latent(conditioning, reference_index)

        current = model.clone()

        def ref_weight_patch(q, k, v, extra_options={}, **kwargs):
            spans = reference_token_spans(extra_options, reference_index)
            if not spans:
                return {}

            k = k.clone()
            v = v.clone()
            for span in spans:
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
                "balance": ("FLOAT", {
                    "default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001,
                    "tooltip": "0.0 = keep reference stronger, 1.0 = let text take over more aggressively.",
                }),
            },
            "optional": {
                "balance_mode": (["attn_patch", "latent_mix"], {
                    "default": "attn_patch",
                    "tooltip": "attn_patch changes reference strength during attention. latent_mix directly reduces reference latent influence before sampling.",
                }),
                "target_reference_index": ("INT", {
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "Which reference to target. Use -1 to affect all references.",
                }),
                "replace_mode": (["zeros", "gaussian_noise", "channel_mean", "lowpass_reference"], {
                    "default": "zeros",
                    "tooltip": "How latent_mix should weaken the reference: remove it, replace it with noise, keep only the mean, or keep only coarse low-frequency structure.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance_streams"
    CATEGORY = "conditioning/flux2klein"

    def balance_streams(self, model, conditioning, balance=0.5, balance_mode="attn_patch", target_reference_index=-1, replace_mode="zeros", debug=False):
        current = model.clone()
        updated_conditioning = []

        if balance <= 0.5:
            text_scale = balance * 2.0
            ref_scale = 1.0
        else:
            text_scale = 1.0
            ref_scale = (1.0 - balance) * 2.0

        if balance_mode == "latent_mix" and ref_scale != 1.0:
            for cond_tensor, meta in _iter_conditioning_meta(conditioning):
                new_meta = clone_meta(meta)
                refs = get_reference_latents(new_meta)
                if refs:
                    indices = reference_indices(len(refs), target_reference_index)
                    if indices:
                        new_refs = []
                        for ref_idx, ref in enumerate(refs):
                            if ref_idx in indices:
                                new_refs.append(
                                    mix_reference_latent(
                                        ref,
                                        reference_keep=ref_scale,
                                        replace_mode=replace_mode,
                                        channel_start=0,
                                        channel_end=ref.shape[1],
                                        spatial_fade="none",
                                        spatial_fade_strength=0.0,
                                    )
                                )
                            else:
                                new_refs.append(ref.clone())
                        new_meta = set_reference_latents(new_meta, new_refs)
                updated_conditioning.append((cond_tensor, new_meta))
            conditioning = updated_conditioning

        def balance_patch(q, k, v, extra_options={}, **kwargs):
            img_slice = extra_options.get("img_slice", None)
            ref_spans = reference_token_spans(extra_options, target_reference_index)

            if img_slice is None and not ref_spans:
                return {}

            k = k.clone()
            v = v.clone()

            if img_slice is not None and text_scale != 1.0:
                txt_end = img_slice[0]
                k[:, :, :txt_end, :] *= text_scale
                v[:, :, :txt_end, :] *= text_scale

            if balance_mode == "attn_patch" and ref_spans and ref_scale != 1.0:
                for span in ref_spans:
                    k[:, :, span["seq_start"] : span["seq_end_idx"], :] *= ref_scale
                    v[:, :, span["seq_start"] : span["seq_end_idx"], :] *= ref_scale

            if debug:
                block_idx = extra_options.get("block_index", "?")
                print(
                    f"[TextRefBalance] block={block_idx} mode={balance_mode} text_scale={text_scale:.3f} ref_scale={ref_scale:.3f}"
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
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly the mask should protect or modify the selected reference regions.",
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask so the untouched area becomes the affected area.",
                }),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Softens mask edges before applying the operation.",
                }),
                "channel_mode": (["all", "low", "high"], {
                    "default": "all",
                    "tooltip": "Which half of the latent channels to affect. Use all unless you are deliberately separating coarse and fine behavior.",
                }),
                "channel_mask_start": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Optional explicit channel range start. Leave 0/0 to fall back to channel_mode.",
                }),
                "channel_mask_end": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Optional explicit channel range end. Leave 0/0 to fall back to channel_mode.",
                }),
                "mask_action": (["scale", "mix"], {
                    "default": "scale",
                    "tooltip": "scale keeps the same reference and dampens it. mix replaces masked regions with a different latent signal.",
                }),
                "reference_keep": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used in mix mode. 1.0 keeps the original reference, 0.0 fully replaces the masked region.",
                }),
                "replace_mode": (["zeros", "gaussian_noise", "channel_mean", "lowpass_reference"], {
                    "default": "zeros",
                    "tooltip": "What to put into masked regions when mask_action=mix.",
                }),
                "target_reference_index": ("INT", {
                    "default": 0, "min": -1, "max": 63, "step": 1,
                    "tooltip": "Which reference to target. Use -1 to affect all references.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/flux2klein"

    def apply_mask(
        self,
        conditioning,
        mask,
        strength=1.0,
        invert_mask=False,
        feather=0,
        channel_mode="all",
        channel_mask_start=0,
        channel_mask_end=0,
        mask_action="scale",
        reference_keep=0.5,
        replace_mode="zeros",
        target_reference_index=0,
        debug=False,
    ):
        if not conditioning or strength == 0.0:
            return (conditioning,)

        output = []

        for idx, (cond_tensor, meta) in enumerate(_iter_conditioning_meta(conditioning)):
            new_meta = meta.copy()
            ref_latents = _extract_reference_latents(meta)

            if not ref_latents:
                if debug:
                    print(f"[MaskRefController] Item {idx}: no reference_latents found")
                output.append((cond_tensor, new_meta))
                continue

            indices = reference_indices(len(ref_latents), target_reference_index)
            if not indices:
                output.append((cond_tensor, new_meta))
                continue

            modified_refs = []
            for ref_idx, ref in enumerate(ref_latents):
                if ref_idx not in indices:
                    modified_refs.append(ref.clone())
                    continue

                if mask_action == "mix":
                    modified = apply_masked_reference_mix(
                        ref,
                        mask,
                        strength=strength,
                        reference_keep=float(reference_keep),
                        replace_mode=replace_mode,
                        invert_mask=invert_mask,
                        feather=feather,
                        channel_mode=channel_mode,
                        channel_start=int(channel_mask_start) if channel_mask_end > channel_mask_start else None,
                        channel_end=int(channel_mask_end) if channel_mask_end > channel_mask_start else None,
                    )
                else:
                    modified = _apply_mask_to_reference_latent(
                        ref,
                        mask,
                        strength=strength,
                        invert_mask=invert_mask,
                        feather=feather,
                        channel_mode=channel_mode,
                        channel_mask_start=channel_mask_start,
                        channel_mask_end=channel_mask_end,
                    )

                if modified is None:
                    modified_refs.append(ref.clone())
                    continue

                modified_refs.append(modified.to(ref.dtype))

            new_meta["reference_latents"] = modified_refs
            output.append((cond_tensor, new_meta))

            if debug:
                _, _, lat_h, lat_w = ref_latents[0].shape
                print(
                    f"[MaskRefController] Item {idx} refs={len(ref_latents)} action={mask_action} "
                    f"mask={tuple(mask.shape)} latent={lat_h}x{lat_w} invert={invert_mask} feather={feather} channel_mode={channel_mode}"
                )

        return (output,)


class Flux2KleinColorAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to pull sampled colors back toward the selected reference palette.",
                }),
            },
            "optional": {
                "ramp_curve": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1,
                    "tooltip": "How quickly color correction ramps in during sampling. Higher values hold correction back longer.",
                }),
                "ref_index": ("INT", {
                    "default": 0, "min": -1, "max": 63,
                    "tooltip": "Which reference to use for color anchoring. Use -1 to average all references together.",
                }),
                "channel_weights": (["uniform", "by_variance"], {
                    "default": "uniform",
                    "tooltip": "uniform treats all channels equally. by_variance trusts stable channels more and noisy channels less.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, conditioning, strength=0.5, ramp_curve=1.5, ref_index=0, channel_weights="uniform", debug=False):
        if strength == 0.0:
            return (model,)

        ref_latent = None
        for _, meta in _iter_conditioning_meta(conditioning):
            refs = _extract_reference_latents(meta)
            if not refs:
                continue
            indices = reference_indices(len(refs), ref_index)
            if not indices:
                continue
            try:
                selected = [refs[i].to(torch.float32) for i in indices]
                ref_latent = torch.stack(selected, dim=0).mean(dim=0).to(dtype=refs[indices[0]].dtype)
            except Exception:
                ref_latent = refs[indices[0]]
            break
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
    "Flux2KleinRefLatentController": "TUZ FLUX.2 Klein Ref Latent Controller",
    "Flux2KleinTextRefBalance": "TUZ FLUX.2 Klein Text/Ref Balance",
    "Flux2KleinMaskRefController": "TUZ FLUX.2 Klein Mask Ref Controller",
    "Flux2KleinColorAnchor": "TUZ FLUX.2 Klein Color Anchor",
}
