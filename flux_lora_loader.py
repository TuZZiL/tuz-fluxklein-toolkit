"""
FLUX LoRA Loader
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
import re
import torch
import comfy.utils
import comfy.lora
import folder_paths
import logging

logger = logging.getLogger(__name__)

N_DOUBLE = 8
N_SINGLE = 24
_MAX_SLOTS = 10


# ── Single loader ──────────────────────────────────────────────────────────────

class FluxLoraLoader:
    """
    Loads a single diffusers-format FLUX LoRA and applies it to a FLUX model.
    The JS graph widget provides per-layer strength bars for img/txt (double) and
    single blocks, serialized into the hidden `layer_strengths` widget as JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01,
                }),
                "auto_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert diffusers→native",
                    "label_off": "Direct load (native only)",
                }),
            },
            # Written by the JS graph widget — never shown as a text box
            "optional": {
                "layer_strengths":    ("STRING", {"default": "{}"}),
                # Wire from FluxLoraAutoStrength so you only pick the LoRA once
                "lora_name_override": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/FLUX"
    TITLE = "FLUX LoRA Loader"

    def load_lora(self, model, lora_name, strength_model,
                  auto_convert=True, layer_strengths="{}", lora_name_override=""):
        if strength_model == 0:
            return (model,)

        # If wired from AutoStrength, use that name — ignore the dropdown
        if lora_name_override and lora_name_override.strip():
            lora_name = lora_name_override.strip()

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        logger.info(f"[FLUX LoRA] Loading: {lora_name}  ({len(lora_sd)} keys)")

        # Parse per-layer strengths from graph widget
        layer_cfg = {}
        try:
            raw = json.loads(layer_strengths)
            if isinstance(raw, dict) and ("db" in raw or "sb" in raw):
                layer_cfg = raw
        except Exception:
            pass

        if auto_convert and self._is_diffusers_format(lora_sd):
            logger.info("[FLUX LoRA] Detected diffusers format — converting")
            lora_sd = self._convert_to_native(lora_sd)

        # Bake per-layer strength into lora_B before patching
        if layer_cfg:
            # Grab a sample lora_B tensor norm BEFORE scaling for verification
            sample_key = next((k for k in lora_sd if k.endswith(".lora_B.weight") or k.endswith(".lora_up.weight")), None)
            norm_before = float(lora_sd[sample_key].float().norm().item()) if sample_key else None

            lora_sd = self._apply_layer_strengths(lora_sd, layer_cfg, strength_model)

            # Norm AFTER — if these differ, scaling was applied
            norm_after = float(lora_sd[sample_key].float().norm().item()) if sample_key else None
            if norm_before is not None:
                ratio = norm_after / norm_before if norm_before > 1e-8 else 0
                logger.warning(
                    f"[FLUX LoRA] ✅ AUTO-STRENGTH APPLIED — "
                    f"sample tensor '{sample_key}' "
                    f"norm: {norm_before:.6f} → {norm_after:.6f} "
                    f"(ratio={ratio:.4f}, expected≠1.0 if scaling active)"
                )

        key_map = self._build_key_map(model)
        patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
        logger.info(f"[FLUX LoRA] Applied {len(patch_dict)} patches")

        model_lora = model.clone()
        # Per-layer scaling already baked → pass 1.0 to avoid double-scaling
        effective_strength = 1.0 if layer_cfg else strength_model
        model_lora.add_patches(patch_dict, strength_patch=effective_strength, strength_model=1.0)

        return (model_lora,)

    # ── Format detection ───────────────────────────────────────────────────────

    @staticmethod
    def _is_diffusers_format(lora_sd):
        markers = (
            ".to_q.",               # separate Q proj
            ".to_k.",               # separate K proj
            ".to_v.",               # separate V proj
            ".add_q_proj.",         # FLUX txt Q
            ".add_k_proj.",         # FLUX txt K
            ".add_v_proj.",         # FLUX txt V
            "transformer_blocks.",  # diffusers double-block prefix
            "single_transformer_blocks.",  # diffusers single-block prefix
            ".ff.net.",             # diffusers FFN naming
            ".ff_context.",         # FLUX txt FFN
            ".to_add_out.",         # FLUX txt out proj
            ".proj_mlp.",           # FLUX single block mlp gate
        )
        return any(m in k for k in lora_sd for m in markers)

    # ── Conversion: diffusers → native ─────────────────────────────────────────

    def _convert_to_native(self, lora_sd):
        norm = self._normalize_keys(lora_sd)
        out = {}
        done = set()

        for i in range(N_DOUBLE):
            db = f"double_blocks.{i}"

            # img stream: fuse to_q + to_k + to_v → img_attn.qkv
            self._fuse_qkv(
                norm, out, done,
                q=f"{db}.attn.to_q",
                k=f"{db}.attn.to_k",
                v=f"{db}.attn.to_v",
                dst=f"diffusion_model.{db}.img_attn.qkv",
            )

            # txt stream: fuse add_q_proj + add_k_proj + add_v_proj → txt_attn.qkv
            self._fuse_qkv(
                norm, out, done,
                q=f"{db}.attn.add_q_proj",
                k=f"{db}.attn.add_k_proj",
                v=f"{db}.attn.add_v_proj",
                dst=f"diffusion_model.{db}.txt_attn.qkv",
            )

            # Simple remaps (output projections + FFN)
            for src, dst in [
                (f"{db}.attn.to_out.0",        f"diffusion_model.{db}.img_attn.proj"),
                (f"{db}.attn.to_add_out",       f"diffusion_model.{db}.txt_attn.proj"),
                (f"{db}.ff.net.0.proj",         f"diffusion_model.{db}.img_mlp.0"),
                (f"{db}.ff.net.2",              f"diffusion_model.{db}.img_mlp.2"),
                (f"{db}.ff_context.net.0.proj", f"diffusion_model.{db}.txt_mlp.0"),
                (f"{db}.ff_context.net.2",      f"diffusion_model.{db}.txt_mlp.2"),
            ]:
                self._remap(norm, out, done, src, dst)

        for i in range(N_SINGLE):
            sb = f"single_blocks.{i}"

            # linear1 = fused [q, k, v, mlp_gate_up] — 36864 = 12288 + 24576
            self._fuse_linear1(norm, out, done, sb)

            # linear2 = proj_out
            self._remap(norm, out, done,
                        f"{sb}.proj_out",
                        f"diffusion_model.{sb}.linear2")

        # Pass through anything not processed (already-native keys, unknowns)
        for key, val in norm.items():
            if key not in done:
                out[key] = val

        n_converted = sum(1 for k in done if k in norm)
        logger.info(f"[FLUX LoRA] Converted {n_converted} diffusers keys → {len(out)} native keys")
        return out

    @staticmethod
    def _normalize_keys(lora_sd):
        """
        Strip prefix variations and remap diffusers layer names to native,
        so all keys are in the form  double_blocks.{i}.* / single_blocks.{i}.*
        """
        out = {}
        for key, val in lora_sd.items():
            k = key
            # Strip leading model prefix
            for pfx in ("transformer.", "diffusion_model.", "unet."):
                if k.startswith(pfx):
                    k = k[len(pfx):]
                    break
            # Remap diffusers layer names to ComfyUI-native
            k = re.sub(r'^transformer_blocks\.',        'double_blocks.', k)
            k = re.sub(r'^single_transformer_blocks\.', 'single_blocks.', k)
            # Normalize lora_down / lora_up → lora_A / lora_B
            k = k.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
            out[k] = val
        return out

    @staticmethod
    def _alpha_scale(norm, base):
        """Return alpha/rank for one LoRA component (defaults to 1.0 if absent)."""
        down_key  = f"{base}.lora_A.weight"
        alpha_key = f"{base}.alpha"
        if alpha_key in norm and down_key in norm:
            rank = norm[down_key].shape[0]
            if rank > 0:
                return float(norm[alpha_key]) / rank
        return 1.0

    def _fuse_qkv(self, norm, out, done, q, k, v, dst):
        """
        Fuse three separate LoRAs (Q, K, V) into one block-diagonal LoRA.

        ΔW_qkv = [B_q@A_q ; B_k@A_k ; B_v@A_v]

        Represented as:
          A_fused = [A_q ; A_k ; A_v]              shape [3r, in]
          B_fused = block_diag(B_q, B_k, B_v)      shape [3·out, 3r]

        alpha/rank scale is pre-baked into each B_i so the stored alpha equals
        the rank (→ effective scale = 1.0, no double-scaling at apply time).
        """
        keys_A = [f"{b}.lora_A.weight" for b in (q, k, v)]
        keys_B = [f"{b}.lora_B.weight" for b in (q, k, v)]

        if not all(kk in norm for kk in keys_A + keys_B):
            return

        A_q, A_k, A_v = [norm[kk] for kk in keys_A]
        B_q, B_k, B_v = (norm[kk] * self._alpha_scale(norm, b)
                          for kk, b in zip(keys_B, (q, k, v)))

        r_q, r_k, r_v   = A_q.shape[0], A_k.shape[0], A_v.shape[0]
        o_q, o_k, o_v   = B_q.shape[0], B_k.shape[0], B_v.shape[0]
        r_total = r_q + r_k + r_v
        o_total = o_q + o_k + o_v

        A_fused = torch.cat([A_q, A_k, A_v], dim=0)  # [3r, in]

        B_fused = torch.zeros(o_total, r_total, dtype=B_q.dtype, device=B_q.device)
        B_fused[0         : o_q,          0         : r_q         ] = B_q
        B_fused[o_q       : o_q + o_k,   r_q       : r_q + r_k   ] = B_k
        B_fused[o_q + o_k : o_total,     r_q + r_k : r_total     ] = B_v

        out[f"{dst}.lora_A.weight"] = A_fused
        out[f"{dst}.lora_B.weight"] = B_fused
        # alpha == rank  →  alpha/rank == 1.0  (scaling already baked into B)
        out[f"{dst}.alpha"] = torch.tensor(float(r_total))

        for kk in keys_A + keys_B:
            done.add(kk)
        for b in (q, k, v):
            done.add(f"{b}.alpha")   # mark as consumed even if absent

    def _fuse_linear1(self, norm, out, done, sb_base):
        """
        Fuse single-block components into linear1 (same block-diag logic as QKV).

        linear1 shape: [36864, 4096]  =  [q+k+v=12288  +  mlp_gate_up=24576,  in=4096]
        """
        components = [
            (f"{sb_base}.attn.to_q",  "q"),
            (f"{sb_base}.attn.to_k",  "k"),
            (f"{sb_base}.attn.to_v",  "v"),
            (f"{sb_base}.proj_mlp",   "mlp"),
        ]
        dst = f"diffusion_model.{sb_base}.linear1"

        present = [
            (base, label)
            for base, label in components
            if f"{base}.lora_A.weight" in norm and f"{base}.lora_B.weight" in norm
        ]
        if not present:
            return

        A_list, B_scaled = [], []
        for base, _ in present:
            A = norm[f"{base}.lora_A.weight"]
            B = norm[f"{base}.lora_B.weight"] * self._alpha_scale(norm, base)
            A_list.append(A)
            B_scaled.append(B)
            done.update([f"{base}.lora_A.weight", f"{base}.lora_B.weight", f"{base}.alpha"])

        ranks  = [A.shape[0] for A in A_list]
        outs   = [B.shape[0] for B in B_scaled]
        r_total, o_total = sum(ranks), sum(outs)

        A_fused = torch.cat(A_list, dim=0)
        B_fused = torch.zeros(o_total, r_total,
                               dtype=B_scaled[0].dtype, device=B_scaled[0].device)
        r_off, o_off = 0, 0
        for A, B in zip(A_list, B_scaled):
            r, o = A.shape[0], B.shape[0]
            B_fused[o_off : o_off + o, r_off : r_off + r] = B
            r_off += r
            o_off += o

        out[f"{dst}.lora_A.weight"] = A_fused
        out[f"{dst}.lora_B.weight"] = B_fused
        out[f"{dst}.alpha"] = torch.tensor(float(r_total))

    @staticmethod
    def _remap(norm, out, done, src_base, dst_base):
        """Copy a LoRA pair (lora_A / lora_B / alpha) from src to dst key name."""
        key_A = f"{src_base}.lora_A.weight"
        key_B = f"{src_base}.lora_B.weight"
        if key_A not in norm or key_B not in norm:
            return
        out[f"{dst_base}.lora_A.weight"] = norm[key_A]
        out[f"{dst_base}.lora_B.weight"] = norm[key_B]
        done.update([key_A, key_B])
        alpha_key = f"{src_base}.alpha"
        if alpha_key in norm:
            out[f"{dst_base}.alpha"] = norm[alpha_key]
            done.add(alpha_key)

    # ── Per-layer strength scaling ─────────────────────────────────────────────

    @staticmethod
    def _apply_layer_strengths(lora_sd, layer_cfg, global_strength):
        """
        Scale lora_B tensors per-layer before patching.

        layer_cfg format (from the JS graph widget):
          {
            "db": { "0": {"img": 1.2, "txt": 0.8}, "1": {...}, ... },
            "sb": { "0": 0.9, "1": 1.1, ... }
          }
        """
        if abs(global_strength) < 1e-8:
            return lora_sd

        db_cfg = {str(k): v for k, v in layer_cfg.get("db", {}).items()}
        sb_cfg = {str(k): v for k, v in layer_cfg.get("sb", {}).items()}

        scaled = {}
        for key, tensor in lora_sd.items():
            if not key.endswith(".lora_B.weight"):
                scaled[key] = tensor
                continue

            parts = key.split(".")
            target = None

            for i, p in enumerate(parts):
                if p == "double_blocks" and i + 1 < len(parts):
                    idx = parts[i + 1]
                    if idx in db_cfg:
                        cfg = db_cfg[idx]
                        is_txt = any(x in parts for x in ("txt_attn", "txt_mlp"))
                        side = "txt" if is_txt else "img"
                        target = (cfg.get(side, global_strength)
                                  if isinstance(cfg, dict) else float(cfg))
                    break
                if p == "single_blocks" and i + 1 < len(parts):
                    idx = parts[i + 1]
                    if idx in sb_cfg:
                        target = float(sb_cfg[idx])
                    break

            if target is not None:
                scaled[key] = tensor * (target / global_strength)
            else:
                scaled[key] = tensor

        return scaled

    # ── Key map ────────────────────────────────────────────────────────────────

    def _build_key_map(self, model):
        """
        Build a dict: {lora_key_base → model_state_dict_key} to handle the
        various prefix conventions a LoRA file might use.
        """
        key_map = {}
        for model_key in model.model.state_dict().keys():
            if not model_key.endswith(".weight"):
                continue

            base = model_key[: -len(".weight")]
            bare = (base[len("diffusion_model."):]
                    if base.startswith("diffusion_model.") else base)

            # Register with multiple prefix styles
            for pfx in ("diffusion_model.", "transformer.", ""):
                key_map[f"{pfx}{bare}"] = model_key

            # Kohya-style  (dots → underscores, lora_unet_ prefix)
            key_map["lora_unet_" + bare.replace(".", "_")] = model_key

        logger.info(f"[FLUX LoRA] Key map: {len(key_map)} entries")
        return key_map


# ── Stack ──────────────────────────────────────────────────────────────────────

class FluxLoraStack(FluxLoraLoader):
    """
    Apply up to 10 FLUX LoRAs in sequence.
    Each slot has its own strength, on/off toggle, and auto-convert flag.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        optional = {}
        for i in range(1, _MAX_SLOTS + 1):
            optional[f"lora_{i}"]     = (loras,)
            optional[f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01,
            })
            optional[f"enabled_{i}"]  = ("BOOLEAN", {
                "default": True, "label_on": "On", "label_off": "Off",
            })
            optional[f"convert_{i}"]  = ("BOOLEAN", {
                "default": True, "label_on": "Auto-convert", "label_off": "Direct",
            })
        return {
            "required": {"model": ("MODEL",)},
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "loaders/FLUX"
    TITLE = "FLUX LoRA Stack"

    def load_loras(self, model, **kwargs):
        key_map = self._build_key_map(model)
        current = model

        for i in range(1, _MAX_SLOTS + 1):
            name     = kwargs.get(f"lora_{i}",     "None")
            strength = kwargs.get(f"strength_{i}", 1.0)
            enabled  = kwargs.get(f"enabled_{i}",  True)
            convert  = kwargs.get(f"convert_{i}",  True)

            if not enabled or name == "None" or strength == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", name)
            lora_sd   = comfy.utils.load_torch_file(lora_path, safe_load=True)
            logger.info(f"[FLUX LoRA Stack] Slot {i}: {name}  ({len(lora_sd)} keys)")

            if convert and self._is_diffusers_format(lora_sd):
                lora_sd = self._convert_to_native(lora_sd)

            patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
            next_model = current.clone()
            next_model.add_patches(patch_dict, strength_patch=strength, strength_model=1.0)
            current = next_model

        return (current,)


# ── Exports ───────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FluxLoraLoader": FluxLoraLoader,
    "FluxLoraStack":  FluxLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraLoader": "FLUX LoRA Loader",
    "FluxLoraStack":  "FLUX LoRA Stack",
}
