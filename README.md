# ComfyUI FLUX.2 Klein LoRA Loader

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Architecture-aware LoRA loading for **FLUX.2 Klein** (9B) in ComfyUI.

[Українська версія](README_UA.md)

---

## The Problem

Most LoRAs are trained with HuggingFace tools and saved in **diffusers format**. The standard ComfyUI LoRA loader **silently drops** most of these weights on Klein 9B — your LoRA loads, but barely works or looks wrong.

On top of that, when you use LoRAs for **image editing** (changing clothes, adding accessories, style transfer), the LoRA often **destroys the face** or changes body proportions.

## The Solution

This node pack does two things:

1. **Auto-converts** any LoRA format to work correctly with Klein 9B
2. **Protects what you want to keep** — face, body, or style — via one dropdown

Just pick a preset, drop in your LoRA, and go.

---

## Quick Start (2 minutes)

### Step 1: Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TuZZiL/Comfyui-flux2klein-Lora-loader.git
# Restart ComfyUI
```

Requires `numpy` (usually already installed with ComfyUI).
For `TUZ Klein Edit Composite`, also install: `pip install opencv-python-headless`

### Step 2: Add the node

Find **TUZ FLUX LoRA Loader** in the node menu under `loaders/FLUX`. Connect your `MODEL` input.

### Step 3: Pick your LoRA and go

1. Select your LoRA file
2. Set `edit_mode` → **Auto** (recommended starting point)
3. Set `strength` → **0.7** (safe default)
4. Generate!

That's it. Auto mode analyzes your LoRA weights and picks the best protection level. Adjust from there.

---

## Node Overview

This pack provides **7 nodes** organized into 3 groups:

### LoRA Loading (core)

| Node | Purpose | When to use |
|---|---|---|
| **TUZ FLUX LoRA Loader** | Single LoRA + interactive graph widget | Your daily driver for 1 LoRA |
| **TUZ FLUX LoRA Multi** | Multiple LoRAs, dynamic slots | Stacking 2-4 LoRAs (editing + consistency + style) |
| **TUZ FLUX LoRA Scheduled** | Per-step strength curves | When you need LoRA to fade in/out during sampling |

### Conditioning (companion tools)

| Node | One-line summary |
|---|---|
| **Ref Latent Controller** | "Make the reference image stronger/weaker in attention" |
| **Text/Ref Balance** | "Push prompt harder or pull reference harder" |
| **Mask Ref Controller** | "Protect or weaken specific regions of the reference" |
| **Color Anchor** | "Prevent color drift during sampling" |

### Analysis & Postprocessing

| Node | Purpose |
|---|---|
| **TUZ FLUX Preflight Advisor** | Analyze LoRA before running — get recommended settings |
| **TUZ Klein Edit Composite** | Merge generated edit back onto original image (post-decode) |

---

## Edit Mode Presets — The Core Concept

Think of `edit_mode` as a **protection dial**, not a category label:

| Preset | What it protects | Best for |
|---|---|---|
| **Auto** ⭐ | Analyzes LoRA weights, picks best | Start here — works for most LoRAs |
| **Preserve Face** | Face & identity | Clothing/accessory/hair edits |
| **Preserve Body** | Face + body proportions | Try-on, outfit replacement |
| **Style Only** | Structure (reduces image stream) | Aesthetic/painterly LoRAs |
| **Edit Subject** | Moderate identity protection | Changing objects while keeping identity |
| **Boost Prompt** | Nothing (strengthens text instead) | When prompt feels too weak |
| **None** | Nothing | Raw LoRA behavior, full freedom |

### The `balance` slider

Interpolates between preset protection and raw LoRA:

```
0.0 ◄━━━━━━━━━━━━━━━━━━━━━► 1.0
Full preset effect          Raw LoRA (no protection)
(maximum protection)        
```

**Rule of thumb:**
- Face keeps getting overwritten? → Lower `balance` toward 0.0
- Edit feels too weak or "too safe"? → Raise `balance` toward 1.0

### Edit vs Generate (`use_case`)

Only affects **Auto** mode:

- **Edit** → Auto is more conservative (preserves identity/structure)
- **Generate** → Auto gives LoRA more freedom (text-to-image, restyling)

Manual presets always behave exactly as selected, regardless of `use_case`.

---

## Detailed Node Reference

### TUZ FLUX LoRA Loader

Single LoRA loader with interactive per-layer graph widget and optional auto-strength.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein / FLUX.1 model |
| `lora_name` | dropdown | LoRA file from `models/loras` |
| `strength` | float | Global LoRA strength (-5.0 to 5.0) |
| `use_case` | dropdown | Tells Auto whether you're editing a reference image or generating freely |
| `auto_convert` | boolean | Convert diffusers-format LoRAs to native FLUX format |
| `auto_strength` | boolean | Auto-compute per-layer strengths from ΔW analysis |
| `edit_mode` | dropdown | Protection level — `Auto` is the recommended start |
| `balance` | float | 0.0 = full preset protection, 1.0 = raw LoRA |

**Graph widget:** 8 double-block columns (img purple / txt teal) + 24 single-block columns (green).
- Drag to adjust individual layer strength
- Shift+drag moves all bars in a section
- Click to toggle a bar on/off

**Auto-strength:** Analyzes the LoRA's weight tensors and auto-fills optimal per-layer strengths. You can still manually tweak afterwards.

### TUZ FLUX LoRA Multi

Dynamic multi-LoRA loader with per-slot control. Click **"+ Add LoRA"** to add slots, **"✕"** to remove.

Each slot has: Enabled toggle, LoRA dropdown, Strength, Use case, Edit mode, Balance.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein / FLUX.1 model |
| `auto_convert` | boolean | Convert diffusers-format LoRAs |

**Recommended multi-slot setup for image editing:**

```
Slot 1: editing LoRA       → edit_mode=Auto, strength=0.6–0.8
Slot 2: consistency LoRA   → edit_mode=Auto, strength=0.4–0.6
Slot 3: enhancer LoRA      → edit_mode=None, strength=0.2–0.4
```

### TUZ FLUX LoRA Scheduled

Per-step LoRA strength control using ComfyUI's native Hook Keyframes. The LoRA effect varies across sampling steps. Takes conditioning as input and returns modified conditioning — no extra utility node needed.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein model |
| `conditioning` | CONDITIONING | Base conditioning |
| `lora_name` | dropdown | LoRA file |
| `strength` | float | Base LoRA strength (0.0–2.0) |
| `schedule` | dropdown | Strength curve profile |
| `edit_mode` | dropdown | Protection level (supports Auto) |
| `balance` | float | Preset effect ↔ raw LoRA |
| `keyframes` | int | Number of keyframes (2–10) |

**Returns:** `MODEL` + `CONDITIONING`

| Schedule | Curve | Best for |
|---|---|---|
| **Constant** | `1.0 → 1.0 → 1.0` | Standard behavior |
| **Fade Out** | `1.0 → 0.7 → 0.3 → 0.0` | Apply changes early, restore reference late |
| **Fade In** | `0.0 → 0.3 → 0.7 → 1.0` | Preserve reference first, add LoRA last |
| **Strong Start** | `1.0 → 0.5 → 0.2 → 0.0` | Aggressive fade-out, max preservation |
| **Pulse** | `0.3 → 1.0 → 1.0 → 0.3` | Peak effect in mid steps |

### TUZ FLUX Preflight Advisor

Analyzes a LoRA file + model compatibility and returns recommendations **without mutating** anything.

| Output | Type | Description |
|---|---|---|
| `report` | STRING | Human-readable summary with warnings |
| `recommended_edit_mode` | STRING | Suggested preset |
| `recommended_balance` | FLOAT | Suggested balance value |
| `recommended_strength` | FLOAT | Safe starting strength |
| `compat_status` | STRING | `ok`, `partial`, or `failed` |
| `matched_modules` | INT | LoRA modules that matched the model |
| `total_modules` | INT | Total complete LoRA modules found |

> **Note:** `recommended_edit_mode=None` means "Raw / No Protection", not "missing value".

The multi-slot advisor (`TUZ FLUX Multi Preflight Advisor`) accepts the same JSON slot format as `TUZ FLUX LoRA Multi`.

---

## Companion Conditioning Nodes

These are **small corrective tools** that sit around your LoRA loader in the graph. They don't replace the loader — they refine how the reference image and text prompt interact.

**Base flow:**
```
reference image → VAE Encode → ReferenceLatent
text prompt → conditioning
LoRA loader → model
companion nodes → conditioning/model
sampler → decode → TUZ Klein Edit Composite
```

### Ref Latent Controller

Controls how strongly one or all reference images influence the model's attention path.

**When to use:** The reference is too dominant or too weak in the output.

| Key parameter | What it does |
|---|---|
| `strength` | Overall reference influence (1.0 = normal, >1 = stronger, <1 = weaker) |
| `reference_index` | Which reference to target (`-1` = all) |
| `appearance_scale` | Boost coarse appearance (color, form) |
| `detail_scale` | Dampen fine detail (textures, small features) |

**Starting values for "keep identity but reduce stiffness":**
- `appearance_scale=1.15`, `detail_scale=0.75`, `blur_radius=2`

### Text/Ref Balance

Single slider to push prompt harder or pull reference harder.

**When to use:** Prompt changes aren't applying, or the reference is too dominant.

| Key parameter | What it does |
|---|---|
| `balance` | 0.0 = reference stronger, 1.0 = text takes over |
| `balance_mode` | `attn_patch` (gentle, default) or `latent_mix` (stronger intervention) |

**Rule of thumb:**
- `attn_patch` for normal edits
- `latent_mix` only when prompt consistently underfires

### Mask Ref Controller

Use a mask to protect, dampen, or replace regions of the reference latent.

**When to use:** You want different reference strength in different image areas.

| Key parameter | What it does |
|---|---|
| `mask_action` | `scale` (dampen) or `mix` (replace with alternate signal) |
| `replace_mode` | For `mix`: `zeros`, `gaussian_noise`, `channel_mean`, `lowpass_reference` |
| `feather` | Smooth mask edges |

**Starting values:** `mask_action=scale`, `strength=0.8`, `feather=12`

### Color Anchor

Keeps reference colors closer to the source during sampling.

**When to use:** Output colors drift too far from the reference.

| Key parameter | What it does |
|---|---|
| `strength` | Color correction intensity (0.25–0.50 is a good start) |
| `ramp_curve` | How quickly correction ramps in (higher = later start) |
| `channel_weights` | `uniform` or `by_variance` (trusts stable channels more) |
| `ref_index` | `-1` to average color from all references |

---

## TUZ Klein Edit Composite

Postprocess node for merging a generated edit back onto the original image. Place this **after VAE Decode**, not inside the LoRA pipeline.

```
original image + generated edit → VAE Decode → TUZ Klein Edit Composite → save
```

**Requires:** `pip install opencv-python-headless`

<details>
<summary><b>Full field reference (click to expand)</b></summary>

| Field | Type | What it does |
|---|---|---|
| `generated_image` | IMAGE | Edited image to composite back |
| `original_image` | IMAGE | Untouched source image |
| `delta_e_threshold` | FLOAT | Change sensitivity (-1 = auto) |
| `flow_quality` | choice | `medium`, `fast`, or `ultrafast` |
| `use_occlusion` | BOOLEAN | Forward-backward flow consistency |
| `occlusion_threshold` | FLOAT | Occlusion sensitivity (-1 = auto) |
| `noise_removal_pct` | FLOAT | Remove mask speckle (% of diagonal) |
| `close_radius_pct` | FLOAT | Morphological close radius (% of diagonal) |
| `fill_holes` | BOOLEAN | Fill enclosed holes in mask |
| `fill_borders` | BOOLEAN | Extend mask into warped border voids |
| `max_islands` | INT | Keep only N largest mask regions (0 = all) |
| `grow_mask_pct` | FLOAT | Grow/shrink final mask (% of diagonal) |
| `feather_pct` | FLOAT | Soften blend edge (% of diagonal) |
| `color_match_blend` | FLOAT | Blend colors toward original background |
| `poisson_blend_edges` | BOOLEAN | Seamless edge blending |
| `custom_mask` | MASK | External mask (optional) |
| `custom_mask_mode` | choice | `replace`, `add`, or `subtract` |
| `enable_debug` | BOOLEAN | Debug gallery + detailed report |

</details>

---

## Practical Recipes

### Recipe 1: Basic clothing edit with face protection

```
LoRA Loader: edit_mode=Preserve Face, strength=0.7, balance=0.3
```

### Recipe 2: Style transfer without structural damage

```
LoRA Loader: edit_mode=Style Only, strength=0.5, balance=0.5
```

### Recipe 3: Multi-LoRA with identity lock

```
Slot 1: clothing LoRA  → edit_mode=Preserve Body, strength=0.7
Slot 2: enhancer       → edit_mode=None, strength=0.3
+ Text/Ref Balance: balance=0.6 (push prompt a bit harder)
+ Color Anchor: strength=0.3 (prevent color shift)
```

### Recipe 4: Prompt feels too weak

```
LoRA Loader: edit_mode=Boost Prompt, strength=0.8, balance=0.4
```

### Recipe 5: Keep overall look but allow edits to "breathe"

```
LoRA Loader: edit_mode=Auto
+ Ref Latent Controller: appearance_scale=1.15, detail_scale=0.7
```

---

## How It Works Under The Hood

<details>
<summary><b>Format conversion (click to expand)</b></summary>

| Format | Source | Example keys | Handled by |
|---|---|---|---|
| **Native** | ComfyUI, kohya | `diffusion_model.double_blocks.0.img_attn.qkv` | Direct passthrough |
| **Diffusers** | HuggingFace | `transformer_blocks.0.attn.to_q` | Block-diagonal QKV fusion |
| **Musubi Tuner / PEFT** | Modelscope, MuseAI | `single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default` | Key remapping |

All formats are auto-detected when `auto_convert` is enabled.

**The core problem:**
LoRAs ship with separate `to_q` / `to_k` / `to_v` projections. FLUX stores these as a single fused QKV matrix. Without conversion, most attention weights never reach the model.

**The math (block-diagonal fusion):**
For fused weight `W = [W_q; W_k; W_v]` and separate LoRAs `B_i @ A_i`:

```
A_fused = cat([A_q, A_k, A_v], dim=0)         [3r × in]
B_fused = block_diag(B_q, B_k, B_v)           [3·out × 3r]
```

Alpha/rank scaling is pre-baked into B_fused.

</details>

<details>
<summary><b>Auto-strength calculation (click to expand)</b></summary>

For every layer pair in the file:

```
ΔW = lora_B @ lora_A
scaled_norm = frobenius_norm(ΔW) × (alpha / rank)
strength = clamp(global × (mean_norm / layer_norm), floor=0.30, ceiling=1.50)
```

Double blocks are processed with img and txt streams independently. The mean layer lands at `global_strength`.

</details>

<details>
<summary><b>Why Auto mode picks what it picks (click to expand)</b></summary>

Auto analyzes each LoRA's weight distribution across architecture layers:

- **High signal in late single blocks** (editing LoRAs) → **Preserve Body**
- **Moderate late-block signal** → **Preserve Face**
- **Uniform distribution** (sliders, enhancers) → **None**

The balance is also computed automatically. Console logs show the decision:
```
[FLUX LoRA Multi slot 1] Auto → Preserve Body (balance=0.25)
[FLUX LoRA Multi slot 2] Auto → Preserve Face (balance=0.40)
```

</details>

<details>
<summary><b>FLUX.2 Klein architecture reference (click to expand)</b></summary>

```
Double blocks (8 layers)
  img stream:
    img_attn.qkv    [12288, 4096]  (fused Q+K+V)
    img_attn.proj   [4096, 4096]
    img_mlp.0       [24576, 4096]
    img_mlp.2       [4096, 12288]
  txt stream:
    txt_attn.qkv    [12288, 4096]
    txt_attn.proj   [4096, 4096]
    txt_mlp.0       [24576, 4096]
    txt_mlp.2       [4096, 12288]

Single blocks (24 layers)
  linear1    [36864, 4096]  (fused Q+K+V+proj_mlp)
  linear2    [4096, 16384]

dim=4096  double_blocks=8  single_blocks=24
```

- **Double blocks (0-7):** Image and text streams are isolated — no cross-modal corruption possible.
- **Single blocks (0-23):** Joint cross-modal processing — text prompt can overwrite reference image. Late single blocks (12-23) are the most aggressive.

</details>

---

## FAQ

**Q: Do I need this for FLUX.1?**
A: It works with FLUX.1 too, but the main value is for Klein 9B where the architecture mismatch is most common.

**Q: What does `None` mean in edit_mode?**
A: "Raw / No Protection" — not "nothing selected". The LoRA runs with all layers at equal strength.

**Q: My LoRA doesn't seem to have any effect.**
A: Check `auto_convert` is ON. If using a GGUF model, ensure [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is installed.

**Q: Auto mode picks the wrong preset for my use case.**
A: Auto can't read your intent. Switch to manual: `Preserve Face` for identity work, `None` for full freedom. Use `balance` to fine-tune.

**Q: How do I know which preset Auto picked?**
A: Check the ComfyUI console. It logs e.g. `Auto → Preserve Body (balance=0.25)`.

**Q: Can I use the conditioning nodes without the LoRA loader?**
A: Yes. They are independent nodes that work on `MODEL` and `CONDITIONING` — they're useful with any FLUX workflow.

---

## Credits

- Original node pack by [capitan01R](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
- Edit-mode presets based on architecture research from [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) (Klein 9B debiaser / layer mapping)
- GGUF support via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
