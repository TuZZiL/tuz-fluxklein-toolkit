# TUZ FluxKlein Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Architecture-aware LoRA loading for **FLUX.2 Klein** (9B) in ComfyUI.

[If you're new, start with the steps below before reading the technical sections.](#start-here)

[Українська версія](README_UA.md)

---

## Start Here

If you only want one working first run:

1. Install the pack in `ComfyUI/custom_nodes`.
2. Restart ComfyUI.
3. Add **TUZ FLUX LoRA Loader** from `loaders/FLUX`.
4. Leave `auto_convert` ON, set `edit_mode=Auto`, `strength=0.7`, and `use_case=Edit`.
5. Run one simple test image with a single LoRA.

If the node appears in the menu and the image changes, the setup worked.

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
git clone https://github.com/TuZZiL/tuz-fluxklein-toolkit.git
# Restart ComfyUI
```

Requires `numpy` (usually already installed with ComfyUI).
For `TUZ Klein Edit Composite`, also install: `pip install opencv-python-headless`

### Step 2: Add the node

Find **TUZ FLUX LoRA Loader** in the node menu under `loaders/FLUX`. Connect your `MODEL` input.

### Step 3: Pick your LoRA and go

1. Select your LoRA file
2. Leave `auto_convert` ON
3. Set `edit_mode` → **Auto** (recommended starting point)
4. Set `strength` → **0.7** (safe default)
5. Leave `auto_strength` OFF for the first test
6. Generate!

That's it. Auto mode analyzes your LoRA weights and picks the best protection level. If you are editing a reference image, keep `use_case=Edit`. Adjust from there.

---

![TUZ FLUX node layout screenshot](images/nodes.png)

*The screenshot shows the loader and graph widget in ComfyUI.*

## Beginner Glossary

These are the terms you will see most often. If you only remember one thing, keep `auto_convert` on and start with `edit_mode=Auto`.

| Term | Simple meaning |
|---|---|
| `edit_mode` | Chooses the protection style. `Auto` is the safest first choice. |
| `protection` | How strong the chosen preset should be. Legacy `balance` still works for old workflows. |
| `use_case` | Tells `Auto` whether you are editing a reference image or generating freely. |
| `auto_convert` | Converts many downloaded FLUX LoRAs into the format Klein expects. Leave it on unless you know the file is already native. |
| `auto_strength` | Automatically spreads strength across layers. Useful later, not required for the first test. |
| `anatomy_profile` | Extra body-preservation preset for clothing or body edits. Leave it off until you need it. |

## Node Overview

If you are new, you only need the loader for the first test. The rest are optional.

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

### The protection dial (`protection`, legacy `balance`)

Acts like a protection dial: `0.0` is raw LoRA, `1.0` is full preset protection.

```
0.0 ◄━━━━━━━━━━━━━━━━━━━━━► 1.0
Raw LoRA (no protection)    Full preset protection
```

**Rule of thumb:**
- Face keeps getting overwritten? → Raise `protection` toward 1.0
- Edit feels too weak or "too safe"? → Lower `protection` toward 0.0

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
| `protection` | float | Protection dial: 0.0 = raw LoRA, 1.0 = full preset protection (`balance` is still accepted as legacy input) |
| `anatomy_profile` | dropdown | Opt-in body-preservation profile for clothing / body edits |
| `anatomy_strength` | float | How strongly the anatomy profile should protect structure |
| `anatomy_strict_zero` | boolean | Advanced. Hard-disable the most sensitive blocks in the profile |

**Graph widget:** 8 double-block columns (img purple / txt teal) + 24 single-block columns (green).
- Drag to adjust individual layer strength
- Shift+drag moves all bars in a section
- Click to toggle a bar on/off

**Auto-strength:** Analyzes the LoRA's weight tensors and auto-fills optimal per-layer strengths. You can still manually tweak afterwards.

**Anatomy Shield:** Use this when a LoRA keeps changing the body shape while you only want the clothes or surface details to move.
- `Undress Safe` is the default starting point for clothing removal.
- `Undress Body Lock` is stricter and keeps the silhouette steadier.
- `Robot Frame Lock` is the safest start for humanoid robots or mech edits.
- Keep `anatomy_profile=None` if you want the old behavior unchanged.
- Start with `anatomy_strength=0.60–0.70`; raise it only if the body still drifts.

**Practical rule:** `edit_mode` controls how much protection the LoRA gets per layer, while `anatomy_profile` adds a higher-level body-preservation preset on top. Use both only when you actually need them.

### TUZ FLUX LoRA Multi

Dynamic multi-LoRA loader with per-slot control. Click **"+ Add LoRA"** to add slots, **"✕"** to remove.

Each slot has: Enabled toggle, LoRA dropdown, Strength, Use case, Edit mode, Protection.

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

If you are doing clothing removal or body-preserving edits, keep the anatomy profile on the main slot only. Suggested starts:
- `Undress Safe` for normal clothing removal.
- `Undress Body Lock` when the body shape must stay stable.
- `Armor Hard Surface` for armor or rigid outfits.

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
| `protection` | float | Raw LoRA ↔ preset protection (`balance` is still accepted as legacy input) |
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
| `recommended_balance` | FLOAT | Suggested protection value (legacy output key) |
| `recommended_strength` | FLOAT | Safe starting strength |
| `compat_status` | STRING | `ok`, `partial`, or `failed` |
| `matched_modules` | INT | LoRA modules that matched the model |
| `total_modules` | INT | Total complete LoRA modules found |

> **Note:** `recommended_edit_mode=None` means "Raw / No Protection", not "missing value".

The multi-slot advisor (`TUZ FLUX Multi Preflight Advisor`) accepts the same JSON slot format as `TUZ FLUX LoRA Multi`.

---

## Companion Conditioning Nodes

These are **small corrective tools** that sit around your LoRA loader in the graph. They do not replace the loader. They refine how the reference image and text prompt interact.

**Base flow:**
```
reference image → VAE Encode → ReferenceLatent
text prompt → conditioning
LoRA loader → model
companion nodes → conditioning/model
sampler → decode → TUZ Klein Edit Composite
```

<details>
<summary><b>Practical guide (click to expand)</b></summary>

### Which node to reach for

| Problem | Use |
|---|---|
| Prompt is getting ignored by the reference | `Text/Ref Balance` |
| Only one region should stay reference-true | `Mask Ref Controller` |
| The whole reference is too rigid or too loose | `Ref Latent Controller` |
| Composition or pose keeps drifting | `Structure Lock` |
| Colors drift while structure is fine | `Color Anchor` |

### Ref Latent Controller

This changes how strongly reference tokens stay present in the attention path. It is the global “reference pressure” dial.

Use it when the reference is too dominant, too weak, or too stiff overall.

Starting values: `strength=1.0`, `appearance_scale=1.15`, `detail_scale=0.75`, `blur_radius=2`.

How to tune:
- Raise `strength` or `appearance_scale` if the prompt is overriding the reference too easily.
- Lower `detail_scale` if the identity is correct but textures and micro-details feel locked in.
- Use `reference_index=-1` when you want the same treatment on every reference latent.
- Use `spatial_fade` when only part of the reference should stay stronger.

Practical use: keep the same face or pose, but loosen the stiffness in clothing texture or fine detail.

### Text/Ref Balance

Note: this node's `balance` is separate from the loader's `protection` dial and its legacy `balance` alias.

This shifts control between prompt and reference. `attn_patch` adjusts attention directly, while `latent_mix` weakens reference influence before sampling.

Use it when the prompt is not landing or the reference keeps winning.

Starting values: `balance=0.45` for prompt-led edits, `balance=0.65` when the prompt needs more authority.

How to tune:
- Lower `balance` to keep the reference stronger.
- Raise `balance` to let text take over more aggressively.
- Keep `attn_patch` as the default.
- Use `latent_mix` only when the prompt still underfires after attention patching.

Practical use: make a stubborn reference listen to the prompt without removing the reference entirely.

### Mask Ref Controller

This applies a spatial mask to the reference latent. It does not localize the prompt. It localizes the reference influence.

Use it when one area should stay reference-true and another area should move.

Starting values: `mask_action=scale`, `strength=0.8`, `feather=12`.

How to tune:
- `scale` dampens the reference under the mask.
- `mix` replaces masked regions with another latent signal.
- `invert_mask=True` flips the protected and affected regions.
- `reference_keep` matters only in `mix` mode.

Practical use: protect the face while changing clothing or background, or invert the mask to push the edit into the subject itself.

### Structure Lock

This keeps coarse spatial structure from the reference while still letting prompts change texture and detail.

Use it when pose, framing, or object placement keeps drifting during edits.

Starting values: `strength=0.35`, `blur_radius=6`, `ramp_start=0.0`, `ramp_end=0.5`.

How to tune:
- Raise `strength` if the composition still drifts too much.
- Raise `blur_radius` if the lock feels too sticky and starts freezing detail.
- Shorten `ramp_end` if you want the structure hold to fade out earlier in sampling.
- Use `reference_index=-1` when multiple references should share the same lock.
- Add a mask when only the face, subject, or background should stay anchored.

Practical use: keep portrait poses stable, preserve product framing, or hold the subject in place while swapping a background.

### Color Anchor

This applies color correction during sampling so the result stays closer to the source palette.

Use it when composition is right but the palette drifts.

Starting values: `strength=0.35`, `ramp_curve=1.5`, `channel_weights=uniform`.

How to tune:
- Raise `strength` if the output still drifts too far from the reference palette.
- Raise `ramp_curve` if the correction kicks in too early and flattens the image.
- Use `by_variance` when some channels are stable and others are noisy.

Practical use: keep skin tones, clothing colors, or background hues closer to the reference without forcing a hard color match.

### Common pairings

| Goal | Pairing |
|---|---|
| Region-specific identity control | `Ref Latent Controller` + `Mask Ref Controller` |
| Prompt lands but colors drift | `Text/Ref Balance` + `Color Anchor` |
| Only part of the frame should be rewritten | `Mask Ref Controller` + `Color Anchor` |

</details>

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

## Beginner Recipes

### Recipe 1: First LoRA, safe default

```
LoRA Loader: auto_convert=ON, use_case=Edit, edit_mode=Auto, strength=0.7, protection=0.5
```

Use this when you just want to confirm the node is working. If the effect is too strong, lower `strength` first.

### Recipe 2: Clothing edit with face protection

```
LoRA Loader: auto_convert=ON, use_case=Edit, edit_mode=Preserve Face, strength=0.65-0.75, protection=0.7
```

Use this when the LoRA changes clothes or accessories but must keep the person recognizable.

### Recipe 3: Simple multi-LoRA stack

```
Slot 1: editing LoRA     → edit_mode=Auto, strength=0.6–0.8
Slot 2: consistency LoRA → edit_mode=Auto, strength=0.4–0.6
Slot 3: enhancer LoRA    → edit_mode=None, strength=0.2–0.4
```

If you need body stability, keep `anatomy_profile` on Slot 1 only. Treat `protection` as the main dial, and ignore legacy `balance` unless you open an older workflow.

---

## Troubleshooting

- LoRA has no visible effect: make sure `auto_convert` is on, the model is loaded, and the LoRA file is in `models/loras`. If the file came from diffusers, leave the converter on.
- Auto picked the wrong preset: switch from `Auto` to `Preserve Face` or `None`. Auto is a best guess, not a mind reader.
- Model/LoRA mismatch: this pack is made for FLUX.2 Klein and also works with FLUX.1. If you are using GGUF, install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
- `TUZ Klein Edit Composite` is missing: install `opencv-python-headless`.
- The image changed, but too little: raise `strength` first before changing the preset.

---

## Advanced Reference (optional)

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

Protection is also computed automatically. Console logs show the decision:
```
[FLUX LoRA Multi slot 1] Auto → Preserve Body (protection=0.75)
[FLUX LoRA Multi slot 2] Auto → Preserve Face (protection=0.60)
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
A: Auto can't read your intent. Switch to manual: `Preserve Face` for identity work, `None` for full freedom. Use the `protection` dial to fine-tune (`balance` still works as a legacy alias).

**Q: How do I know which preset Auto picked?**
A: Check the ComfyUI console. It logs e.g. `Auto → Preserve Body (protection=0.75)`.

**Q: Can I use the conditioning nodes without the LoRA loader?**
A: Yes. They are independent nodes that work on `MODEL` and `CONDITIONING` — they're useful with any FLUX workflow.

---

## Credits

- Original node pack by [capitan01R](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
- Edit-mode presets based on architecture research from [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) (Klein 9B debiaser / layer mapping)
- GGUF support via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
