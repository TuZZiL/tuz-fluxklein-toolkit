# ComfyUI FLUX.2 Klein LoRA Loader
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Architecture-aware LoRA loading for **FLUX.2 Klein** (9B) in ComfyUI.

[Українська версія](README_UA.md)

## Why use this instead of the standard LoRA loader?

Most LoRAs you download were trained with HuggingFace tools and saved in **diffusers format**. The standard ComfyUI LoRA loader **silently drops** most of these weights on Klein 9B — your LoRA loads, but it barely works or looks wrong.

This pack **automatically converts** any LoRA format to work correctly with Klein 9B. Just drop in your LoRA and it works at full power.

But there's more. When you use LoRAs for **image editing** (changing clothes, adding accessories, style transfer on a reference photo), the LoRA often **destroys the face** or changes body proportions. This pack solves that with **edit-mode presets** — one dropdown that tells the loader which parts of the image to protect:

- **Preserve Face** — edit freely while keeping the person's face intact
- **Preserve Body** — protect both face and body proportions (figure, pose)
- **Auto** — the loader analyzes your LoRA and picks the best protection automatically

The result: your edits apply where you want them, and everything else stays untouched.

## Key Features

- **3 focused nodes + 1 advisor**: Loader (single LoRA + graph), Multi (dynamic slots), Scheduled (temporal control), Preflight Advisor (pre-run recommendations)
- **Format auto-detection**: Supports native, diffusers, and Musubi Tuner (PEFT) LoRA formats
- **Block-diagonal QKV fusion**: Correctly maps separate Q/K/V projections to fused matrices
- **Edit-mode presets**: Preserve face, body, or style during LoRA-based image editing
- **Auto mode**: Analyzes LoRA weight norms and automatically selects the best preset
- **Auto-strength**: Built-in per-layer strength calibration from ΔW forensic analysis
- **Dynamic multi-slot**: Add/remove LoRA slots with per-slot edit_mode (rgthree-style)
- **Strength scheduling**: Varies LoRA strength across sampling steps
- **GGUF compatible**: Works with ComfyUI-GGUF quantized models

## Background

LoRAs trained against FLUX models are commonly shipped in diffusers format — separate `to_q`, `to_k`, `to_v` projections per attention layer. FLUX's native architecture stores these as a single fused QKV matrix, and single blocks fuse attention and MLP gate into a single `linear1` projection. Loading these LoRAs without conversion means most attention weights never reach the model.

| What the LoRA ships with | What FLUX expects | What this pack does |
|---|---|---|
| Separate `to_q` / `to_k` / `to_v` | Fused `img_attn.qkv` / `txt_attn.qkv` | Block-diagonal fusion at load time |
| Separate single block components | Fused `linear1` `[36864, 4096]` | Fuses `[q, k, v, proj_mlp]` correctly |
| Musubi Tuner `.default.` keys | Standard LoRA keys | Auto-strips and remaps |
| Global strength only | Independent img/txt + per-single-block | Interactive graph widget + auto-calibration |
| LoRA affects everything equally | Different layers control different aspects | **Edit-mode presets** for selective control |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TuZZiL/Comfyui-flux2klein-Lora-loader.git
```

Requires `numpy` (usually already installed with ComfyUI).

## Nodes

### TUZ FLUX Preflight Advisor

Advisory node for pre-run LoRA inspection. It analyzes the LoRA file plus model key compatibility and returns recommendations without mutating the workflow.

| Output | Type | Description |
|---|---|---|
| `report` | STRING | Human-readable summary with warnings and rationale. |
| `recommended_edit_mode` | STRING | Suggested edit protection preset. |
| `recommended_balance` | FLOAT | Suggested preset balance. |
| `recommended_strength` | FLOAT | Safe starting strength for the LoRA. |
| `compat_status` | STRING | `ok`, `partial`, or `failed`. |
| `matched_modules` | INT | How many LoRA modules matched the model. |
| `total_modules` | INT | Total complete LoRA modules found. |

The multi-slot advisor uses the same slot JSON shape as `TUZ FLUX LoRA Multi` and returns `recommended_slot_data_json` for each active slot. It is advisory-only and does not inspect or recommend `schedule`.

Quick terminology:
- `recommended_edit_mode=None` means **Raw / No Protection**, not "missing value"
- `recommended_balance` is the amount of preset behavior to keep:
  `0.0 = strongest preset effect`, `1.0 = raw LoRA behavior`

### TUZ FLUX LoRA Loader

Single LoRA loader with interactive per-layer graph widget and optional auto-strength.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein / FLUX.1 model |
| `lora_name` | dropdown | LoRA file from `models/loras` |
| `strength` | float | Global LoRA strength (-5.0 to 5.0) |
| `use_case` | dropdown | Tells Auto whether you are editing a reference image or doing freer generation |
| `auto_convert` | boolean | Convert diffusers-format LoRAs to native FLUX format |
| `auto_strength` | boolean | When ON: normalize uneven LoRAs by auto-computing per-layer strengths from ΔW analysis |
| `edit_mode` | dropdown | How protective the loader should be; `Auto` is the recommended starting point |
| `balance` | float | 0.0 = strongest preset effect, 1.0 = raw LoRA behavior |

**Graph widget:** Shows double blocks (8 columns, img purple / txt teal, split top/bottom) and single blocks (24 columns, green). Drag to adjust. Shift-drag moves all bars in a section. Click to toggle a bar on/off.

**Auto-strength:** When enabled, the node analyzes the LoRA's weight tensors and computes optimal per-layer strengths automatically. The graph bars auto-populate — you can still manually tweak them afterwards.

### TUZ FLUX LoRA Multi

**Dynamic multi-LoRA loader** with per-slot control. Click **"+ Add LoRA"** to add slots, **"✕"** to remove.

Each slot has:
- **Enabled** toggle
- **LoRA** dropdown
- **Strength** (-5.0 to 5.0)
- **Use case** (Edit or Generate)
- **Edit mode** (None, Preserve Face, Preserve Body, Style Only, Edit Subject, Boost Prompt, Auto)
- **Balance** (0.0 to 1.0)

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein / FLUX.1 model |
| `auto_convert` | boolean | Convert diffusers-format LoRAs to native FLUX format |

Recommended setup for image editing:
```
Slot 1: editing LoRA       → edit_mode=Auto (or Preserve Body), strength=0.6-0.8
Slot 2: consistency LoRA   → edit_mode=Auto (or None),          strength=0.4-0.6
Slot 3: enhancer LoRA      → edit_mode=Auto (or None),          strength=0.2-0.4
```

### TUZ FLUX LoRA Scheduled

**Per-step LoRA strength control** using ComfyUI's native Hook Keyframes system. Instead of constant strength, the LoRA effect varies across sampling steps. Takes conditioning as input and returns modified conditioning directly — no extra utility node needed.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein model |
| `conditioning` | CONDITIONING | Base conditioning (hooks attached automatically) |
| `lora_name` | dropdown | LoRA file |
| `strength` | float | Base LoRA strength (0.0 to 2.0) |
| `use_case` | dropdown | Tells Auto whether to prioritize reference preservation or freer generation |
| `schedule` | dropdown | Strength curve profile |
| `edit_mode` | dropdown | How protective the loader should be (supports Auto) |
| `balance` | float | 0.0 = strongest preset effect, 1.0 = raw LoRA behavior |
| `keyframes` | int | Number of keyframes (2-10, default 5) |

**Returns:** `MODEL` + `CONDITIONING`

```
FluxLoraScheduled → MODEL → CFGGuider
                  → CONDITIONING → ReferenceLatent → CFGGuider
```

Available schedules:

| Schedule | Curve | Use case |
|---|---|---|
| **Constant** | `1.0 → 1.0 → 1.0 → 1.0` | Standard behavior (no scheduling) |
| **Fade Out** | `1.0 → 0.7 → 0.3 → 0.0` | Editing: apply changes early, restore reference details late |
| **Fade In** | `0.0 → 0.3 → 0.7 → 1.0` | Detailing: preserve reference first, add LoRA effect last |
| **Strong Start** | `1.0 → 0.5 → 0.2 → 0.0` | Aggressive fade-out for maximum reference preservation |
| **Pulse** | `0.3 → 1.0 → 1.0 → 0.3` | Peak effect in mid steps |

### Companion Conditioning Nodes

These nodes control reference-latent behavior and prompt-conditioning without changing the LoRA loader pipeline.

| Node | What it does |
|---|---|
| `TUZ FLUX.2 Klein Ref Latent Controller` | Scales one or all reference images inside the model attention path, with optional appearance/detail rebalancing before patching. |
| `TUZ FLUX.2 Klein Text/Ref Balance` | Balances text vs reference influence with a single slider, either through attention-patch scaling or direct latent mixing. |
| `TUZ FLUX.2 Klein Mask Ref Controller` | Uses a mask to protect, dampen, or partially replace areas of one or many reference latents. |
| `TUZ FLUX.2 Klein Color Anchor` | Keeps reference colors closer to the source during sampling, with support for averaging multiple references. |

These nodes are designed to be simple ComfyUI-style controls: one job per node, plain widgets, and no extra visual chrome.

Practical upgrades in the current version:
- **Multi-reference aware**: `reference_index` / `target_reference_index` can address one reference or all references (`-1`) depending on the node.
- **Mask mix modes**: `TUZ FLUX.2 Klein Mask Ref Controller` now supports `mask_action=scale|mix`. `mix` can replace masked regions with `zeros`, `gaussian_noise`, `channel_mean`, or `lowpass_reference`.
- **Prompt/reference balance modes**: `TUZ FLUX.2 Klein Text/Ref Balance` keeps the original `attn_patch` behavior by default and also supports `latent_mix` for direct reference-latent control.
- **Appearance/detail rebalance**: `TUZ FLUX.2 Klein Ref Latent Controller` can optionally boost coarse appearance and dampen fine detail before attention-path scaling.

Quick terminology:
- `reference_index` / `target_reference_index`:
  `0` = first reference, `1` = second, `-1` = all references
- `attn_patch`:
  change reference strength inside attention during sampling
- `latent_mix`:
  weaken or replace part of the reference latent before sampling
- `mask_action=scale`:
  keep the same reference latent and dampen it
- `mask_action=mix`:
  replace masked regions with a different latent signal
- `channel_mask_start=0` and `channel_mask_end=0`:
  use the full latent channel range unless you deliberately want a narrower slice

### Practical Companion Workflows

These nodes are most useful when you treat them as **small corrective tools** around a normal Klein edit graph, not as replacements for the LoRA loader itself.

Recommended base flow:
```text
reference image -> VAE Encode -> ReferenceLatent
text prompt -> conditioning
LoRA loader -> model
conditioning companion nodes -> conditioning/model
sampler -> decode -> TUZ Klein Edit Composite
```

#### 1. Keep identity stronger during clothing or accessory edits

Use:
```text
ReferenceLatent -> TUZ FLUX.2 Klein Text/Ref Balance -> CFGGuider
                                      \-> TUZ FLUX.2 Klein Ref Latent Controller -> model
```

Start with:
- `Text/Ref Balance`: `balance=0.55` to `0.70`
- `Ref Latent Controller`: `strength=1.1` to `1.4`
- If you have multiple references, set `reference_index=-1` only when all references should stay equally strong

Practical effect:
- prompt changes still apply
- face and core structure drift less
- useful for try-on, accessory swaps, and identity-sensitive edits

#### 2. Protect only one region of the reference

Use:
```text
conditioning -> TUZ FLUX.2 Klein Mask Ref Controller -> sampler
```

Start with:
- `mask_action=scale`
- `strength=0.7` to `1.0`
- `channel_mode=all`
- `feather=8` to `20`

When to switch to `mask_action=mix`:
- the masked region still leaks too much structure
- you want harder removal or replacement inside the masked area

Good `mix` starting points:
- `replace_mode=zeros` for strongest suppression
- `replace_mode=lowpass_reference` for softer cleanup that keeps coarse structure
- `target_reference_index=-1` only when all references should be masked the same way

#### 3. Push prompt harder or pull reference harder

Use:
```text
conditioning -> TUZ FLUX.2 Klein Text/Ref Balance
```

Modes:
- `balance_mode=attn_patch`:
  best default; light-touch control during sampling
- `balance_mode=latent_mix`:
  stronger intervention; directly reduces reference latent influence

Start with:
- `attn_patch` for normal edits
- `latent_mix` only when the prompt keeps underfiring or the reference dominates too much

Rule of thumb:
- lower `balance` toward `0.25` to weaken text and keep reference stronger
- raise `balance` toward `0.75` to let text take over more aggressively

#### 4. Keep overall look but soften detail rigidity

Use:
```text
model + conditioning -> TUZ FLUX.2 Klein Ref Latent Controller
```

Start with:
- `appearance_scale=1.10` to `1.25`
- `detail_scale=0.60` to `0.85`
- `blur_radius=2` or `3`

Practical effect:
- preserves coarse appearance and broad color/form
- reduces fine-detail lock that can make edits feel stiff
- especially useful when the reference is over-constraining fabrics, skin texture, or small accessories

#### 5. Reduce color drift late in sampling

Use:
```text
model -> TUZ FLUX.2 Klein Color Anchor -> sampler
```

Start with:
- `strength=0.25` to `0.50`
- `ramp_curve=1.5`
- `channel_weights=uniform`

Use `channel_weights=by_variance` when:
- some channels are clearly noisier than others
- you want gentler color anchoring with less overcorrection

Use `ref_index=-1` when:
- multiple references define the palette together
- you want an averaged color anchor instead of a single dominant source

#### 6. Multi-reference practical rule

If one reference is the main identity source and another is only style/support:
- keep `reference_index` / `target_reference_index` on the primary ref first
- avoid `-1` until you know you want symmetric behavior across all refs

If all references are intended to cooperate equally:
- `-1` is the cleanest option
- use lower strengths than in single-reference mode because the aggregate conditioning is already stronger

### TUZ Klein Edit Composite

Postprocess node for merging a generated edit back onto the original image. This is the right place for final cleanup after decode, not inside the LoRA pipeline.

Recommended flow:
```text
original image + edit generation -> VAE Decode -> TUZ Klein Edit Composite -> save
```

Install note:
```bash
pip install opencv-python-headless
```

Fields:

| Field | Type | Meaning |
|---|---|---|
| `generated_image` | IMAGE | The edited / generated image to composite back onto the source. |
| `original_image` | IMAGE | The untouched source image used as the base. |
| `delta_e_threshold` | FLOAT | Change sensitivity. Set `-1` for auto-thresholding. |
| `flow_quality` | choice | Optical flow precision: `medium`, `fast`, or `ultrafast`. |
| `use_occlusion` | BOOLEAN | Adds forward-backward flow consistency to the mask. |
| `occlusion_threshold` | FLOAT | Occlusion sensitivity. Set `-1` for auto-thresholding. |
| `noise_removal_pct` | FLOAT | Removes speckle noise from the mask as a % of image diagonal. |
| `close_radius_pct` | FLOAT | Morphological close radius as a % of image diagonal. |
| `fill_holes` | BOOLEAN | Fills enclosed holes inside the detected mask. |
| `fill_borders` | BOOLEAN | Extends the mask into warped border voids. |
| `max_islands` | INT | Keeps only the largest N connected mask regions. `0` disables pruning. |
| `grow_mask_pct` | FLOAT | Grows or shrinks the final mask as a % of image diagonal. |
| `feather_pct` | FLOAT | Softens the final blend edge as a % of image diagonal. |
| `color_match_blend` | FLOAT | Blends generated colors toward the original background. |
| `poisson_blend_edges` | BOOLEAN | Uses Poisson/seamless blending for edges when possible. |
| `custom_mask` | MASK | Optional external mask to replace or adjust auto-detection. |
| `custom_mask_mode` | choice | How to combine the external mask: `replace`, `add`, or `subtract`. |
| `enable_debug` | BOOLEAN | Emits a debug gallery and richer report text. |

Best use cases:
- local clothing/accessory edits
- face-preserving touchups
- controlled object replacement
- cleanup of background drift after generation

## Edit Mode Presets

Think of `edit_mode` as a **protection level**, not as a category label for the LoRA itself. Different edit LoRAs can belong to the same practical bucket:

- Some LoRAs mostly change clothing, accessories, or makeup
- Some push pose, body shape, or camera changes much harder
- Some are really style LoRAs that people reuse in image editing
- Some are consistency / enhancer LoRAs that already behave well without protection

When using LoRAs for image editing (e.g., changing clothing on a reference photo), the LoRA can corrupt parts of the image you want to preserve — most commonly the face/identity. This happens because FLUX.2 Klein processes image and text in different ways across its layers:

- **Double blocks (0-7):** Image and text streams are isolated — they can't cause text-driven image corruption on their own.
- **Single blocks (0-23):** Joint cross-modal processing — this is where the text prompt overwrites the reference image. Late single blocks (12-23) are the most aggressive.

### Available Presets

| Preset | What it does | Use case |
|---|---|---|
| **None** | Standard LoRA (all layers equal) | Default behavior |
| **Preserve Face** | Dampens late single blocks, keeps img stream intact | Editing while keeping face/identity |
| **Preserve Body** | Aggressively dampens mid+late single blocks | Editing while keeping face + body proportions |
| **Style Only** | Reduces img stream in double blocks, dampens late singles | Applying style changes without structural edits |
| **Edit Subject** | Moderate protection on late blocks, slight txt boost | Changing clothing/objects while preserving identity |
| **Boost Prompt** | Strengthens txt stream and mid single blocks | When the prompt isn't being followed strongly enough |
| **Auto** | Analyzes LoRA weights, picks best preset + balance automatically | Zero-config — recommended for most users |

### Which Mode To Start With

| If your LoRA feels like... | Start with | Why |
|---|---|---|
| Clothing / accessories / hair / makeup edit | **Auto** or **Preserve Face** | Usually keeps identity steadier while still allowing local edits |
| Outfit replacement / try-on / body-sensitive edit | **Auto** or **Preserve Body** | Best when face, silhouette, or proportions drift too much |
| Style / aesthetic / painterly look | **Auto** or **Style Only** | Lets the look change while reducing structural rewrites |
| Consistency / enhancer / "fixer" LoRA | **None** or **Auto** | These often behave well already and do not need extra protection |
| Prompt feels weak or the LoRA is under-following | **Boost Prompt** | Gives text-driven changes more authority |
| You intentionally want the LoRA to freely change face/body | **None** | Raw LoRA behavior, no protection layer applied |

### Use Case: Edit vs Generate

`use_case` only affects **Auto**. Manual modes always behave exactly as selected.

- **Edit**: Best for reference-driven image editing. Auto will be more conservative and favor identity / structure preservation.
- **Generate**: Best for text-to-image, loose restyling, or when there is no strong reference image to protect. Auto will allow more raw LoRA behavior.

This matters because Klein edit LoRAs are not all the same. A style LoRA, a clothing edit LoRA, and a consistency LoRA can all be valid on Klein, but they benefit from different starting assumptions.

### Auto Mode

When `edit_mode` is set to **Auto**, the node analyzes each LoRA's weight distribution and selects the optimal preset:

- High training signal in late single blocks (editing LoRAs) → **Preserve Body**
- Moderate late-block signal → **Preserve Face**
- Uniform distribution (sliders, enhancers) → **None**

Auto is a strong starting point, but it still cannot read user intent. If you want to deliberately change pose, face, or body, switch to **None** or raise `balance` toward `1.0`.

The balance is also computed automatically based on signal concentration. Console logs show which preset was selected:
```
[FLUX LoRA Multi slot 1] Auto → Preserve Body (balance=0.25)
[FLUX LoRA Multi slot 2] Auto → Preserve Face (balance=0.40)
```

### Balance Slider

The `balance` slider interpolates between the preset and standard behavior:
- **0.0** — full preset effect (maximum protection/boost)
- **0.5** — halfway between preset and standard
- **1.0** — standard LoRA (preset has no effect)

Practical rule of thumb:
- Lower `balance` if the LoRA keeps overwriting the face, body, or reference structure.
- Raise `balance` if the edit feels too weak or too "safe".

## How Auto Strength Works

For every layer pair in the file:

```
ΔW = lora_B @ lora_A
scaled_norm = frobenius_norm(ΔW) * (alpha / rank)
strength = clamp(global * (mean_norm / layer_norm), floor=0.30, ceiling=1.50)
```

Double blocks are processed with img and txt streams independently. Mean layer lands at `global_strength`.

## Supported LoRA Formats

| Format | Source | Example keys | Handled by |
|---|---|---|---|
| **Native** | ComfyUI, kohya | `diffusion_model.double_blocks.0.img_attn.qkv` | Direct passthrough |
| **Diffusers** | HuggingFace | `transformer_blocks.0.attn.to_q` | Block-diagonal QKV fusion |
| **Musubi Tuner / PEFT** | Modelscope, MuseAI | `single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default` | Key remapping + direct remap |

All formats are auto-detected when `auto_convert` is enabled.

## FLUX.2 Klein Architecture Reference

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

## Credits

- Original node pack by [capitan01R](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
- Edit-mode presets based on architecture research from [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) (Klein 9B debiaser / layer mapping)
- GGUF support via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
