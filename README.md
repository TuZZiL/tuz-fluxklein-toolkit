# ComfyUI FLUX.2 Klein LoRA Loader
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Architecture-aware LoRA loading for **FLUX.2 Klein** (9B) in ComfyUI, with automatic per-layer strength calibration, **semantic edit-mode presets** for identity-preserving image editing, and **Auto mode** that picks the best preset for each LoRA.

[Українська версія](README_UA.md)

![](images/nodes.png)

## Key Features

- **Format auto-detection**: Supports native, diffusers, and Musubi Tuner (PEFT) LoRA formats
- **Block-diagonal QKV fusion**: Correctly maps separate Q/K/V projections to fused matrices
- **Edit-mode presets**: Preserve face, body, or style during LoRA-based image editing
- **Auto mode**: Analyzes LoRA weight norms and automatically selects the best preset
- **Per-slot control**: FluxLoraQuad node with independent edit_mode per LoRA
- **Strength scheduling**: FluxLoraScheduled node varies LoRA strength across sampling steps
- **GGUF compatible**: Works with ComfyUI-GGUF quantized models
- **Example workflow**: Ready-to-use Klein 9B editing workflow included

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

## Nodes

### FLUX LoRA Loader

Single LoRA loader with edit-mode support.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein / FLUX.1 model |
| `lora_name` | dropdown | LoRA file from `models/loras` |
| `strength_model` | float | Global LoRA strength (-20.0 to 20.0) |
| `auto_convert` | boolean | Convert diffusers-format LoRAs to native FLUX format |
| `edit_mode` | dropdown | Semantic edit preset — None, Preserve Face, Preserve Body, Style Only, Edit Subject, Boost Prompt, or **Auto** |
| `balance` | float | 0.0 = full preset effect, 1.0 = standard LoRA (0.0 to 1.0) |
| `lora_name_override` | string (link) | Optional — overrides the dropdown when connected |
| `layer_strengths` | string (link) | Optional — per-layer JSON from Auto Strength node |

The graph widget shows double blocks (8 columns, img purple / txt teal, split top/bottom) and single blocks (24 columns, green). Drag to adjust. Shift-drag moves all bars in a section.

### FLUX LoRA Quad

**4 LoRA slots with independent edit_mode and balance per slot.** Designed for editing workflows where different LoRAs need different protection levels.

Each slot has: `lora_name`, `strength`, `edit_mode`, `balance`, `enabled`, `auto_convert`.

Recommended setup for image editing:
```
Slot 1: editing LoRA       → edit_mode=Auto (or Preserve Body), strength=0.6-0.8
Slot 2: consistency LoRA   → edit_mode=Auto (or None),          strength=0.4-0.6
Slot 3: enhancer LoRA      → edit_mode=Auto (or None),          strength=0.2-0.4
Slot 4: (optional)         → as needed
```

### FLUX LoRA Stack

Apply up to 10 LoRAs in sequence with independent strength, enable toggle, and auto-convert per slot. Supports global `edit_mode` and `balance` applied to all slots. When set to Auto, each LoRA is analyzed individually.

### FLUX LoRA Auto Strength

Reads the LoRA's weight tensors directly and computes per-layer strengths from the actual training signal in the file. Double blocks are analyzed with img and txt streams independently. One knob: `global_strength`.

### FLUX LoRA Auto Loader

Self-contained version of the above — analysis and application in one node. `model` in, patched `model` out.

### FLUX LoRA Scheduled

**Per-step LoRA strength control** using ComfyUI's native Hook Keyframes system. Instead of constant strength, the LoRA effect varies across sampling steps.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | FLUX.2 Klein model |
| `lora_name` | dropdown | LoRA file |
| `strength` | float | Base LoRA strength (0.0 to 2.0) |
| `schedule` | dropdown | Strength curve profile |
| `edit_mode` | dropdown | Semantic edit preset (supports Auto) |
| `balance` | float | Preset balance (0.0 to 1.0) |
| `keyframes` | int | Number of keyframes (2-10, default 5) |

**Returns:** `MODEL` + `HOOKS`. The HOOKS output must be connected to conditioning via the **FLUX Set Cond Hooks** node.

Available schedules:

| Schedule | Curve | Use case |
|---|---|---|
| **Constant** | `1.0 → 1.0 → 1.0 → 1.0` | Standard behavior (no scheduling) |
| **Fade Out** | `1.0 → 0.7 → 0.3 → 0.0` | Editing: apply changes early, restore reference details late |
| **Fade In** | `0.0 → 0.3 → 0.7 → 1.0` | Detailing: preserve reference first, add LoRA effect last |
| **Strong Start** | `1.0 → 0.5 → 0.2 → 0.0` | Aggressive fade-out for maximum reference preservation |
| **Pulse** | `0.3 → 1.0 → 1.0 → 0.3` | Peak effect in mid steps |

### FLUX Set Cond Hooks

Utility node that attaches HOOKS (from FluxLoraScheduled) to conditioning. **Required** for per-step scheduling to work.

```
FluxLoraScheduled → MODEL → CFGGuider
                  → HOOKS → FluxSetCondHooks → conditioning → ReferenceLatent → CFGGuider
```

## Example Workflow

An example workflow file `workflow_scheduled_lora.json` is included. Load it in ComfyUI via drag & drop or File → Load.

The workflow uses: UnetLoaderGGUF → FluxLoraScheduled (Fade Out + Auto edit_mode) → FluxSetCondHooks → ReferenceLatent → CFGGuider → SamplerCustomAdvanced.

## Edit Mode Presets

When using LoRAs for image editing (e.g., changing clothing on a reference photo), the LoRA can corrupt parts of the image you want to preserve — most commonly the face/identity. This happens because FLUX.2 Klein processes image and text in different ways across its layers:

- **Double blocks (0-7):** Image and text streams are isolated — they can't cause text-driven image corruption on their own.
- **Single blocks (0-23):** Joint cross-modal processing — this is where the text prompt overwrites the reference image. Late single blocks (12-23) are the most aggressive.

### Available Presets

| Preset | What it does | Use case |
|---|---|---|
| **None** | Standard LoRA (all layers equal) | Default behavior |
| **Preserve Face** | Dampens late single blocks, keeps img stream intact | Editing while keeping face/identity |
| **Preserve Body** | Aggressively dampens mid+late single blocks | Editing while keeping face + body proportions (figure, breast size, waist) |
| **Style Only** | Reduces img stream in double blocks, dampens late singles | Applying style changes without structural edits |
| **Edit Subject** | Moderate protection on late blocks, slight txt boost | Changing clothing/objects while preserving identity |
| **Boost Prompt** | Strengthens txt stream and mid single blocks | When the prompt isn't being followed strongly enough |
| **Auto** | Analyzes LoRA weights, picks best preset + balance automatically | Zero-config — recommended for most users |

### Auto Mode

When `edit_mode` is set to **Auto**, the node analyzes each LoRA's weight distribution and selects the optimal preset:

- High training signal in late single blocks (editing LoRAs) → **Preserve Body**
- Moderate late-block signal → **Preserve Face**
- Uniform distribution (sliders, enhancers) → **None**

The balance is also computed automatically based on signal concentration. Console logs show which preset was selected:
```
[FLUX LoRA Quad] Slot 1: Auto → Preserve Body (balance=0.25)
[FLUX LoRA Quad] Slot 2: Auto → Preserve Face (balance=0.40)
```

### Balance Slider

The `balance` slider interpolates between the preset and standard behavior:
- **0.0** — full preset effect (maximum protection/boost)
- **0.5** — halfway between preset and standard
- **1.0** — standard LoRA (preset has no effect)

Edit mode works on top of Auto Strength — you can combine automatic ΔW-based calibration with semantic presets.

## Supported LoRA Formats

| Format | Source | Example keys | Handled by |
|---|---|---|---|
| **Native** | ComfyUI, kohya | `diffusion_model.double_blocks.0.img_attn.qkv` | Direct passthrough |
| **Diffusers** | HuggingFace | `transformer_blocks.0.attn.to_q` | Block-diagonal QKV fusion |
| **Musubi Tuner / PEFT** | Modelscope, MuseAI | `single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default` | Key remapping + direct remap |

All formats are auto-detected when `auto_convert` is enabled.

## How Auto Strength works

For every layer pair in the file:

```
ΔW = lora_B @ lora_A
scaled_norm = frobenius_norm(ΔW) * (alpha / rank)
strength = clamp(global * (mean_norm / layer_norm), floor=0.30, ceiling=1.50)
```

Double blocks are processed with img and txt streams independently. Mean layer lands at `global_strength`.

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
