# Project Report

## 2026-04-16
- context: Reviewed README clarity for end users.
- done: Assessed README structure, onboarding flow, and terminology density.
- done: Implemented beginner-friendly README refresh in `README.md` and `README_UA.md`.
- resolved: Added start-here flow, screenshot, glossary, beginner recipes, troubleshooting, and optional advanced reference labels.
- resolved: Clarified loader `protection` vs `balance` and the separate `Text/Ref Balance` control.
- done: Fixed `Structure Lock` device mismatch by moving reference latent onto the sampler tensor device before blending.
- resolved: `apply_structure_lock` now works with CUDA `denoised` tensors and CPU-origin reference latents.
- next: If desired, do a final read-through on GitHub-rendered markdown for spacing/visual balance.

## 2026-04-15
- context: Start of session.
- done: Initialized project memory file.
- next: Capture concrete task context when work starts.
- done: Investigated Flux Multi state-loss after workflow JSON transfer between machines.
- resolved: Added persistence fallback for Multi slot state in `node.properties.slot_data_json` and restore logic in `js/flux_lora_multi.js` (`onConfigure` + `onSerialize`).
- context: Root risk is hidden-widget (`slot_data`) serialization inconsistency across ComfyUI installs; fallback now keeps selected LoRAs/values recoverable.
- done: Added anatomy controls to `FluxLoraMulti` expanded cards (`anatomy_profile`, `anatomy_strength`, `anatomy_strict_zero`, custom JSON editor for `Custom` profile).
- done: Added compact anatomy status hint per slot and profile-list loading from `FluxLoraLoader` object info with local fallback values.
- resolved: Multi UI now exposes anatomy slot fields already supported by backend `slot_data` contract.
- resolved: Fixed visual misalignment in Multi anatomy row by unifying `Strict zero` and `Custom JSON` geometry/typography (same height and baseline rhythm).
- done: Performed focused UI quality review for `FluxLoraMulti` card widget (`js/flux_lora_multi.js`) with usability/clarity/risk findings and improvement priorities.
- done: Rebranded project metadata for Comfy Manager submission to `TUZ FluxKlein Toolkit` (`pyproject` DisplayName + normalized package name + README titles).
- done: Updated repository/documentation/issue URLs and install clone commands to renamed repo `https://github.com/TuZZiL/tuz-fluxklein-toolkit`.
- done: Aligned Loader docs to new `protection` naming with explicit legacy `balance` compatibility notes (EN/UA).
- resolved: Unified protection field naming across Loader/Multi/Scheduled (`protection` primary, legacy `balance` accepted for backward compatibility in Python + Multi JS slot contract).
