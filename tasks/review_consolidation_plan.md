# Review Consolidation Plan

This file merges the three external reviews into one implementation roadmap.
OPUS is the priority baseline. GPT-5.3 and QWEN are used as confirmation.

## Phase 1 - UX Clarity and Contract Safety

- Add tooltips to all conditioning-node fields.
- Clarify loader field semantics:
  - `auto_strength`
  - `balance`
  - `edit_mode=None`
- Add backend validation and warnings for:
  - `slot_data`
  - `layer_strengths`
- Keep saved-graph compatibility. No breaking field renames in this phase.

## Phase 2 - Shared Architecture Extraction

- Extract shared constants.
- Deduplicate `_build_key_map`.
- Extract loader pipeline helpers from `flux_lora_loader.py`.
- Deduplicate mask preprocessing in conditioning reference helpers.

## Phase 3 - Runtime Hardening and Test Coverage

- Replace silent fallbacks with explicit warnings or structured failure paths.
- Add bounds checking in `lora_meta.py`.
- Add integration/smoke tests for loader flows and malformed JSON handling.
- Bound `_ANALYSIS_CACHE`.

## Phase 4 - Deep Refactors

- Split `lora_meta.py` into reader / runtime API / CLI concerns.
- Split `flux_image_postprocess.py` into smaller pipeline stages.
- Revisit potentially confusing serialized UI values only after migration policy is decided.

