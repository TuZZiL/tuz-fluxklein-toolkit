# Preflight Advisor MVP

- [x] Add a pure policy module for LoRA preflight scoring and recommendations.
- [x] Add an isolated `Preflight Advisor` node for single LoRA analysis.
- [x] Add an isolated `Multi Preflight Advisor` node for slot-based analysis.
- [x] Keep the MVP advisory-only and exclude any `schedule` logic.
- [x] Add unit tests for policy heuristics and multi-slot overlap handling.
- [x] Update README files with the new advisor node and its outputs.
- [x] Run syntax checks and available tests.
- [x] Summarize implementation notes and residual risks.
- [x] Fix hidden `slot_data` widgets so ComfyUI keeps them in `widgets_values` during workflow serialization.

## Summary

- The advisor is now implemented as an isolated, advisory-only path.
- Single-LoRA and multi-slot flows both emit structured recommendations.
- Schedule logic was intentionally left untouched.
- Hidden data widgets that must survive reload should preserve their original type instead of being forced to `converted-widget`.

## Balance UX Inversion

- [x] Invert the preset mix slider so `0.0` means raw LoRA and `1.0` means full preset protection.
- [x] Update loader tooltips, multi-slot labels, and README examples to match the new semantics.
- [x] Add a regression test that locks the inverted interpolation behavior.
- [x] Verify the generated logs still read clearly after the semantic flip.

## Summary

- The preset mix dial now behaves intuitively: `0.0` is raw LoRA and `1.0` is full preset protection.
- Loader tooltips, multi-slot labels, advisor output, and README examples were updated to match the new scale.
- Targeted unit tests passed; full `unittest discover` is still blocked by missing optional `torch` and `numpy` dependencies in unrelated tests.

## Companion Conditioning Docs

- [x] Add a collapsible practical guide for the companion conditioning nodes in both README files.
- [x] Expand each companion conditioning node with concrete usage guidance, starting values, and workflow notes.
- [x] Keep the English and Ukrainian guides structurally aligned.

## Summary

- Both README files now include a collapsible practical guide for the companion conditioning nodes.
- Each companion node now has a practical “what it changes / when to use / how to tune” section.
- The English and Ukrainian docs were kept structurally aligned so they can be maintained together.

## OPUS Brainstorm Brief

- [ ] Hand the new companion conditioning brief to Opus and collect its markdown research output.
- [ ] Review the ranked node ideas and extract the top candidates for implementation.
- [ ] Decide which ideas should extend existing nodes versus become new nodes.

## StructureLock

- [x] Implement `StructureLock` as a sampler-adjacent companion node.
- [x] Add shared sigma-progress and reference-selection helpers for reuse.
- [x] Add unit tests for structure locking, masking, and no-op behavior.
- [x] Update README docs in both languages with practical usage guidance.

## Anatomy Shield v1

- [ ] Finalize profile registry schema (`anatomy_profiles.py`) with intent-based presets.
- [ ] Implement anatomy profile application stage in `lora_pipeline.py` after `edit_mode` merge.
- [ ] Add `strict_zero` handling for selected block ranges.
- [ ] Add anatomy inputs to `FluxLoraLoader`, `FluxLoraMulti`, and Composer slot schema.
- [ ] Add unit tests for profile interpolation, band expansion, and merge precedence.
- [ ] Add contract tests for new JSON fields in slot payloads.
- [ ] Update README EN/UA with practical “choose profile” guide and troubleshooting.
- [ ] Validate on regression matrix (human male/female, robot, stylized, armor scenarios).

## Summary

- Added implementation plan draft: `tasks/anatomy_shield_implementation_plan.md`.
- Plan includes scope, architecture, profile catalog with starter values, phased rollout, and user-facing practical workflow.
