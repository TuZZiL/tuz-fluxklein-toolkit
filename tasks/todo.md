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
