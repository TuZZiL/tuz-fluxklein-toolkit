# Lessons

- Keep the preflight advisor schedule-free. The advisor should stay focused on compatibility, edit mode, balance, and starter strength only.
- Put advisory heuristics in a pure module first. That keeps the logic testable without ComfyUI runtime dependencies.
- When UX labels are already serialized in saved ComfyUI graphs, improve tooltips and validation first; postpone breaking renames to a later migration phase.
- Hidden state widgets like `slot_data` should preserve their original widget type when ComfyUI needs to serialize them into `widgets_values`; only purely decorative widgets should be converted to `converted-widget`.
- When a user-facing dial is semantically inverted, flip the math, the tooltip, and the log language together. Renaming only the label leaves the UI misleading.
- For companion conditioning docs, describe the actual surface being changed (`conditioning`, `model`, or `reference_latents`), then give starting values and a concrete workflow example. Parameter tables alone are not enough.
- For LLM brainstorming briefs, pin the current baseline, the non-overlap constraints, the scoring rubric, and a few seed directions. Otherwise you get generic ideas that are hard to action.
- For structure-preserving sampler hooks, keep the model narrow: low-pass blend, optional mask, and a time ramp are enough for v1. Extra controls are usually noise unless users ask for them.
- For anatomy-preserving LoRA UX, prefer intent-based profile packs (undress/body-lock/robot-frame/etc.) over rigid demographic-only presets; keep the registry extensible so users can add custom profiles without changing node architecture.
