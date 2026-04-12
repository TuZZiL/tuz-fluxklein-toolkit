# Lessons

- Keep the preflight advisor schedule-free. The advisor should stay focused on compatibility, edit mode, balance, and starter strength only.
- Put advisory heuristics in a pure module first. That keeps the logic testable without ComfyUI runtime dependencies.
- When UX labels are already serialized in saved ComfyUI graphs, improve tooltips and validation first; postpone breaking renames to a later migration phase.
