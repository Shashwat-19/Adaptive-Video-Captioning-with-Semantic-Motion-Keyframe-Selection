# Working rules

## Work loop
1. Read TODO.md and changes.md. Pick the first unchecked item.
2. Implement it in the main session. Keep the diff to that one item.
3. Run the reviewer subagent on the diff.
4. If the reviewer returns BLOCKER or WARN items, fix them in the main session and re-run the reviewer on the new diff.
5. Stop after two review rounds and report what is still open.
6. Only check off the TODO item when the verdict is READY TO COMMIT.

## Reviewer model selection
- Default: run the reviewer with model haiku.
- Use model opus instead when the diff touches any of:
  - src/video_captioning/keyframes/
  - src/video_captioning/evaluation/
  - src/video_captioning/training/
  - splits, seeds, normalization, or metric code
  - README numbers or anything under results/
- Also use opus for any re-review after a BLOCKER fix.
- Do not use the reviewer's verdict alone for leakage or metric questions. Verify those yourself if the reviewer marks them UNVERIFIED.
- The main session chooses the model by passing a model override when invoking the reviewer. A subagent cannot change the model of the session that called it.

## Hard rules
- Never edit results/ or generated notebooks. Change scripts/build_notebook.py instead.
- Never commit or push without my explicit go-ahead.