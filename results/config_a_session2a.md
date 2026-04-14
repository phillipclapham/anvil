# Config A Session 2a — Vertical-Slice Validation

**Session type:** Session 2a (vertical slice of Session 2 scope).
**Cell:** bug_03_retention_wiring × gemma4:e4b-it-q4_K_M.
**Date:** April 14, 2026.
**Goal:** Validate that `harness/loop_anneal.py` integrates anneal-memory
end-to-end on one cell before scaling to the full N≥3 matrix in Session 2b.
**Verdict:** ✓ All 5 pipeline-validation criteria met. Compression delta
structurally verifiable in prompt-length data. N=1 outcome does NOT satisfy
the Session 2 kill criterion evaluation (that's N=3).

## Summary

| Metric | Value |
|---|---|
| Config | A (anneal-memory integration) |
| Max iters | 5 |
| Iterations used | 5 / 5 |
| Termination | `max_iters` |
| Passed | False |
| Failure classes | `hallucination × 5` |
| Fix hashes | all distinct (no cycles) |
| Consolidations | 1 (fired at iter 3 as designed) |
| Episodes compressed | 4 (seed + iters 1-3) |
| Continuity chars | 1101 |
| Total latency | 1857.0s (31 min) |
| Total tokens | 25,914 |
| Store path (forensic) | `/var/folders/3_/.../anvil_store_salfnq6w` |

## The compression delta is real and structurally verifiable

The most important Session 2a data point is NOT pass/fail — it's the
**prompt-length signature** across iterations. The compression layer
rebuilds iter 4's prompt from compressed continuity instead of raw
history. The numbers match the design to ±15 chars.

### Per-iteration prompt lengths (Config A vs Config E)

| Iter | Config A prompt | Config E prompt (first run) | Delta |
|---|---|---|---|
| 1 | 18,142 | 18,142 | 0 |
| 2 | 19,209 | 19,209 | 0 |
| 3 | 20,016 | 20,016 | 0 |
| 4 | **19,488** ← consolidated | 20,823 | **-1,335** |
| 5 | 20,342 | 21,630 | **-1,288** |

(Config E numbers are from the first run of this cell, before the
`'dict' object has no attribute 'status'` bug fix — that run effectively
ran Config E semantics because consolidation crashed silently. The fix
is committed in loop_anneal.py.)

### Math verification

- Base prompt ≈ 18,142c (iter 1 prompt, with zero prior history, equals the base).
- Iter 4 under Config A = `base + "# Consolidated prior attempts\n\n" + continuity + closing instruction`
  - 18,142 + ~30 (header) + 1,101 (continuity) + ~200 (instructions) ≈ **19,473**
  - Observed: **19,488** (delta = 15c, matches within instruction-string precision)
- Iter 5 under Config A = `base + continuity + 1 recent unwrapped (iter 4 episode)`
  - iter 4 episode contributes ~850c of def/indented lines via `build_config_a_prompt`'s episode formatter
  - 19,488 + ~850 ≈ **20,338**
  - Observed: **20,342** (delta = 4c, matches exactly)

The compression architecture is **structurally verified** — the bytes
entering the model match the design, not just the pipeline executing
without errors.

### Compression scaling note

At N=5 iterations the delta is modest (~6% reduction) because (a) only
3 iterations of material were available to compress by the time iter 4
prompt was built, and (b) the rule-based continuity template is itself
~1,100 chars so it partially replaces rather than fully compressing the
raw material. At 10+ iterations the ratio widens fast — continuity
grows sub-linearly (template + a few lines per new failure class)
while raw history grows linearly (~800c per iteration).

## Per-iteration forensics

| Iter | Termination prompt | Latency | Tokens | fix_hash | Failure class |
|---|---|---|---|---|---|
| 1 | 18,142c | 335.1s | 5,276 | `55cb91ccc336` | hallucination |
| 2 | 19,209c | 351.6s | 5,294 | `1bc8ce6030c7` | hallucination |
| 3 | 20,016c | 368.2s | 5,182 | `33f0692c2149` | hallucination |
| 4 | **19,488c** ← compressed | 420.5s | 5,409 | `1dd1fcb3a361` | hallucination |
| 5 | 20,342c | 381.6s | 4,753 | `0f42a74a7b63` | hallucination |

**All 5 fix hashes distinct** → `detect_no_progress` correctly did NOT
fire (the window detector is a cycle-by-hash check, not a failure-class
check). No termination before `max_iters`. Gemma at `temperature=0.2`
produces enough per-call variance that even similar wrong answers don't
exact-match their own prior attempts.

### The consolidated continuity document

The rule-based template produced this at iter 3:

```markdown
## State

- Bug: bug_03_retention_wiring
- Model: gemma4:e4b-it-q4_K_M
- Total attempts: 4 (0 pass, 4 fail)
- Unique failure modes by fix_hash: 4

## Context

Prior distinct failure attempts (most recent first):

- Iter 3 (hallucination, 33f0692c): def wrap_completed(
- Iter 2 (hallucination, 1bc8ce60): def wrap_completed(
- Iter 1 (hallucination, 55cb91cc): def wrap_completed(
- Iter ? (unknown, ): Problem: fix Store.wrap_completed in anneal_memory/store.py...

## Patterns

- hallucination: 3 distinct attempt(s)
  - Failure mode: fix referenced a symbol that doesn't exist on the
    target class. Verify every attribute access against the method's
    class definition before returning.
- unknown: 1 distinct attempt(s)

## Decisions

- Do NOT repeat any of the fix_hashes listed in Context. These
  attempts were already tried and failed.
- Read the failing test in the prompt carefully — it shows exactly
  what the fixed method must satisfy.
- Every attribute referenced in the fix must exist on the class. Do
  not invent methods or attempts.
```

**Known quirk:** the seed CONTEXT episode (recorded at loop start as
the problem statement) leaked into the Context section as
`Iter ? (unknown, )` because `synthesize_continuity_text` treated it
as a failed attempt with missing metadata. Minor clutter, doesn't
break anything, 2-line fix logged for 2b.

## Methodology finding: no determinism shortcut available

Before running the main Session 2a cell, a throwaway `determinism_probe.py`
tested Ollama with `temperature=0.0` + `seed=42` on bug_01 × Gemma,
calling twice with byte-identical prompts:

- Call 1: latency 81.2s, 1832 tokens, fix_hash `62f695c7e6f3`
- Call 2: latency 95.4s, 2138 tokens, fix_hash `12b31019398e`

**Different fix_hashes. Different token counts. Different latencies.**

Ollama does not honor `seed` determinism across separate calls at the
backend level for Gemma 4 E4B quantized weights. Session 2b MUST
commit to N≥3 replication at `temperature=0.2` as the only honest
variance control. "Just turn the temperature down" is not a valid
substitute for replication at this model/quantization layer.

*(Side observation: both deterministic calls preserved the buggy
`or` pattern on bug_01 and missed the bug entirely, adding another
data point to the Session 0.5 → Session 1 variance flip story. Gemma
is sometimes finding bug_01's fix and sometimes not, and it's mostly
noise rather than prompt-dependent signal.)*

## The hallucination-persistence finding (paper-relevant)

**All 5 Config A iterations classified as hallucination with distinct
symbols.** The continuity document explicitly warned in its Decisions
section: *"Every attribute referenced in the fix must exist on the
class. Do not invent methods or attributes."* Gemma invented different
fake attributes on each subsequent iteration anyway.

**This is a real Section 5 finding, not a Config A failure.** Structural
memory of "you hallucinated before, don't do it again" is NOT sufficient
to prevent future hallucination on the same class of bug. The
`hallucination` class appears to be grounding-bound, not memory-bound:
what would actually help is putting the target class's real attribute
list in the prompt (which is grounding, not memory).

Different iterative-harness failure classes may benefit from different
interventions:

- **Drift** (operator mutation during whole-function regen): memory
  probably helps — "iter 1 changed `<=` to `<` in the SQL query" is
  actionable structural guidance the model can respect.
- **Syntax** (unparseable response): memory probably helps — "iter 1's
  response had no code block" is actionable.
- **Hallucination** (invented symbols): memory does NOT seem to help.
  Grounding (class definition in context) is probably the right
  intervention. Memory would help on the meta-layer: "when you see a
  hallucination-class failure, fetch grounding before retrying."

This splits the memory-without-grounding thesis into a more nuanced
claim: **memory-without-grounding is sufficient for some failure
classes and insufficient for others.** Session 2b's N=3 matrix across
drift-heavy cells (bug_01) and hallucination-heavy cells (bug_03) will
generate the first data point on this nuance. Paper Section 5 should
discuss it explicitly rather than hand-waving.

## Kill criterion NOT triggered

The Session 2 kill criterion states: *If Config A pass rate is not
strictly better than Config E pass rate on bug_01 × Gemma OR bug_03 ×
Gemma at **N=3**, the memory-without-grounding thesis at this scale is
in trouble.*

bug_03 × Gemma at **N=1** Config A failed. Config E at N=1 also failed
this cell in Session 1 (no_progress after 3 hallucinations). At N=1
the two arms are tied at 0 pass / 0 pass. This is below the noise floor
for any comparison — exactly the methodology problem Session 1 surfaced.

The kill criterion evaluation happens in Session 2b after the N=3
matrix runs on both arms. Do NOT update paper framing based on Session
2a N=1 data.

## Session 2b scope locked

Session 2a validated the pipeline. Session 2b runs the experiment. The
scope is unchanged from Session 1's scope-addition list with two new
items from 2a findings:

1. **N≥3 per cell minimum** (Session 1 finding). At `temperature=0.2`
   because determinism probe showed `temp=0.0 + seed` is not deterministic
   across calls.
2. **Single-model cell sequencing** (Session 1 finding). All Gemma cells
   first, unload, then all Qwen cells.
3. **Subprocess cell isolation** (Session 1 + 2 finding). OS-level kill
   recovers repo state when threading.Timer's hard-timeout still isn't
   enough.
4. **Refined classifier** (Session 1 finding). AST-diff separation of
   semantic hallucination vs operator drift — Session 2a's
   `bug_03 × Gemma` case shows the current classifier is conflating
   distinct hallucinated-symbol attempts under one "hallucination" label,
   which is actually OK but we want finer granularity for 2b reporting.
5. **NEW — Seed episode filter in continuity synthesizer**. Session 2a
   finding: `synthesize_continuity_text` leaks seed CONTEXT episodes
   into the Context section as `"Iter ?"` entries. 2-line fix: filter
   on `metadata.get("seed")` before iterating.
6. **NEW — Per-iteration stderr progress logging**. Session 2a finding:
   the 32-min bug_03 × Gemma cell was opaque to monitoring because
   `loop_anneal.py` prints only at completion. Add per-iter "iter N/M:
   backend call..." lines to stderr. Helps distinguish "hung at iter 3"
   from "normally slow at iter 3" during long matrix runs.
7. **Full Config A vs Config E matrix** on 7 feasible cells (asymmetric,
   drops bug_03 × Qwen as infeasible, accepted in Session 1).
8. **Load-bearing cells:** bug_01 × Gemma (drift-class exhibit), bug_03
   × Gemma (hallucination-class exhibit). Kill criterion evaluated on
   these at N=3.
9. **Saturation controls preserved:** bug_04 × both + bug_05 × Gemma
   must not regress under Config A.

## Files changed

- `harness/loop_anneal.py` — NEW. 527 lines. Config A iteration loop.
- `harness/determinism_probe.py` — NEW. 170 lines. One-shot methodology probe.
- `results/raw/config_a_bug_03_retention_wiring_gemma4_e4b-it-q4_K_M.json` —
  NEW. Full LoopResultA JSON dump from the validated Session 2a run.

## Files NOT changed

- `harness/loop.py` — unchanged. Config E arm stays as-is for Session 2b comparison.
- `harness/backend.py` — unchanged. Backend interface stable.
- `harness/baseline.py` — unchanged. Primitives reused cleanly.
- No changes in `anneal-memory/`. Library used via read-only API.
