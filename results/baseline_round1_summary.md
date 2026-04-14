# Baseline Round 1 — Single-Shot Results Across 4 Bugs × 2 Models

**Session:** Anvil Session 0.5
**Date run:** April 13, 2026
**Pipeline:** `harness/baseline.py` (single-shot, Config 0 — no iteration, no memory)
**Purpose:** Scale Session 0's vertical slice horizontally to produce cross-bug signal for Session 1 iteration experiments.

---

## Summary Matrix

| Bug | Difficulty | Module | Gemma 4 E4B | Qwen3.5-9B |
|---|---|---|---|---|
| 1 — `prune(older_than_days=0)` falsy | easy-medium | store.py | **PASS** ✓ | **FAIL** ✗ (drift) |
| 3 — `retention_days` auto-prune wiring | medium | store.py | **FAIL** ✗ (hallucination) | **TIMEOUT** ⏱ |
| 4 — `_initialized` flag placement | hard | audit.py | **PASS** ✓ | **PASS** ✓ |
| 5 — double-orphan dedup (`.gz` vs `.jsonl`) | hard | audit.py | **PASS** ✓ | **PASS** ✓ |

**Pass rate:** Gemma 3/4 (75%) · Qwen 2/4 (50%, one case incomplete due to timeout).

**Bug 2 (LIKE wildcards) was dropped from the benchmark** after investigation revealed the existing tests (`test_percent_in_keyword`, `test_underscore_in_keyword`) are non-discriminating under the bug — both happen to produce the same result count buggy vs fixed because the test inputs don't trigger the specific false-positive case the LIKE-wildcard bug creates. This is itself a useful finding: a test-design gap that survived anneal-memory's 4-layer review pipeline. The bug is real, the fix is correct, the tests don't catch it.

---

## Per-Bug Detail

### Bug 3 — `retention_days` not wired into `wrap_completed`

**Gemma 4 E4B:** FAIL — **hallucinated API that doesn't exist**

- Latency: 194.5s
- Tokens: 4,171
- Fix applied cleanly (no syntax error, AST surgery succeeded)
- Test failure at runtime:
  ```
  E   AttributeError: 'Store' object has no attribute 'has_cleanup_enabled'
  ```
- Gemma invented two attributes — `self.has_cleanup_enabled` and `self.cleanup_enabled` — neither of which exist on the `Store` class. It fabricated plausible-looking API surface around its fix logic instead of using the actual `self._retention_days` instance variable that the buggy code showed it.

**This is a fundamentally different failure mode from Qwen's Bug 1 drift.** Drift is *silent mutation of correct unrelated code* during whole-function regeneration. Hallucination is *fabrication of API that doesn't exist anywhere in the context*. Both slip past AST-based fix application because neither produces a syntax error; both fail at runtime against the test oracle. An iterative harness with episodic memory can correct either:

- **Drift → episode:** `finding: "Qwen changed <= to < in unrelated SQL query; test failed on microsecond edge"` → next iteration preserves operators.
- **Hallucination → episode:** `finding: "Gemma referenced self.has_cleanup_enabled which doesn't exist; test failed with AttributeError"` → next iteration uses only attributes present in the buggy code.

The paper's core mechanism — "feed the failure back as a typed episode, consolidate, next iteration avoids it" — is validated in shape by this run before we've even started iterating.

**Qwen3.5-9B:** TIMEOUT — exceeded 900s `call_ollama` budget (bumped mid-session from 300s)

- Qwen's thinking mode scales badly with prompt size. `wrap_completed` is ~180 lines after the 10.5c.4→10.5c.6 refactor sequence (TOCTOU fix + two-phase commit + SQLite error wrapping); the pre-fix function body fed into the prompt is almost as large. Exact 15:00.32 wall time on the retry confirms it was hitting the socket timeout, not completing.
- `next.md` already has the fix scoped for Session 1: `/no_think` prompt directive to disable Qwen's reasoning mode. This is the first empirical case demonstrating why that Session 1 task is load-bearing, not cosmetic.
- Bug 3 × Qwen remains an empty cell for Round 1; will be re-run in Session 1 with `/no_think` as the first validation of the directive.

### Bug 4 — `_initialized` flag set before init complete

Both models PASSED on the first try.

| Model | Latency | Tokens |
|---|---|---|
| Gemma 4 E4B | 168.7s | 2,886 |
| Qwen3.5-9B | 148.0s | 1,283 |

This is the "hard" bug in the set by spec (multi-site edit, subtle init-retry semantics, monkeypatch-based test). Both models correctly identified that the test was asserting `_initialized is False` after a failure and moved the flag to the end of the method. The first real datum suggesting that bug difficulty from a human author's perspective (this was Diogenes Sweep 4-7 MEDIUM) does NOT directly map to model difficulty — the reasoning structure ("flag set too early → move to end") is compact even if the crash-recovery semantics are subtle.

Notable: Bug 4 required the **multi-site `bug_reintroduction`** support added to `baseline.py` this session (three coordinated edits: add flag at top + remove from the fresh-start branch + remove from the main branch end). First validation that the harness-extension pattern works.

### Bug 5 — double-orphan adoption (`.gz` vs `.jsonl` for same period)

Both models PASSED.

| Model | Latency | Tokens |
|---|---|---|
| Gemma 4 E4B | 172.3s | 3,063 |
| Qwen3.5-9B | 350.9s | 3,073 |

This substituted for the original Bug 5 (raw-bytes hashing) after pre-session investigation showed the latter was not testable in Python-only — `_compute_hash` strips whitespace, making reverted `json.dumps` vs on-disk hashing byte-identical under CPython. The commit message itself says the fix is defensive against cross-language verifiers. Substituted with the double-orphan dedup fix from the same Apr 3 commit (`65ade7e`), which HAS a direct regression test (`test_double_orphan_prefers_gz_and_removes_jsonl`).

Both models correctly reasoned through the crash scenario (both `.gz` and `.jsonl` exist for the same period when rotation crashes between gzip completion and unlink), grouped orphans by period, preferred `.gz`, removed the `.jsonl` duplicate, and preserved the downstream manifest-building loop. Qwen's 350.9s run is the case that forced the mid-session timeout bump from 300s to 900s — it finished on the retry without intervention because each `python3 -m harness.baseline` invocation is a fresh subprocess that reads the updated source file.

---

## Cross-Bug Signal for Session 1

The four-bug matrix produces a clean decision for which bugs advance to iteration experiments:

| Bug | Gemma | Qwen | Harness signal? |
|---|---|---|---|
| 1 prune falsy | PASS | FAIL (drift) | ✓ — cross-family divergence, Qwen needs harness |
| 3 retention wiring | FAIL (hallucination) | TIMEOUT | ✓✓ — Gemma needs harness, Qwen needs `/no_think` + harness |
| 4 _initialized flag | PASS | PASS | ✗ — both solve single-shot, useless for harness delta |
| 5 double-orphan dedup | PASS | PASS | ✗ — both solve single-shot, useless for harness delta |

**Recommended Session 1 focus:** Bugs 1 and 3. Both produce clean per-model failure signal on single-shot, which means iteration + memory has room to show measurable improvement. Bugs 4 and 5 are saturated at single-shot — running an iteration loop on them would measure overhead, not correction value.

**Do not drop Bugs 4 and 5 from the corpus.** Both are useful as **control cases** — they set the ceiling for how much single-shot already achieves on "hard by human assessment" bugs. Keeping them in the Round 2 matrix lets us report "iteration helps on Bug 1 and Bug 3, and does not regress on Bug 4 and Bug 5" — that's a stronger claim than reporting only the bugs where iteration helps.

## Two Distinct Failure Classes Captured

The most unexpectedly rich finding from this run: we already have **two structurally distinct single-shot failure modes** from TWO different models on TWO different bugs, both before the iteration loop has been implemented.

| Failure class | Where | Mechanism | Why AST surgery doesn't catch it |
|---|---|---|---|
| **Drift** (Qwen Bug 1) | Whole-function regeneration touches correct code | Unrelated SQL operator `<=` → `<` during regen | Syntactically valid; test fails at runtime on edge case |
| **Hallucination** (Gemma Bug 3) | Fix references invented API surface | `self.has_cleanup_enabled` doesn't exist on `Store` | Syntactically valid; test fails at runtime with AttributeError |

Both failure classes produce `fix_ok=True` in the baseline pipeline (the pipeline only checks "can I insert this into the class?"), and both fail only against the test oracle. This validates the decision to use **code execution** as the instrument layer from Session 0 onward — no LLM-as-judge would reliably detect either failure mode (Gemma's hallucinated API would look plausible to another LLM; Qwen's one-character drift would read as "minor refactor").

## Harness-Effect Framing for the Paper

Both failure modes feed cleanly into the iterative harness design:

- Iteration 1: model attempts fix → instrument layer (tests) returns structured failure → typed episode recorded (`type: finding`, `content: "AttributeError: Store has no attribute has_cleanup_enabled"` or `"test failed: pruned count 0 != expected 1, query operator changed <= to <"`).
- Consolidation fires after N attempts → compressed continuity surfaces "when fixing Store methods, reference only attributes present in the current class body" or "preserve query operators during whole-function regeneration."
- Iteration 2+: model sees compressed guidance, avoids the failure class.

The iterative loop is specifically a **failure-class-to-structural-guidance converter** driven by the episodic→continuity compression pipeline in anneal-memory. Session 0.5 provides the empirical data for that claim: different models exhibit different failure classes on the same corpus, and each failure class is precisely the kind of correction the harness is designed to feed back through its memory layer.

## Latency Observations

| | Min | Median | Max |
|---|---|---|---|
| Gemma 4 E4B | 104.8s (Bug 1) | 170.5s | 194.5s (Bug 3) |
| Qwen3.5-9B | 112.2s (Bug 1) | 249.5s | 900s+ timeout (Bug 3) |

Gemma is stable in the 100–200s range across bugs regardless of prompt size. Qwen is 100–350s on smaller prompts (Bugs 1, 4, 5) but exceeds 900s on the largest prompt (Bug 3's `wrap_completed`, ~180 lines). This is a direct function of Qwen3's thinking mode generating `<think>` tokens proportional to input complexity with no hard cap. `/no_think` directive is the intended remediation and is now empirically justified for Session 1.

Sustained benchmark cost estimate at current latency (Round 1, 4 bugs × 2 models × single-shot): ~25 minutes total. Round 2 with a 20-iteration harness budget per bug: ~3–5 hours per full sweep. Feasible as background runs.

## Harness Extensions This Session

1. **Multi-site `bug_reintroduction`** — `reintroduce_bug()` in `harness/baseline.py` now accepts either a single `{find, replace}` dict (Bugs 1, 3, 5) OR a list of `{find, replace}` dicts (Bug 4). All edits must succeed or reintroduction fails atomically. Validated on Bug 4 which needs three coordinated edits across `_initialize()`.
2. **`call_ollama` timeout 300s → 900s** — Bumped mid-session after Bug 3 × Qwen timeout. Bug 5 × Qwen (350.9s) finished on the retry because each baseline invocation is a fresh subprocess. 900s is not enough for Qwen on the largest prompts; `/no_think` (Session 1) is the real fix.

## Deferred to Session 1

- **Bug 3 × Qwen rerun** with `/no_think` directive. First empirical test that `/no_think` preserves Qwen's fix-generation quality while eliminating thinking-mode latency.
- **Iteration loop** (Config E naive context — control arm). Run on Bugs 1 and 3 where single-shot has real failure signal; Bugs 4 and 5 as saturation controls.
- **Anneal-memory integration** (Config A experimental arm). Same bug corpus.
- **Prompt strategy experiments** — whole-function regeneration vs targeted-patch vs diff-format. Directly motivated by Bug 1 drift + Bug 3 hallucination both being whole-function-regeneration artifacts.

## Files Produced

- `bugs/bug_03_retention_wiring.py`
- `bugs/bug_04_initialized_flag.py` (multi-site)
- `bugs/bug_05_double_orphan_dedup.py` (substituted for original raw-bytes hashing framing)
- `harness/baseline.py` — extended for multi-site `bug_reintroduction`, timeout bump
- `results/raw/session_0_5/*.json` — per-run raw output (6 runs + 1 Qwen retry timeout)
- `results/session_0_5_aggregated.json` — programmatic aggregated matrix
- `results/baseline_round1_summary.md` — this file

---

*Captured: April 13, 2026*
*Next: Session 1 — Core iteration loop (Config E naive context) on Bugs 1, 3, 4, 5. First test of `/no_think` directive for Qwen. First cross-model cross-bug iteration-vs-single-shot comparison.*
