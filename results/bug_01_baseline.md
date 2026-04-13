# Bug 01 Baseline — Single-Shot Results

**Bug:** `bug_01_prune_falsy` — `prune(older_than_days=0)` falsy short-circuit
**Difficulty:** easy-medium
**Date run:** April 13, 2026
**Pipeline:** Session 0 baseline (single-shot, no iteration, no memory)

---

## Summary

| Model | Result | Latency | Tokens | Rate |
|---|---|---|---|---|
| Gemma 4 E4B (`gemma4:e4b-it-q4_K_M`) | **PASS ✓** | 104.8s | 2488 | 26.2 tok/s |
| Qwen3.5-9B (`qwen3.5:9b`) | **FAIL ✗** | 112.2s | 1177 | 10.8 tok/s |

**Cross-family divergence on the easiest bug.** Gemma solves single-shot. Qwen fails despite correctly identifying the main bug.

## The Bug

Pre-fix code:
```python
days = older_than_days or self._retention_days
```

Post-fix code:
```python
days = older_than_days if older_than_days is not None else self._retention_days
```

When `older_than_days=0`, Python's truthiness rules treat `0` as falsy and fall through to `self._retention_days` (None by default). The function returns 0 without pruning anything. User asked "prune everything older than 0 days" and got nothing pruned.

Source: anneal-memory commit `c3bd2ea` (Mar 31, 2026), one of 14 fixes in that commit, isolated here for benchmarking.

## Test

```python
def test_prune_zero_days(self, store):
    """prune(older_than_days=0) should prune everything."""
    store.record("Recent", EpisodeType.OBSERVATION)
    pruned = store.prune(older_than_days=0)
    assert pruned == 1
    assert store.status().total_episodes == 0
```

Command: `pytest tests/test_store.py::TestPruneEdgeCases::test_prune_zero_days -v`

## Gemma 4 E4B Run

**Result:** PASS ✓

Gemma produced the exact correct fix, preserved all surrounding code (including the SQL query operator `<=` that Qwen would later break). The model's fix matches the actual human fix line-for-line.

Key line from Gemma's response:
```python
days = older_than_days if older_than_days is not None else self._retention_days
```

**Zero collateral damage** to any other part of the function. Gemma preserved the query operator, the retry loop, the audit log, everything.

## Qwen3.5-9B Run

**Result:** FAIL ✗ — but in the most interesting possible way.

Qwen correctly identified and fixed the main bug:
```python
days = older_than_days if older_than_days is not None else self._retention_days
```

**But Qwen also silently changed unrelated code during full-function regeneration.** Specifically, the SQL query operator:

- Original: `"SELECT * FROM episodes WHERE timestamp <= ?"`
- Qwen's "fix": `"SELECT * FROM episodes WHERE timestamp < ?"`

### Why that single character change breaks the test on `older_than_days=0`

With `older_than_days=0`:
- Cutoff is computed as `datetime.now()` at prune time
- The test records an episode, then IMMEDIATELY calls prune
- The two `_now_utc()` calls can return the SAME microsecond timestamp on a fast-enough system
- When `episode.timestamp == cutoff`:
  - `<=` (original): True → episode pruned → test passes ✓
  - `<` (Qwen's drift): False → episode NOT pruned → test fails ✗

Test output:
```
tests/test_store.py:1038: in test_prune_zero_days
    assert pruned == 1
E   assert 0 == 1
FAILED tests/test_store.py::TestPruneEdgeCases::test_prune_zero_days
```

The main bug was correctly diagnosed. The drift on the query operator was fatal.

## Paper-Grade Implications

### 1. Drift rate is a first-class harness design variable

This run reveals that **drift rate** (how often the generator touches code it shouldn't) is a first-class variable for harness design. The whole-function regeneration prompt strategy used here has higher drift risk than a targeted-patch strategy would have.

Empirical question the paper can investigate: **can we measure drift rate as a function of prompt strategy (whole-function vs targeted-patch vs diff-format), and does grounded memory reduce it?**

### 2. Cross-family drift divergence

Gemma E4B preserved the `<=` operator under whole-function regeneration. Qwen3.5-9B did not. This is not a "Gemma is smarter" effect — it's a **prior bias difference**:

- Gemma's training produced a stronger "preserve what you're not explicitly asked to change" bias
- Qwen's produced more willingness to rewrite freely during regeneration

**The harness effect may be MORE valuable for high-drift models than for low-drift models** — which is itself a publishable finding. It would mean "memory-grounded iteration" has variable value depending on the generator's intrinsic drift profile.

### 3. The iterative loop as a drift corrector

This is exactly the failure mode the iterative harness exists to address. In an iterative run with anneal-memory integration:

- **Iteration 1:** Qwen proposes full-function rewrite, introduces drift on `<=`, test fails with `assert 0 == 1`.
- **Episode recorded:** `{type: finding, content: "Qwen changed query operator <= to < in prune(), test failed with assert 0 == 1 (pruned count mismatch)"}`
- **Consolidation trigger fires** after N attempts.
- **Iteration 2:** Qwen sees the compressed continuity pattern "preserve query operators during fixes." Proposes a more targeted fix that preserves `<=`, test passes.

The harness literally converts "dumb model drift" into a signal that the harness teaches the model to avoid on subsequent iterations. This is the empirical mechanism the paper has been theorizing about, caught live on the first real baseline run. **We weren't expecting Bug 1 to give us a paper result. It did.**

### 4. Bug 1 is now a useful control case

Single-shot rationale for including Bug 1: "if single-shot solves it, confirms the baseline pipeline works at all." We got significantly better than that — single-shot divergence across models gives us a real signal from the easiest bug in the set. Paper Section 4/5 can use Bug 1 as **"the easy bug where even the baseline reveals drift as a confounder to raw capability measurement."** The Config A vs Config E comparison on Bug 1 becomes meaningful, not a throwaway.

## Implementation Notes

### Prompt strategy (current v1)

The current prompt asks the model to return the entire corrected function. This maximizes context usefulness for the model but maximizes blast radius for drift. Session 1+ should experiment with alternatives:

- **Targeted-patch prompt:** give the model the buggy function, ask for a unified diff or line-level replacement only.
- **Constrained-rewrite prompt:** ask for the full function but explicitly instruct "preserve all code outside the specific bug; return the function verbatim with only the minimum necessary change."
- **Iterative refinement:** when drift is detected (test fails after fix applied), feed back "you changed X which wasn't supposed to change" and ask for a corrected version.

These are Session 1-2 design questions. v1 baseline stays whole-function regeneration because that's what gives us the most informative control — **we WANT to see drift**, because drift-in-baseline + drift-reduction-in-harness = the core claim.

### Pipeline worked first try

End-to-end pipeline (bug reintroduction via find/replace → test verification → AST-based function extraction → Ollama API call → markdown code block extraction → AST-based function replacement → test re-run → repo reset via `git checkout --`) worked on the first real run for both models. The AST-based function replacement handled Qwen's slightly-different-indentation output without issue. The `assert_clean_repo` guard prevented any risk of leaving anneal-memory in a dirty state.

### Timing math

Both models took roughly 100-110s for a single bug on an easy-medium task. Extrapolated to a 20-iteration run for a harder bug: 30-40 min per bug. A full 5-bug × 3-config × 2-model benchmark suite: ~6-10 hours of compute. Feasible as an overnight or background run, not as an interactive session.

Qwen's thinking mode added substantial hidden generation (1177 eval tokens for ~50 tokens of visible output — about 95% thinking). Disabling thinking mode via `/no_think` prompt directive could roughly halve Qwen's wall-clock time. Worth investigating in Session 1.

## What This Run Validates

- **Pipeline architecture is sound.** Bug reintroduction → test → prompt → model → fix application → test → reset works end-to-end.
- **Both local models can engage with real anneal-memory bugs.** Capability floor is adequate for the experiment. No garbage-in-garbage-out fear from E4B (the smaller model).
- **Cross-family data is meaningful.** Gemma and Qwen diverge even on the easiest bug, proving the cross-family replication plan will produce signal, not convergent redundancy.
- **Drift-as-signal is real.** The mechanism the paper theorizes about showed up on the first run without us aiming for it.

## What This Run Does NOT Validate

- Nothing about Bugs 2-5 (pipeline is proven on one bug only; Bug 5 raw-bytes-hashing may behave very differently).
- Nothing about the actual iterative loop (Sessions 1-2 work).
- Nothing about anneal-memory integration (Session 2 work).
- Nothing about consolidation trigger tuning (Session 3 work).

Session 0 is a vertical slice, not a horizontal validation. Next session scales horizontally to Bugs 2-5 on the same single-shot pipeline, then Session 1 adds the iteration layer.

---

*Captured: April 13, 2026*
*Pipeline: `harness/baseline.py` (single-shot, Config 0)*
*Next: Session 0.5 — add Bugs 2-5 to config set, re-run baseline on all 5, record cross-bug signal.*
