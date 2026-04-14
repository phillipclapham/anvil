# Config E Round 1 — Session 1 Forensic

**Session:** 1
**Config:** E (naive context iteration — no anneal-memory, full prior-attempt accumulation)
**Date:** April 14, 2026
**Matrix:** 7 cells (4 bugs × 2 models, minus infeasible Bug 3 × Qwen)
**Driver:** `harness/loop.py` + `harness/run_session1.py`
**Backend:** Ollama (Gemma 4 E4B, Qwen3.5-9B), `num_ctx=16384`, `temperature=0.2`, `keep_alive=30m`
**Hard wall-clock guard:** 600s per generate() call via `threading.Timer` (added after the first matrix run hung the urllib socket past its per-socket-op timeout)

---

## Headline

| Metric | Value |
|---|---|
| Cells run | 7 |
| Cells passed | 4 / 7 |
| Termination mix | pass: 4 · no_progress: 2 · error: 1 |
| Failure class mix | drift: 6 · hallucination: 2 · timeout: 1 |
| Total wall latency | 44.2 min |
| Total tokens generated | 37,362 |

**One-line claim:** Config E naive iteration **does not rescue the failure cases** and **reinforces hallucinated mental models** in the worst case. Iteration value without compressed memory is at best a wash, at worst actively harmful.

---

## Matrix

| Bug | Model | Role | Term | Passed | Iters | Wall | Classes |
|---|---|---|---|---|---|---|---|
| bug_01_prune_falsy | gemma4:e4b-it-q4_K_M | iter_target | **no_progress** | False | 5/5 | 838s | drift×5 |
| bug_01_prune_falsy | qwen3.5:9b | iter_target | pass | True | 1/1 | 172s | — |
| bug_03_retention_wiring | gemma4:e4b-it-q4_K_M | iter_target | **no_progress** | False | 3/3 | 1074s | hallucination, hallucination, drift |
| bug_03_retention_wiring | qwen3.5:9b | iter_target | — (infeasible) | — | — | — | dropped before Session 1 — 900s single-shot timeout persists with `/no_think` (validated Apr 14) |
| bug_04_initialized_flag | gemma4:e4b-it-q4_K_M | saturation_ctl | pass | True | 1/1 | 167s | — |
| bug_04_initialized_flag | qwen3.5:9b | saturation_ctl | pass | True | 1/1 | 208s | — |
| bug_05_double_orphan_dedup | gemma4:e4b-it-q4_K_M | saturation_ctl | pass | True | 1/1 | 190s | — |
| bug_05_double_orphan_dedup | qwen3.5:9b | saturation_ctl | **error** | False | 1/1 | 0s | timeout (VRAM deadlock — see §4) |

Saturation controls held for every cell they ran on (Bug 4 × both, Bug 5 × Gemma). Bug 5 × Qwen failed for infrastructure reasons, not a legitimate iteration regression.

---

## 1. The paper exhibit — Bug 3 × Gemma

This is the load-bearing finding of Session 1. Bug 3 × Gemma exemplifies exactly the failure mode the memory-without-grounding thesis predicts for naive context accumulation.

| Iter | Class | Prompt size | Latency | Tokens | Fix hash |
|---|---|---|---|---|---|
| 1 | hallucination | 18,143c | 307s | 5125 | `a722938660a0` |
| 2 | hallucination | 34,242c | 385s | 5234 | **`a722938660a0`** (identical) |
| 3 | drift | 50,090c | 382s | 4866 | `df9819a795ad` (diverged, still wrong) |

**Iter 1 and iter 2 produced byte-for-byte identical fixes**, even though iter 2's prompt contained iter 1's failed attempt PLUS iter 1's test failure output. Gemma saw its own hallucinated fix, saw that it failed, and generated the exact same fix again. Only at iter 3 did Gemma diverge from the hallucinated mental model — but then failed via drift on a different axis.

The mechanism is the one the paper describes:

1. Naive accumulation appends full prior attempts to context.
2. Prior attempts carry the failure mode in them (the hallucinated symbol, the drifted operator).
3. Context pollution reinforces the failure mode rather than providing signal to correct it.
4. Model echoes its own mistakes back at higher weight than the actual ground-truth signals (failing test output).

The no_progress detector correctly fired at iter 3 because iter 1 and iter 2's fix hashes matched within the window. Termination was clean. This is the kind of cell Session 2 (Config A with anneal-memory compression) needs to rescue to show the thesis holds.

Bloat was also visible: 18K → 34K → 50K chars per iter. Latency stayed roughly flat at 307-385s per call — Gemma was not context-window-limited in the absolute sense, but the signal-to-noise ratio degraded as the prompt grew.

## 2. The variance finding (the methodology caveat)

Two cells flipped between Session 0.5 (single-shot) and Session 1 (Config E iter 1):

| Cell | Session 0.5 | Session 1 iter 1 |
|---|---|---|
| bug_01 × Gemma | PASS (correct fix, 104.8s) | **FAIL** (drift, hallucinated 1900-fallback) |
| bug_01 × Qwen | FAIL (drift `<=` → `<`) | **PASS** (correct fix, 172s) |

Same prompts, same models, temperature=0.2 (low but not zero). N=1 per cell per session is too noisy to distinguish "iteration effect" from "baseline variance" on this corpus. The Session 0.5 "canonical" failure classes (Qwen drift on Bug 1, Gemma hallucination on Bug 3) are the *modal* failures at N=1, not deterministic outcomes.

**The real methodology claim Session 1 surfaces is:** pass/fail at N=1 per cell cannot carry a Config A vs Config E claim. Session 2 must either (a) commit to N≥3 per cell with explicit variance reporting, or (b) use a fixed-seed harness if the Ollama backend supports deterministic generation at `temperature=0.0`.

This finding is more important than any specific per-cell result — it reshapes the experimental design for Session 2+.

The deeper consequence: the Session 0.5 "iter target" labels (Bug 1 × Qwen = drift case, Bug 3 × Gemma = hallucination case) were pinned to single observations. Bug 1 × Qwen passed first-shot in Session 1 without any iteration, making it a *non-target* by accident. Bug 3 × Gemma reliably hallucinated on iter 1 AND iter 2 (same fix), so it holds up as a real iter target for Session 2 even at N=1.

## 3. Detailed per-cell notes

### bug_01_prune_falsy × Gemma — no_progress, 5/5 iters

Canonical `prune()` in anneal-memory has NO `if days == 0` special case — it relies on `cutoff = datetime.now() - timedelta(days=0) = now()`, and `SELECT WHERE timestamp < cutoff` matches episodes recorded earlier in the same call chain (the test records an episode and immediately calls `prune(older_than_days=0)`).

Gemma iter 1 fixed the `or` → `is not None` bug correctly on the first line but added a hallucinated special case:

```python
if days == 0:
    cutoff_dt = datetime(1900, 1, 1, tzinfo=timezone.utc)  # prune things older than 1900 = nothing
else:
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
```

This produces `pruned == 0` on the test because no episodes are older than 1900. Iter 2 and iter 3 produced the same pattern. Iters 4-5 kept drifting on variations of the same hallucinated structure.

The classifier labeled all 5 iters as `drift` because the extracted code contained a `def` and applied cleanly (no AttributeError / NameError in the post-fix test output), and the failure was an `assert 0 == 1` from the test, not a symbol-level error. This is a classifier honest-limit: "drift" is the default for applied-but-failed, which conflates operator-mutation drift with semantic-hallucination drift. Session 2+ should refine the classifier to separate semantic hallucination from operator drift, likely via AST diff against the buggy function.

### bug_01_prune_falsy × Qwen — pass, 1/1 iter

Produced the exact correct fix on iter 1, no drift, no hallucination. Contradicts Session 0.5's observation of Qwen drifting `<=` → `<`. Temperature-variance noise.

### bug_03_retention_wiring × Gemma — no_progress, 3/3 iters

Covered in §1. Paper exhibit.

### bug_04_initialized_flag × both — pass, 1/1 iter each

Saturation controls held. Single-shot Session 0.5 already showed both models pass this three-site `_initialized` revert correctly. Config E didn't regress. Contributes the "iteration doesn't break already-working cells" half of the control claim.

### bug_05_double_orphan_dedup × Gemma — pass, 1/1 iter

Saturation control held. Gemma produced a correct fix for the single-site dedup logic revert.

### bug_05_double_orphan_dedup × Qwen — **error (timeout), 0 iters completed**

Infrastructure failure, not an iteration finding.

After cell 6 (Bug 5 × Gemma, iter 1 pass), Ollama had BOTH Gemma (10.5GB VRAM) and Qwen (8.8GB VRAM) resident simultaneously because the `keep_alive=30m` directive was preventing eviction. The cell 7 request hung — Python held an ESTABLISHED TCP socket to Ollama but Qwen's `expires_at` never updated, meaning Ollama never processed the request. Working memory almost certainly saturated with both models + macOS + flow system.

The `threading.Timer` hard timeout (600s) fired cleanly, the backend raised `TimeoutError`, the loop caught it as a cell-level backend failure, and the matrix moved on. **The safety net worked**, which is the second most important Session 1 finding after the variance story.

Infrastructure decision for Session 2: either (a) unload the previous model between cells via Ollama's `keep_alive=0` request at cell boundaries, (b) set `keep_alive=5m` so models can time out naturally between cells, or (c) run the matrix single-model-at-a-time (all 4 Gemma cells, then all 4 Qwen cells) to avoid ever needing both resident simultaneously. Option (c) is cleanest for Session 2.

## 4. The `/no_think` detour

Session 0.5 deferred the `/no_think` experiment to Session 1 as the remediation for Bug 3 × Qwen's 900s single-shot timeout. Session 1 validated the hypothesis **and it failed on both axes:**

1. **Quality regression on Bug 1 × Qwen:** /no_think caused Qwen to miss the bug entirely (preserved the original `or` operator, added a dead-code `if days == 0` wrapper). Session 0.5 without /no_think correctly identified the bug (then drifted on an unrelated operator).
2. **No latency rescue on Bug 3 × Qwen:** still timed out at 900s. The generation wall is on total token output time through a 6.6GB model with an 18K-char prompt, not on thinking-mode overhead.

`make_backend()` was updated to demote `/no_think` from the default to opt-in. The real remediation for Bug 3 × Qwen is **targeted-patch prompting** — ask for a unified-diff or a surgical replacement instead of a full whole-function regen. That's a Session 1.5 experiment, explicitly out of current scope.

## 5. Infrastructure findings (the non-paper bank)

These are code-level learnings that land back in the harness but are not paper-relevant on their own:

- **urllib timeout is a lie for slow-drip generations.** The `timeout` parameter is per-socket-op, not total-time. The first matrix run hung Python for 6+ minutes past the 900s "timeout" on cell 3 because data was trickling through fast enough to reset the per-op timer. The `threading.Timer` wall-clock guard in `backend.py` is now the only trustworthy escape hatch. Any future slow-generation code path should copy the pattern.
- **SIGTERM on a hung urlib read is effectively SIGKILL from Python's perspective** — the `finally: reset_repo()` in `loop.py` never ran when the first matrix was killed, leaving `anneal_memory/store.py` in its buggy state. Manual `git checkout -- .` was required. For Session 2, the loop should add a best-effort `atexit` handler on the repo reset AND the `try/finally` should be wrapped in a short subprocess so the OS-level kill cleans up the child without losing the repo state in the parent.
- **VRAM deadlock on multi-model runs.** Long `keep_alive` values plus multiple models in a single matrix can saturate working memory without exceeding VRAM ceilings on paper. Prefer short keep_alive + single-model-per-cell-sequence or explicit cell-boundary unload.
- **Classifier coarse-graining.** The three classes (drift, hallucination, syntax) are good enough to label episodes but conflate real distinctions — "semantic hallucination" (Gemma's 1900-fallback) and "operator drift" (Qwen's `<=` → `<`) both land as "drift" if the post-fix test output lacks AttributeError/NameError markers. Session 2+ classifier refinement target.

## 6. What this means for Session 2 (Config A with anneal-memory)

Session 2 runs the same matrix (7 cells, same corpus) with the same loop architecture and `num_ctx=16384`, plus anneal-memory integration via `Store` + `prepare_wrap` + `validated_save_continuity`. The key comparison cells are:

- **bug_03 × Gemma** — the hallucination-reinforcement cell. If Config A rescues this (pass in ≤4 iters) while Config E cycles, that's the core memory-without-grounding claim materialized.
- **bug_01 × Gemma** — the drift-cycle cell. Same logic. But given the variance finding, this cell needs N≥3 to be load-bearing.
- **bug_01 × Qwen, bug_04 × both, bug_05 × Gemma** — "does Config A regress saturation controls?" must be no. If Config A adds overhead that prevents iter-1 passes on these, the thesis is in trouble even on working cells.

Session 2 scope additions driven by Session 1 findings:

1. **N≥3 per cell** — run each cell 3 times at minimum, report pass rate as a fraction, not a boolean. Temperature variance makes N=1 unpublishable.
2. **Single-model cell ordering** — run all Gemma cells, unload, run all Qwen cells. Avoid VRAM deadlock.
3. **Cell boundary repo reset** — belt-and-suspenders: run the loop in a subprocess with OS-level cleanup so a hung iteration cannot leave the anneal-memory repo dirty.
4. **Classifier refinement** — AST-diff-based drift vs hallucination separation. Currently both land as `drift` when the post-fix test fails without an attribute error, losing signal.
5. **bug_03 × Qwen is a standing problem** — before Session 2 runs Config A, either (a) accept bug_03 × Qwen is infeasible at the current prompt strategy and carry the asymmetry explicitly, or (b) run a Session 1.5 prompt-strategy experiment (targeted patch instead of whole-function regen) to unblock that cell before locking Session 2 matrix.

## 7. Methodology caveats (the honest-report section)

- **N=1 per cell** is below the noise floor for this corpus. All pass/fail claims in this document must be read as "modal at N=1," not deterministic.
- **Max iterations asymmetry.** Cells 1-2 were salvaged from an earlier run with `max_iters=5`. Cells 3-7 ran with `max_iters=4`. The matrix is not internally consistent on this knob. The Gemma Bug 1 cell hit `no_progress` at iter 5 on the max_iters=5 run; on max_iters=4 it would have hit `no_progress` at iter 4 with the same pattern (same drift class every iter). The asymmetry matters for exact iter counts but not for termination reason or failure class distribution.
- **num_ctx=16384.** The baseline single-shot runs in Session 0.5 used `num_ctx=8192`. Different KV cache sizes can produce different Gemma/Qwen generations at the same temperature. Some of the variance between Session 0.5 and Session 1 baselines may be attributable to this, not to pure stochastic sampling. Session 2 should pin a single num_ctx and carry it consistently.
- **Bug 3 × Qwen is not in the matrix at all** — dropped as infeasible per Session 0.5's 900s timeout observation and Session 1's /no_think validation. The Config E Round 1 "4/7 passed" headline counts 7 cells, not 8. Any comparison to Config A Round 1 must hold this same 7-cell scope.

## 8. Raw data

- `results/config_e_round1.json` — aggregated matrix
- `results/raw/config_e_<bug>_<model>.json` — per-cell LoopResult with all episodes, fix hashes, test outputs, latencies

## 9. What we shipped (code)

- `harness/backend.py` — Backend ABC + `OllamaBackend` with per-model prompt transform hook and threading.Timer hard wall-clock guard. Factory `make_backend()` routes model tags to the right backend.
- `harness/loop.py` — Config E iteration loop. Reuses `baseline.py` primitives, adds naive context accumulator, failure classifier, episode records, termination logic. CLI for single-cell runs.
- `harness/run_session1.py` — Matrix runner. Skip-existing salvage mode for resuming after an infrastructure failure.

Anvil code repo state: `~/Documents/anvil/` — clean working tree, new files tracked under `harness/`.

anneal-memory repo state: clean, HEAD at v0.2.1 (`f6613fa`). The one-time manual repo reset after the first matrix kill was clean.

---

*Session 1 complete. Session 2 builds on this infrastructure with anneal-memory integration, variance-aware replication, and single-model cell sequencing. The Config A vs Config E comparison lands there.*
