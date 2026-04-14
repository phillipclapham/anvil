"""Core iteration loop — Config A (anneal-memory integration).

Session 2 deliverable. This is the TREATMENT arm for the memory-without-
grounding thesis. Config E (loop.py) appends every prior attempt into the
next prompt. Config A stores attempts as typed episodes in anneal-memory,
consolidates episodes into a compressed continuity doc via a rule-based
template, and prompts the next iteration with (base_prompt + compressed
continuity + recent unwrapped episodes) — NOT the full raw history.

Session 2a scope (this file):

    - Fresh anneal-memory Store in a per-run temp dir (never pollutes flow's
      real memory store).
    - Per-iteration Episode recording with structured metadata (iter,
      fix_hash, test_passed, failure_class, bug_id, model).
    - Consolidation trigger: every 3 iterations OR when no_progress
      detected. Tunable in Session 3.
    - Context strategy: base_prompt + Store.load_continuity() + episodes
      since last wrap. Raw history never enters the prompt.
    - Compression is rule-based structural synthesis (dedup by fix_hash,
      grouping by failure class, one-line-per-attempt summary). NO
      meta-reasoning. NO generator used for compression. NO cloud model
      smuggled in. The compression IS the claim being tested — does a
      structural filter on prior attempts beat raw accumulation?

Reuses loop.py primitives where possible: IterationEpisode dataclass,
LoopResult dataclass, classify_failure, _fix_hash, detect_no_progress,
build_naive_prompt is NOT reused (that's the Config E strategy).
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure anneal-memory and harness are importable
HARNESS_ROOT = Path(__file__).parent.parent
ANNEAL_SRC = Path.home() / "Documents" / "anneal-memory"
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(ANNEAL_SRC))

from anneal_memory import (  # noqa: E402
    AnnealMemoryError,
    Episode,
    EpisodeType,
    Store,
    prepare_wrap,
    validated_save_continuity,
)

from harness.backend import Backend, make_backend  # noqa: E402
from harness.baseline import (  # noqa: E402
    ANNEAL_REPO,
    assert_clean_repo,
    build_prompt,
    extract_code_from_response,
    extract_function_source,
    extract_test_source,
    reintroduce_bug,
    replace_function_in_file,
    reset_repo,
    run_test,
)
from harness.loop import (  # noqa: E402
    IterationEpisode,
    LoopResult,
    _fix_hash,
    classify_failure,
    detect_no_progress,
)

RESULTS_RAW = HARNESS_ROOT / "results" / "raw"


# -- Episode content shaping --------------------------------------------------


def format_episode_content(
    iter_num: int,
    extracted_fix: str,
    fix_hash: str,
    test_passed: bool,
    failure_class: str | None,
    post_test_output: str,
    error: str | None,
) -> str:
    """Shape an IterationEpisode into anneal-memory Episode content.

    Deliberately structured (not prose). The compression downstream
    depends on this being parseable and deduplicable.
    """
    status = "PASS" if test_passed else "FAIL"
    fc = failure_class or ("pass" if test_passed else "unknown")
    parts = [
        f"Iter {iter_num}: {status} ({fc}) fix_hash={fix_hash}",
        "",
        "Fix (truncated):",
        extracted_fix.strip()[:600] if extracted_fix else "(no fix extracted)",
    ]
    if error:
        parts += ["", f"Error: {error}"]
    if not test_passed and post_test_output:
        parts += ["", "Test output tail:", post_test_output.strip()[-600:]]
    return "\n".join(parts)


# -- Rule-based continuity synthesis ------------------------------------------


def synthesize_continuity_text(
    bug_id: str,
    model: str,
    episodes: list[Episode],
) -> str:
    """Build a continuity document from episodes via pure structural rules.

    No meta-reasoning. No generator calls. No cloud. Pure filter+group+
    dedup+template. The four required sections (state, context, patterns,
    decisions) are populated from episode metadata in a deterministic way.

    If Config A wins Session 2b's matrix, the paper claim is: *structural
    filtering of prior attempts outperforms raw accumulation*. This
    function is the claim's operational definition.
    """
    total = len(episodes)
    passes = [ep for ep in episodes if (ep.metadata or {}).get("test_passed")]
    fails = [ep for ep in episodes if not (ep.metadata or {}).get("test_passed")]

    # Dedup failed attempts by fix_hash (collapses repeated hallucinations)
    seen_hashes: dict[str, int] = {}  # hash -> count
    unique_fails: list[Episode] = []
    for ep in fails:
        meta = ep.metadata or {}
        h = meta.get("fix_hash", "")
        if h and h in seen_hashes:
            seen_hashes[h] += 1
        else:
            seen_hashes[h] = 1
            unique_fails.append(ep)

    # Group failures by class
    by_class: dict[str, list[Episode]] = {}
    for ep in unique_fails:
        fc = (ep.metadata or {}).get("failure_class") or "unknown"
        by_class.setdefault(fc, []).append(ep)

    lines: list[str] = []

    # ## State
    lines += [
        "## State",
        "",
        f"- Bug: {bug_id}",
        f"- Model: {model}",
        f"- Total attempts: {total} ({len(passes)} pass, {len(fails)} fail)",
        f"- Unique failure modes by fix_hash: {len(unique_fails)}",
    ]
    if any(c > 1 for c in seen_hashes.values()):
        repeats = [(h, c) for h, c in seen_hashes.items() if c > 1]
        lines.append(
            f"- Repeated fix_hashes (identical attempts): "
            f"{', '.join(f'{h[:8]}x{c}' for h, c in repeats)}"
        )
    lines.append("")

    # ## Context
    lines += ["## Context", ""]
    if unique_fails:
        lines.append("Prior distinct failure attempts (most recent first):")
        lines.append("")
        for ep in reversed(unique_fails[-6:]):  # cap to most recent 6 unique
            meta = ep.metadata or {}
            iter_num = meta.get("iter_num", "?")
            fc = meta.get("failure_class") or "unknown"
            h = (meta.get("fix_hash") or "")[:8]
            summary = _extract_fix_summary(ep.content)
            lines.append(f"- Iter {iter_num} ({fc}, {h}): {summary}")
        lines.append("")
    else:
        lines.append("No prior failed attempts recorded.")
        lines.append("")

    # ## Patterns
    lines += ["## Patterns", ""]
    if by_class:
        for fc in sorted(by_class.keys()):
            count = len(by_class[fc])
            lines.append(f"- {fc}: {count} distinct attempt(s)")
            # Pull the failure-class-specific lesson
            if fc == "hallucination":
                lines.append(
                    "  - Failure mode: fix referenced a symbol that doesn't "
                    "exist on the target class. Verify every attribute "
                    "access against the method's class definition before "
                    "returning."
                )
            elif fc == "drift":
                lines.append(
                    "  - Failure mode: fix changed unrelated code during "
                    "whole-function regeneration. Preserve existing operators "
                    "and logic except where the bug explicitly requires "
                    "modification."
                )
            elif fc == "syntax":
                lines.append(
                    "  - Failure mode: response could not be parsed as a "
                    "function. Return ONLY a single python code block "
                    "containing the full method from def through the last "
                    "return statement."
                )
    else:
        lines.append("- No failure patterns recorded yet.")
    lines.append("")

    # ## Decisions
    lines += ["## Decisions", ""]
    if unique_fails:
        lines.append(
            "- Do NOT repeat any of the fix_hashes listed in Context. "
            "These attempts were already tried and failed."
        )
        lines.append(
            "- Read the failing test in the prompt carefully — it shows "
            "exactly what the fixed method must satisfy."
        )
        if "hallucination" in by_class:
            lines.append(
                "- Every attribute referenced in the fix must exist on the "
                "class. Do not invent methods or attributes."
            )
    else:
        lines.append("- First attempt. No prior decisions to preserve.")
    lines.append("")

    return "\n".join(lines)


def _extract_fix_summary(episode_content: str) -> str:
    """One-line summary of the fix from the episode content.

    The content format is set by ``format_episode_content``. We extract
    the def signature line if available, otherwise fall back to the
    first non-header line. Truncate at 120 chars.
    """
    lines = episode_content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[:120]
    # Fallback: first non-iter-header line of substance
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("Iter ", "Fix ", "Test ", "Error", "(no fix")):
            return stripped[:120]
    return "(no signature extracted)"


# -- Prompt construction ------------------------------------------------------


def build_config_a_prompt(
    base_prompt: str,
    store: Store,
    recent_unwrapped: list[Episode],
) -> str:
    """Config A context strategy: compressed continuity + recent episodes.

    Deliberately does NOT include the raw prior-attempt text from
    already-consolidated episodes. That's the whole point — the
    consolidation step filtered them through the rule-based template and
    the model sees the distilled result, not the noise.
    """
    continuity = store.load_continuity() or ""
    parts = [base_prompt.rstrip()]

    if continuity:
        parts += [
            "",
            "# Consolidated prior attempts",
            "",
            continuity.strip(),
        ]

    if recent_unwrapped:
        parts += [
            "",
            "# Recent attempts (since last consolidation)",
            "",
        ]
        for ep in recent_unwrapped:
            meta = ep.metadata or {}
            iter_num = meta.get("iter_num", "?")
            fc = meta.get("failure_class") or "unknown"
            h = (meta.get("fix_hash") or "")[:8]
            passed = meta.get("test_passed", False)
            label = "PASS" if passed else "FAIL"
            parts.append(f"## Iter {iter_num} — {label} ({fc}, {h})")
            parts.append("")
            # Pull the truncated fix from content
            for line in ep.content.split("\n"):
                if line.strip().startswith("def ") or line.startswith("    "):
                    parts.append(line)
            parts.append("")

    if continuity or recent_unwrapped:
        parts.append(
            "Review the consolidated state and recent attempts above. "
            "Produce a corrected method that avoids the failure modes of the "
            "prior attempts. Return ONLY the fixed method in a single python "
            "code block, no explanation."
        )

    return "\n".join(parts)


# -- Consolidation ------------------------------------------------------------


def should_consolidate(
    iter_num: int,
    stalled: bool,
    trigger_every: int = 3,
) -> bool:
    """Adaptive consolidation trigger.

    Session 2a heuristic:
      - Every ``trigger_every`` iterations
      - OR when no_progress detection fires

    Session 3 will tune this empirically. For 2a vertical slice, the goal
    is "compression runs at least once during a cell run" so we can
    validate the integration end-to-end.
    """
    if stalled:
        return True
    return iter_num > 0 and iter_num % trigger_every == 0


def do_consolidate(
    store: Store,
    bug_id: str,
    model: str,
) -> dict[str, Any]:
    """Run one consolidation pass: prepare_wrap + validated_save_continuity.

    Returns a forensic dict with what happened. Swallows AnnealMemoryError
    into the dict so the iteration loop can continue even if memory-layer
    bugs surface.
    """
    forensic: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "prepare_status": None,
        "episodes_compressed": 0,
        "chars": 0,
        "error": None,
    }
    try:
        # prepare_wrap + validated_save_continuity return TypedDicts
        # (dict subclasses), not dataclasses — use subscript access.
        pre = prepare_wrap(store)
        forensic["prepare_status"] = pre["status"]
        if pre["status"] == "empty":
            forensic["ok"] = True
            return forensic

        unwrapped = store.episodes_since_wrap()
        text = synthesize_continuity_text(bug_id, model, unwrapped)
        save_result = validated_save_continuity(
            store, text, wrap_token=pre["wrap_token"]
        )
        forensic["ok"] = True
        forensic["episodes_compressed"] = save_result["episodes_compressed"]
        forensic["chars"] = save_result["chars"]
    except AnnealMemoryError as e:
        forensic["error"] = f"AnnealMemoryError: {e}"
    except Exception as e:
        forensic["error"] = f"{type(e).__name__}: {e}"
    return forensic


# -- The loop -----------------------------------------------------------------


@dataclass
class LoopResultA(LoopResult):
    """Config A result adds consolidation forensics."""

    store_path: str = ""
    consolidations: list[dict[str, Any]] = field(default_factory=list)


def run_loop_a(
    bug_module_name: str,
    backend: Backend,
    *,
    max_iters: int = 5,
    no_progress_window: int = 3,
    consolidate_every: int = 3,
    repo: Path = ANNEAL_REPO,
    store_dir: Path | None = None,
) -> LoopResultA:
    """Config A iteration loop on one bug + one backend.

    Creates a fresh anneal-memory Store in ``store_dir`` (or a tempdir).
    Records each iteration as an episode. Triggers consolidation per
    ``should_consolidate``. Builds prompts from compressed continuity +
    recent unwrapped episodes (never raw history).

    Always resets the anneal-memory repo to clean state on exit. Does
    NOT clean up the Store directory — that's forensic data.
    """
    sys.path.insert(0, str(HARNESS_ROOT))
    bug_module = importlib.import_module(bug_module_name)
    bug = bug_module.BUG

    if store_dir is None:
        store_dir = Path(tempfile.mkdtemp(prefix="anvil_store_"))
    else:
        store_dir.mkdir(parents=True, exist_ok=True)
    # Store expects a DB file path; it will create the parent dir.
    store_db_path = store_dir / "anvil.db"

    result = LoopResultA(
        bug_id=bug["id"],
        model=backend.name,
        config="A",
        termination_reason="error",
        passed=False,
        iterations_used=0,
        total_latency_s=0.0,
        total_tokens=0,
        store_path=str(store_dir),
    )

    store: Store | None = None
    repo_was_clean = False
    try:
        assert_clean_repo(repo)
        repo_was_clean = True

        store = Store(
            store_db_path,
            project_name=f"anvil_{bug['id']}_{backend.name.replace(':', '_')}",
            audit=False,  # Skip audit infra for benchmark runs
        )

        # Seed episode: the problem statement itself
        store.record(
            content=(
                f"Problem: fix {bug['target_class']}.{bug['target_function']} "
                f"in {bug['source_file']}. Failing test: {bug['test_name']} "
                f"in {bug['test_file']}."
            ),
            episode_type=EpisodeType.CONTEXT,
            source=f"anvil/{backend.name}",
            metadata={"seed": True, "bug_id": bug["id"]},
        )

        # One-time setup
        if not reintroduce_bug(bug, repo):
            result.error = "Bug reintroduction find-string not present."
            return result

        pre_passed, pre_output = run_test(bug, repo)
        if pre_passed:
            result.error = "Bug reintroduction did not break the test."
            return result

        src_path = repo / bug["source_file"]
        test_path = repo / bug["test_file"]
        buggy_function = extract_function_source(
            src_path, bug["target_class"], bug["target_function"]
        )
        test_source = extract_test_source(test_path, bug["test_name"])
        base_prompt = build_prompt(bug, buggy_function, test_source, pre_output)
        buggy_file_state = src_path.read_text()

        # Iteration loop
        for iter_num in range(1, max_iters + 1):
            recent_unwrapped = store.episodes_since_wrap()
            # Filter out the seed CONTEXT episode from recent prompt display
            recent_unwrapped = [
                ep for ep in recent_unwrapped
                if not (ep.metadata or {}).get("seed")
            ]
            prompt = build_config_a_prompt(base_prompt, store, recent_unwrapped)

            try:
                gen = backend.generate(prompt)
            except Exception as e:
                ep = IterationEpisode(
                    iter_num=iter_num,
                    prompt_len_chars=len(prompt),
                    raw_response="",
                    extracted_fix="",
                    extracted_fix_hash="",
                    fix_applied_ok=False,
                    test_passed=False,
                    failure_class="syntax",
                    latency_s=0.0,
                    eval_tokens=0,
                    eval_rate_tok_s=0.0,
                    post_test_output="",
                    error=f"Backend generate failed: {e}",
                )
                result.episodes.append(ep)
                result.iterations_used = iter_num
                result.termination_reason = "error"
                result.error = f"Backend failure at iter {iter_num}: {e}"
                return result

            extracted = extract_code_from_response(gen.text)
            fh = _fix_hash(extracted) if extracted else ""

            ep = IterationEpisode(
                iter_num=iter_num,
                prompt_len_chars=len(prompt),
                raw_response=gen.text,
                extracted_fix=extracted,
                extracted_fix_hash=fh,
                fix_applied_ok=False,
                test_passed=False,
                failure_class=None,
                latency_s=gen.latency_s,
                eval_tokens=gen.eval_tokens,
                eval_rate_tok_s=gen.eval_rate_tok_s,
                post_test_output="",
            )
            result.total_latency_s += gen.latency_s
            result.total_tokens += gen.eval_tokens

            # Revert to buggy state before applying this iteration's fix
            src_path.write_text(buggy_file_state)

            if "def " not in extracted:
                ep.error = "Model response contained no function definition."
                ep.failure_class = "syntax"
                _record_iteration_episode(store, ep, bug["id"], backend.name)
                result.episodes.append(ep)
                if detect_no_progress(result.episodes, no_progress_window):
                    result.iterations_used = iter_num
                    result.termination_reason = "no_progress"
                    _maybe_consolidate(
                        store, result, bug["id"], backend.name,
                        iter_num, stalled=True, every=consolidate_every,
                    )
                    return result
                _maybe_consolidate(
                    store, result, bug["id"], backend.name,
                    iter_num, stalled=False, every=consolidate_every,
                )
                continue

            # Apply fix
            try:
                replace_function_in_file(
                    src_path,
                    bug["target_class"],
                    bug["target_function"],
                    extracted,
                )
                ep.fix_applied_ok = True
            except Exception as e:
                ep.error = f"Fix application failed: {e}"
                ep.failure_class = "syntax"
                _record_iteration_episode(store, ep, bug["id"], backend.name)
                result.episodes.append(ep)
                src_path.write_text(buggy_file_state)
                if detect_no_progress(result.episodes, no_progress_window):
                    result.iterations_used = iter_num
                    result.termination_reason = "no_progress"
                    _maybe_consolidate(
                        store, result, bug["id"], backend.name,
                        iter_num, stalled=True, every=consolidate_every,
                    )
                    return result
                _maybe_consolidate(
                    store, result, bug["id"], backend.name,
                    iter_num, stalled=False, every=consolidate_every,
                )
                continue

            # Run test
            post_passed, post_output = run_test(bug, repo)
            ep.test_passed = post_passed
            ep.post_test_output = post_output[-1500:]
            ep.failure_class = classify_failure(
                ep.fix_applied_ok, post_passed, post_output
            )
            _record_iteration_episode(store, ep, bug["id"], backend.name)
            result.episodes.append(ep)

            if post_passed:
                result.iterations_used = iter_num
                result.termination_reason = "pass"
                result.passed = True
                return result

            src_path.write_text(buggy_file_state)

            stalled = detect_no_progress(result.episodes, no_progress_window)
            if stalled:
                result.iterations_used = iter_num
                result.termination_reason = "no_progress"
                _maybe_consolidate(
                    store, result, bug["id"], backend.name,
                    iter_num, stalled=True, every=consolidate_every,
                )
                return result

            _maybe_consolidate(
                store, result, bug["id"], backend.name,
                iter_num, stalled=False, every=consolidate_every,
            )

        # Fell through loop without passing
        result.iterations_used = max_iters
        result.termination_reason = "max_iters"
        return result

    except Exception as e:
        result.error = f"Unhandled loop exception: {type(e).__name__}: {e}"
        return result

    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        # Only reset repo if we successfully asserted cleanliness. If
        # assert_clean_repo raised, the repo had uncommitted work we must NOT
        # wipe.
        if repo_was_clean:
            try:
                reset_repo(repo)
            except Exception as e:
                print(f"WARNING: repo reset failed: {e}", file=sys.stderr)


def _record_iteration_episode(
    store: Store,
    ep: IterationEpisode,
    bug_id: str,
    model: str,
) -> None:
    """Write an IterationEpisode to the anneal-memory store."""
    content = format_episode_content(
        iter_num=ep.iter_num,
        extracted_fix=ep.extracted_fix,
        fix_hash=ep.extracted_fix_hash,
        test_passed=ep.test_passed,
        failure_class=ep.failure_class,
        post_test_output=ep.post_test_output,
        error=ep.error,
    )
    episode_type = EpisodeType.OUTCOME if ep.test_passed else EpisodeType.OBSERVATION
    try:
        store.record(
            content=content,
            episode_type=episode_type,
            source=f"anvil/{model}",
            metadata={
                "iter_num": ep.iter_num,
                "fix_hash": ep.extracted_fix_hash,
                "test_passed": ep.test_passed,
                "failure_class": ep.failure_class,
                "bug_id": bug_id,
                "prompt_len_chars": ep.prompt_len_chars,
                "latency_s": ep.latency_s,
                "eval_tokens": ep.eval_tokens,
            },
        )
    except Exception as e:
        print(
            f"WARNING: store.record failed at iter {ep.iter_num}: {e}",
            file=sys.stderr,
        )


def _maybe_consolidate(
    store: Store,
    result: LoopResultA,
    bug_id: str,
    model: str,
    iter_num: int,
    *,
    stalled: bool,
    every: int,
) -> None:
    """Call do_consolidate if the trigger fires. Append forensic dict to result."""
    if not should_consolidate(iter_num, stalled, trigger_every=every):
        return
    forensic = do_consolidate(store, bug_id, model)
    forensic["iter_num"] = iter_num
    forensic["stalled"] = stalled
    result.consolidations.append(forensic)


# -- Persistence --------------------------------------------------------------


def write_result(result: LoopResultA, out_dir: Path = RESULTS_RAW) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = result.model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"config_a_{result.bug_id}_{safe_model}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return out_path


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Config A (anneal-memory) iteration loop on one cell."
    )
    parser.add_argument("bug", help="Bug module, e.g. bugs.bug_03_retention_wiring")
    parser.add_argument("model", help="Model tag, e.g. gemma4:e4b-it-q4_K_M")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--no-progress-window", type=int, default=3)
    parser.add_argument("--consolidate-every", type=int, default=3)
    parser.add_argument("--repo", default=str(ANNEAL_REPO))
    parser.add_argument("--store-dir", default=None, help="Persist store here (default: tempdir)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    backend = make_backend(args.model)
    result = run_loop_a(
        args.bug,
        backend,
        max_iters=args.max_iters,
        no_progress_window=args.no_progress_window,
        consolidate_every=args.consolidate_every,
        repo=Path(args.repo),
        store_dir=Path(args.store_dir) if args.store_dir else None,
    )

    if args.save:
        path = write_result(result)
        print(f"Saved: {path}")

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print(f"=== Config A: {result.bug_id} on {result.model} ===")
        print(f"Store path: {result.store_path}")
        print(f"Termination: {result.termination_reason}")
        print(f"Passed: {result.passed}")
        print(f"Iterations used: {result.iterations_used} / {args.max_iters}")
        print(f"Total latency: {result.total_latency_s:.1f}s")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Consolidations: {len(result.consolidations)}")
        for i, c in enumerate(result.consolidations):
            status = "OK" if c.get("ok") else "FAIL"
            print(
                f"  [{i}] iter={c.get('iter_num')} stalled={c.get('stalled')} "
                f"{status} status={c.get('prepare_status')} "
                f"episodes_compressed={c.get('episodes_compressed')} "
                f"chars={c.get('chars')}"
                + (f" error={c.get('error')}" if c.get("error") else "")
            )
        for ep in result.episodes:
            label = "PASS" if ep.test_passed else "FAIL"
            fc = f" ({ep.failure_class})" if ep.failure_class else ""
            print(
                f"  iter {ep.iter_num}: {label}{fc}  "
                f"prompt={ep.prompt_len_chars}c  "
                f"latency={ep.latency_s:.1f}s  "
                f"tokens={ep.eval_tokens}  "
                f"hash={ep.extracted_fix_hash}"
            )
        if result.error:
            print(f"Error: {result.error}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
