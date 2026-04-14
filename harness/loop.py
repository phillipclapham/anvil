"""Core iteration loop — Config E (naive context control arm).

Session 1 deliverable. This is the CONTROL arm for the memory-without-
grounding thesis: iterate attempt → evaluate → record, accumulating full
prior-attempt text into the next prompt. Deliberately bad context strategy,
because measuring what anneal-memory's compression adds requires a baseline
where nothing is compressed.

Config E does NOT integrate anneal-memory. That's Session 2 (Config A).
Config E records episodes to local JSONL for post-hoc analysis.

Design notes:

- Reuses baseline.py primitives (assert_clean_repo, reintroduce_bug,
  run_test, extract/replace, build_prompt, extract_code_from_response).
  No duplication — baseline.py is the single-shot primitive, loop.py wraps
  it.

- Buggy function is extracted ONCE at loop start and reused for every
  iteration's prompt. The file state is reverted to buggy BEFORE each
  iteration's fix application, so each iteration sees a fresh buggy
  working tree. The "memory" is in the prompt, not the filesystem.

- Termination conditions:
  * pass — post-fix test passed
  * max_iters — hit iteration cap
  * no_progress — N consecutive iterations with no new passing test AND
    extracted-fix hash matches a prior attempt's hash

- Failure classification (post-fix test failed but fix applied cleanly):
  * hallucination — post_test_output contains AttributeError /
    NameError / UnboundLocalError referencing an undefined attribute
  * drift — default for applied-but-failed (weaker signal; lacks
    structural diff against buggy function; refinement in Session 3+)
  * syntax — fix could not be applied (extraction returned no def, or
    replace_function_in_file raised)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Reuse the baseline primitives
from harness.baseline import (
    ANNEAL_REPO,
    assert_clean_repo,
    reset_repo,
    reintroduce_bug,
    run_test,
    extract_function_source,
    extract_test_source,
    replace_function_in_file,
    extract_code_from_response,
    build_prompt,
)
from harness.backend import Backend, make_backend


HARNESS_ROOT = Path(__file__).parent.parent
RESULTS_RAW = HARNESS_ROOT / "results" / "raw"


# -- Episode + run records ----------------------------------------------------


@dataclass
class IterationEpisode:
    """One iteration's attempt + evaluation result."""

    iter_num: int
    prompt_len_chars: int
    raw_response: str
    extracted_fix: str
    extracted_fix_hash: str  # sha1 of the extracted fix (dedup/cycle detection)
    fix_applied_ok: bool
    test_passed: bool
    failure_class: str | None  # hallucination | drift | syntax | None
    latency_s: float
    eval_tokens: int
    eval_rate_tok_s: float
    post_test_output: str  # last 1500 chars
    error: str | None = None


@dataclass
class LoopResult:
    """The whole iteration run."""

    bug_id: str
    model: str
    config: str  # "E" for naive, "A" for anneal-memory (future)
    termination_reason: str  # pass | max_iters | no_progress | error
    passed: bool
    iterations_used: int
    total_latency_s: float
    total_tokens: int
    episodes: list[IterationEpisode] = field(default_factory=list)
    error: str | None = None


# -- Classifier ---------------------------------------------------------------


# Attribute-error-class signatures: fix-generated code referenced a symbol
# that doesn't exist. "self.has_cleanup_enabled" in Session 0.5's Gemma
# hallucination is the prototype.
_HALLUCINATION_PATTERNS = [
    r"AttributeError: .*object has no attribute",
    r"NameError: name .* is not defined",
    r"UnboundLocalError: .*referenced before assignment",
]


def classify_failure(
    fix_applied_ok: bool,
    test_passed: bool,
    post_test_output: str,
) -> str | None:
    """Best-effort classification of a failed iteration.

    This is a v1 heuristic. Session 3+ will refine it with structural diff
    against the buggy function and cross-reference with the target-test
    error location. For now, coarse-grained is enough to label episodes.
    """
    if test_passed:
        return None
    if not fix_applied_ok:
        return "syntax"
    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, post_test_output):
            return "hallucination"
    return "drift"


# -- Naive context accumulator ------------------------------------------------


def build_naive_prompt(
    base_prompt: str,
    history: list[IterationEpisode],
) -> str:
    """Config E context strategy: append every prior attempt and result.

    Deliberately unbounded. The control arm exists to show what happens
    when the loop has no compression — context grows linearly, prior
    failed approaches keep dominating the prompt, and the model starts
    echoing its own mistakes. This is the mem0-trap made visible.

    Bound is max_iters at the driver level + the context window at the
    backend level. If context overruns cause degradation, that's valid
    data for the paper.
    """
    if not history:
        return base_prompt

    parts = [base_prompt.rstrip(), "", "# Previous attempts (chronological)", ""]
    for ep in history:
        label = "PASS" if ep.test_passed else "FAIL"
        fc = ep.failure_class
        header = f"## Attempt {ep.iter_num} — {label}"
        if fc:
            header += f" ({fc})"
        parts.append(header)
        parts.append("")
        parts.append("```python")
        parts.append(ep.extracted_fix.rstrip())
        parts.append("```")
        parts.append("")
        parts.append("Test output (tail):")
        parts.append("```")
        parts.append(ep.post_test_output.rstrip()[-1200:])
        parts.append("```")
        parts.append("")

    parts.append(
        "Review the previous attempts above. The test still fails. "
        "Produce a corrected method that avoids the failure modes of the "
        "prior attempts. Return ONLY the fixed method in a single python "
        "code block, no explanation."
    )
    return "\n".join(parts)


# -- No-progress detection ----------------------------------------------------


def detect_no_progress(history: list[IterationEpisode], window: int = 3) -> bool:
    """Return True if the last ``window`` iterations show no new progress.

    Working definition (Session 1):
      1. No iteration in the window passed the test.
      2. At least one extracted_fix_hash in the window matches a fix_hash
         from an earlier window iteration (cycle detection).

    This is a deliberately conservative check. Refinement target in
    Session 3 after we have real data on what stall actually looks like.
    """
    if len(history) < window:
        return False

    recent = history[-window:]
    if any(ep.test_passed for ep in recent):
        return False

    seen: set[str] = set()
    for ep in recent:
        if ep.extracted_fix_hash in seen:
            return True
        seen.add(ep.extracted_fix_hash)
    return False


# -- The loop -----------------------------------------------------------------


def _fix_hash(extracted_fix: str) -> str:
    """sha1 of whitespace-normalized fix. Used for cycle detection."""
    normalized = re.sub(r"\s+", " ", extracted_fix.strip())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def run_loop(
    bug_module_name: str,
    backend: Backend,
    *,
    max_iters: int = 5,
    no_progress_window: int = 3,
    repo: Path = ANNEAL_REPO,
) -> LoopResult:
    """Config E iteration loop on one bug + one backend.

    Raises nothing — packages errors into LoopResult.error and returns.
    Always resets the anneal-memory repo to clean state on exit.
    """
    sys.path.insert(0, str(HARNESS_ROOT))
    bug_module = importlib.import_module(bug_module_name)
    bug = bug_module.BUG

    result = LoopResult(
        bug_id=bug["id"],
        model=backend.name,
        config="E",
        termination_reason="error",
        passed=False,
        iterations_used=0,
        total_latency_s=0.0,
        total_tokens=0,
    )

    try:
        assert_clean_repo(repo)

        # One-time setup: reintroduce bug, extract buggy function, build base prompt
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

        # Snapshot the buggy file state as text so we can revert between iterations
        buggy_file_state = src_path.read_text()

        # Iteration loop
        for iter_num in range(1, max_iters + 1):
            # Prompt = base + full prior history (Config E naive accumulation)
            prompt = build_naive_prompt(base_prompt, result.episodes)

            # Call model
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
            fix_hash = _fix_hash(extracted) if extracted else ""

            ep = IterationEpisode(
                iter_num=iter_num,
                prompt_len_chars=len(prompt),
                raw_response=gen.text,
                extracted_fix=extracted,
                extracted_fix_hash=fix_hash,
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
                result.episodes.append(ep)
                if detect_no_progress(result.episodes, no_progress_window):
                    result.iterations_used = iter_num
                    result.termination_reason = "no_progress"
                    return result
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
                result.episodes.append(ep)
                # Revert before next iter so the file isn't half-modified
                src_path.write_text(buggy_file_state)
                if detect_no_progress(result.episodes, no_progress_window):
                    result.iterations_used = iter_num
                    result.termination_reason = "no_progress"
                    return result
                continue

            # Run test
            post_passed, post_output = run_test(bug, repo)
            ep.test_passed = post_passed
            ep.post_test_output = post_output[-1500:]
            ep.failure_class = classify_failure(
                ep.fix_applied_ok, post_passed, post_output
            )
            result.episodes.append(ep)

            if post_passed:
                result.iterations_used = iter_num
                result.termination_reason = "pass"
                result.passed = True
                return result

            # Revert file for next iteration
            src_path.write_text(buggy_file_state)

            if detect_no_progress(result.episodes, no_progress_window):
                result.iterations_used = iter_num
                result.termination_reason = "no_progress"
                return result

        # Fell through loop without passing
        result.iterations_used = max_iters
        result.termination_reason = "max_iters"
        return result

    except Exception as e:
        result.error = f"Unhandled loop exception: {e}"
        return result

    finally:
        try:
            reset_repo(repo)
        except Exception as e:
            print(f"WARNING: repo reset failed: {e}", file=sys.stderr)


# -- Persistence --------------------------------------------------------------


def write_result(result: LoopResult, out_dir: Path = RESULTS_RAW) -> Path:
    """Write a LoopResult as JSON to results/raw/config_e_<bug>_<model>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = result.model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"config_e_{result.bug_id}_{safe_model}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    return out_path


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Config E iteration loop on one bug + one model.")
    parser.add_argument("bug", help="Bug module name, e.g. bugs.bug_01_prune_falsy")
    parser.add_argument("model", help="Ollama model tag, e.g. gemma4:e4b-it-q4_K_M")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--no-progress-window", type=int, default=3)
    parser.add_argument("--repo", default=str(ANNEAL_REPO))
    parser.add_argument("--json", action="store_true", help="Print full LoopResult as JSON")
    parser.add_argument("--save", action="store_true", help="Save to results/raw/")
    args = parser.parse_args()

    backend = make_backend(args.model)
    result = run_loop(
        args.bug,
        backend,
        max_iters=args.max_iters,
        no_progress_window=args.no_progress_window,
        repo=Path(args.repo),
    )

    if args.save:
        path = write_result(result)
        print(f"Saved: {path}")

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(f"=== Config E: {result.bug_id} on {result.model} ===")
        print(f"Termination: {result.termination_reason}")
        print(f"Passed: {result.passed}")
        print(f"Iterations used: {result.iterations_used} / {args.max_iters}")
        print(f"Total latency: {result.total_latency_s:.1f}s")
        print(f"Total tokens: {result.total_tokens}")
        for ep in result.episodes:
            label = "PASS" if ep.test_passed else "FAIL"
            fc = f" ({ep.failure_class})" if ep.failure_class else ""
            print(
                f"  iter {ep.iter_num}: {label}{fc}  "
                f"prompt={ep.prompt_len_chars}c  "
                f"latency={ep.latency_s:.1f}s  "
                f"tokens={ep.eval_tokens}"
            )
        if result.error:
            print(f"Error: {result.error}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
