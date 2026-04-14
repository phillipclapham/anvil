"""Session 1 matrix runner — Config E Round 1.

Runs the Config E iteration loop across all 7 feasible cells (4 bugs × 2
models minus the infeasible Bug 3 × Qwen cell, which single-shot times
out at 900s even with /no_think per Session 1 validation).

Writes:
  - results/raw/config_e_<bug>_<model>.json — per-cell LoopResult
  - results/config_e_round1.json — aggregated matrix
  - results/config_e_round1.md — human forensic (written separately,
    this runner just produces the data)

Per-cell failures (timeout, backend errors) are caught and recorded as
error cells rather than crashing the whole matrix — we want partial data
even when some cells blow up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from harness.backend import make_backend
from harness.loop import run_loop, write_result, LoopResult


HARNESS_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = HARNESS_ROOT / "results"


# 4 bugs × 2 models, minus the known-infeasible Bug 3 × Qwen cell.
MATRIX: list[tuple[str, str, str]] = [
    # (bug_module, model, role)
    ("bugs.bug_01_prune_falsy",       "gemma4:e4b-it-q4_K_M", "iter_target"),   # Gemma passed single-shot; iter 1 loop variance may differ
    ("bugs.bug_01_prune_falsy",       "qwen3.5:9b",           "iter_target"),   # Qwen drifted single-shot — real iter target
    ("bugs.bug_03_retention_wiring",  "gemma4:e4b-it-q4_K_M", "iter_target"),   # Gemma hallucinated single-shot — real iter target
    # bug_03 × qwen — INFEASIBLE (900s timeout), dropped
    ("bugs.bug_04_initialized_flag",  "gemma4:e4b-it-q4_K_M", "saturation_ctl"),
    ("bugs.bug_04_initialized_flag",  "qwen3.5:9b",           "saturation_ctl"),
    ("bugs.bug_05_double_orphan_dedup", "gemma4:e4b-it-q4_K_M", "saturation_ctl"),
    ("bugs.bug_05_double_orphan_dedup", "qwen3.5:9b",           "saturation_ctl"),
]


def _cell_result_path(bug: str, model: str) -> Path:
    from harness.loop import RESULTS_RAW
    bug_id = bug.split(".")[-1]
    safe_model = model.replace(":", "_").replace("/", "_")
    return RESULTS_RAW / f"config_e_{bug_id}_{safe_model}.json"


def run_matrix(max_iters: int, no_progress_window: int, skip_existing: bool = True) -> list[LoopResult]:
    results: list[LoopResult] = []
    total_t0 = time.time()

    for i, (bug, model, role) in enumerate(MATRIX, 1):
        print(f"\n[{i}/{len(MATRIX)}] {bug} × {model}  ({role})", flush=True)

        # Skip cells we've already run (salvage mode)
        existing = _cell_result_path(bug, model)
        if skip_existing and existing.exists():
            try:
                data = json.loads(existing.read_text())
                from harness.loop import LoopResult as _LR, IterationEpisode as _IE
                eps = [_IE(**e) for e in data.get("episodes", [])]
                result = _LR(
                    bug_id=data["bug_id"],
                    model=data["model"],
                    config=data["config"],
                    termination_reason=data["termination_reason"],
                    passed=data["passed"],
                    iterations_used=data["iterations_used"],
                    total_latency_s=data["total_latency_s"],
                    total_tokens=data["total_tokens"],
                    episodes=eps,
                    error=data.get("error"),
                )
                print(
                    f"  SKIP (existing): {result.termination_reason}  "
                    f"passed={result.passed}  iters={result.iterations_used}",
                    flush=True,
                )
                results.append(result)
                continue
            except Exception as e:
                print(f"  existing result unreadable ({e}) — re-running", flush=True)
        t0 = time.time()
        backend = make_backend(model)
        try:
            result = run_loop(
                bug,
                backend,
                max_iters=max_iters,
                no_progress_window=no_progress_window,
            )
        except KeyboardInterrupt:
            print("  INTERRUPTED — stopping matrix", flush=True)
            raise
        except Exception as e:
            print(f"  CELL ERROR: {e}", flush=True)
            result = LoopResult(
                bug_id=bug.split(".")[-1],
                model=model,
                config="E",
                termination_reason="matrix_error",
                passed=False,
                iterations_used=0,
                total_latency_s=0.0,
                total_tokens=0,
                error=str(e),
            )
        dt = time.time() - t0
        print(
            f"  → {result.termination_reason}  passed={result.passed}  "
            f"iters={result.iterations_used}  wall={dt:.0f}s  "
            f"tokens={result.total_tokens}",
            flush=True,
        )
        path = write_result(result)
        print(f"  saved: {path.name}", flush=True)
        results.append(result)

    total_wall = time.time() - total_t0
    print(f"\n=== Matrix complete in {total_wall:.0f}s ({total_wall/60:.1f} min) ===")
    return results


def write_aggregate(results: list[LoopResult], out_dir: Path = RESULTS_ROOT) -> Path:
    """Write an aggregated JSON of the full matrix run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config_e_round1.json"
    payload = {
        "config": "E",
        "round": 1,
        "session": "1",
        "max_iters_used": results[0].iterations_used if results else None,
        "cells": [asdict(r) for r in results],
        "summary": summarize(results),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def summarize(results: list[LoopResult]) -> dict:
    by_reason: dict[str, int] = {}
    total_latency = 0.0
    total_tokens = 0
    passed = 0
    failure_classes: dict[str, int] = {}

    for r in results:
        by_reason[r.termination_reason] = by_reason.get(r.termination_reason, 0) + 1
        total_latency += r.total_latency_s
        total_tokens += r.total_tokens
        if r.passed:
            passed += 1
        for ep in r.episodes:
            if ep.failure_class:
                failure_classes[ep.failure_class] = failure_classes.get(ep.failure_class, 0) + 1

    return {
        "cells_run": len(results),
        "cells_passed": passed,
        "pass_rate": f"{passed}/{len(results)}",
        "termination_reasons": by_reason,
        "total_latency_s": round(total_latency, 1),
        "total_latency_min": round(total_latency / 60, 1),
        "total_tokens": total_tokens,
        "failure_classes": failure_classes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Session 1 Config E matrix.")
    parser.add_argument("--max-iters", type=int, default=5,
                        help="Max iterations per cell (default 5).")
    parser.add_argument("--no-progress-window", type=int, default=3,
                        help="Cycle detection window (default 3).")
    parser.add_argument("--only", help="Run only cells whose bug module contains this substring.")
    args = parser.parse_args()

    if args.only:
        global MATRIX
        MATRIX = [row for row in MATRIX if args.only in row[0] or args.only in row[1]]
        print(f"Filtered matrix: {len(MATRIX)} cells")

    results = run_matrix(args.max_iters, args.no_progress_window)
    path = write_aggregate(results)
    print(f"\nAggregate: {path}")

    # Terse headline summary
    summary = summarize(results)
    print("\n=== Headline ===")
    print(f"Passed: {summary['pass_rate']}")
    print(f"Termination: {summary['termination_reasons']}")
    print(f"Failure classes: {summary['failure_classes']}")
    print(f"Wall: {summary['total_latency_min']} min  |  Tokens: {summary['total_tokens']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
