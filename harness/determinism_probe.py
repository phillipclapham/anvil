"""Determinism probe — answers the Session 2 methodology question.

Does Ollama produce byte-identical output across calls at temperature=0.0
with a fixed seed? If yes, Session 2 can use determinism as a belt-and-
suspenders complement to N=3 (free variance reduction). If no, Session 2
is committed to N>=3 at temperature=0.2 because "just turn the temperature
down" isn't a valid substitute for replication.

Session 1 found that bug_01 x both models flipped pass/fail between
Session 0.5 (N=1 single-shot) and Session 1 (N=1 Config E iter 1) at
temperature=0.2. This probe runs the same Session 0.5 single-shot prompt
on bug_01 x Gemma twice at temperature=0.0 + seed=42 and compares:

    1. fix_hash (sha1 of whitespace-normalized extracted fix)
    2. raw response byte equality

If fix_hashes match across calls we have genuine determinism. If only the
raw responses vary in whitespace but the extracted fix is identical, that
still counts for the benchmark since we classify by fix_hash.

Usage:
    python3 harness/determinism_probe.py

Writes nothing. Prints verdict. ~3-4 minutes of wall time.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HARNESS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HARNESS_ROOT))

from harness.baseline import (  # noqa: E402
    ANNEAL_REPO,
    assert_clean_repo,
    reset_repo,
    reintroduce_bug,
    run_test,
    extract_function_source,
    extract_test_source,
    extract_code_from_response,
    build_prompt,
)

import importlib  # noqa: E402


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-q4_K_M"
BUG_MODULE = "bugs.bug_01_prune_falsy"
SEED = 42


def fix_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def call_ollama_deterministic(prompt: str) -> dict:
    """Call Ollama with temperature=0.0 + fixed seed. Returns raw API dict."""
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "seed": SEED,
                "num_ctx": 16384,
            },
        }
    ).encode("utf-8")
    req = Request(
        OLLAMA_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urlopen(req, timeout=600) as resp:
        api = json.loads(resp.read().decode("utf-8"))
    api["_latency_s"] = time.time() - t0
    return api


def main() -> int:
    bug_module = importlib.import_module(BUG_MODULE)
    bug = bug_module.BUG

    assert_clean_repo(ANNEAL_REPO)
    try:
        if not reintroduce_bug(bug, ANNEAL_REPO):
            print("FAIL: bug reintroduction find-string missing (stale config)")
            return 2

        pre_passed, pre_output = run_test(bug, ANNEAL_REPO)
        if pre_passed:
            print("FAIL: bug reintroduction did not break the test")
            return 2

        src_path = ANNEAL_REPO / bug["source_file"]
        test_path = ANNEAL_REPO / bug["test_file"]
        buggy_fn = extract_function_source(
            src_path, bug["target_class"], bug["target_function"]
        )
        test_src = extract_test_source(test_path, bug["test_name"])
        prompt = build_prompt(bug, buggy_fn, test_src, pre_output)
    finally:
        reset_repo(ANNEAL_REPO)

    print(f"Probe: {MODEL} on {bug['id']} at temp=0.0 + seed={SEED}")
    print(f"Prompt length: {len(prompt)} chars")
    print()

    print("Call 1...")
    api1 = call_ollama_deterministic(prompt)
    resp1 = api1.get("response", "")
    fix1 = extract_code_from_response(resp1)
    h1 = fix_hash(fix1)
    print(
        f"  latency={api1['_latency_s']:.1f}s  "
        f"eval_tokens={api1.get('eval_count', 0)}  "
        f"fix_hash={h1}  "
        f"len(fix)={len(fix1)}"
    )

    print("Call 2...")
    api2 = call_ollama_deterministic(prompt)
    resp2 = api2.get("response", "")
    fix2 = extract_code_from_response(resp2)
    h2 = fix_hash(fix2)
    print(
        f"  latency={api2['_latency_s']:.1f}s  "
        f"eval_tokens={api2.get('eval_count', 0)}  "
        f"fix_hash={h2}  "
        f"len(fix)={len(fix2)}"
    )
    print()

    raw_match = resp1 == resp2
    fix_match = fix1 == fix2
    hash_match = h1 == h2

    print("=== Verdict ===")
    print(f"  raw response byte-equal:    {raw_match}")
    print(f"  extracted fix byte-equal:   {fix_match}")
    print(f"  fix_hash equal:             {hash_match}")
    print()

    if hash_match:
        print(
            "OUTCOME: Ollama temperature=0.0 + fixed seed IS deterministic "
            "for this model. Session 2 can use determinism as a variance-"
            "reduction layer on top of N>=3."
        )
        return 0
    else:
        print(
            "OUTCOME: Ollama temperature=0.0 + fixed seed is NOT deterministic "
            "for this model. Session 2 must rely on N>=3 replication at "
            "temperature=0.2 as the only honest variance control."
        )
        print()
        print("First 400 chars of each extracted fix for inspection:")
        print("--- fix 1 ---")
        print(fix1[:400])
        print("--- fix 2 ---")
        print(fix2[:400])
        return 1


if __name__ == "__main__":
    sys.exit(main())
