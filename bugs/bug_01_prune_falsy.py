"""Bug 01 — prune(older_than_days=0) falsy short-circuit.

The prune() function in store.py uses `older_than_days or self._retention_days`
to resolve which day threshold to use. When older_than_days=0, Python's truthiness
rules treat 0 as falsy and fall through to self._retention_days (None by default),
so the function returns 0 without pruning anything. User asked "prune everything
older than 0 days" (= prune everything) and got nothing pruned.

The fix is trivial (replace `or` with an explicit `is not None` check), but the
bug requires the model to reason about Python truthiness semantics from the test
failure alone. This is a good baseline bug: low difficulty but non-trivial
reasoning about Python's semantics.

Source: anneal-memory commit c3bd2ea (Mar 31, 2026) — one of 14 fixes in that
commit, isolated here for harness benchmarking.
"""

BUG = {
    "id": "bug_01_prune_falsy",
    "title": "prune(older_than_days=0) falsy short-circuit",
    "difficulty": "easy-medium",

    "source_file": "anneal_memory/store.py",
    "target_class": "Store",
    "target_function": "prune",

    # Bug reintroduction: simple find/replace on the fixed line.
    # `find` is the current (post-fix) line; `replace` is the buggy line.
    "bug_reintroduction": {
        "find": "        days = older_than_days if older_than_days is not None else self._retention_days",
        "replace": "        days = older_than_days or self._retention_days",
    },

    # Test that verifies the fix.
    "test_file": "tests/test_store.py",
    "test_command": "pytest tests/test_store.py::TestPruneEdgeCases::test_prune_zero_days -v --tb=short",

    # Name of the test for prompt context.
    "test_name": "test_prune_zero_days",

    # Commit where this bug was fixed (for reference only; not used by pipeline).
    "fix_commit": "c3bd2ea",
}
