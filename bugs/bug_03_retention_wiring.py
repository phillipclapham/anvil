"""Bug 03 — retention_days not wired into wrap_completed.

Store supports a ``retention_days`` constructor parameter and a
``prune()`` method, but in the buggy version ``wrap_completed()`` did
not actually call ``prune()`` at the end. Users who configured
``retention_days=30`` expected old episodes to be pruned on every
wrap; instead episodes accumulated forever because nothing ever
called prune.

The fix adds an auto-prune step at the end of wrap_completed when
``retention_days`` is configured. The 10.5c.5 two-phase-commit refactor
added a ``not self._defer_commit`` guard so the auto-prune is skipped
inside a batch (pipeline callers can invoke prune after the batch
exits).

This is a medium-difficulty bug: the fix is short, but the model has
to notice that the test is configuring ``retention_days=30``, wraps,
and then expects ``total_episodes == 1`` — the test is asking
wrap_completed to prune, and the current wrap_completed code in the
buggy version never calls prune. The reasoning chain is "test expects
pruning on wrap → wrap_completed has no prune call → add one, guarded
by retention_days."

Source: anneal-memory commit e03417b (Mar 31, 2026) — one of 6 fixes
in that commit, isolated here for harness benchmarking.
"""

BUG = {
    "id": "bug_03_retention_wiring",
    "title": "retention_days not wired into wrap_completed auto-prune",
    "difficulty": "medium",

    "source_file": "anneal_memory/store.py",
    "target_class": "Store",
    "target_function": "wrap_completed",

    "bug_reintroduction": {
        "find": (
            "        pruned = 0\n"
            "        if self._retention_days is not None and not self._defer_commit:\n"
            "            pruned = self.prune()"
        ),
        "replace": (
            "        pruned = 0"
        ),
    },

    "test_file": "tests/test_store.py",
    "test_command": "pytest tests/test_store.py::TestWrapLifecycle::test_wrap_completed_auto_prunes_when_retention_set -v --tb=short",
    "test_name": "test_wrap_completed_auto_prunes_when_retention_set",

    "fix_commit": "e03417b",
}
