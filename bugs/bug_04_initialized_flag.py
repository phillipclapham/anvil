"""Bug 04 — AuditTrail._initialized flag set before init complete.

AuditTrail uses lazy initialization: the first call to ``log()`` calls
``_initialize()``, which adopts orphaned sealed files, reads the last
valid entry, and recovers ``seq`` + ``prev_hash`` from on-disk state.
Only after all recovery steps succeed is the audit trail ready to
append to the chain.

In the buggy version, ``_initialize()`` set ``self._initialized = True``
as its FIRST statement. If any recovery step then raised (disk full,
permission error during orphan adoption, corrupt manifest), the flag
was already set. The next ``log()`` call would see ``_initialized=True``
and skip re-running ``_initialize()``, writing with broken state:
``seq=0`` and ``prev_hash=GENESIS_HASH``. This breaks the hash chain
invariant and silently corrupts the audit trail across process
lifetimes.

The fix moves ``self._initialized = True`` to the END of ``_initialize()``
— after both code paths (fresh-start and recovery-from-existing-file)
have finished their work. If any step raises, the flag stays False and
the next ``log()`` call retries initialization.

This is a hard bug because the model has to:
1. Notice the test is monkeypatching ``_adopt_orphaned_files`` to fail.
2. Read the test assertion: ``_initialized`` should be False after
   the failure, and the next ``log()`` should succeed with correct seq.
3. Realize this means the init method is setting the flag too early.
4. Move the flag to the end of ALL code paths — including the early-
   return branch for fresh starts, not just the main path.

Multi-site reintroduction: this bug reverts THREE locations
simultaneously (add flag at top, remove from end of fresh-start
branch, remove from end of main branch). The harness's multi-edit
support handles this.

Source: anneal-memory commit 65ade7e (Apr 3, 2026) — one of 6 fixes
in that commit, isolated here for harness benchmarking.
"""

BUG = {
    "id": "bug_04_initialized_flag",
    "title": "AuditTrail._initialized flag set before init complete",
    "difficulty": "hard",

    "source_file": "anneal_memory/audit.py",
    "target_class": "AuditTrail",
    "target_function": "_initialize",

    "bug_reintroduction": [
        {
            "find": (
                "        \"\"\"Lazy init: recover seq and prev_hash from existing audit file.\n"
                "\n"
                "        Sets _initialized only after all recovery steps complete. If any\n"
                "        step raises (disk full, permission error during orphan adoption),\n"
                "        the next log() call retries init instead of writing with broken state.\n"
                "        \"\"\"\n"
                "        # Adopt orphaned sealed files"
            ),
            "replace": (
                "        \"\"\"Lazy init: recover seq and prev_hash from existing audit file.\"\"\"\n"
                "        self._initialized = True\n"
                "\n"
                "        # Adopt orphaned sealed files"
            ),
        },
        {
            "find": (
                "            self._last_week = _iso_week_now()\n"
                "            self._initialized = True\n"
                "            return"
            ),
            "replace": (
                "            self._last_week = _iso_week_now()\n"
                "            return"
            ),
        },
        {
            "find": (
                "        else:\n"
                "            self._last_week = _iso_week_now()\n"
                "\n"
                "        self._initialized = True\n"
                "\n"
                "    def _adopt_orphaned_files(self) -> None:"
            ),
            "replace": (
                "        else:\n"
                "            self._last_week = _iso_week_now()\n"
                "\n"
                "    def _adopt_orphaned_files(self) -> None:"
            ),
        },
    ],

    "test_file": "tests/test_audit.py",
    "test_command": "pytest tests/test_audit.py::TestDiogenesBugFixes::test_init_failure_allows_retry -v --tb=short",
    "test_name": "test_init_failure_allows_retry",

    "fix_commit": "65ade7e",
}
