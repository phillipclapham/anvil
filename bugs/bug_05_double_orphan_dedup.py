"""Bug 05 — Double-orphan adoption causes false verify() failures.

*** SUBSTITUTED from the original "raw-bytes hashing" framing ***

The originally-scoped Bug 5 (raw-bytes hashing in audit.py, commit 96aa8e4)
turned out to be untestable in Python-only: _compute_hash() strips
whitespace before hashing, so reverting to json.dumps(entry, sort_keys=True)
produces byte-identical output under CPython. The commit message itself
says the fix is defensive against hypothetical cross-language verifiers
("byte-identical in CPython today but fragile for cross-language"). No
Python test fails on reversion → not benchmarkable.

This substitute uses a different "hard" bug from the same commit window
(65ade7e, Apr 3, 2026) — the double-orphan deduplication fix.

## The bug

AuditTrail._adopt_orphaned_files() scans the audit directory for sealed
files the manifest doesn't know about. If a crash happens during rotation
between gzip completion and sealed_path.unlink(), BOTH a .gz file and a
.jsonl file exist for the same period.

In the buggy version, the code adopted both files into the manifest. That
produces TWO entries for the same period in the hash chain, and a
subsequent verify() call sees the duplication as a broken chain.

The fix groups orphans by period in a dict (`orphans_by_period`), then
deduplicates: if both .gz and .jsonl exist for the same period, prefer
the .gz (gzip completed = later state) and unlink the .jsonl duplicate.
It also sorts orphans by period so manifest entries land in chronological
order across mixed orphan types.

## Why it's hard

The model has to:
1. Read the test, which creates both a .gz AND a .jsonl file for the same
   period `2026-W13`, then initializes a trail and asserts:
   - .jsonl file should no longer exist (got removed)
   - manifest should have exactly one entry for period 2026-W13
   - verify() should return valid=True
2. Notice the current code just appends every orphan it finds (no dedup).
3. Understand the crash scenario: .gz file exists because gzip completed;
   .jsonl file exists because the unlink() after the rename/gzip was
   interrupted. Both describe the SAME logical period.
4. Implement grouping + dedup + preference for .gz + unlink of the .jsonl.
5. Preserve the existing manifest-building loop downstream (the `orphans`
   variable is consumed later in the method to build manifest entries).

Source: anneal-memory commit 65ade7e (Apr 3, 2026), MEDIUM finding
"Double-orphan adoption causing false verify() — deduplicate by period,
prefer .gz, remove .jsonl duplicate."
"""

BUG = {
    "id": "bug_05_double_orphan_dedup",
    "title": "Double-orphan adoption causes false verify() chain break",
    "difficulty": "hard",

    "source_file": "anneal_memory/audit.py",
    "target_class": "AuditTrail",
    "target_function": "_adopt_orphaned_files",

    "bug_reintroduction": {
        "find": (
            "        # Collect orphans grouped by period to detect duplicates\n"
            "        orphans_by_period: dict[str, list[Path]] = {}\n"
            "        for pattern in [f\"{prefix}*.jsonl.gz\", f\"{prefix}*.jsonl\"]:\n"
            "            for path in sorted(audit_dir.glob(pattern)):\n"
            "                if path.name == active_name:\n"
            "                    continue  # Skip the active file\n"
            "                if path.name not in known_files:\n"
            "                    # Extract period from filename\n"
            "                    period = path.name.removeprefix(prefix)\n"
            "                    period = period.removesuffix(\".jsonl.gz\").removesuffix(\".jsonl\")\n"
            "                    orphans_by_period.setdefault(period, []).append(path)\n"
            "\n"
            "        if not orphans_by_period:\n"
            "            return\n"
            "\n"
            "        # Deduplicate: if both .gz and .jsonl exist for same period,\n"
            "        # prefer .gz (gzip completed) and remove the .jsonl duplicate.\n"
            "        # Sort by period to ensure manifest entries are chronological —\n"
            "        # without sorting, two-pass glob inserts all .gz periods before\n"
            "        # all .jsonl periods, breaking chronological order in the manifest\n"
            "        # when mixed orphan types span non-adjacent periods.\n"
            "        orphans: list[Path] = []\n"
            "        for period, paths in sorted(orphans_by_period.items()):\n"
            "            if len(paths) > 1:\n"
            "                gz_paths = [p for p in paths if p.name.endswith(\".gz\")]\n"
            "                jsonl_paths = [p for p in paths if not p.name.endswith(\".gz\")]\n"
            "                if gz_paths:\n"
            "                    orphans.append(gz_paths[0])\n"
            "                    for dup in jsonl_paths:\n"
            "                        try:\n"
            "                            dup.unlink()\n"
            "                            logger.info(\"Removed duplicate orphan: %s (preferring .gz)\", dup.name)\n"
            "                        except OSError:\n"
            "                            logger.warning(\"Failed to remove duplicate orphan: %s\", dup.name)\n"
            "                else:\n"
            "                    orphans.append(paths[0])\n"
            "            else:\n"
            "                orphans.append(paths[0])"
        ),
        "replace": (
            "        orphans: list[Path] = []\n"
            "        # Scan for both .gz and uncompressed .jsonl orphans\n"
            "        for pattern in [f\"{prefix}*.jsonl.gz\", f\"{prefix}*.jsonl\"]:\n"
            "            for path in sorted(audit_dir.glob(pattern)):\n"
            "                if path.name == active_name:\n"
            "                    continue  # Skip the active file\n"
            "                if path.name not in known_files:\n"
            "                    orphans.append(path)\n"
            "\n"
            "        if not orphans:\n"
            "            return"
        ),
    },

    "test_file": "tests/test_audit.py",
    "test_command": "pytest tests/test_audit.py::TestDiogenesBugFixes::test_double_orphan_prefers_gz_and_removes_jsonl -v --tb=short",
    "test_name": "test_double_orphan_prefers_gz_and_removes_jsonl",

    "fix_commit": "65ade7e",
}
