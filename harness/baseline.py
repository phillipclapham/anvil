"""Single-shot baseline runner for the iterative harness benchmark.

For a given bug + model, this script:
1. Verifies the anneal-memory repo is clean.
2. Reintroduces the bug via file find/replace.
3. Runs the bug's test → expects failure (confirms reintroduction worked).
4. Builds a prompt containing the buggy function + failing test + test output.
5. Calls the Ollama API to get the model's proposed fix.
6. Applies the fix by replacing the target function in the source file.
7. Runs the test again → hopefully passes.
8. Resets the repo to clean state.
9. Records the result.

This is the SINGLE-SHOT baseline (Config 0). The iterative loop (Config E naive +
Config A anneal-memory) builds on top of this in later sessions.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ANNEAL_REPO = Path.home() / "Documents" / "anneal-memory"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
HARNESS_ROOT = Path(__file__).parent.parent


# -- Result record -----------------------------------------------------------


@dataclass
class BaselineResult:
    bug_id: str
    model: str
    reintroduction_ok: bool
    pre_fix_test_passed: bool  # Should be False (bug reproduced)
    post_fix_test_passed: bool  # Should be True (model fixed it)
    model_latency_s: float
    model_eval_tokens: int
    model_eval_rate_tok_s: float
    fix_applied_ok: bool
    error: str | None = None
    raw_response: str = ""
    extracted_fix: str = ""
    pre_test_output: str = ""
    post_test_output: str = ""


# -- Repo state management ----------------------------------------------------


def assert_clean_repo(repo: Path) -> None:
    """Fail loudly if the anneal-memory repo isn't clean. We're about to mutate it."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            f"anneal-memory repo at {repo} has uncommitted changes. "
            f"Refusing to run baseline — would lose work. Output:\n{result.stdout}"
        )


def reset_repo(repo: Path) -> None:
    """Hard-reset the working tree to HEAD. Safe because we asserted cleanliness first."""
    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=repo,
        check=True,
        capture_output=True,
    )


# -- Bug reintroduction -------------------------------------------------------


def reintroduce_bug(bug: dict, repo: Path) -> bool:
    """Apply the bug-reintroduction find/replace to the source file.

    Returns True if the replacement was made, False if the `find` string
    wasn't present (indicates stale config or repo drift).
    """
    src_path = repo / bug["source_file"]
    content = src_path.read_text()

    find = bug["bug_reintroduction"]["find"]
    replace = bug["bug_reintroduction"]["replace"]

    if find not in content:
        return False

    new_content = content.replace(find, replace, 1)
    src_path.write_text(new_content)
    return True


# -- Test execution -----------------------------------------------------------


def run_test(bug: dict, repo: Path) -> tuple[bool, str]:
    """Run the bug's test command. Returns (passed, combined_output)."""
    result = subprocess.run(
        bug["test_command"].split(),
        cwd=repo,
        capture_output=True,
        text=True,
    )
    passed = result.returncode == 0
    output = result.stdout + result.stderr
    return passed, output


# -- Function extraction (AST-based) ------------------------------------------


def extract_function_source(file_path: Path, class_name: str, function_name: str) -> str:
    """Extract the source lines of a method from a class, preserving indentation."""
    source = file_path.read_text()
    tree = ast.parse(source)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                    target = item
                    break
            break

    if target is None:
        raise ValueError(f"{class_name}.{function_name} not found in {file_path}")

    lines = source.splitlines(keepends=True)
    # Include any decorators above the def
    start = (target.decorator_list[0].lineno if target.decorator_list else target.lineno) - 1
    end = target.end_lineno  # 1-indexed, inclusive of last line

    return "".join(lines[start:end])


def extract_test_source(file_path: Path, test_name: str) -> str:
    """Find a test function anywhere in the file (inside any class or top-level)."""
    source = file_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            lines = source.splitlines(keepends=True)
            start = node.lineno - 1
            end = node.end_lineno
            return "".join(lines[start:end])

    raise ValueError(f"Test {test_name} not found in {file_path}")


# -- Fix application ----------------------------------------------------------


def replace_function_in_file(
    file_path: Path, class_name: str, function_name: str, new_function_source: str
) -> None:
    """Replace a method in a class with new source, preserving class-body indentation."""
    source = file_path.read_text()
    tree = ast.parse(source)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                    target = item
                    break
            break

    if target is None:
        raise ValueError(f"{class_name}.{function_name} not found in {file_path}")

    lines = source.splitlines(keepends=True)
    start = (target.decorator_list[0].lineno if target.decorator_list else target.lineno) - 1
    end = target.end_lineno

    # Figure out the indentation of the existing def line (methods are typically 4 spaces in)
    def_line = lines[target.lineno - 1]
    existing_indent = def_line[: len(def_line) - len(def_line.lstrip())]

    # Normalize new source: strip leading whitespace from each line, then re-add existing indent
    new_lines = new_function_source.strip().splitlines()
    # Detect the new source's own base indent (from the def line inside it)
    new_def_line = next((l for l in new_lines if l.lstrip().startswith("def ")), new_lines[0])
    new_base_indent = new_def_line[: len(new_def_line) - len(new_def_line.lstrip())]

    reindented = []
    for line in new_lines:
        if line.strip() == "":
            reindented.append("\n")
        else:
            # Strip the new source's base indent, re-add the target's existing indent
            if line.startswith(new_base_indent):
                stripped = line[len(new_base_indent):]
            else:
                stripped = line.lstrip()
            reindented.append(existing_indent + stripped + "\n")

    # Preserve trailing blank line if it existed in the original block
    new_block = "".join(reindented)

    new_source = "".join(lines[:start]) + new_block + "".join(lines[end:])
    file_path.write_text(new_source)


# -- Response parsing ---------------------------------------------------------


def extract_code_from_response(response: str) -> str:
    """Pull a Python code block out of a markdown response.

    Handles triple-backtick fences with or without language tag.
    Falls back to the raw response stripped.
    """
    # Prefer fenced code blocks
    fence_pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = fence_pattern.findall(response)
    if matches:
        # Prefer the longest match — usually the full function
        return max(matches, key=len).strip()
    return response.strip()


# -- Prompt construction ------------------------------------------------------


def build_prompt(bug: dict, buggy_function: str, test_source: str, test_output: str) -> str:
    """Construct the single-shot prompt for the model.

    Deliberately does NOT include the bug description — we're testing whether
    the model can diagnose the bug from code + test + failure alone.
    """
    return f"""You are debugging a Python method. A test is failing. Your task: identify the bug, \
return the corrected method.

# The method under test

```python
{buggy_function.rstrip()}
```

# The failing test

```python
{test_source.rstrip()}
```

# The test output

```
{test_output.rstrip()}
```

# Your task

Analyze the code, the test, and the test output. Identify the bug. Return ONLY \
the corrected method — a single Python code block containing the full method from \
`def` through the return statement. No explanation. No prose. No additional \
functions. No tests. Just the fixed method inside a python code block.
"""


# -- Ollama API call ----------------------------------------------------------


def call_ollama(model: str, prompt: str, timeout: int = 300) -> dict[str, Any]:
    """Call Ollama generate API. Returns the parsed JSON response."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
    ).encode("utf-8")

    req = Request(
        OLLAMA_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# -- Main pipeline ------------------------------------------------------------


def run_baseline(bug_module_name: str, model: str, repo: Path = ANNEAL_REPO) -> BaselineResult:
    """Run the single-shot baseline pipeline for one bug + one model."""
    # Load bug config
    sys.path.insert(0, str(HARNESS_ROOT))
    bug_module = importlib.import_module(bug_module_name)
    bug = bug_module.BUG

    result = BaselineResult(
        bug_id=bug["id"],
        model=model,
        reintroduction_ok=False,
        pre_fix_test_passed=False,
        post_fix_test_passed=False,
        model_latency_s=0.0,
        model_eval_tokens=0,
        model_eval_rate_tok_s=0.0,
        fix_applied_ok=False,
    )

    try:
        assert_clean_repo(repo)

        # 1. Reintroduce the bug
        if not reintroduce_bug(bug, repo):
            result.error = "Bug reintroduction find-string not present — config may be stale."
            return result
        result.reintroduction_ok = True

        # 2. Run the test — we expect it to fail
        pre_passed, pre_output = run_test(bug, repo)
        result.pre_fix_test_passed = pre_passed
        result.pre_test_output = pre_output[-2000:]  # Last 2000 chars
        if pre_passed:
            result.error = "Bug reintroduction did not break the test. Something is wrong."
            reset_repo(repo)
            return result

        # 3. Extract the buggy function + test source for the prompt
        src_path = repo / bug["source_file"]
        test_path = repo / bug["test_file"]
        buggy_function = extract_function_source(src_path, bug["target_class"], bug["target_function"])
        test_source = extract_test_source(test_path, bug["test_name"])

        # 4. Build prompt + call model
        prompt = build_prompt(bug, buggy_function, test_source, pre_output)
        t0 = time.time()
        api_response = call_ollama(model, prompt)
        result.model_latency_s = time.time() - t0

        result.raw_response = api_response.get("response", "")
        eval_ns = api_response.get("eval_duration", 0)
        eval_count = api_response.get("eval_count", 0)
        result.model_eval_tokens = eval_count
        if eval_ns:
            result.model_eval_rate_tok_s = eval_count / (eval_ns / 1e9)

        # 5. Extract the code from the response
        extracted = extract_code_from_response(result.raw_response)
        result.extracted_fix = extracted

        if "def " not in extracted:
            result.error = "Model response contained no function definition."
            reset_repo(repo)
            return result

        # 6. Apply the fix
        try:
            replace_function_in_file(src_path, bug["target_class"], bug["target_function"], extracted)
            result.fix_applied_ok = True
        except Exception as e:
            result.error = f"Fix application failed: {e}"
            reset_repo(repo)
            return result

        # 7. Run the test again
        post_passed, post_output = run_test(bug, repo)
        result.post_fix_test_passed = post_passed
        result.post_test_output = post_output[-2000:]

    finally:
        # Always reset the repo, even on exception, so we don't leave anneal-memory dirty.
        try:
            reset_repo(repo)
        except Exception as e:
            print(f"WARNING: repo reset failed: {e}", file=sys.stderr)

    return result


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-shot baseline for one bug + one model.")
    parser.add_argument("bug", help="Bug module name, e.g. bugs.bug_01_prune_falsy")
    parser.add_argument("model", help="Ollama model tag, e.g. gemma4:e4b-it-q4_K_M")
    parser.add_argument(
        "--repo",
        default=str(ANNEAL_REPO),
        help=f"Path to anneal-memory repo (default: {ANNEAL_REPO})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON (default: human-readable summary)",
    )
    args = parser.parse_args()

    result = run_baseline(args.bug, args.model, Path(args.repo))

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(f"=== Baseline: {result.bug_id} on {result.model} ===")
        print(f"Bug reintroduction: {'OK' if result.reintroduction_ok else 'FAILED'}")
        print(f"Pre-fix test (should fail): {'FAIL ✓' if not result.pre_fix_test_passed else 'PASS ✗'}")
        print(f"Model response: {result.model_latency_s:.1f}s, {result.model_eval_tokens} tokens @ {result.model_eval_rate_tok_s:.1f} tok/s")
        print(f"Fix applied: {'OK' if result.fix_applied_ok else 'FAILED'}")
        print(f"Post-fix test: {'PASS ✓' if result.post_fix_test_passed else 'FAIL ✗'}")
        if result.error:
            print(f"Error: {result.error}")
        print()
        print(f"=== Model's proposed fix ===")
        print(result.extracted_fix[:2000])
        print()
        if not result.post_fix_test_passed and result.fix_applied_ok:
            print("=== Post-fix test output (last 500 chars) ===")
            print(result.post_test_output[-500:])

    return 0 if result.post_fix_test_passed else 1


if __name__ == "__main__":
    sys.exit(main())
