"""Safe execution of generated Python code for coding-prompt evaluation."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from typing import Any


def extract_python_code(text: str) -> str | None:
    """Extract the first Python code block from *text*.

    Looks for fenced code blocks (```python ... ``` or ``` ... ```) first,
    then falls back to the entire text if it looks like valid Python.
    """
    # Try fenced blocks with language tag
    pattern = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # Try plain fenced blocks
    pattern = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # Fall back: if the text contains def/class, extract from first def/class to end
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ")):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()

    return None


def run_code_with_tests(
    code: str,
    test_cases: list[dict[str, str]],
    timeout: int = 5,
) -> dict[str, Any]:
    """Execute *code* in a subprocess and test it against *test_cases*.

    Each test case has ``input`` (arguments as a Python literal) and
    ``expected`` (the expected repr of the return value).  The function
    name is inferred from the first ``def`` statement in *code*.

    Returns a dict with keys ``passed``, ``failed``, ``errors``,
    ``details`` (list of per-test dicts), and ``verdict``
    (``'pass'``/``'fail'``/``'error'``).
    """
    if not test_cases:
        return {"passed": 0, "failed": 0, "errors": 0, "details": [], "verdict": "pass"}

    # Discover function name
    func_match = re.search(r"def\s+(\w+)\s*\(", code)
    if func_match is None:
        return {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "details": [{"error": "no function definition found in generated code"}],
            "verdict": "error",
        }
    func_name = func_match.group(1)

    passed = 0
    failed = 0
    errors = 0
    details: list[dict[str, Any]] = []

    for tc in test_cases:
        args_literal = tc["input"]
        expected_literal = tc["expected"]

        test_script = (
            "import json, sys\n"
            + code + "\n\n"
            + f"args = {args_literal}\n"
            + "if not isinstance(args, (list, tuple)):\n"
            + "    args = [args]\n"
            + f"result = {func_name}(*args)\n"
            + "print(repr(result))\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            errors += 1
            details.append({"input": args_literal, "expected": expected_literal, "error": "timeout"})
            continue

        if result.returncode != 0:
            errors += 1
            details.append({
                "input": args_literal,
                "expected": expected_literal,
                "error": result.stderr.strip()[-500:],
            })
            continue

        actual = result.stdout.strip()
        # Flexible comparison: check if expected appears in actual repr
        if expected_literal == "hasattr":
            # Special: just check the code defined the expected class/function
            passed += 1
            details.append({"input": args_literal, "expected": expected_literal, "actual": actual, "match": True})
        elif _values_match(actual, expected_literal):
            passed += 1
            details.append({"input": args_literal, "expected": expected_literal, "actual": actual, "match": True})
        else:
            failed += 1
            details.append({"input": args_literal, "expected": expected_literal, "actual": actual, "match": False})

    if errors > 0:
        verdict = "error"
    elif failed > 0:
        verdict = "fail"
    else:
        verdict = "pass"

    return {"passed": passed, "failed": failed, "errors": errors, "details": details, "verdict": verdict}


def _values_match(actual_repr: str, expected_literal: str) -> bool:
    """Compare actual output repr against expected literal, tolerating minor formatting diffs."""
    # Normalize whitespace
    a = actual_repr.strip()
    e = expected_literal.strip()

    if a == e:
        return True

    # Try evaluating both as literals and comparing
    try:
        actual_val = ast.literal_eval(a)
        expected_val = ast.literal_eval(e)
        if actual_val == expected_val:
            return True
        # Float tolerance
        if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
            if abs(actual_val - expected_val) < 1e-6:
                return True
        # List of floats tolerance
        if isinstance(actual_val, list) and isinstance(expected_val, list) and len(actual_val) == len(expected_val):
            if all(
                isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-6
                for a, b in zip(actual_val, expected_val)
            ):
                return True
    except Exception:
        pass

    return False
