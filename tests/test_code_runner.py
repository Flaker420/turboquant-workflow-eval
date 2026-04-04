from __future__ import annotations

from turboquant_workflow_eval.code_runner import extract_python_code, run_code_with_tests, _values_match


class TestExtractPythonCode:
    def test_fenced_python(self) -> None:
        text = 'Here is the code:\n```python\ndef foo():\n    return 42\n```\nDone.'
        code = extract_python_code(text)
        assert code is not None
        assert "def foo" in code

    def test_fenced_plain(self) -> None:
        text = 'Code:\n```\ndef bar():\n    pass\n```'
        code = extract_python_code(text)
        assert code is not None
        assert "def bar" in code

    def test_raw_def(self) -> None:
        text = "Some explanation.\ndef baz(x):\n    return x + 1\n"
        code = extract_python_code(text)
        assert code is not None
        assert "def baz" in code

    def test_no_code(self) -> None:
        assert extract_python_code("Just plain text.") is None


class TestValuesMatch:
    def test_exact(self) -> None:
        assert _values_match("42", "42") is True

    def test_none_repr(self) -> None:
        assert _values_match("None", "None") is True

    def test_list_match(self) -> None:
        assert _values_match("[2.0, 3.0, 4.0]", "[2.0, 3.0, 4.0]") is True

    def test_float_tolerance(self) -> None:
        assert _values_match("2.0000001", "2.0") is True

    def test_mismatch(self) -> None:
        assert _values_match("42", "43") is False


class TestRunCodeWithTests:
    def test_passing(self) -> None:
        code = "def add(a, b):\n    return a + b"
        cases = [
            {"input": "[1, 2]", "expected": "3"},
            {"input": "[0, 0]", "expected": "0"},
        ]
        result = run_code_with_tests(code, cases)
        assert result["verdict"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_failing(self) -> None:
        code = "def add(a, b):\n    return a - b"
        cases = [{"input": "[1, 2]", "expected": "3"}]
        result = run_code_with_tests(code, cases)
        assert result["verdict"] == "fail"

    def test_error(self) -> None:
        code = "def add(a, b):\n    raise ValueError('oops')"
        cases = [{"input": "[1, 2]", "expected": "3"}]
        result = run_code_with_tests(code, cases)
        assert result["verdict"] == "error"

    def test_timeout(self) -> None:
        code = "import time\ndef slow():\n    time.sleep(10)\n    return 1"
        cases = [{"input": "[]", "expected": "1"}]
        result = run_code_with_tests(code, cases, timeout=1)
        assert result["verdict"] == "error"
        assert result["errors"] == 1

    def test_no_function(self) -> None:
        code = "x = 42"
        cases = [{"input": "[]", "expected": "42"}]
        result = run_code_with_tests(code, cases)
        assert result["verdict"] == "error"

    def test_empty_cases(self) -> None:
        result = run_code_with_tests("def f(): pass", [])
        assert result["verdict"] == "pass"
