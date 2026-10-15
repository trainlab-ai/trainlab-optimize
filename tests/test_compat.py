"""The package declares requires-python >= 3.10, so the source must parse on 3.10.

Python 3.12 relaxed f-string rules (PEP 701): an f-string may reuse its own quote character
inside a replacement field. Older versions reject that with a SyntaxError at import time,
so a test suite run only on 3.12+ never sees it. This test finds that pattern with the
current tokenizer, independent of which interpreter runs the tests.
"""
import io
import sys
import tokenize
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _quote(prefix_and_quote: str) -> str:
    body = prefix_and_quote.lstrip("rRfFbBuU")
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def _reused_quotes(text: str):
    """Yield line numbers of f-strings whose fields contain the f-string's own quote."""
    stack = []  # [quote, start_line, bad]
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.FSTRING_START:
            if stack and _quote(tok.string) == stack[-1][0]:
                stack[-1][2] = True  # nested f-string with the same quote
            stack.append([_quote(tok.string), tok.start[0], False])
        elif tok.type == tokenize.FSTRING_END:
            quote, line, bad = stack.pop()
            if bad:
                yield line
        elif stack and tok.type == tokenize.STRING:
            outer = stack[-1][0]
            inner = _quote(tok.string)
            if outer == inner or (len(outer) == 1 and outer in inner):
                stack[-1][2] = True


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs the 3.12 tokenizer; older versions fail at import instead")
def test_source_parses_on_python_310():
    problems = []
    for path in sorted(SRC.rglob("*.py")):
        for line in _reused_quotes(path.read_text(encoding="utf-8")):
            problems.append(f"{path.relative_to(SRC.parent)}:{line}")
    assert not problems, "f-string reuses its own quote (SyntaxError before Python 3.12): " + ", ".join(problems)


def test_detector_catches_the_pattern():
    if sys.version_info < (3, 12):
        pytest.skip("needs the 3.12 tokenizer")
    bad = 'x = f"day{"s" if n else ""}"\n'
    good = "x = f\"day{'s' if n else ''}\"\n"
    assert list(_reused_quotes(bad)) == [1]
    assert list(_reused_quotes(good)) == []
