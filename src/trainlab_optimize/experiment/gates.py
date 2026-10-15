"""Quality gates: is a cheaper model's output as good as the current model's, sample by sample?

Each sample is scored pass/fail against the reference: the known-correct answer when the
dataset has one, otherwise the current (baseline) model's output.

  classification  same label (normalised: case, punctuation, surrounding text)
  extraction      valid JSON with every field equal to the reference (numbers numerically,
                  strings case/space-normalised; prose fields such as "summary" by ROUGE-L >= 0.5);
                  field-level agreement is reported too
  generation      an LLM judge compares the two answers in both orders; the candidate passes
                  if it is not judged worse in either. Without a judge, generation results
                  are reported but never validated: similarity to the baseline is not quality.

A candidate PASSES only if the one-sided 95% lower confidence bound (Wilson) of its pass
rate is at or above the threshold, FAILS if the upper bound is below it, and is
INCONCLUSIVE otherwise (with an estimate of how many samples would settle it).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

TASKS = ("classification", "extraction", "generation")
DEFAULT_THRESHOLD = {"classification": 0.95, "extraction": 0.95, "generation": 0.90}
Z = 1.645                                     # one-sided 95%
CLASS_HINTS = {"classification", "routing", "sentiment", "moderation", "labeling", "tagging", "intent"}
EXTRACT_HINTS = {"extraction", "structured_output", "json", "parsing"}


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def verdict(k: int, n: int, threshold: float, min_n: int = 20) -> tuple[str, int | None]:
    """(PASS | FAIL | INCONCLUSIVE, samples needed to settle it when inconclusive)."""
    lo, hi = wilson(k, n)
    if n >= min_n and lo >= threshold:
        return "PASS", None
    if n >= 10 and hi < threshold:
        return "FAIL", None
    rate = k / n if n else 0.0
    if rate <= threshold:
        return "INCONCLUSIVE", None                     # at this rate, more samples won't make it pass
    for m in range(max(n, min_n), 20001):
        if wilson(round(rate * m), m)[0] >= threshold:
            return "INCONCLUSIVE", m
    return "INCONCLUSIVE", None


# ------------------------------------------------------------------ normalisation
def norm_text(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return " ".join(s.split())


def norm_label(s: str | None) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```\w*|```$", "", s).strip()
    line = next((ln for ln in s.splitlines() if ln.strip()), "")
    line = re.sub(r"^(label|category|class|answer|intent|sentiment)\s*[:=-]\s*", "", line.strip(), flags=re.I)
    line = line.strip().strip("\"'`*").strip()
    line = re.sub(r"[.!。]+$", "", line)
    return norm_text(line)


def parse_json(s: str | None):
    if not s:
        return None
    t = s.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            return None
    return None


def _flatten(x, prefix="") -> dict:
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flatten(v, f"{prefix}.{str(k).casefold()}" if prefix else str(k).casefold()))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        if not x:
            out[prefix] = []
    else:
        out[prefix] = x
    return out


def _same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return math.isclose(float(str(a).replace(",", "")), float(str(b).replace(",", "")), rel_tol=1e-6)
        except ValueError:
            return False
    if a is None or b is None:
        return a is None and b is None
    return norm_text(str(a)) == norm_text(str(b))


# Prose fields inside structured output ("summary": "...") are never worded the same way twice,
# not even by the same model. They are compared by overlap (ROUGE-L >= 0.5) instead of equality;
# every other field must match exactly.
PROSE_KEY = re.compile(r"(^|[._\]])(summary|description|reason|rationale|explanation|justification|notes?|"
                       r"comments?|message|reply|answer|details)$")


def _is_prose(key: str, a, b) -> bool:
    return bool(PROSE_KEY.search(key)) and isinstance(a, str) and isinstance(b, str) \
        and max(len(a.split()), len(b.split())) >= 4


def json_diff_fields(ref, cand) -> list[str]:
    """The fields (flattened keys) on which cand disagrees with ref; ["(invalid JSON)"] if unparsable."""
    if cand is None or ref is None:
        return ["(invalid JSON)"]
    fr, fc = _flatten(ref), _flatten(cand)
    out = []
    for k in sorted(set(fr) | set(fc)):
        if k not in fr or k not in fc:
            out.append(k)
        elif _is_prose(k, fr[k], fc[k]):
            if rouge_l(fr[k], fc[k]) < 0.5:
                out.append(k)
        elif not _same_value(fr[k], fc[k]):
            out.append(k)
    return out


def json_agreement(ref, cand) -> tuple[bool, float]:
    fr, fc = _flatten(ref), _flatten(cand)
    keys = set(fr) | set(fc)
    if not keys:
        return True, 1.0

    def same(k):
        if k not in fr or k not in fc:
            return False
        if _is_prose(k, fr[k], fc[k]):
            return rouge_l(fr[k], fc[k]) >= 0.5
        return _same_value(fr[k], fc[k])
    n = sum(1 for k in keys if same(k))
    return n == len(keys), n / len(keys)


def rouge_l(ref: str, cand: str) -> float:
    a, b = norm_text(ref).split()[:1500], norm_text(cand).split()[:1500]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


_REFUSAL = re.compile(r"^(i'?m sorry|i cannot|i can'?t|as an ai|i am unable|i'm unable)", re.I)


# ------------------------------------------------------------------ task detection and scoring
def detect_task(refs: list[str], hints: list[str] | None = None, json_mode: bool = False) -> str:
    h = {x.lower() for x in hints or []}
    if json_mode or h & EXTRACT_HINTS:
        return "extraction"
    if h & CLASS_HINTS:
        return "classification"
    refs = [r for r in refs if r]
    if not refs:
        return "generation"
    if sum(1 for r in refs if isinstance(parse_json(r), (dict, list))) >= 0.9 * len(refs):
        return "extraction"
    labels = [norm_label(r) for r in refs]
    if sum(1 for r in refs if len(r.split()) <= 6) >= 0.9 * len(refs) and len(set(labels)) <= max(20, len(refs) // 3):
        return "classification"
    return "generation"


def score(task: str, reference: str | None, candidate: str | None) -> dict:
    """{'pass': bool|None, ...details}. pass is None when it needs the judge (generation)."""
    if candidate is None:
        return {"pass": False, "reason": "error"}
    if not candidate.strip() or _REFUSAL.match(candidate.strip()):
        return {"pass": False, "reason": "empty or refusal"}
    if task == "classification":
        ok = norm_label(candidate) == norm_label(reference)
        return {"pass": ok, "reason": None if ok else "different label"}
    if task == "extraction":
        c = parse_json(candidate)
        if c is None:
            return {"pass": False, "reason": "invalid JSON", "field_score": 0.0}
        r = parse_json(reference)
        if r is None:
            return {"pass": False, "reason": "reference is not JSON", "field_score": 0.0}
        exact, fs = json_agreement(r, c)
        return {"pass": exact, "reason": None if exact else "fields differ", "field_score": round(fs, 4)}
    return {"pass": None, "rouge_l": round(rouge_l(reference or "", candidate), 4)}


JUDGE_PROMPT = """You are grading two responses to the same request.

<request>
{request}
</request>

<response_a>
{a}
</response_a>

<response_b>
{b}
</response_b>

Which response better fulfils the request (correctness first, then completeness and following the
instructions)? Reply with exactly one word: A, B, or TIE (use TIE when they are equally good)."""


def judge_messages(request_messages: list[dict], a: str, b: str) -> list[dict]:
    req = "\n\n".join(f"[{m['role']}] {m['content']}" for m in request_messages)
    return [{"role": "user", "content": JUDGE_PROMPT.format(request=req[:20000], a=a[:12000], b=b[:12000])}]


def judge_choice(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"\b(TIE|A|B)\b", text.strip().upper())
    return m.group(1) if m else None
