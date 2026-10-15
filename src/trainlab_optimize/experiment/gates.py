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


# ------------------------------------------------------------------ extraction: fields by role
# Every field of a structured answer gets a role, decided once per experiment from the reference
# answers and the field names (never from how candidates did, so a role can't excuse a mistake):
#   fact      must be equal (numbers numerically, text case- and space-insensitive)
#   prose     free text ("summary", "reason", or long strings): equal enough by word overlap
#             (ROUGE-L >= 0.5), otherwise an LLM judge decides; without a judge it fails (approximate)
#   judgment  a rating the current model has no claim to be right about ("urgency", "priority",
#             "sentiment"...): agreement is measured and shown, never part of pass/fail
#   ignore    named with --ignore-fields: not compared at all
# --strict-fields makes any field a fact.
PROSE_KEY = re.compile(r"(^|[._\]])(summary|description|reason|rationale|explanation|justification|notes?|"
                       r"comments?|message|reply|answer|details)$")
JUDGMENT_KEY = re.compile(r"(^|[._\]])(urgency|priority|severity|sentiment|mood|tone|emotion|confidence|"
                          r"rating|score|importance|risk|risk_level|quality|difficulty)$")
PROSE_MIN_WORDS = 15         # strings that average this many words in the references are prose
                             # (an address or a name is a fact; a sentence is prose)
PROSE_OVERLAP = 0.5


def _leaf(key: str) -> str:
    return re.sub(r"\[\d+\]", "", key).rsplit(".", 1)[-1]


def _matches(key: str, names) -> bool:
    return any(key == n or _leaf(key) == n or key.startswith(n + ".") or key.startswith(n + "[")
               for n in names)


def field_roles(references: list, ignore=(), strict=()) -> dict[str, str]:
    """Role of every flattened field seen in the parsed reference answers."""
    words: dict[str, list[int]] = {}
    for ref in references:
        for k, v in _flatten(ref).items():
            words.setdefault(k, []).append(len(v.split()) if isinstance(v, str) else 0)
    roles = {}
    for k, ws in words.items():
        if _matches(k, ignore):
            roles[k] = "ignore"
        elif _matches(k, strict):
            roles[k] = "fact"
        elif JUDGMENT_KEY.search(k):
            roles[k] = "judgment"
        elif PROSE_KEY.search(k) or (ws and sum(ws) / len(ws) >= PROSE_MIN_WORDS):
            roles[k] = "prose"
        else:
            roles[k] = "fact"
    return roles


def _role(key: str, roles: dict | None, a, b) -> str:
    if roles and key in roles:
        return roles[key]
    if roles is not None:                 # a field the references never had: compare it strictly
        return "fact"
    if JUDGMENT_KEY.search(key):
        return "judgment"
    if isinstance(a, str) and isinstance(b, str) and (
            PROSE_KEY.search(key) or max(len(a.split()), len(b.split())) >= PROSE_MIN_WORDS):
        return "prose"
    return "fact"


def compare_fields(ref, cand, roles: dict | None = None) -> dict[str, tuple[str, bool | None]]:
    """{field: (role, agrees)} for every field in either answer. For prose, agrees is None when the
    wording differs too much to decide without a judge."""
    fr, fc = _flatten(ref), _flatten(cand)
    out = {}
    for k in sorted(set(fr) | set(fc)):
        a, b = fr.get(k), fc.get(k)
        role = _role(k, roles, a, b)
        if k not in fr or k not in fc:
            out[k] = (role, False)
        elif role == "prose" and isinstance(a, str) and isinstance(b, str):
            out[k] = (role, True if rouge_l(a, b) >= PROSE_OVERLAP or norm_text(a) == norm_text(b) else None)
        else:
            out[k] = (role, _same_value(a, b))
    return out


def extraction_verdict(fields: dict[str, tuple[str, bool | None]], judged: dict[str, bool | None] | None = None
                       ) -> bool | None:
    """Pass when every fact agrees and every prose field agrees (by overlap, or by the judge when it
    was asked). Judgment and ignored fields don't count. None = waiting for the judge."""
    judged = judged or {}
    pending = False
    for k, (role, ok) in fields.items():
        if role == "fact" and not ok:
            return False
        if role == "prose":
            if ok is None:
                ok = judged.get(k, False) if k in judged else None
            if ok is False:
                return False
            if ok is None:
                pending = True
    return None if pending else True


def json_diff_fields(ref, cand, roles: dict | None = None) -> list[str]:
    """The fields on which cand disagrees with ref (prose: by word overlap); ["(invalid JSON)"] if unparsable."""
    if cand is None or ref is None:
        return ["(invalid JSON)"]
    return [k for k, (role, ok) in compare_fields(ref, cand, roles).items() if ok is not True and role != "ignore"]


def json_agreement(ref, cand, roles: dict | None = None) -> tuple[bool, float]:
    """(passes on facts and prose by overlap, share of all fields that agree)."""
    f = compare_fields(ref, cand, roles)
    if not f:
        return True, 1.0
    scored = {k: v for k, v in f.items() if v[0] != "ignore"}
    same = sum(1 for _, ok in scored.values() if ok is True)
    return extraction_verdict(f) is True, (same / len(scored)) if scored else 1.0


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


def score(task: str, reference: str | None, candidate: str | None, roles: dict | None = None,
          judged: dict | None = None) -> dict:
    """{'pass': bool|None, ...details}. pass is None when it needs the judge (generation, or
    extraction prose that differs in wording and hasn't been judged yet)."""
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
        fields = compare_fields(r, c, roles)
        ok = extraction_verdict(fields, judged)
        scored = {k: v for k, v in fields.items() if v[0] != "ignore"}
        fs = sum(1 for _, a in scored.values() if a is True) / len(scored) if scored else 1.0
        return {"pass": ok, "reason": None if ok else ("fields differ" if ok is False else "prose needs the judge"),
                "field_score": round(fs, 4), "fields": {k: [role, a] for k, (role, a) in fields.items()}}
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


FIELD_JUDGE_PROMPT = """You are grading one field of two structured answers to the same request.

<request>
{request}
</request>

The field is `{field}`.

<field_a>
{a}
</field_a>

<field_b>
{b}
</field_b>

Judge only this field: which value better fulfils what the request asks for it (correct and
faithful to the input first, then complete and following the instructions)? Wording differences
alone don't matter. Reply with exactly one word: A, B, or TIE (use TIE when they are equally good)."""


def field_judge_messages(request_messages: list[dict], field: str, a: str, b: str) -> list[dict]:
    req = "\n\n".join(f"[{m['role']}] {m['content']}" for m in request_messages)
    return [{"role": "user", "content": FIELD_JUDGE_PROMPT.format(request=req[:20000], field=field,
                                                                  a=str(a)[:6000], b=str(b)[:6000])}]


def judge_messages(request_messages: list[dict], a: str, b: str) -> list[dict]:
    req = "\n\n".join(f"[{m['role']}] {m['content']}" for m in request_messages)
    return [{"role": "user", "content": JUDGE_PROMPT.format(request=req[:20000], a=a[:12000], b=b[:12000])}]


def judge_choice(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"\b(TIE|A|B)\b", text.strip().upper())
    return m.group(1) if m else None
