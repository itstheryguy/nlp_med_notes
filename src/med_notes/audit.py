"""Tier-1 audit: cheap, deterministic checks that run on every generated note.

- Template compliance: the note has exactly the four fields, in order, and nothing else.
- N/A agreement: per field, does the note say N/A where the reference does (and fill it where it doesn't)?
- Grounding: content terms and numbers in the note that never appear in the dialogue. A rough
  hallucination signal: synonyms ("high blood pressure" -> "hypertension") are false positives, which
  is why tier 2 (the LLM judge) audits a sample and measures how often this flag agrees with it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FIELDS = ["Symptoms", "Diagnosis", "History of Patient", "Plan of Action"]
_FIELD_RE = re.compile(r"^\s*(Symptoms|Diagnosis|History of Patient|Plan of Action)\s*:\s*(.*)$", re.IGNORECASE)
_NA_VALUES = {"n/a", "na", "none", "not applicable", "not mentioned", ""}

# Words that carry no clinical claim: template labels, discourse verbs, generic nouns.
_IGNORE = set("""
symptoms symptom diagnosis diagnoses history patient patients plan action actions reports reported report
states stated denies denied notes noted mentions mentioned says said describes described current currently
also with without that this have has having been were from they their there which other about some more
none past prior previous previously recent recently since well when what will would should could does doing
over under into onto after before while very much many most just only still being because ongoing
complaint complaints chief evaluation evaluated significant approximately further including experiencing
experienced experiences status known discussed during possible associated management assessment continue
continued received placed then year-old years-old old male female onset unknown noncontributory underwent
treated treatment presents presented presenting follow follow-up regarding related due consistent
""".split())

_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                      "eighty": "80", "ninety": "90", "hundred": "100", "once": "1", "twice": "2", "single": "1",
                      "couple": "2", "half": "0.5"})

GROUNDING_FLAG_MIN = 3  # flag a note with this many unsupported words, or any unsupported number


def parse_template(text: str) -> dict | None:
    """Field -> value if the note follows the 4-field template exactly; else None."""
    found: dict[str, str] = {}
    order: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _FIELD_RE.match(line)
        if not m:
            return None  # content outside the template
        name = next(f for f in FIELDS if f.lower() == m.group(1).lower())
        if name in found:
            return None
        found[name] = m.group(2).strip()
        order.append(name)
    return found if order == FIELDS else None


def is_na(value: str) -> bool:
    return value.strip().rstrip(".").strip().lower() in _NA_VALUES


_stemmer = None


def _stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        from nltk.stem.porter import PorterStemmer

        _stemmer = PorterStemmer()
    return _stemmer.stem(word)


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'\-]+|\d+(?:\.\d+)?", text.lower())


def unsupported_terms(note: str, dialogue: str) -> tuple[list[str], int]:
    """(terms in the note not found in the dialogue, number of content terms checked)."""
    dialogue_terms = _terms(dialogue)
    dia_stems = {_stem(t) for t in dialogue_terms if not t[0].isdigit()}
    dia_prefixes = {s[:5] for s in dia_stems if len(s) >= 5}
    dia_numbers = {t for t in dialogue_terms if t[0].isdigit()} | {_NUMBER_WORDS[t] for t in dialogue_terms if t in _NUMBER_WORDS}
    checked, missing = 0, []
    for t in dict.fromkeys(_terms(note)):  # unique, in order
        if t[0].isdigit():
            checked += 1
            if t not in dia_numbers:
                missing.append(t)
            continue
        if len(t) < 4 or t in _IGNORE or t.replace("-", "") in ("na",):
            continue
        checked += 1
        s = _stem(t)
        if s not in dia_stems and not (len(s) >= 5 and s[:5] in dia_prefixes):
            missing.append(t)
    return missing, checked


def tier1(predictions: list[str], references: list[str], dialogues: list[str]) -> pd.DataFrame:
    rows = []
    for pred, ref, dia in zip(predictions, references, dialogues):
        p, r = parse_template(pred), parse_template(ref)
        na_agree = None
        if p is not None and r is not None:
            na_agree = float(np.mean([is_na(p[f]) == is_na(r[f]) for f in FIELDS]))
        missing, checked = unsupported_terms(pred, dia)
        numbers = [t for t in missing if t[0].isdigit()]
        rows.append({
            "template_ok": p is not None,
            "na_agreement": na_agree,
            "unsupported_terms": missing,
            "unsupported_count": len(missing),
            "unsupported_numbers": len(numbers),
            "unsupported_rate": len(missing) / checked if checked else 0.0,
            # Numbers (doses, durations, ages) are the most precise signal: a number absent from the dialogue
            # is almost always invented. Words need several misses because clinical paraphrase is common.
            "grounding_flag": bool(numbers) or len(missing) - len(numbers) >= GROUNDING_FLAG_MIN,
        })
    return pd.DataFrame(rows)


def select_audit_sample(silver: pd.DataFrame, n: int, seed: int = 7, split: str = "test") -> list[int]:
    """Fixed example_ids for the LLM audit, proportional across section headers.

    Every run is judged on the same examples, so judge metrics are comparable between models and seeds.
    """
    pool = silver[silver["split"] == split]
    rng = np.random.default_rng(seed)
    pool = pool.assign(_r=rng.random(len(pool))).sort_values(["section_header", "_r"]).reset_index(drop=True)
    if n >= len(pool):
        return sorted(pool["example_id"].tolist())
    # Systematic sample over header-sorted rows = proportional allocation per header.
    idx = np.floor((np.arange(n) + rng.random()) * len(pool) / n).astype(int)
    return sorted(pool.loc[idx, "example_id"].astype(int).tolist())
