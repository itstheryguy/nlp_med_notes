"""Tier-2 audit: an LLM judge on a fixed sample of notes, via a Databricks Foundation Model endpoint.

The judge reads the dialogue (source of truth), the clinician's reference note (for omissions only),
and a generated note, and labels every claim in the generated note. It runs on a fixed sample, not on
every output: tier 1 (audit.py) covers every note for free, and the sample measures how far tier 1 can
be trusted. The judges are open-weight models served inside Databricks, so audit data never leaves the
platform.

Prompt v2 (after #346, where identical all-N/A notes got scores of 2 and 3 and inconsistent labels):
- the judge only labels claims and omissions; the 1-5 score and "faithful" are computed from the labels by a
  fixed rule (score_from_labels), so identical labels always give the identical score
- explicit rules: "N/A" is never a claim; omitted pertinent negatives have a fixed severity
- temperature 0, and notes are judged in a canonical form (fields on separate lines, whitespace normalized), so
  T5/PEGASUS one-line notes and BioBART multi-line notes look the same to the judge
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ENDPOINT = "databricks-gpt-oss-120b"
SECOND_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You audit AI-generated clinical note sections for faithfulness to the doctor-patient dialogue they summarize.

The DIALOGUE is the only source of truth. A REFERENCE note written by a clinician is provided so you can spot important omissions; do not penalize the generated note for wording that differs from the reference.

The generated note has four fields: Symptoms, Diagnosis, History of Patient, Plan of Action. "N/A" means the field has nothing to report.

1. Split the generated note into atomic claims (one fact each) and label each:
- supported: stated or directly implied by the dialogue
- unsupported: not in the dialogue (a hallucination), including invented specifics such as doses, durations, body sites, or severity
- contradicted: conflicts with the dialogue
- misplaced: true, but in the wrong field (for example a medication listed as a diagnosis, or past history listed as a current symptom)
"N/A" is never a claim: do not list it, whatever the field.

2. List omissions: clinically important facts from the dialogue that belong in this section but are missing from the generated note.
Pertinent negatives count: if the patient explicitly denies conditions, symptoms, or exposures that this section records (for example "no history of diabetes" in a past medical history section), list them as ONE omission when the note leaves them out.

Severity, for claims and omissions alike:
- major: acting on the note could change clinical care (wrong or missing drug, dose, allergy, diagnosis, laterality, a symptom escalated such as "decreased vision" becoming "blindness"), or the note writes N/A for a field whose content the dialogue explicitly covers (including pertinent negatives)
- minor: anything else

Respond with one JSON object and nothing else:
{"claims": [{"field": "...", "claim": "...", "label": "supported|unsupported|contradicted|misplaced", "severity": "minor|major"}],
 "omissions": [{"fact": "...", "severity": "minor|major"}],
 "rationale": "<one sentence>"}"""

USER_TEMPLATE = """SECTION: {section}

<dialogue>
{dialogue}
</dialogue>

<reference_note>
{reference}
</reference_note>

<generated_note>
{note}
</generated_note>"""

LABELS = {"supported", "unsupported", "contradicted", "misplaced"}


def canonical_note(text: str) -> str:
    """The note as the judge sees it: one field per line, whitespace collapsed. A note outside the template is
    whitespace-collapsed as-is. T5/PEGASUS write fields on one line; BioBART uses newlines; same content, same text."""
    from .audit import FIELDS, parse_template

    fields = parse_template(text or "")
    if fields:
        return "\n".join(f"{f}: {' '.join(fields[f].split())}" for f in FIELDS)
    return " ".join((text or "").split())


def verdict_key(example_id: int, canonical: str, endpoint: str, prompt_version: str = PROMPT_VERSION) -> str:
    """Cache key: same conversation + same note content + same judge + same prompt -> same verdict."""
    import hashlib

    return hashlib.sha256(f"{example_id}\x1f{canonical}\x1f{endpoint}\x1f{prompt_version}".encode()).hexdigest()


def _text_from_content(content) -> str:
    """Endpoints return a string, or (reasoning models) a list of parts; keep only the answer text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "output_text"))
    return ""


def parse_verdict(text: str) -> dict:
    """Extract and validate the judge's JSON. Raises ValueError if it's unusable."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in judge response")
    v = json.loads(m.group(0))
    if not isinstance(v.get("claims"), list) or not isinstance(v.get("omissions"), list):
        raise ValueError("judge JSON needs 'claims' and 'omissions' lists")
    # Enforce the "N/A is never a claim" rule even when a judge slips.
    claims = [c for c in v["claims"] if isinstance(c, dict) and c.get("label") in LABELS
              and not _is_na_claim(c.get("claim", ""))]
    omissions = [o for o in v["omissions"] if isinstance(o, dict) and o.get("fact")]
    return {"claims": claims, "omissions": omissions, "rationale": str(v.get("rationale", ""))}


def _is_na_claim(text: str) -> bool:
    t = re.sub(r"[^a-z/ ]", " ", str(text).lower())
    return t.strip() in ("n/a", "na", "none") or bool(re.fullmatch(r"\s*(no\b.*\bto report\b.*)?\s*\(?n/a\)?\s*", t))


def score_from_labels(n_major_issues: int, n_minor_issues: int) -> int:
    """5 = no issues, 4 = only minor issues, 3 = 1 major issue, 2 = 2 major issues, 1 = 3+ major issues."""
    if n_major_issues == 0:
        return 4 if n_minor_issues else 5
    return max(1, 4 - n_major_issues)


def summarize_verdict(v: dict) -> dict:
    """Flat per-note metrics for the judge_results table."""
    claims = v["claims"]
    count = lambda label: sum(c["label"] == label for c in claims)  # noqa: E731
    major_errors = sum(c["label"] != "supported" and c.get("severity") == "major" for c in claims)
    minor_errors = sum(c["label"] != "supported" and c.get("severity") != "major" for c in claims)
    major_omissions = sum(o.get("severity") == "major" for o in v["omissions"])
    minor_omissions = len(v["omissions"]) - major_omissions
    return {
        "n_claims": len(claims),
        "n_supported": count("supported"),
        "n_unsupported": count("unsupported"),
        "n_contradicted": count("contradicted"),
        "n_misplaced": count("misplaced"),
        "n_major_errors": major_errors,
        "n_omissions": len(v["omissions"]),
        "n_major_omissions": major_omissions,
        # Both computed from the labels, never asked of the judge, so they can't disagree with each other.
        "faithfulness_score": score_from_labels(major_errors + major_omissions, minor_errors + minor_omissions),
        "faithful": major_errors == 0 and major_omissions == 0,
        "hallucinated": count("unsupported") + count("contradicted") > 0,
    }


class Judge:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, max_tokens: int = 4000, client=None):
        from databricks.sdk import WorkspaceClient

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.client = client or WorkspaceClient()

    def judge_one(self, section: str, dialogue: str, reference: str, note: str, attempts: int = 3) -> dict:
        """note should already be canonical (canonical_note) so identical content looks identical."""
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        messages = [
            ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=ChatMessageRole.USER, content=USER_TEMPLATE.format(
                section=section, dialogue=dialogue, reference=reference, note=note)),
        ]
        prompt_tokens = completion_tokens = 0
        error = None
        for attempt in range(attempts):
            try:
                r = self.client.serving_endpoints.query(name=self.endpoint, messages=messages, max_tokens=self.max_tokens,
                                                        temperature=0.0)
                if r.usage:
                    prompt_tokens += r.usage.prompt_tokens or 0
                    completion_tokens += r.usage.completion_tokens or 0
                verdict = parse_verdict(_text_from_content(r.choices[0].message.content))
                return {**summarize_verdict(verdict), "verdict_json": json.dumps(verdict), "error": None,
                        "attempts": attempt + 1, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
            except Exception as e:  # parse failures and transient API errors both retry
                error = f"{type(e).__name__}: {e}"[:500]
                time.sleep(2 ** attempt)
        return {"verdict_json": None, "error": error, "attempts": attempts,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    def judge_many(self, items: list[dict], workers: int = 4) -> list[dict]:
        """items: dicts with section, dialogue, reference, note. Results come back in input order."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda it: self.judge_one(it["section"], it["dialogue"], it["reference"], it["note"]), items))


_COUNT_COLS = ["n_claims", "n_supported", "n_unsupported", "n_contradicted", "n_misplaced", "n_major_errors",
               "n_omissions", "n_major_omissions", "faithfulness_score", "attempts", "prompt_tokens", "completion_tokens"]
_BOOL_COLS = ["faithful", "hallucinated"]


def results_frame(results: list[dict]) -> "pd.DataFrame":
    """Fixed columns and dtypes whether a call succeeded or failed, so Spark never has to guess types."""
    import pandas as pd

    df = pd.DataFrame(results)
    out = pd.DataFrame(index=df.index)
    for c in _COUNT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64") if c in df else float("nan")
    for c in _BOOL_COLS:
        out[c] = [None if pd.isna(v) else bool(v) for v in df[c]] if c in df else None
    for c in ("verdict_json", "error"):
        out[c] = [None if pd.isna(v) else str(v) for v in df[c]] if c in df else None
    return out
