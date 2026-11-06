"""Tier-2 audit: an LLM judge on a fixed sample of notes, via a Databricks Foundation Model endpoint.

The judge reads the dialogue (source of truth), the clinician's reference note (for omissions only),
and a generated note, and labels every claim in the generated note. It runs on a fixed sample, not on
every output: tier 1 (audit.py) covers every note for free, and the sample measures how far tier 1 can
be trusted. The default endpoint is an open-weight model served inside Databricks, so audit data never
leaves the platform.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ENDPOINT = "databricks-gpt-oss-120b"
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You audit AI-generated clinical note sections for faithfulness to the doctor-patient dialogue they summarize.

The DIALOGUE is the only source of truth. A REFERENCE note written by a clinician is provided so you can spot important omissions; do not penalize the generated note for wording that differs from the reference.

The generated note uses four fields: Symptoms, Diagnosis, History of Patient, Plan of Action. "N/A" means nothing to report.

Split the generated note into atomic claims (one fact each; skip "N/A") and label each:
- supported: stated or directly implied by the dialogue
- unsupported: not in the dialogue (a hallucination), including invented specifics such as doses, durations, body sites, or severity
- contradicted: conflicts with the dialogue
- misplaced: true, but in the wrong field (for example a medication listed as a diagnosis, or past history listed as a current symptom)

Severity is "major" if acting on the note could change clinical care (wrong drug, dose, allergy, diagnosis, laterality, or a symptom escalated, such as "decreased vision" becoming "blindness"); otherwise "minor".

Then list clinically important facts from the dialogue that belong in this section but are missing from the generated note.

Respond with one JSON object and nothing else:
{"claims": [{"field": "...", "claim": "...", "label": "supported|unsupported|contradicted|misplaced", "severity": "minor|major"}],
 "omissions": [{"fact": "...", "severity": "minor|major"}],
 "faithfulness_score": <integer 1-5, 5 = every claim supported and nothing important missing>,
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
    claims = [c for c in v.get("claims", []) if isinstance(c, dict) and c.get("label") in LABELS]
    omissions = [o for o in v.get("omissions", []) if isinstance(o, dict)]
    score = int(v["faithfulness_score"])
    if not 1 <= score <= 5:
        raise ValueError(f"faithfulness_score out of range: {score}")
    return {"claims": claims, "omissions": omissions, "faithfulness_score": score, "rationale": str(v.get("rationale", ""))}


def summarize_verdict(v: dict) -> dict:
    """Flat per-note metrics for the judge_results table."""
    claims = v["claims"]
    count = lambda label: sum(c["label"] == label for c in claims)  # noqa: E731
    major_errors = sum(c["label"] != "supported" and c.get("severity") == "major" for c in claims)
    return {
        "n_claims": len(claims),
        "n_supported": count("supported"),
        "n_unsupported": count("unsupported"),
        "n_contradicted": count("contradicted"),
        "n_misplaced": count("misplaced"),
        "n_major_errors": major_errors,
        "n_omissions": len(v["omissions"]),
        "n_major_omissions": sum(o.get("severity") == "major" for o in v["omissions"]),
        "faithfulness_score": v["faithfulness_score"],
        # A note with no claims (all N/A) is faithful only if it also omits nothing major.
        "faithful": major_errors == 0 and not any(o.get("severity") == "major" for o in v["omissions"]),
        "hallucinated": count("unsupported") + count("contradicted") > 0,
    }


class Judge:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, max_tokens: int = 4000, client=None):
        from databricks.sdk import WorkspaceClient

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.client = client or WorkspaceClient()

    def judge_one(self, section: str, dialogue: str, reference: str, note: str, attempts: int = 3) -> dict:
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
                r = self.client.serving_endpoints.query(name=self.endpoint, messages=messages, max_tokens=self.max_tokens)
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
