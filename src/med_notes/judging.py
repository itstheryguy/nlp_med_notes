"""Planning and bookkeeping for the LLM-judge audit: cache, multiple judges, agreement.

Pure pandas, so it's testable without Spark or an endpoint. The notebook supplies the tables and the calls.

- Every note is judged in canonical form (judge.canonical_note), keyed by (conversation, canonical note,
  judge endpoint, prompt version). A key already judged, in this run or any earlier one, is never re-judged:
  identical notes always get identical verdicts, and repeats cost nothing.
- Several judges can audit the same notes; agreement between them, and between a judge and itself on a
  deliberate re-judge (bypassing the cache), measures how far the judge itself can be trusted.
"""

from __future__ import annotations

import pandas as pd

from .judge import PROMPT_VERSION, canonical_note, verdict_key

RESULT_COLS = ["n_claims", "n_supported", "n_unsupported", "n_contradicted", "n_misplaced", "n_major_errors",
               "n_omissions", "n_major_omissions", "faithfulness_score", "attempts", "prompt_tokens",
               "completion_tokens", "faithful", "hallucinated", "verdict_json", "error"]


def build_items(sample: pd.DataFrame, model_notes: pd.DataFrame) -> pd.DataFrame:
    """One row per note to judge: the clinician references plus the run's model notes, with canonical text.

    sample: example_id, section_header, dialogue, reference. model_notes: experiment, family, seed, example_id,
    note, grounding_flag.
    """
    refs = sample.assign(experiment="reference", family="reference", seed=0, note=sample["reference"],
                         grounding_flag=None)
    models = model_notes.merge(sample, on="example_id", how="inner")
    items = pd.concat([refs, models], ignore_index=True)
    items["seed"] = items["seed"].astype("int64")
    items["canonical"] = items["note"].map(canonical_note)
    return items


def plan_calls(items: pd.DataFrame, endpoints: list[str], cached_keys: set[str],
               prompt_version: str = PROMPT_VERSION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (keyed, calls).

    keyed: items x endpoints with a verdict_key column. calls: one row per key that still needs judging
    (unique, and not in cached_keys).
    """
    keyed = pd.concat([items.assign(endpoint=e) for e in endpoints], ignore_index=True)
    keyed["verdict_key"] = [verdict_key(x, c, e, prompt_version)
                            for x, c, e in zip(keyed["example_id"], keyed["canonical"], keyed["endpoint"])]
    calls = keyed[~keyed["verdict_key"].isin(cached_keys)].drop_duplicates("verdict_key").reset_index(drop=True)
    return keyed, calls


def assemble(keyed: pd.DataFrame, verdicts: pd.DataFrame, new_keys: set[str]) -> pd.DataFrame:
    """One row per (note, judge) with its verdict. verdicts: one row per verdict_key with RESULT_COLS.

    `cached` is True when the verdict came from an earlier job run (not judged in this one).
    """
    out = keyed.merge(verdicts[["verdict_key", *RESULT_COLS]], on="verdict_key", how="left")
    out["cached"] = ~out["verdict_key"].isin(new_keys)
    return out


def cohen_kappa(a: pd.Series, b: pd.Series) -> float | None:
    """Agreement beyond chance for two boolean labelings of the same notes."""
    a, b = a.astype(bool), b.astype(bool)
    n = len(a)
    if n == 0:
        return None
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """Faithful-label agreement between two judgments of the same notes."""
    return {"n": int(len(a)), "agreement": float((a.astype(bool) == b.astype(bool)).mean()) if len(a) else None,
            "kappa": cohen_kappa(a, b)}
