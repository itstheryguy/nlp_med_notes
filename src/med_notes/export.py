"""Build the static snapshot the Hugging Face Space reads (no Databricks access from the Space).

Everything the app shows is precomputed here, so the Space needs only pandas and the parquet files:
per-field parses, which N/A fields a note filled or missed, judge claims, and model-level summaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .audit import FIELDS, is_na, parse_template
from .judging import agreement

NOTE_COLS = ["experiment", "family", "seed", "example_id", "prediction", "rougeL", "bertscore_f1", "template_ok",
             "na_agreement", "unsupported_terms", "unsupported_numbers", "grounding_flag", "empty_prediction"]
JUDGE_COLS = ["faithful", "hallucinated", "n_claims", "n_unsupported", "n_contradicted", "n_misplaced",
              "n_major_errors", "n_major_omissions", "faithfulness_score", "verdict_json"]
J2_COLS = ["faithful", "hallucinated", "n_major_errors", "n_major_omissions", "faithfulness_score", "verdict_json"]


def field_diff(note: str, reference: str) -> dict:
    """Per-field parse of a note, and where its N/A choices disagree with the reference."""
    p, r = parse_template(note or ""), parse_template(reference or "")
    out = {"fields_json": json.dumps(p) if p else None, "filled_na_fields": [], "missed_fields": []}
    if p and r:
        out["filled_na_fields"] = [f for f in FIELDS if is_na(r[f]) and not is_na(p[f])]  # wrote content where the clinician wrote N/A
        out["missed_fields"] = [f for f in FIELDS if not is_na(r[f]) and is_na(p[f])]     # N/A where the clinician wrote content
    return out


def build_snapshot(silver: pd.DataFrame, audit: pd.DataFrame, predictions: pd.DataFrame, judge: pd.DataFrame,
                   reference_checks: pd.DataFrame, runs: pd.DataFrame, training_jobs: pd.DataFrame,
                   meta: dict, judge2: pd.DataFrame | None = None, consistency: pd.DataFrame | None = None,
                   judge_names: dict | None = None) -> dict[str, pd.DataFrame | dict]:
    """Inputs are the Delta tables (already filtered to the job runs being published). Test split only.

    judge: the primary judge's rows; judge2: the second judge's (optional, columns prefixed j2_ in the output);
    consistency: judge_consistency rows (self-agreement on deliberate re-judges)."""
    training_jobs = training_jobs if training_jobs is not None else pd.DataFrame()
    test = silver[silver["split"] == "test"][["example_id", "section_header", "dialogue", "section_text"]]
    test = test.rename(columns={"section_text": "reference"})
    test = test.merge(audit, on="example_id", how="left")
    test["in_audit"] = test["audit_rank"].notna()

    ref_judge = judge[judge["experiment"] == "reference"].drop_duplicates("example_id", keep="last")
    examples = (test.merge(reference_checks[["example_id", "template_ok", "unsupported_terms", "unsupported_numbers", "grounding_flag"]]
                           .rename(columns=lambda c: c if c == "example_id" else f"ref_{c}"), on="example_id", how="left")
                    .merge(ref_judge[["example_id", *JUDGE_COLS]].rename(columns=lambda c: c if c == "example_id" else f"ref_{c}"),
                           on="example_id", how="left"))
    examples["ref_fields_json"] = [json.dumps(p) if (p := parse_template(t)) else None for t in examples["reference"]]
    if judge2 is not None and len(judge2):
        r2 = judge2[judge2["experiment"] == "reference"].drop_duplicates("example_id", keep="last")
        examples = examples.merge(r2[["example_id", *J2_COLS]].rename(
            columns=lambda c: c if c == "example_id" else f"ref_j2_{c}"), on="example_id", how="left")

    notes = predictions[predictions["split"] == "test"][NOTE_COLS].merge(
        judge[judge["experiment"] != "reference"][["experiment", "example_id", *JUDGE_COLS]],
        on=["experiment", "example_id"], how="left")
    if judge2 is not None and len(judge2):
        notes = notes.merge(judge2[judge2["experiment"] != "reference"][["experiment", "example_id", *J2_COLS]].rename(
            columns=lambda c: c if c in ("experiment", "example_id") else f"j2_{c}"), on=["experiment", "example_id"], how="left")
    refs = examples.set_index("example_id")["reference"]
    diffs = pd.DataFrame([field_diff(n, refs.get(e, "")) for n, e in zip(notes["prediction"], notes["example_id"])])
    notes = pd.concat([notes.reset_index(drop=True), diffs], axis=1)
    notes["unsupported_terms"] = notes["unsupported_terms"].apply(lambda v: list(v) if v is not None else [])

    # One row per run: overlap metrics plus tier-1 and judge rates on the test split.
    per_run = notes.groupby(["experiment", "family", "seed"]).agg(
        rougeL=("rougeL", "mean"), bertscore_f1=("bertscore_f1", "mean"), template_rate=("template_ok", "mean"),
        na_agreement=("na_agreement", "mean"), empty_rate=("empty_prediction", "mean"),
        grounding_flag_rate=("grounding_flag", "mean"),
        filled_na_rate=("filled_na_fields", lambda s: np.mean([len(x) > 0 for x in s])),
    ).reset_index()
    judged = notes[notes["faithful"].notna()]
    jr = judged.groupby("experiment").agg(
        judged=("faithful", "size"), faithful_rate=("faithful", "mean"), hallucination_rate=("hallucinated", "mean"),
        major_errors_per_note=("n_major_errors", "mean"), major_omissions_per_note=("n_major_omissions", "mean"),
        misplaced_per_note=("n_misplaced", "mean"), judge_score=("faithfulness_score", "mean")).reset_index()
    per_run = per_run.merge(jr, on="experiment", how="left").merge(
        runs[["experiment", "minutes", "best_epoch", "epochs_run", "num_params_m", "model_name"]], on="experiment", how="left")

    metrics = [c for c in per_run.columns if c not in ("experiment", "family", "seed", "model_name", "judged")]
    families = []
    for fam, g in per_run.groupby("family"):
        row = {"family": fam, "model_name": g["model_name"].iloc[0], "seeds": int(g["seed"].nunique())}
        for m in metrics:
            vals = g[m].astype(float)
            row[m] = float(vals.mean())
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        families.append(row)

    ref_j = examples[examples["ref_faithful"].notna()]
    calib = []
    for fam, g in judged.groupby("family"):
        flag, hal = g["grounding_flag"].astype(bool), g["hallucinated"].astype(bool)
        calib.append({"family": fam, "judged": int(len(g)), "flag_rate": float(flag.mean()),
                      "judge_hallucination_rate": float(hal.mean()),
                      "precision": float((flag & hal).sum() / flag.sum()) if flag.sum() else None,
                      "recall": float((flag & hal).sum() / hal.sum()) if hal.sum() else None})

    judges = None
    if judge2 is not None and len(judge2) and "j2_faithful" in notes:
        both = notes[notes["faithful"].notna() & notes["j2_faithful"].notna()]
        rboth = examples[examples["ref_faithful"].notna() & examples.get("ref_j2_faithful", pd.Series(dtype=object)).notna()]
        by_family = [{"family": f, **agreement(g["faithful"], g["j2_faithful"]),
                      "primary_faithful": float(g["faithful"].astype(bool).mean()),
                      "secondary_faithful": float(g["j2_faithful"].astype(bool).mean())} for f, g in both.groupby("family")]
        judges = {
            **(judge_names or {}),
            "cross_overall": agreement(both["faithful"], both["j2_faithful"]),
            "cross_by_family": by_family,
            "cross_reference": {**agreement(rboth["ref_faithful"], rboth["ref_j2_faithful"]),
                                "primary_faithful": float(rboth["ref_faithful"].astype(bool).mean()) if len(rboth) else None,
                                "secondary_faithful": float(rboth["ref_j2_faithful"].astype(bool).mean()) if len(rboth) else None},
            "self_consistency": [
                {"endpoint": e, "n": int(len(g)),
                 "faithful_agreement": float((g["faithful"].astype(bool) == g["faithful_repeat"].astype(bool)).mean()),
                 "score_agreement": float((g["faithfulness_score"] == g["score_repeat"]).mean())}
                for e, g in (consistency.groupby("endpoint") if consistency is not None and len(consistency) else [])],
        }

    summary = {
        **meta,
        "judges": judges,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_test_examples": int(len(examples)), "n_audit_examples": int(examples["in_audit"].sum()),
        "families": families,
        "reference": {
            "template_rate": float(examples["ref_template_ok"].mean()),
            "grounding_flag_rate": float(examples["ref_grounding_flag"].mean()),
            "faithful_rate": float(ref_j["ref_faithful"].mean()) if len(ref_j) else None,
            "hallucination_rate": float(ref_j["ref_hallucinated"].mean()) if len(ref_j) else None,
            "judge_score": float(ref_j["ref_faithfulness_score"].mean()) if len(ref_j) else None,
        },
        "calibration": calib,
        "training_cost": {
            "billable_minutes": float(training_jobs["billable_seconds"].sum() / 60) if len(training_jobs) else None,
            "estimated_usd": float(training_jobs["estimated_usd"].sum()) if len(training_jobs) else None,
            "instance_type": training_jobs["instance_type"].iloc[0] if len(training_jobs) else None,
        },
    }
    return {"examples": examples, "notes": notes, "runs": per_run, "summary": summary}
