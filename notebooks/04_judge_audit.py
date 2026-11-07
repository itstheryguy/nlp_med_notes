# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · LLM judge audit (CPU + Foundation Model API)
# MAGIC
# MAGIC **Tier 2 of the audit.** Tier 1 (`03`) checks every note for free; this task sends a *fixed sample* of test
# MAGIC notes to LLM judges (`src/med_notes/judge.py`) that label every claim as supported / unsupported /
# MAGIC contradicted / misplaced, with severity, and list important omissions. The 1–5 score and "faithful" are
# MAGIC computed from those labels, not asked of the judge.
# MAGIC
# MAGIC - **Same examples for every run** (`audit_sample`), so judge metrics compare models and seeds fairly.
# MAGIC - **References are judged too**: how often a judge objects to a clinician's note is the noise floor.
# MAGIC - **Two judges** (`judge_endpoints`, comma-separated; the first is primary), both open-weight models served
# MAGIC   inside Databricks. Agreement between them measures the judge, not the models.
# MAGIC - **Verdict cache:** notes are judged in canonical form and keyed by conversation + note + judge + prompt
# MAGIC   version. A key judged before (in any run) is reused, so identical notes always get identical verdicts.
# MAGIC - **Self-consistency:** `consistency_sample` keys are judged a second time on purpose (bypassing the cache)
# MAGIC   and written to `judge_consistency`, to measure how often a judge agrees with itself.
# MAGIC - **Cost control:** `judge_sample_size` caps examples per run and `judge_max_calls` caps the whole task.

# COMMAND ----------

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
dbutils.widgets.text("judge_endpoints", "databricks-gpt-oss-120b,databricks-qwen3-next-80b-a3b-instruct",
                     "Judge endpoints (first = primary)")
dbutils.widgets.text("judge_sample_size", "40", "Examples judged per run")
dbutils.widgets.text("consistency_sample", "40", "Keys re-judged per judge for self-consistency (0 = skip)")
dbutils.widgets.text("judge_max_calls", "2000", "Max judge calls this task")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
job_run_id = dbutils.widgets.get("job_run_id")
endpoints = [e.strip() for e in dbutils.widgets.get("judge_endpoints").split(",") if e.strip()]
sample_size = int(dbutils.widgets.get("judge_sample_size"))
consistency_n = int(dbutils.widgets.get("consistency_sample"))
max_calls = int(dbutils.widgets.get("judge_max_calls"))
spark.sql(f"USE {catalog}.{schema}")

if not spark.catalog.tableExists("predictions"):
    dbutils.notebook.exit("No predictions table yet.")

# COMMAND ----------

from med_notes.judge import PROMPT_VERSION, Judge, results_frame
from med_notes.judging import RESULT_COLS, assemble, build_items, plan_calls

def pdf(sql: str) -> pd.DataFrame:
    df = spark.sql(sql).toPandas()
    df.attrs = {}
    return df

sample = pdf(f"""
    SELECT a.example_id, s.section_header, s.dialogue, s.section_text AS reference
    FROM audit_sample a JOIN silver_dialog_notes s USING (example_id) WHERE a.audit_rank < {sample_size}""")
model_notes = pdf(f"""
    SELECT experiment, family, seed, example_id, prediction AS note, grounding_flag
    FROM predictions WHERE job_run_id = '{job_run_id}' AND split = 'test'""")
items = build_items(sample, model_notes)

cached = pd.DataFrame(columns=["verdict_key", *RESULT_COLS])
if spark.catalog.tableExists("judge_results") and "verdict_key" in spark.table("judge_results").columns:
    cached = pdf(f"""
        SELECT * EXCEPT (rn) FROM (
          SELECT verdict_key, {', '.join(RESULT_COLS)},
                 row_number() OVER (PARTITION BY verdict_key ORDER BY judged_at DESC) AS rn
          FROM judge_results WHERE error IS NULL AND verdict_key IS NOT NULL AND prompt_version = '{PROMPT_VERSION}'
        ) WHERE rn = 1""")

keyed, calls = plan_calls(items, endpoints, set(cached["verdict_key"]))
print(f"{len(items)} notes x {len(endpoints)} judges = {len(keyed)} verdicts; "
      f"{len(keyed) - len(calls)} reused (cache or identical notes), {len(calls)} new calls")
if len(calls) > max_calls:
    raise RuntimeError(f"{len(calls)} calls planned > judge_max_calls={max_calls}; raise the cap to proceed")

# COMMAND ----------

def judge_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Call each row's endpoint; returns one row per input with RESULT_COLS."""
    parts = []
    for endpoint, group in rows.groupby("endpoint", sort=False):
        res = Judge(endpoint=endpoint).judge_many(
            [{"section": r.section_header, "dialogue": r.dialogue, "reference": r.reference, "note": r.canonical}
             for r in group.itertuples()], workers=6)
        parts.append(pd.concat([group[["verdict_key", "endpoint"]].reset_index(drop=True), results_frame(res)], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["verdict_key", "endpoint", *RESULT_COLS])

new = judge_rows(calls)
failed = new["error"].notna().sum()
print(f"Judged {len(new) - failed}/{len(new)} new · tokens in {new['prompt_tokens'].sum():,.0f} "
      f"out {new['completion_tokens'].sum():,.0f}")
if failed:
    display(new[new["error"].notna()][["endpoint", "error"]].head(10))

verdicts = pd.concat([cached, new[["verdict_key", *RESULT_COLS]]], ignore_index=True).drop_duplicates("verdict_key", keep="last")
out = assemble(keyed, verdicts, set(new["verdict_key"]))
out = out[["experiment", "family", "seed", "example_id", "grounding_flag", "endpoint", "verdict_key", "canonical",
           "cached", *RESULT_COLS]].rename(columns={"canonical": "judged_note"})
out["grounding_flag"] = [None if pd.isna(v) else bool(v) for v in out["grounding_flag"]]
for c in ("faithful", "hallucinated"):
    out[c] = [None if pd.isna(v) else bool(v) for v in out[c]]
out.insert(0, "job_run_id", job_run_id)
out["prompt_version"], out["judged_at"] = PROMPT_VERSION, datetime.now(timezone.utc)

if spark.catalog.tableExists("judge_results"):
    spark.sql(f"DELETE FROM judge_results WHERE job_run_id = '{job_run_id}' AND prompt_version = '{PROMPT_VERSION}'")
spark.createDataFrame(out).write.mode("append").option("mergeSchema", "true").saveAsTable("judge_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Self-consistency: re-judge a sample on purpose (no cache)

# COMMAND ----------

if consistency_n > 0:
    first = out[out["error"].isna()].drop_duplicates("verdict_key")
    pick = pd.concat([g.sample(min(consistency_n, len(g)), random_state=7) for _, g in first.groupby("endpoint")])
    pick = pick.merge(keyed[["verdict_key", "section_header", "dialogue", "reference", "canonical"]].drop_duplicates("verdict_key"),
                      on="verdict_key")
    again = judge_rows(pick)
    cons = pick[["verdict_key", "endpoint", "experiment", "example_id", "faithful", "faithfulness_score"]].merge(
        again[["verdict_key", "faithful", "faithfulness_score", "error"]].rename(
            columns={"faithful": "faithful_repeat", "faithfulness_score": "score_repeat"}), on="verdict_key")
    cons = cons[cons["error"].isna()].drop(columns="error")
    cons["faithful"] = cons["faithful"].astype(bool)
    cons["faithful_repeat"] = cons["faithful_repeat"].astype(bool)
    cons.insert(0, "job_run_id", job_run_id)
    cons["prompt_version"], cons["judged_at"] = PROMPT_VERSION, datetime.now(timezone.utc)
    if spark.catalog.tableExists("judge_consistency"):
        spark.sql(f"DELETE FROM judge_consistency WHERE job_run_id = '{job_run_id}' AND prompt_version = '{PROMPT_VERSION}'")
    spark.createDataFrame(cons).write.mode("append").option("mergeSchema", "true").saveAsTable("judge_consistency")
    display(cons.groupby("endpoint").apply(lambda g: pd.Series({
        "n": len(g), "faithful_agreement": (g["faithful"] == g["faithful_repeat"]).mean(),
        "score_agreement": (g["faithfulness_score"] == g["score_repeat"]).mean()})).reset_index())

# COMMAND ----------

display(spark.sql(f"""
    SELECT endpoint, family, count(*) AS notes, round(avg(CAST(faithful AS INT)), 3) AS faithful_rate,
           round(avg(CAST(hallucinated AS INT)), 3) AS hallucination_rate, round(avg(faithfulness_score), 2) AS score
    FROM judge_results
    WHERE error IS NULL AND job_run_id = '{job_run_id}' AND prompt_version = '{PROMPT_VERSION}'
    GROUP BY endpoint, family ORDER BY endpoint, faithful_rate DESC
"""))
