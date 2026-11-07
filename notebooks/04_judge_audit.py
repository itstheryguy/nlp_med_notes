# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · LLM judge audit (CPU + Foundation Model API)
# MAGIC
# MAGIC **Tier 2 of the audit.** Tier 1 (`03`) checks every note for free; this task sends a *fixed sample* of test
# MAGIC notes to an LLM judge (`src/med_notes/judge.py`) that labels every claim as supported / unsupported /
# MAGIC contradicted / misplaced, with severity, and lists important omissions.
# MAGIC
# MAGIC - **Same examples for every run** (`audit_sample`), so judge metrics compare models and seeds fairly.
# MAGIC - **References are judged too**, once per endpoint and prompt version. That shows how often the judge
# MAGIC   objects to a clinician's note: the noise floor every model is measured against.
# MAGIC - **Cost control:** `judge_sample_size` caps examples per run and `judge_max_calls` caps the whole task.
# MAGIC   Token counts are stored per call.
# MAGIC - **Default endpoint** is `databricks-gpt-oss-120b`, an open-weight model served inside Databricks, so audit
# MAGIC   data stays on the platform. Any chat endpoint works via `judge_endpoint`.

# COMMAND ----------

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
dbutils.widgets.text("judge_endpoint", "databricks-gpt-oss-120b", "Judge endpoint")
dbutils.widgets.text("judge_sample_size", "40", "Examples judged per run")
dbutils.widgets.text("judge_max_calls", "600", "Max judge calls this task")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
job_run_id = dbutils.widgets.get("job_run_id")
endpoint = dbutils.widgets.get("judge_endpoint")
sample_size = int(dbutils.widgets.get("judge_sample_size"))
max_calls = int(dbutils.widgets.get("judge_max_calls"))
spark.sql(f"USE {catalog}.{schema}")

if not spark.catalog.tableExists("predictions"):
    dbutils.notebook.exit("No predictions table yet.")

# COMMAND ----------

from med_notes.judge import PROMPT_VERSION, Judge, results_frame

sample = spark.sql(f"""
    SELECT a.example_id, a.audit_rank, s.section_header, s.dialogue, s.section_text AS reference
    FROM audit_sample a JOIN silver_dialog_notes s USING (example_id)
    WHERE a.audit_rank < {sample_size}
""").toPandas()

model_items = spark.sql(f"""
    SELECT p.experiment, p.family, p.seed, p.example_id, p.prediction AS note, p.grounding_flag
    FROM predictions p JOIN audit_sample a USING (example_id)
    WHERE p.job_run_id = '{job_run_id}' AND p.split = 'test' AND a.audit_rank < {sample_size}
""").toPandas().merge(sample, on="example_id")

already = set()
if spark.catalog.tableExists("judge_results"):
    already = {r.example_id for r in spark.sql(f"""
        SELECT example_id FROM judge_results
        WHERE experiment = 'reference' AND endpoint = '{endpoint}' AND prompt_version = '{PROMPT_VERSION}'
    """).collect()}
# Filter first, then take `note` from the filtered frame: assigning a Series from the unfiltered frame onto
# an empty one makes pandas expand it to that Series' rows (phantom references with NaN columns).
ref_items = sample[~sample["example_id"].isin(already)]
ref_items = ref_items.assign(experiment="reference", family="reference", seed=0, note=ref_items["reference"],
                             grounding_flag=None)

items = pd.concat([ref_items, model_items], ignore_index=True)
if len(items) > max_calls:
    print(f"WARNING: {len(items)} calls planned, capped at judge_max_calls={max_calls}")
    items = items.head(max_calls)
print(f"{len(model_items)} model notes + {len(ref_items)} references → {len(items)} judge calls on {endpoint}")

# COMMAND ----------

judge = Judge(endpoint=endpoint)
results = judge.judge_many(
    [{"section": r.section_header, "dialogue": r.dialogue, "reference": r.reference, "note": r.note}
     for r in items.itertuples()],
    workers=4,
)
out = pd.concat([items[["experiment", "family", "seed", "example_id", "grounding_flag"]].reset_index(drop=True),
                 results_frame(results)], axis=1)
out["grounding_flag"] = [None if pd.isna(v) else bool(v) for v in out["grounding_flag"]]
out["seed"] = out["seed"].astype("int64")
out.insert(0, "job_run_id", job_run_id)
out["endpoint"], out["prompt_version"], out["judged_at"] = endpoint, PROMPT_VERSION, datetime.now(timezone.utc)

failed = out["error"].notna().sum()
print(f"Judged {len(out) - failed}/{len(out)} · tokens in {out['prompt_tokens'].sum():,} out {out['completion_tokens'].sum():,}")
if failed:
    display(out[out["error"].notna()][["experiment", "example_id", "error"]].head(10))

# COMMAND ----------

if spark.catalog.tableExists("judge_results"):
    spark.sql(f"DELETE FROM judge_results WHERE job_run_id = '{job_run_id}' AND experiment != 'reference'")
spark.createDataFrame(out).write.mode("append").option("mergeSchema", "true").saveAsTable("judge_results")

display(spark.sql(f"""
    SELECT family, count(*) AS notes, round(avg(CAST(faithful AS INT)), 3) AS faithful_rate,
           round(avg(CAST(hallucinated AS INT)), 3) AS hallucination_rate, round(avg(faithfulness_score), 2) AS score
    FROM judge_results
    WHERE error IS NULL AND (job_run_id = '{job_run_id}' OR experiment = 'reference') AND endpoint = '{endpoint}'
    GROUP BY family ORDER BY faithful_rate DESC
"""))
