# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Prepare data and the run plan
# MAGIC
# MAGIC Loads [MTS-Dialog](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note) into Delta, freezes the
# MAGIC splits and the audit sample, then writes the handoff files the GPU task reads.
# MAGIC
# MAGIC | Output | Contents |
# MAGIC |---|---|
# MAGIC | `bronze_mts_dialog` | rows exactly as the HF dataset returns them, plus a row-order `example_id` |
# MAGIC | `silver_dialog_notes` | normalized text, `split` (train/val/test 80/10/10, stratified by section, repeated dialogues on one side) |
# MAGIC | `audit_sample` | the fixed test examples the LLM judge scores for every run |
# MAGIC | `pipeline/runs/<job_run_id>/plan.json` | the runs to train, the eval splits, the GPU budget |
# MAGIC | `pipeline/runs/<job_run_id>/splits.parquet` | a snapshot of `silver_dialog_notes` for the GPU task (it has no Spark) |
# MAGIC
# MAGIC Tables are only rebuilt when `refresh_data` is `true`, so an upstream dataset change can't silently
# MAGIC reshuffle the splits between runs.

# COMMAND ----------

# MAGIC %pip install -q datasets pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("experiments", "group:pilot", "Experiments")
dbutils.widgets.text("eval_splits", "val,test", "Eval splits")
dbutils.widgets.text("max_gpu_minutes", "30", "GPU budget (minutes)")
dbutils.widgets.text("audit_sample_size", "40", "Audit sample size (used when audit_sample is built)")
dbutils.widgets.dropdown("refresh_data", "false", ["false", "true"], "Rebuild tables")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
refresh = dbutils.widgets.get("refresh_data") == "true"
job_run_id = dbutils.widgets.get("job_run_id")
spark.sql(f"USE {catalog}.{schema}")

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("hf", "token")
except Exception:
    pass

# COMMAND ----------

from med_notes.audit import select_audit_sample
from med_notes.data import DATASET_ID, build_silver, load_raw

if refresh or not spark.catalog.tableExists("silver_dialog_notes"):
    raw = load_raw()
    silver = build_silver(raw)
    (spark.createDataFrame(raw).write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable("bronze_mts_dialog"))
    (spark.createDataFrame(silver).write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable("silver_dialog_notes"))
    spark.sql(f"COMMENT ON TABLE silver_dialog_notes IS 'MTS-Dialog ({DATASET_ID}) with a frozen train/val/test split'")

    audit_ids = select_audit_sample(silver, int(dbutils.widgets.get("audit_sample_size")))
    (spark.createDataFrame([(i, r) for r, i in enumerate(audit_ids)], "example_id BIGINT, audit_rank INT")
        .write.mode("overwrite").saveAsTable("audit_sample"))
    print(f"Wrote {len(raw)} rows and an audit sample of {len(audit_ids)} test examples")
else:
    print("Tables exist; set refresh_data=true to rebuild them")

display(spark.sql("""
    SELECT split, count(*) AS rows, count(DISTINCT section_header) AS sections,
           round(avg(dialogue_chars)) AS avg_dialogue_chars, round(avg(target_chars)) AS avg_target_chars,
           round(avg(CAST(has_template AS INT)), 3) AS template_rate
    FROM silver_dialog_notes GROUP BY split ORDER BY split
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run plan
# MAGIC The GPU task can't take task values or run Spark, so it gets everything it needs as files in a
# MAGIC per-run folder. Rerunning this task for the same job run overwrites the plan.

# COMMAND ----------

from med_notes.config import load_experiments, select_experiments

experiments = load_experiments("../configs/experiments.yml")
names = select_experiments(experiments, dbutils.widgets.get("experiments"))
eval_splits = [s.strip() for s in dbutils.widgets.get("eval_splits").split(",") if s.strip()]

run_dir = f"/Volumes/{catalog}/{schema}/pipeline/runs/{job_run_id}"
os.makedirs(run_dir, exist_ok=True)
plan = {
    "job_run_id": job_run_id,
    "experiments": [experiments[n].to_dict() for n in names],
    "eval_splits": eval_splits,
    "max_gpu_minutes": float(dbutils.widgets.get("max_gpu_minutes")),
    "models_dir": f"/Volumes/{catalog}/{schema}/models",
}
with open(f"{run_dir}/plan.json", "w") as f:
    json.dump(plan, f, indent=2)
spark.table("silver_dialog_notes").toPandas().to_parquet(f"{run_dir}/splits.parquet")

est = sum(experiments[n].est_minutes for n in names)
print(f"Plan: {len(names)} runs, estimated {est:.0f} GPU min (budget {plan['max_gpu_minutes']:.0f}) → {run_dir}")
for n in names:
    c = experiments[n]
    print(f"  {n:24s} {c.model_name:32s} lr={c.learning_rate:g} ~{c.est_minutes:g} min")
