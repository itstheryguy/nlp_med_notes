# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Ingest results (CPU)
# MAGIC
# MAGIC Loads the GPU session's output folder into Delta and runs the **tier-1 audit** on every generated note.
# MAGIC
# MAGIC | Table | Grain | Contents |
# MAGIC |---|---|---|
# MAGIC | `runs` | one row per run per job run | status, GPU minutes (actual vs estimated), best epoch, val/test ROUGE + BERTScore, config |
# MAGIC | `predictions` | one row per run per example | generated note, reference, per-example scores, tier-1 checks |
# MAGIC | `training_losses` | one row per logged step | train and val loss curves |
# MAGIC | `reference_checks` | one row per example | tier-1 checks on the clinician references (the ceiling) |
# MAGIC
# MAGIC Idempotent: rows for this `job_run_id` are deleted before they're appended, so rerunning the task
# MAGIC never duplicates results.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
job_run_id = dbutils.widgets.get("job_run_id")
run_dir = f"/Volumes/{catalog}/{schema}/pipeline/runs/{job_run_id}"
spark.sql(f"USE {catalog}.{schema}")

if not os.path.exists(f"{run_dir}/status.json"):
    dbutils.notebook.exit(f"No status.json in {run_dir}: the GPU task produced nothing to ingest.")

# COMMAND ----------

from med_notes.ingest import load_run_dir, reference_tier1

silver = spark.table("silver_dialog_notes").toPandas()
frames = load_run_dir(run_dir, silver, job_run_id)

for table, df in frames.items():
    if df.empty:
        continue
    if spark.catalog.tableExists(table):
        spark.sql(f"DELETE FROM {table} WHERE job_run_id = '{job_run_id}'")
    spark.createDataFrame(df).write.mode("append").option("mergeSchema", "true").saveAsTable(table)
    print(f"{table}: {len(df)} rows")

if not spark.catalog.tableExists("reference_checks"):
    spark.createDataFrame(reference_tier1(silver)).write.saveAsTable("reference_checks")

# COMMAND ----------

cols = [c for c in ["experiment", "status", "minutes", "estimated_minutes", "best_epoch", "epochs_run",
                    "val_rougeL", "test_rougeL", "error"] if c in frames["runs"].columns]
display(frames["runs"][cols].sort_values("experiment"))
