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
# MAGIC | `training_jobs` | one row per job run trained on SageMaker | instance, spot, billable seconds, spot savings, estimated cost |
# MAGIC
# MAGIC It also logs one **MLflow run** per training run (params, val/test metrics, loss curves). Training on
# MAGIC SageMaker can't log to Databricks MLflow without sending a Databricks token to AWS, so it's logged here.
# MAGIC
# MAGIC Idempotent: rows for this `job_run_id` are deleted before they're appended, so rerunning the task
# MAGIC never duplicates results.

# COMMAND ----------

# MAGIC %pip install -q nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
dbutils.widgets.dropdown("run_volume", "sagemaker", ["sagemaker", "pipeline"], "Run folder volume")
dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment path (blank = skip MLflow)")
job_run_id = dbutils.widgets.get("job_run_id")
run_dir = f"/Volumes/{catalog}/{schema}/{dbutils.widgets.get('run_volume')}/runs/{job_run_id}"
spark.sql(f"USE {catalog}.{schema}")

if not os.path.exists(f"{run_dir}/status.json"):
    dbutils.notebook.exit(f"No status.json in {run_dir}: the training task produced nothing to ingest.")

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

import json
from datetime import datetime, timezone

import pandas as pd

training_job = None
if os.path.exists(f"{run_dir}/sagemaker_job.json"):
    with open(f"{run_dir}/sagemaker_job.json") as f:
        training_job = json.load(f)
    tj = pd.DataFrame([{**training_job, "job_run_id": job_run_id, "ingested_at": datetime.now(timezone.utc)}])
    if spark.catalog.tableExists("training_jobs"):
        spark.sql(f"DELETE FROM training_jobs WHERE job_run_id = '{job_run_id}'")
    spark.createDataFrame(tj).write.mode("append").option("mergeSchema", "true").saveAsTable("training_jobs")
    print(f"training_jobs: {training_job['training_job_name']} {training_job['billable_seconds']}s billable, "
          f"{training_job['spot_savings_pct']}% spot savings, ≈${training_job['estimated_usd']}")

# COMMAND ----------

from med_notes.ingest import log_runs_to_mlflow

if dbutils.widgets.get("mlflow_experiment"):
    import mlflow

    mlflow.set_experiment(dbutils.widgets.get("mlflow_experiment"))
    ids = log_runs_to_mlflow(frames["runs"], frames["training_losses"], job_run_id, training_job)
    print(f"Logged {len(ids)} MLflow runs")

# COMMAND ----------

cols = [c for c in ["experiment", "status", "minutes", "estimated_minutes", "best_epoch", "epochs_run",
                    "val_rougeL", "test_rougeL", "error"] if c in frames["runs"].columns]
display(frames["runs"][cols].sort_values("experiment"))
