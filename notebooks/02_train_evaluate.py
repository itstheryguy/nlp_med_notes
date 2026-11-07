# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train and evaluate
# MAGIC
# MAGIC The only training task. One session trains every run in `plan.json` back to back, so startup and
# MAGIC package install are paid once. It runs on serverless CPU by default; on Serverless GPU where the account
# MAGIC allows it (see `databricks.yml`). Logic lives in `src/med_notes/runner.py`:
# MAGIC
# MAGIC - **Budget guard:** a run is skipped (not started) when `elapsed + estimate > max_train_minutes`. The estimate
# MAGIC   is the family's `est_minutes` until its first seed finishes, then that run's measured time.
# MAGIC - **Failure isolation:** a crashed run is recorded in `status.json` and the session moves on.
# MAGIC - **Outputs:** per run, `predictions.parquet` and `losses.parquet` in the run folder; `status.json` is
# MAGIC   rewritten after every run. `03_ingest_results` loads them into Delta on CPU.
# MAGIC - **Weights:** the seed with the lowest validation loss for each model is copied to the `models` volume.
# MAGIC
# MAGIC This task needs no Spark. Interactively: run `01` first, then this notebook with `job_run_id` set to the
# MAGIC folder `01` wrote (`manual` by default).

# COMMAND ----------

# torch first, matched to the hardware: the CPU-only build is ~200 MB; the default (CUDA) build is ~2.5 GB.
import importlib.util
import os
import subprocess
import sys

has_gpu = os.path.exists("/dev/nvidia0")
pip = [sys.executable, "-m", "pip", "install", "-q"]
if importlib.util.find_spec("torch") is None:
    subprocess.check_call(pip + ["torch"] + ([] if has_gpu else ["--index-url", "https://download.pytorch.org/whl/cpu"]))
subprocess.check_call(pip + ["-r", "../requirements-train.txt"])

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment path (blank = per-user default)")

catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
job_run_id = dbutils.widgets.get("job_run_id")
dbutils.widgets.dropdown("run_volume", "pipeline", ["sagemaker", "pipeline"], "Run folder volume")
run_dir = f"/Volumes/{catalog}/{schema}/{dbutils.widgets.get('run_volume')}/runs/{job_run_id}"

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("hf", "token")
except Exception:
    pass

with open(f"{run_dir}/plan.json") as f:
    plan = json.load(f)
splits = pd.read_parquet(f"{run_dir}/splits.parquet")

import torch

device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU x{torch.get_num_threads()} threads"
print(f"Device: {device} · torch {torch.__version__} · {os.cpu_count()} cores")
print(f"{len(plan['experiments'])} runs, budget {plan['max_train_minutes']:.0f} min, eval on {plan['eval_splits']}")

# COMMAND ----------

import mlflow

if dbutils.widgets.get("mlflow_experiment"):  # the job sets this; interactive runs log to the notebook's experiment
    mlflow.set_experiment(dbutils.widgets.get("mlflow_experiment"))
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "false"  # weights go to the volume, not MLflow artifacts

from med_notes.runner import run_plan

status = run_plan(plan, splits, run_dir, use_mlflow=True)

# COMMAND ----------

summary = pd.DataFrame([{k: s.get(k) for k in ("experiment", "status", "minutes", "best_epoch", "epochs_run", "error")}
                        for s in status])
display(summary)
print(f"Training minutes used: {summary['minutes'].sum():.1f} of {plan['max_train_minutes']:.0f}")
if not (summary["status"] == "succeeded").any():
    raise RuntimeError("No run succeeded; see status.json in " + run_dir)
