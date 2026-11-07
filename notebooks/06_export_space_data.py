# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Export the Space snapshot
# MAGIC
# MAGIC Writes the files the Hugging Face Space reads to `pipeline/space_export/<export_id>/`. The Space never
# MAGIC queries Databricks: it ships this snapshot, so it needs no credentials and keeps working after the
# MAGIC workspace is gone.
# MAGIC
# MAGIC | File | Grain |
# MAGIC |---|---|
# MAGIC | `examples.{json,parquet}` | one row per test example: dialogue, reference, reference audit results |
# MAGIC | `notes.{json,parquet}` | one row per run per test example: generated note, tier-1 checks, judge verdict, per-field N/A diff |
# MAGIC | `runs.{json,parquet}` | one row per run: overlap, tier-1, and judge rates |
# MAGIC | `summary.json` | per-model mean ± sd, reference ceiling, audit calibration, training cost, provenance |
# MAGIC
# MAGIC The Space ships the JSON files (it runs in the browser via stlite); parquet is for analysis.
# MAGIC
# MAGIC `job_run_ids` picks which job runs to publish (comma-separated). Publish from the prod target for the
# MAGIC final version, so the snapshot traces back to one tagged commit and one schema.

# COMMAND ----------

# MAGIC %pip install -q nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_ids", "", "Job run IDs to publish (comma-separated)")
dbutils.widgets.text("export_id", "", "Export ID (blank = timestamp)")
dbutils.widgets.text("code_version", "", "Git tag/commit the runs came from (shown in the Space)")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.sql(f"USE {catalog}.{schema}")
ids = [i.strip() for i in dbutils.widgets.get("job_run_ids").split(",") if i.strip()]
assert ids, "Set job_run_ids"
id_list = ", ".join(f"'{i}'" for i in ids)

def table(sql: str):
    df = spark.sql(sql).toPandas()
    df.attrs = {}  # Spark Connect metrics break to_parquet
    return df

# COMMAND ----------

from datetime import datetime, timezone

from med_notes.export import build_snapshot
from med_notes.judge import PROMPT_VERSION

runs = table(f"SELECT * FROM runs WHERE job_run_id IN ({id_list}) AND status = 'succeeded'")
dupes = runs["experiment"].duplicated()
assert not dupes.any(), f"Same experiment in several job runs: {runs.loc[dupes, 'experiment'].tolist()}"
endpoints = table(f"SELECT DISTINCT endpoint FROM judge_results WHERE job_run_id IN ({id_list})")["endpoint"].tolist()

snap = build_snapshot(
    silver=table("SELECT * FROM silver_dialog_notes"),
    audit=table("SELECT * FROM audit_sample"),
    predictions=table(f"SELECT * FROM predictions WHERE job_run_id IN ({id_list})"),
    judge=table(f"SELECT * FROM judge_results WHERE (job_run_id IN ({id_list}) OR experiment = 'reference') "
                f"AND error IS NULL AND prompt_version = '{PROMPT_VERSION}'"),
    reference_checks=table("SELECT * FROM reference_checks"),
    runs=runs,
    training_jobs=table(f"SELECT * FROM training_jobs WHERE job_run_id IN ({id_list})") if spark.catalog.tableExists("training_jobs") else None,
    meta={"catalog": catalog, "schema": schema, "job_run_ids": ids, "code_version": dbutils.widgets.get("code_version"), "judge_endpoints": endpoints,
          "judge_prompt_version": PROMPT_VERSION, "dataset": "har1/MTS_Dialogue-Clinical_Note (MTS-Dialog, CC BY 4.0)"},
)

export_id = dbutils.widgets.get("export_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = f"/Volumes/{catalog}/{schema}/pipeline/space_export/{export_id}"
os.makedirs(out, exist_ok=True)
for name in ("examples", "notes", "runs"):
    df = snap[name].copy()
    df.to_parquet(f"{out}/{name}.parquet", index=False)  # for analysis
    for c in df.columns:  # numpy arrays -> lists, so the JSON is plain
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.tolist() if hasattr(v, "tolist") else v)
    # The static Space (stlite, in the browser) reads JSON: pyarrow may not be available under Pyodide.
    df.to_json(f"{out}/{name}.json", orient="records")
with open(f"{out}/summary.json", "w") as f:
    json.dump(snap["summary"], f, indent=2, default=str)
print(f"Exported to {out}: {len(snap['examples'])} examples, {len(snap['notes'])} notes, {len(snap['runs'])} runs")
display(snap["runs"])
