# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train and evaluate on SageMaker (AWS GPU)
# MAGIC
# MAGIC Runs the same `runner.run_plan` as `02_train_evaluate`, inside a SageMaker training job on an A10G
# MAGIC (`ml.g5.xlarge`). This workspace can't start Serverless GPU; the AWS side is set up by the
# MAGIC [`aws-ml-infra`](../../aws-ml-infra) stack.
# MAGIC
# MAGIC | Step | How |
# MAGIC |---|---|
# MAGIC | Share files | The run folder is on the `sagemaker` **external volume**, so `/Volumes/.../sagemaker/runs/<id>` here is `s3://<bucket>/<prefix>/runs/<id>` there. No copying |
# MAGIC | Authenticate | Unity Catalog **service credential** → short-lived AWS credentials. No keys |
# MAGIC | Ship code | `src/med_notes` + `sagemaker/entry.py` + requirements as a tarball in the run folder |
# MAGIC | Limits | None self-imposed: the job requests SageMaker's default 24 h ceiling. The IAM policy limits instance types. On-demand by default; spot is optional (`sagemaker_use_spot`) but can wait a long time for capacity |
# MAGIC | Watch | Job status and key log lines (`START` / `DONE` / `FAIL`) stream into this notebook |
# MAGIC | Resume | If the job restarts (e.g. a spot interruption), it skips runs that already succeeded |
# MAGIC | Record | `sagemaker_job.json` in the run folder: billable seconds, spot savings, estimated cost |
# MAGIC
# MAGIC If this notebook is cancelled, it stops the training job so nothing keeps billing.

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "med_notes", "Schema")
dbutils.widgets.text("job_run_id", "manual", "Job run ID")
dbutils.widgets.text("aws_region", "us-east-2", "AWS region")
dbutils.widgets.text("aws_service_credential", "aws-ml-training", "UC service credential")
dbutils.widgets.text("sagemaker_s3_root", "", "S3 URI behind the sagemaker volume")
dbutils.widgets.text("sagemaker_role_arn", "", "SageMaker execution role ARN")
dbutils.widgets.text("sagemaker_instance_type", "ml.g5.xlarge", "Instance type")
dbutils.widgets.dropdown("sagemaker_use_spot", "false", ["true", "false"], "Spot pricing")
dbutils.widgets.text("sagemaker_usd_per_hour", "1.2575", "On-demand $/hour (for the cost estimate)")
dbutils.widgets.text("sagemaker_max_runtime_hours", "24", "SageMaker max runtime, hours (24 = SageMaker's default)")

w = {k: dbutils.widgets.get(k) for k in ["catalog", "schema", "job_run_id", "aws_region", "aws_service_credential",
                                         "sagemaker_s3_root", "sagemaker_role_arn", "sagemaker_instance_type",
                                         "sagemaker_use_spot", "sagemaker_usd_per_hour", "sagemaker_max_runtime_hours"]}
volume_root = f"/Volumes/{w['catalog']}/{w['schema']}/sagemaker"
run_rel, models_rel = f"runs/{w['job_run_id']}", "models"
run_dir = f"{volume_root}/{run_rel}"
s3_root = w["sagemaker_s3_root"].rstrip("/")
assert s3_root.startswith("s3://") and w["sagemaker_role_arn"], "Set sagemaker_s3_root and sagemaker_role_arn"

with open(f"{run_dir}/plan.json") as f:
    plan = json.load(f)
# No self-imposed time cap: request SageMaker's own default ceiling (24 h). The IAM policy in aws-ml-infra
# must allow it (MaxRuntimeSeconds >= this).
max_runtime_s = int(float(dbutils.widgets.get("sagemaker_max_runtime_hours")) * 3600)
budget = plan["max_train_minutes"]
print(f"{len(plan['experiments'])} runs · " + (f"budget guard {budget:.0f} min · " if budget > 0 else "no budget guard · ")
      + f"SageMaker max runtime {max_runtime_s / 3600:.0f} h")

# COMMAND ----------

import boto3
from med_notes.sagemaker_job import DEFAULT_IMAGE, build_sourcedir, job_name, summarize_job, training_job_request, wait_for_job

session = boto3.Session(botocore_session=dbutils.credentials.getServiceCredentialsProvider(w["aws_service_credential"]),
                        region_name=w["aws_region"])
sm, logs = session.client("sagemaker"), session.client("logs")

os.makedirs(f"{run_dir}/code", exist_ok=True)
with open(f"{run_dir}/code/sourcedir.tar.gz", "wb") as f:
    f.write(build_sourcedir(os.path.abspath("..")))

name = job_name("nlp-med-notes", w["job_run_id"])
request = training_job_request(
    name=name, image=DEFAULT_IMAGE.format(region=w["aws_region"]), role_arn=w["sagemaker_role_arn"],
    instance_type=w["sagemaker_instance_type"], s3_root=s3_root, volume_root=volume_root, run_rel=run_rel,
    models_rel=models_rel, sourcedir_s3=f"{s3_root}/{run_rel}/code/sourcedir.tar.gz",
    max_runtime_s=max_runtime_s, use_spot=w["sagemaker_use_spot"] == "true", region=w["aws_region"],
    tags={"project": "nlp_med_notes", "databricks_job_run_id": w["job_run_id"], "schema": w["schema"]},
)
sm.create_training_job(**request)
print(f"Started {name} ({w['sagemaker_instance_type']}, spot={w['sagemaker_use_spot']})")
print(f"https://{w['aws_region']}.console.aws.amazon.com/sagemaker/home?region={w['aws_region']}#/jobs/{name}")

# COMMAND ----------

try:
    final = wait_for_job(sm, logs, name)
except BaseException:  # notebook cancelled or the poll loop broke: don't leave a GPU running
    print(f"Stopping {name}")
    sm.stop_training_job(TrainingJobName=name)
    raise

summary = summarize_job(final, float(w["sagemaker_usd_per_hour"]))
with open(f"{run_dir}/sagemaker_job.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
if summary["status"] != "Completed":
    raise RuntimeError(f"Training job {summary['status']}: {summary['failure_reason']}")
