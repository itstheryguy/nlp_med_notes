"""Start and watch a SageMaker training job from Databricks.

The job runs `sagemaker/entry.py` in AWS's PyTorch GPU container (script mode: the container downloads
the code tarball, pip-installs its requirements.txt, then runs the entry point).
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import time

DEFAULT_IMAGE = ("763104351884.dkr.ecr.{region}.amazonaws.com/"
                 "pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker")
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
# Lines worth echoing into the Databricks notebook; everything else stays in CloudWatch.
LOG_PATTERN = re.compile(r"^(START|DONE|FAIL|SKIP|RESUME|Nothing left|REMOTE)|Traceback|Error|error:|Killed|OOM")
TERMINAL = {"Completed", "Failed", "Stopped"}


def build_sourcedir(repo_root: str) -> bytes:
    """tar.gz with entry.py and requirements.txt at the root, plus the med_notes package."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(repo_root, "sagemaker", "entry.py"), arcname="entry.py")
        tar.add(os.path.join(repo_root, "requirements-train.txt"), arcname="requirements.txt")
        tar.add(os.path.join(repo_root, "src", "med_notes"), arcname="med_notes",
                filter=lambda t: None if "__pycache__" in t.name else t)
    return buf.getvalue()


def job_name(project: str, job_run_id: str) -> str:
    """Unique, ≤63 chars, [A-Za-z0-9-]."""
    base = re.sub(r"[^A-Za-z0-9-]", "-", f"{project}-{job_run_id}")[:48].strip("-")
    return f"{base}-{int(time.time()) % 10**8}"


def training_job_request(*, name: str, image: str, role_arn: str, instance_type: str, s3_root: str,
                         volume_root: str, run_rel: str, models_rel: str, sourcedir_s3: str,
                         max_runtime_s: int, use_spot: bool, region: str, tags: dict | None = None) -> dict:
    req = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "RoleArn": role_arn,
        "HyperParameters": {  # script-mode contract of the SageMaker training toolkit (values must be strings)
            "sagemaker_program": "entry.py",
            "sagemaker_submit_directory": sourcedir_s3,
            "sagemaker_region": region,
        },
        "Environment": {
            "S3_ROOT": s3_root, "VOLUME_ROOT": volume_root, "RUN_REL": run_rel, "MODELS_REL": models_rel,
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
        },
        "OutputDataConfig": {"S3OutputPath": f"{s3_root.rstrip('/')}/sagemaker-output/"},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 60},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_runtime_s)},
        "EnableManagedSpotTraining": bool(use_spot),
        "Tags": [{"Key": k, "Value": str(v)} for k, v in (tags or {}).items()],
    }
    if use_spot:
        # How long to wait for spot capacity, plus the runtime itself. Must be >= MaxRuntimeInSeconds.
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = int(max_runtime_s) + 3600
    return req


def wait_for_job(sm, logs, name: str, log=print, poll_seconds: int = 60) -> dict:
    """Poll until the job ends, echoing status changes and key log lines. Returns the final description."""
    last_status, token, since = None, None, int(time.time() * 1000) - 60_000
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        status = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if status != last_status:
            log(f"[{time.strftime('%H:%M:%S')}] {status[0]} / {status[1]}"
                + (f": {d['SecondaryStatusTransitions'][-1].get('StatusMessage', '')}" if d.get("SecondaryStatusTransitions") else ""))
            last_status = status
        try:
            kw = {"logGroupName": LOG_GROUP, "logStreamNamePrefix": name, "startTime": since}
            while True:
                page = logs.filter_log_events(**kw, **({"nextToken": token} if token else {}))
                for e in page.get("events", []):
                    msg = e["message"].strip()
                    if LOG_PATTERN.search(msg):
                        log("  " + msg[:400])
                    since = max(since, e["timestamp"] + 1)
                token = page.get("nextToken")
                if not token:
                    break
        except Exception as e:  # log group doesn't exist until the container starts
            if "ResourceNotFound" not in type(e).__name__ and "ResourceNotFound" not in str(e):
                log(f"  (log read failed: {type(e).__name__})")
        if d["TrainingJobStatus"] in TERMINAL:
            return d
        time.sleep(poll_seconds)


def summarize_job(d: dict, usd_per_hour: float) -> dict:
    """Billing-relevant facts. BillableTimeInSeconds already reflects spot savings."""
    billable = d.get("BillableTimeInSeconds") or 0
    training = d.get("TrainingTimeInSeconds") or 0
    return {
        "training_job_name": d["TrainingJobName"],
        "status": d["TrainingJobStatus"],
        "secondary_status": d.get("SecondaryStatus"),
        "failure_reason": d.get("FailureReason"),
        "instance_type": d["ResourceConfig"]["InstanceType"],
        "spot": bool(d.get("EnableManagedSpotTraining")),
        "training_seconds": training,
        "billable_seconds": billable,
        "spot_savings_pct": round(100 * (1 - billable / training), 1) if d.get("EnableManagedSpotTraining") and training else 0.0,
        "usd_per_hour_on_demand": usd_per_hour,
        "estimated_usd": round(billable / 3600 * usd_per_hour, 4),
        "started_at": str(d.get("TrainingStartTime")),
        "ended_at": str(d.get("TrainingEndTime")),
    }
