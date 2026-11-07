"""Run a plan inside a remote training job (SageMaker) against files that Databricks sees as a volume.

The run folder lives in S3 under an external volume, so `/Volumes/<c>/<s>/<vol>/runs/<id>/plan.json` on
Databricks is `s3://<bucket>/<prefix>/runs/<id>/plan.json` here. This module:

  1. downloads plan.json + splits.parquet to local disk
  2. skips runs that already succeeded (a spot interruption restarts the job; finished runs aren't repeated)
  3. runs runner.run_plan on local disk, uploading each run's folder, the best weights, and a merged
     status.json as soon as that run finishes, so Databricks can watch progress and a crash loses nothing
"""

from __future__ import annotations

import io
import json
import os
import shutil

import pandas as pd


class LocalStore:
    """A directory standing in for the S3 prefix (tests)."""

    def __init__(self, root: str):
        self.root = root

    def get(self, rel: str) -> bytes | None:
        p = os.path.join(self.root, rel)
        return open(p, "rb").read() if os.path.exists(p) else None

    def put(self, rel: str, data: bytes) -> None:
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    def put_dir(self, rel: str, local_dir: str) -> None:
        dest = os.path.join(self.root, rel)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(local_dir, dest)


class S3Store:
    """s3://bucket/prefix, using the training job's own (execution role) credentials."""

    def __init__(self, s3_root: str, client=None):
        import boto3

        bucket, _, prefix = s3_root.removeprefix("s3://").partition("/")
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.s3 = client or boto3.client("s3")

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def get(self, rel: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(rel))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, rel: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(rel), Body=data)

    def put_dir(self, rel: str, local_dir: str) -> None:
        prefix = self._key(rel).rstrip("/") + "/"
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
        for dirpath, _, files in os.walk(local_dir):
            for name in files:
                path = os.path.join(dirpath, name)
                self.s3.upload_file(path, self.bucket, prefix + os.path.relpath(path, local_dir))


def run_remote(store, run_rel: str, models_rel: str, volume_root: str, work_dir: str, log=print,
               bertscore_model: str | None = None) -> list[dict]:
    """Returns the merged status (earlier sessions' successes + this session's runs)."""
    from .runner import run_plan

    run_dir, models_dir = os.path.join(work_dir, "run"), os.path.join(work_dir, "models")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    plan = json.loads(store.get(f"{run_rel}/plan.json"))
    splits = pd.read_parquet(io.BytesIO(store.get(f"{run_rel}/splits.parquet")))

    prior_raw = store.get(f"{run_rel}/status.json")
    done = [r for r in json.loads(prior_raw)["runs"] if r["status"] == "succeeded"] if prior_raw else []
    done_names = {r["experiment"] for r in done}
    if done:
        log(f"RESUME: {len(done)} run(s) already succeeded, skipping: {sorted(done_names)}")
    plan["experiments"] = [e for e in plan["experiments"] if e["name"] not in done_names]
    plan["models_dir"] = models_dir
    # Weights from an earlier session are only replaced by a seed with lower val loss.
    plan["best_val_loss_by_family"] = {}
    for r in done:
        if r.get("model_path"):
            prev = plan["best_val_loss_by_family"].get(r["family"], float("inf"))
            plan["best_val_loss_by_family"][r["family"]] = min(prev, r["best_val_loss"])

    current: list[dict] = []

    def publish():
        used = sum(r.get("minutes") or 0 for r in done + current)
        store.put(f"{run_rel}/status.json",
                  json.dumps({"train_minutes_used": round(used, 2), "runs": done + current}, indent=2, default=str).encode())

    def on_run_complete(entry: dict, out_dir: str | None) -> None:
        if out_dir and os.path.isdir(out_dir):
            store.put_dir(f"{run_rel}/{entry['experiment']}", out_dir)
        if entry.get("model_path"):
            store.put_dir(f"{models_rel}/{entry['family']}", entry["model_path"])
            entry["model_path"] = f"{volume_root.rstrip('/')}/{models_rel}/{entry['family']}"  # as Databricks sees it
            for r in done:  # an earlier session's weights for this family were just replaced
                if r["family"] == entry["family"]:
                    r["model_path"] = None
        current.append(entry)
        publish()

    if not plan["experiments"]:
        log("Nothing left to run.")
        return done
    run_plan(plan, splits, run_dir, use_mlflow=False, log=log, bertscore_model=bertscore_model,
             on_run_complete=on_run_complete)
    return done + current
