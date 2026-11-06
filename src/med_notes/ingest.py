"""Turn a GPU session's output folder into table-shaped DataFrames (runs, predictions, losses)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .audit import tier1


def load_run_dir(run_dir: str, silver: pd.DataFrame, job_run_id: str) -> dict[str, pd.DataFrame]:
    with open(os.path.join(run_dir, "status.json")) as f:
        status = json.load(f)["runs"]
    now = datetime.now(timezone.utc)

    run_rows = []
    for s in status:
        row = {
            "job_run_id": job_run_id, "experiment": s["experiment"], "family": s["family"], "seed": int(s["seed"]),
            "model_name": s["model_name"], "status": s["status"], "gpu": s.get("gpu"),
            "minutes": float(s.get("minutes") or 0.0), "estimated_minutes": s.get("estimated_minutes"),
            "train_minutes": s.get("train_minutes"), "best_val_loss": s.get("best_val_loss"),
            "best_epoch": s.get("best_epoch"), "epochs_run": s.get("epochs_run"), "num_params_m": s.get("num_params_m"),
            "source_truncated_frac": s.get("source_truncated_frac"), "target_truncated_frac": s.get("target_truncated_frac"),
            "model_path": s.get("model_path"), "error": s.get("error"),
            "config_json": json.dumps(s.get("config")) if s.get("config") else None, "ingested_at": now,
        }
        for split, metrics in (s.get("metrics") or {}).items():
            row.update({f"{split}_{k}": float(v) for k, v in metrics.items()})
        run_rows.append(row)
    runs = pd.DataFrame(run_rows)

    pred_frames, loss_frames = [], []
    for s in status:
        if s["status"] != "succeeded":
            continue
        d = os.path.join(run_dir, s["experiment"])
        p = pd.read_parquet(os.path.join(d, "predictions.parquet"))
        p = p.merge(silver[["example_id", "section_header", "dialogue", "section_text"]], on="example_id", how="left")
        checks = tier1(p["prediction"].tolist(), p["section_text"].tolist(), p["dialogue"].tolist())
        p = pd.concat([p.reset_index(drop=True), checks], axis=1)
        p = p.rename(columns={"section_text": "reference"}).drop(columns=["dialogue"])
        p.insert(0, "job_run_id", job_run_id)
        p["family"], p["seed"], p["ingested_at"] = s["family"], int(s["seed"]), now
        pred_frames.append(p)
        loss_frames.append(pd.read_parquet(os.path.join(d, "losses.parquet")).assign(job_run_id=job_run_id))

    return {
        "runs": runs,
        "predictions": pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
        "training_losses": pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame(),
    }


def reference_tier1(silver: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 checks on the clinician references themselves: the ceiling a model is compared against."""
    checks = tier1(silver["section_text"].tolist(), silver["section_text"].tolist(), silver["dialogue"].tolist())
    return pd.concat([silver[["example_id", "split", "section_header"]].reset_index(drop=True), checks], axis=1)
