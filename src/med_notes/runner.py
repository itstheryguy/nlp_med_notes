"""The training session: train + evaluate every planned run back to back, inside one compute allocation.

Contract with the CPU tasks (all files under run_dir, a UC volume folder per job run):
  in:  plan.json, splits.parquet                  (written by 01_prepare_data)
  out: <run>/predictions.parquet, <run>/losses.parquet, status.json   (read by 03_ingest_results)

Production habits for a shared budget:
  - budget guard: a run is skipped, not started, if its estimated duration would pass max_train_minutes
  - failure isolation: one run crashing is recorded and the session moves on to the next
  - incremental status: status.json is rewritten after every run, so a timeout keeps finished work
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from .config import ExperimentConfig
from .data import get_split


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def run_plan(plan: dict, splits_df: pd.DataFrame, run_dir: str, use_mlflow: bool = False, log=print,
             bertscore_model: str | None = None) -> list[dict]:
    from .evaluation import generate, score, summarize
    from .train import loss_history, train, truncation_stats

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu x{torch.get_num_threads()}"
    max_minutes = float(plan["max_train_minutes"])
    session_start = time.time()
    family_minutes: dict[str, float] = {}  # measured duration of the first finished run per family
    best_by_family: dict[str, float] = {}  # lowest val loss saved per family this session
    status: list[dict] = []
    status_path = os.path.join(run_dir, "status.json")

    for cfg_dict in plan["experiments"]:
        cfg = ExperimentConfig.from_dict(cfg_dict)
        elapsed = (time.time() - session_start) / 60
        estimate = family_minutes.get(cfg.family, cfg.est_minutes)
        entry = {"experiment": cfg.name, "family": cfg.family, "seed": cfg.seed, "model_name": cfg.model_name,
                 "gpu": gpu, "estimated_minutes": round(estimate, 1)}
        if elapsed + estimate > max_minutes:
            entry.update(status="skipped_budget", minutes=0.0)
            log(f"SKIP {cfg.name}: {elapsed:.0f} min used + ~{estimate:.0f} min > budget {max_minutes:.0f} min")
            status.append(entry)
            _write_json(status_path, {"train_minutes_used": round(elapsed, 2), "runs": status})
            continue

        run_start = time.time()
        out_dir = os.path.join(run_dir, cfg.name)
        os.makedirs(out_dir, exist_ok=True)
        local_dir = tempfile.mkdtemp(prefix=f"{cfg.name}_")
        mlflow_run = None
        try:
            if use_mlflow:
                import mlflow

                mlflow_run = mlflow.start_run(run_name=cfg.name)
                mlflow.log_params({f"cfg.{k}": v for k, v in cfg.to_flat_dict().items()})
                mlflow.set_tags({"job_run_id": plan["job_run_id"], "family": cfg.family, "seed": cfg.seed, "gpu": gpu})

            train_df, val_df = get_split(splits_df, "train"), get_split(splits_df, "val")
            log(f"START {cfg.name} ({cfg.model_name}), train={len(train_df)} val={len(val_df)}")
            model, tokenizer, trainer, info = train(cfg, train_df, val_df, local_dir,
                                                    report_to="mlflow" if use_mlflow else "none")
            num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
            train_minutes = info["train_seconds"] / 60
            loss_history(trainer).assign(experiment=cfg.name).to_parquet(os.path.join(out_dir, "losses.parquet"))

            frames, metrics_by_split = [], {}
            for split in plan["eval_splits"]:
                df = get_split(splits_df, split)
                preds, sec_per_ex = generate(model, tokenizer, df, cfg)
                per_ex = score(preds, df["section_text"].tolist(), bertscore_model=bertscore_model)
                metrics_by_split[split] = {**summarize(per_ex), "gen_seconds_per_example": round(sec_per_ex, 4)}
                frames.append(pd.concat([pd.DataFrame({
                    "experiment": cfg.name, "split": split,
                    "example_id": df["example_id"].astype("int64"), "prediction": preds,
                }), per_ex], axis=1))
            pd.concat(frames, ignore_index=True).to_parquet(os.path.join(out_dir, "predictions.parquet"))

            model_path = None
            if cfg.save_model and info["best_val_loss"] < best_by_family.get(cfg.family, float("inf")):
                model_path = os.path.join(plan["models_dir"], cfg.family)
                tmp_model = os.path.join(local_dir, "best_model")
                trainer.save_model(tmp_model)
                shutil.rmtree(model_path, ignore_errors=True)
                shutil.copytree(tmp_model, model_path)
                _write_json(os.path.join(model_path, "run_info.json"),
                            {"experiment": cfg.name, "job_run_id": plan["job_run_id"], **info, "config": cfg.to_flat_dict()})
                best_by_family[cfg.family] = info["best_val_loss"]

            minutes = (time.time() - run_start) / 60
            family_minutes.setdefault(cfg.family, minutes)
            entry.update(status="succeeded", minutes=round(minutes, 2), train_minutes=round(train_minutes, 2),
                         num_params_m=round(num_params_m, 1), metrics=metrics_by_split, model_path=model_path,
                         config=cfg.to_flat_dict(), **info, **truncation_stats(train_df, tokenizer, cfg))
            if use_mlflow:
                import mlflow

                mlflow.log_metrics({"best_val_loss": info["best_val_loss"], "epochs_run": info["epochs_run"],
                                    "run_minutes": minutes, "num_params_m": num_params_m})
                for split, m in metrics_by_split.items():
                    mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            log(f"DONE  {cfg.name} in {minutes:.1f} min, best epoch {info['best_epoch']:.0f}/{info['epochs_run']:.0f}, "
                + ", ".join(f"{s}: rougeL {m['rougeL']:.3f}" for s, m in metrics_by_split.items()))
            del model, trainer
        except Exception as e:
            minutes = (time.time() - run_start) / 60
            entry.update(status="failed", minutes=round(minutes, 2), error=f"{type(e).__name__}: {e}",
                         traceback=traceback.format_exc()[-4000:])
            log(f"FAIL  {cfg.name} after {minutes:.1f} min: {type(e).__name__}: {e}")
        finally:
            if mlflow_run is not None:
                import mlflow

                mlflow.end_run(status="FINISHED" if entry.get("status") == "succeeded" else "FAILED")
            shutil.rmtree(local_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        status.append(entry)
        _write_json(status_path, {"train_minutes_used": round((time.time() - session_start) / 60, 2), "runs": status})

    return status
