"""Fine-tune a seq2seq model with early stopping on validation loss."""

from __future__ import annotations

import math
import os
import time

import pandas as pd

from .config import ExperimentConfig
from .data import format_source


def resolve_precision(precision: str) -> dict:
    import torch

    if precision == "auto":
        if not torch.cuda.is_available():
            precision = "fp32"
        elif torch.cuda.is_bf16_supported():
            precision = "bf16"  # fp32's exponent range; T5 overflows in fp16
        else:
            precision = "fp16"
    return {"bf16": precision == "bf16", "fp16": precision == "fp16"}


def to_hf_dataset(df: pd.DataFrame, tokenizer, cfg: ExperimentConfig):
    from datasets import Dataset

    sources = [format_source(d, h, cfg) for d, h in zip(df["dialogue"], df["section_header"])]
    ds = Dataset.from_dict({"source": sources, "target": df["section_text"].tolist()})

    def tokenize(batch):
        # No padding here: DataCollatorForSeq2Seq pads per batch and masks label padding with -100.
        enc = tokenizer(batch["source"], max_length=cfg.max_source_length, truncation=True)
        enc["labels"] = tokenizer(text_target=batch["target"], max_length=cfg.max_target_length, truncation=True)[
            "input_ids"
        ]
        return enc

    return ds.map(tokenize, batched=True, remove_columns=["source", "target"])


def truncation_stats(df: pd.DataFrame, tokenizer, cfg: ExperimentConfig) -> dict:
    src = [format_source(d, h, cfg) for d, h in zip(df["dialogue"], df["section_header"])]
    src_len = [len(x) for x in tokenizer(src)["input_ids"]]
    tgt_len = [len(x) for x in tokenizer(text_target=df["section_text"].tolist())["input_ids"]]
    return {
        "source_truncated_frac": sum(n > cfg.max_source_length for n in src_len) / len(src_len),
        "target_truncated_frac": sum(n > cfg.max_target_length for n in tgt_len) / len(tgt_len),
    }


def train(cfg: ExperimentConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str, report_to: str = "none"):
    """Train up to cfg.max_epochs, evaluating val loss each epoch; returns the best epoch's model.

    Returns (model, tokenizer, trainer, info) where info has train_seconds, best_val_loss, epochs_run, best_epoch.
    """
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")

    train_ds = to_hf_dataset(train_df, tokenizer, cfg)
    val_ds = to_hf_dataset(val_df, tokenizer, cfg)

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    # Warmup is a fraction of the max schedule; the linear decay also spans max_epochs, so stopping
    # early leaves the LR partway down. That's the usual trade for not knowing the right epoch count.
    warmup_steps = int(round(steps_per_epoch * cfg.max_epochs * cfg.warmup_ratio))

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler,
        label_smoothing_factor=cfg.label_smoothing,
        seed=cfg.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,       # best + latest
        save_only_model=True,     # no optimizer state: checkpoints are ~1/3 the size
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # generation-based eval happens once, after training
        logging_steps=10,
        report_to=report_to,
        disable_tqdm=True,
        # Worker processes deadlock with the tokenizer's thread pool after fork, and a hung run keeps
        # billing until the task timeout. The data is pre-tokenized and tiny, so workers buy nothing.
        dataloader_num_workers=0,
        **resolve_precision(cfg.precision),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    start = time.time()
    trainer.train()
    evals = [e for e in trainer.state.log_history if "eval_loss" in e]
    best = min(evals, key=lambda e: e["eval_loss"])
    info = {
        "train_seconds": time.time() - start,
        "best_val_loss": float(best["eval_loss"]),
        "best_epoch": float(best["epoch"]),
        "epochs_run": float(evals[-1]["epoch"]),
    }
    return trainer.model, tokenizer, trainer, info


def loss_history(trainer) -> pd.DataFrame:
    rows = [
        {"step": e["step"], "epoch": e.get("epoch"), "train_loss": e.get("loss"), "eval_loss": e.get("eval_loss")}
        for e in trainer.state.log_history
        if "loss" in e or "eval_loss" in e
    ]
    return pd.DataFrame(rows)
