"""Batched generation and per-example scoring (ROUGE + BERTScore)."""

from __future__ import annotations

import time

import pandas as pd

from .config import ExperimentConfig
from .data import format_source

METRICS = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]


def generate(model, tokenizer, df: pd.DataFrame, cfg: ExperimentConfig, batch_size: int = 16) -> tuple[list[str], float]:
    """Generate one note per row. Returns (predictions in df order, seconds per example).

    Rows are generated longest-first in batches, so each batch pads to similar lengths.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    sources = [format_source(d, h, cfg) for d, h in zip(df["dialogue"], df["section_header"])]
    order = sorted(range(len(sources)), key=lambda i: -len(sources[i]))
    preds: list[str | None] = [None] * len(sources)
    gen_kwargs = cfg.generation_kwargs()

    start = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tokenizer(
            [sources[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_source_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        for i, text in zip(idx, tokenizer.batch_decode(out, skip_special_tokens=True)):
            preds[i] = text.strip()
    seconds_per_example = (time.time() - start) / max(len(sources), 1)
    return preds, seconds_per_example


_bert_scorers: dict = {}


def _bert_scorer(model_type: str | None):
    """Load each BERTScore model once per process (roberta-large is ~1.4 GB)."""
    key = model_type or "default"
    if key not in _bert_scorers:
        import bert_score

        kwargs = {"model_type": model_type} if model_type else {"lang": "en"}
        _bert_scorers[key] = bert_score.BERTScorer(**kwargs)
    return _bert_scorers[key]


def score(predictions: list[str], references: list[str], bertscore_model: str | None = None,
          batch_size: int = 32) -> pd.DataFrame:
    """Per-example ROUGE-1/2/L (stemmed) and BERTScore F1 (roberta-large by default).

    Kept for continuity and as a cheap regression signal. They barely separate models on this task;
    the audit metrics (audit.py, judge.py) are what measure note quality.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)  # (target, prediction)
        rows.append({k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")})
    df = pd.DataFrame(rows)
    # An empty note scores 0. bert_score crashes on empty strings under transformers 5, and a
    # model emitting nothing is a real failure mode worth counting, not skipping.
    df["bertscore_f1"] = 0.0
    keep = [i for i, p in enumerate(predictions) if p.strip()]
    if keep:
        _, _, f1 = _bert_scorer(bertscore_model).score(
            [predictions[i] for i in keep], [references[i] for i in keep], batch_size=batch_size)
        df.loc[keep, "bertscore_f1"] = f1.tolist()
    df["empty_prediction"] = [not p.strip() for p in predictions]
    return df


def summarize(per_example: pd.DataFrame) -> dict:
    return {m: round(float(per_example[m].mean()), 4) for m in METRICS}
