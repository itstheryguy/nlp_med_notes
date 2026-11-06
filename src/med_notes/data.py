"""Load MTS-Dialog from Hugging Face, assign stable IDs and splits, and format model inputs.

Pure pandas, no Spark, so the same code runs in a Databricks notebook and in the local smoke test.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

DATASET_ID = "har1/MTS_Dialogue-Clinical_Note"

# The HF dataset concatenates two CSVs into one "train" split: MTS-Dialog's official training set
# (1,201 rows, IDs 0-1200) followed by its official validation set (100 rows, IDs 0-99).
OFFICIAL_TRAIN_ROWS = 1201

VAL_FRAC = 0.1
TEST_FRAC = 0.1
SPLIT_SEED = 2026


def load_raw() -> pd.DataFrame:
    """Rows exactly as the HF dataset returns them, plus a stable row-order `example_id`."""
    from datasets import load_dataset

    df = load_dataset(DATASET_ID, split="train").to_pandas()
    df.insert(0, "example_id", np.arange(len(df), dtype="int64"))
    df = df.rename(columns={"ID": "source_id"})
    df["source_split"] = np.where(df["example_id"] < OFFICIAL_TRAIN_ROWS, "official_train", "official_validation")
    return df


def assign_split(df: pd.DataFrame) -> pd.Series:
    """80/10/10 train/val/test, stratified by section header.

    Rows that share a dialogue always land in the same split, so the model never sees a test dialogue
    during training (3 dialogues appear twice with different section targets).
    """
    rng = np.random.default_rng(SPLIT_SEED)
    dialogue_key = df["dialogue"].map(lambda s: hashlib.sha1(s.strip().encode()).hexdigest())
    groups = df.assign(_key=dialogue_key).groupby("_key").agg(header=("section_header", "first")).reset_index()

    assignment: dict[str, str] = {}
    for _, g in groups.groupby("header"):
        keys = g["_key"].to_numpy()
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Tiny headers (3-4 rows) still put one example in test before val so every header is evaluated.
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for i, k in enumerate(keys):
            assignment[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return dialogue_key.map(assignment)


def build_silver(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean text and attach the frozen split."""
    df = raw.copy()
    for col in ("dialogue", "section_text"):
        # Targets mix \r\n and \n line endings across the two source CSVs.
        df[col] = df[col].str.replace("\r\n", "\n", regex=False).str.strip()
    df["split"] = assign_split(df)
    df["dialogue_chars"] = df["dialogue"].str.len()
    df["target_chars"] = df["section_text"].str.len()
    df["has_template"] = df["section_text"].str.contains("Plan of Action:", regex=False)
    return df


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}")
    return df[df["split"] == split].sort_values("example_id").reset_index(drop=True)


def format_source(dialogue: str, section_header: str, cfg) -> str:
    """Model input text. Used for training AND generation so the two can't drift apart."""
    text = dialogue
    if cfg.include_section_header:
        text = f"Section: {section_header}\n{text}"
    return f"{cfg.input_prefix}{text}"
