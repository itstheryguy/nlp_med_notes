"""Experiment configuration, loaded from configs/experiments.yml.

One YAML entry describes a model + training recipe. Its `seeds` list expands it into one run per seed
(`<name>_s<seed>`), so a model's result is a mean over repetitions, not a single lucky run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# Generation settings every run starts from; an experiment's `generation:` block overrides them.
# no_repeat_ngram_size=0 and min_length=0 matter: some checkpoints (bart-large-cnn) ship news-summary
# defaults that block repeated trigrams, which forbids writing "N/A." in more than one field.
DEFAULT_GENERATION = {
    "num_beams": 4,
    "max_new_tokens": 256,
    "early_stopping": True,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
}


@dataclass
class ExperimentConfig:
    name: str          # run name, e.g. biobart_s42
    family: str        # YAML entry name shared by all seeds, e.g. biobart
    model_name: str
    seed: int = 42
    groups: list[str] = field(default_factory=list)
    description: str = ""

    # Input formatting. format_source() builds inputs for training AND generation, so they can't drift.
    input_prefix: str = ""
    include_section_header: bool = True

    # Optimization. Training runs up to max_epochs and keeps the epoch with the lowest val loss.
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_epochs: int = 10
    early_stopping_patience: int = 2
    batch_size: int = 8
    grad_accum: int = 2
    lr_scheduler: str = "linear"
    label_smoothing: float = 0.1
    precision: str = "auto"  # auto = bf16 if the GPU supports it, else fp16; or "bf16" | "fp16" | "fp32"

    max_source_length: int = 1024
    max_target_length: int = 256

    generation: dict = field(default_factory=dict)
    save_model: bool = True   # keep the best seed of each family in the models volume
    est_minutes: float = 15   # budget guard's estimate before the first run of this family finishes

    def generation_kwargs(self) -> dict:
        return {**DEFAULT_GENERATION, **self.generation}

    def to_flat_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["groups"] = ",".join(self.groups)
        d["generation"] = str(self.generation_kwargs())
        return d

    def to_dict(self) -> dict:
        """Lossless dict for plan.json; from_dict() reverses it."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)


def load_experiments(path: str | Path) -> dict[str, ExperimentConfig]:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {}) or {}
    known = {f.name for f in dataclasses.fields(ExperimentConfig)} - {"name", "family", "seed"}
    out: dict[str, ExperimentConfig] = {}
    for entry in raw["experiments"]:
        merged = {**defaults, **entry}
        family = merged.pop("name")
        seeds = merged.pop("seeds", [42])
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"Experiment {family!r} has unknown keys: {sorted(unknown)}")
        for seed in seeds:
            cfg = ExperimentConfig(name=f"{family}_s{seed}", family=family, seed=seed, **merged)
            if cfg.name in out:
                raise ValueError(f"Duplicate run name {cfg.name!r}")
            out[cfg.name] = cfg
    return out


def select_experiments(experiments: dict[str, ExperimentConfig], selector: str) -> list[str]:
    """Resolve the `experiments` job parameter into run names.

    "group:main" | "all" | family names ("biobart") | run names ("biobart_s42"), comma-separated.
    """
    selector = selector.strip()
    if selector == "all":
        return list(experiments)
    names: list[str] = []
    for part in [s.strip() for s in selector.split(",") if s.strip()]:
        if part.startswith("group:"):
            group = part.split(":", 1)[1]
            match = [n for n, c in experiments.items() if group in c.groups]
        else:
            match = [n for n, c in experiments.items() if part in (n, c.family)]
        if not match:
            raise ValueError(f"{part!r} matched no experiments. Runs: {list(experiments)}")
        names += [m for m in match if m not in names]
    return names
