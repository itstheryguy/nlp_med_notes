"""Local CPU smoke test of everything except Spark and the live judge endpoint.

    pip install torch transformers datasets accelerate rouge_score bert_score sentencepiece pyyaml pandas pyarrow databricks-sdk
    python tests/smoke_test.py

Takes a few minutes, mostly downloads. Uses tiny random models, so metrics are meaningless; this checks the code paths.
"""

import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from med_notes.audit import parse_template, select_audit_sample, tier1, unsupported_terms  # noqa: E402
from med_notes.config import load_experiments, select_experiments  # noqa: E402
from med_notes.data import build_silver, get_split, load_raw  # noqa: E402
from med_notes.ingest import load_run_dir, reference_tier1  # noqa: E402
from med_notes.judge import Judge, _text_from_content, parse_verdict, results_frame, summarize_verdict  # noqa: E402
from med_notes.runner import run_plan  # noqa: E402


def check_configs():
    exps = load_experiments(ROOT / "configs" / "experiments.yml")
    main = select_experiments(exps, "group:main")
    assert len(main) == 9 and {exps[n].family for n in main} == {"biobart", "bart_large_cnn", "t5_base"}, main
    assert select_experiments(exps, "biobart") == ["biobart_s13", "biobart_s42", "biobart_s2026"]
    assert select_experiments(exps, "group:pilot") == ["pilot_biobart_s42"]
    assert exps["t5_base_s42"].input_prefix == "summarize: " and exps["t5_base_s42"].learning_rate == 3e-4
    assert exps["bart_large_cnn_s13"].generation_kwargs()["no_repeat_ngram_size"] == 0
    c = exps["biobart_s42"]
    assert type(c).from_dict(json.loads(json.dumps(c.to_dict()))) == c  # plan.json round trip
    print("configs ok:", len(exps), "runs")
    return exps


def check_data():
    silver = build_silver(load_raw())
    counts = silver["split"].value_counts().to_dict()
    assert counts == {"train": 1038, "val": 131, "test": 132}, counts
    assert (silver.groupby("dialogue")["split"].nunique() == 1).all(), "dialogue leaks across splits"
    print("data ok:", counts)
    return silver


def check_audit(silver):
    ok = "Symptoms: cough\nDiagnosis: N/A\nHistory of Patient: smoker\nPlan of Action: N/A."
    assert parse_template(ok) == {"Symptoms": "cough", "Diagnosis": "N/A", "History of Patient": "smoker", "Plan of Action": "N/A."}
    assert parse_template(ok + "\nPatient: extra text") is None          # content outside the template
    assert parse_template("Diagnosis: x\nSymptoms: y\nHistory of Patient: z\nPlan of Action: w") is None  # order
    missing, _ = unsupported_terms("Allergy: penicillin (hives), sulfa. Dose 500 mg.",
                                   "Doctor: Any allergies? Patient: Penicillin gives me hives. I take 250 a day.")
    assert "sulfa" in missing and "500" in missing and "penicillin" not in missing and "hives" not in missing, missing

    ref = reference_tier1(silver)
    assert 0.9 < ref["template_ok"].mean() < 0.96, ref["template_ok"].mean()  # ~93% of references use the template
    sample = select_audit_sample(silver, 40)
    assert sample == select_audit_sample(silver, 40) and len(set(sample)) == 40
    assert set(silver.set_index("example_id").loc[sample, "split"]) == {"test"}
    print(f"audit ok: reference template rate {ref['template_ok'].mean():.3f}, "
          f"reference grounding-flag rate {ref['grounding_flag'].mean():.3f}")


def check_judge():
    good = ('Here you go: {"claims": [{"field": "Diagnosis", "claim": "asthma", "label": "supported", "severity": "minor"},'
            '{"field": "Plan of Action", "claim": "albuterol 5 mg", "label": "unsupported", "severity": "major"}],'
            '"omissions": [], "faithfulness_score": 2, "rationale": "invented dose"}')
    s = summarize_verdict(parse_verdict(good))
    assert s["n_claims"] == 2 and s["n_major_errors"] == 1 and s["hallucinated"] and not s["faithful"]
    assert _text_from_content([{"type": "reasoning", "summary": []}, {"type": "text", "text": "{}"}]) == "{}"

    class FakeServing:  # first reply is unparseable, second (list-shaped, like reasoning models) is valid
        def __init__(self):
            self.calls = 0

        def query(self, **kw):
            self.calls += 1
            content = "sorry" if self.calls == 1 else [{"type": "reasoning"}, {"type": "text", "text": good}]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                                   usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))

    import med_notes.judge as judge_mod
    judge_mod.time.sleep = lambda s: None
    j = Judge(client=SimpleNamespace(serving_endpoints=FakeServing()))
    r = j.judge_one("PLAN", "dialogue", "reference", "note")
    assert r["error"] is None and r["attempts"] == 2 and r["prompt_tokens"] == 200, r
    frame = results_frame([r, {"error": "boom", "attempts": 3, "prompt_tokens": 0, "completion_tokens": 0}])
    assert list(frame["faithful"]) == [False, None] and frame["n_claims"].dtype == "float64"
    print("judge ok: parse, retry on bad JSON, list-shaped content, fixed result schema")


def check_runner_and_ingest(exps, silver):
    small = pd.concat([get_split(silver, "train").head(16), get_split(silver, "val").head(6),
                       get_split(silver, "test").head(6)], ignore_index=True)
    tiny = dict(max_source_length=96, max_target_length=32, max_epochs=2, early_stopping_patience=1,
                batch_size=4, grad_accum=1, generation={"max_new_tokens": 16, "num_beams": 2})
    runs = [
        dataclasses.replace(exps["biobart_s13"], model_name="hf-internal-testing/tiny-random-BartForConditionalGeneration",
                            est_minutes=1, **tiny),
        dataclasses.replace(exps["biobart_s42"], model_name="hf-internal-testing/tiny-random-BartForConditionalGeneration",
                            est_minutes=1, **tiny),
        dataclasses.replace(exps["t5_base_s13"], model_name="hf-internal-testing/tiny-random-T5ForConditionalGeneration",
                            est_minutes=1, **tiny),
        dataclasses.replace(exps["t5_base_s42"], model_name="does-not-exist/model", est_minutes=1, **tiny),  # fails
        dataclasses.replace(exps["bart_large_cnn_s13"], est_minutes=10_000, **tiny),                    # over budget
    ]
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, models_dir = os.path.join(tmp, "run"), os.path.join(tmp, "models")
        os.makedirs(run_dir)
        plan = {"job_run_id": "test", "experiments": [c.to_dict() for c in runs], "eval_splits": ["val", "test"],
                "max_gpu_minutes": 60, "models_dir": models_dir}
        status = run_plan(json.loads(json.dumps(plan)), small, run_dir, bertscore_model="distilbert-base-uncased")
        got = {s["experiment"]: s["status"] for s in status}
        assert got == {"biobart_s13": "succeeded", "biobart_s42": "succeeded", "t5_base_s13": "succeeded",
                       "t5_base_s42": "failed", "bart_large_cnn_s13": "skipped_budget"}, got
        assert os.path.exists(os.path.join(models_dir, "biobart", "run_info.json"))
        saved = json.load(open(os.path.join(models_dir, "biobart", "run_info.json")))
        best = min((s for s in status if s["family"] == "biobart"), key=lambda s: s["best_val_loss"])
        assert saved["experiment"] == best["experiment"], "models volume should hold the best seed"

        frames = load_run_dir(run_dir, silver, "test")
        assert len(frames["runs"]) == 5 and len(frames["predictions"]) == 3 * 12
        assert {"template_ok", "unsupported_terms", "grounding_flag", "val_rougeL", "test_rougeL"} <= (
            set(frames["predictions"].columns) | set(frames["runs"].columns))
    print("runner + ingest ok:", got)


if __name__ == "__main__":
    exps = check_configs()
    silver = check_data()
    check_audit(silver)
    check_judge()
    check_runner_and_ingest(exps, silver)
    print("SMOKE TEST PASSED")
