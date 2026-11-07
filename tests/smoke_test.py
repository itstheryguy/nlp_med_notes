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
from med_notes.remote import LocalStore, run_remote  # noqa: E402
from med_notes.runner import run_plan  # noqa: E402
from med_notes.judge import canonical_note, score_from_labels, verdict_key  # noqa: E402
from med_notes.judging import agreement, assemble, build_items, cohen_kappa, plan_calls  # noqa: E402
from med_notes.export import build_snapshot  # noqa: E402
from med_notes.sagemaker_job import build_sourcedir, job_name, summarize_job, training_job_request  # noqa: E402


def check_configs():
    exps = load_experiments(ROOT / "configs" / "experiments.yml")
    main = select_experiments(exps, "group:main")
    assert len(main) == 12 and {exps[n].family for n in main} == {"biobart", "t5_base", "pegasus_large", "biobart_large"}, main
    assert {exps[n].family for n in select_experiments(exps, "group:large")} == {"pegasus_large", "biobart_large"}
    assert {exps[n].family for n in select_experiments(exps, "group:archive")} == {"bart_large_cnn", "bart_large_cnn_minlen"}
    assert select_experiments(exps, "biobart") == ["biobart_s13", "biobart_s42", "biobart_s2026"]
    assert select_experiments(exps, "group:pilot") == ["pilot_biobart_s42"]
    assert exps["t5_base_s42"].input_prefix == "summarize: " and exps["t5_base_s42"].learning_rate == 3e-4
    assert exps["bart_large_cnn_s13"].generation_kwargs()["no_repeat_ngram_size"] == 0
    c = exps["biobart_s42"]
    assert type(c).from_dict(json.loads(json.dumps(c.to_dict()))) == c  # plan.json round trip
    assert select_experiments(exps, "group:bart_followup") == [
        "bart_large_cnn_minlen_s13", "bart_large_cnn_minlen_s42", "bart_large_cnn_minlen_s2026"]
    assert exps["bart_large_cnn_minlen_s42"].generation_kwargs()["min_length"] == 10
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
    assert parse_template(ok.replace("\n", " ")) == parse_template(ok)  # T5 writes one line (no newline token)
    assert parse_template("Note: " + ok) is None                          # content before the template
    # Trailing text after the last label belongs to Plan of Action (labels, not lines, delimit fields).
    assert parse_template(ok + " Follow up in 2 weeks.")["Plan of Action"] == "N/A. Follow up in 2 weeks."
    assert parse_template("Diagnosis: x\nSymptoms: y\nHistory of Patient: z\nPlan of Action: w") is None  # order
    missing, _ = unsupported_terms("Allergy: penicillin (hives), sulfa. Dose 500 mg.",
                                   "Doctor: Any allergies? Patient: Penicillin gives me hives. I take 250 a day.")
    assert "sulfa" in missing and "500" in missing and "penicillin" not in missing and "hives" not in missing, missing
    # Spoken numbers (from the pilot): the model wrote 1961 / 135 for "nineteen sixty eight" / "one hundred and eighty three".
    dia = "Patient: I quit in nineteen sixty eight. I am at one hundred and eighty three pounds. I was five foot ten."
    assert {"1961", "135"} <= set(unsupported_terms("Quit in 1961, weighing 135 pounds", dia)[0])
    assert not [t for t in unsupported_terms("Quit in 1968, 183 pounds, 5 feet 10", dia)[0] if t[0].isdigit()]

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
                "max_train_minutes": 60, "models_dir": models_dir}
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


def check_remote(exps, silver):
    """SageMaker path without AWS: a LocalStore stands in for the S3 prefix behind the volume."""
    import io
    import tarfile

    small = pd.concat([get_split(silver, "train").head(16), get_split(silver, "val").head(6),
                       get_split(silver, "test").head(6)], ignore_index=True)
    tiny = dict(max_source_length=96, max_target_length=32, max_epochs=1, early_stopping_patience=1,
                batch_size=4, grad_accum=1, generation={"max_new_tokens": 8, "num_beams": 2}, est_minutes=1,
                model_name="hf-internal-testing/tiny-random-BartForConditionalGeneration")
    runs = [dataclasses.replace(exps[n], **tiny) for n in ("biobart_s13", "biobart_s42")]
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalStore(os.path.join(tmp, "s3"))
        plan = {"job_run_id": "r1", "experiments": [c.to_dict() for c in runs], "eval_splits": ["val"],
                "max_train_minutes": 60, "models_dir": "/Volumes/c/s/sagemaker/models"}
        store.put("runs/r1/plan.json", json.dumps(plan).encode())
        buf = io.BytesIO(); small.to_parquet(buf); store.put("runs/r1/splits.parquet", buf.getvalue())
        # A previous (interrupted) session already finished biobart_s13.
        store.put("runs/r1/status.json", json.dumps({"runs": [{"experiment": "biobart_s13", "family": "biobart",
                  "seed": 13, "status": "succeeded", "minutes": 1.0, "best_val_loss": 99.0, "model_path": None}]}).encode())

        status = run_remote(store, "runs/r1", "models", "/Volumes/c/s/sagemaker", os.path.join(tmp, "work"),
                            log=lambda m: None, bertscore_model="distilbert-base-uncased")
        assert [r["experiment"] for r in status] == ["biobart_s13", "biobart_s42"], status
        published = json.loads(store.get("runs/r1/status.json"))["runs"]
        assert [r["status"] for r in published] == ["succeeded", "succeeded"]
        assert published[1]["model_path"] == "/Volumes/c/s/sagemaker/models/biobart"  # Databricks' view of the path
        assert store.get("runs/r1/biobart_s42/predictions.parquet") is not None
        assert store.get("models/biobart/run_info.json") is not None
        assert store.get("runs/r1/biobart_s13/predictions.parquet") is None  # not re-run

    tar = tarfile.open(fileobj=io.BytesIO(build_sourcedir(str(ROOT))))
    names = tar.getnames()
    assert {"entry.py", "requirements.txt", "med_notes/remote.py"} <= set(names) and not any("__pycache__" in n for n in names)
    name = job_name("nlp-med-notes", "1234567890123")
    assert len(name) <= 63 and name.replace("-", "").isalnum()
    req = training_job_request(name=name, image="img", role_arn="arn:role", instance_type="ml.g5.xlarge",
                               s3_root="s3://b/p", volume_root="/Volumes/c/s/sagemaker", run_rel="runs/1",
                               models_rel="models", sourcedir_s3="s3://b/p/runs/1/code/sourcedir.tar.gz",
                               max_runtime_s=7200, use_spot=True, region="us-east-2", tags={"k": 1})
    assert req["StoppingCondition"] == {"MaxRuntimeInSeconds": 7200, "MaxWaitTimeInSeconds": 10800}
    assert all(isinstance(v, str) for v in [*req["HyperParameters"].values(), *req["Environment"].values()])
    cost = summarize_job({"TrainingJobName": name, "TrainingJobStatus": "Completed", "EnableManagedSpotTraining": True,
                          "ResourceConfig": {"InstanceType": "ml.g5.xlarge"}, "TrainingTimeInSeconds": 1000,
                          "BillableTimeInSeconds": 300}, 1.2575)
    assert cost["spot_savings_pct"] == 70.0 and cost["estimated_usd"] == round(300 / 3600 * 1.2575, 4)
    print("remote ok: resume skips finished runs, per-run upload, model path mapping, tarball, job request, cost")


def check_judging():
    # #346: the same all-N/A note, one-line (T5/PEGASUS) vs multi-line (BioBART), must look identical to the judge.
    one = "Symptoms: N/A Diagnosis: N/A History of Patient: N/A Plan of Action: N/A"
    multi = "Symptoms: N/A\nDiagnosis: N/A\nHistory of Patient:  N/A\nPlan of Action: N/A"
    assert canonical_note(one) == canonical_note(multi)
    assert verdict_key(346, canonical_note(one), "j") == verdict_key(346, canonical_note(multi), "j")
    assert verdict_key(346, canonical_note(one), "j") != verdict_key(346, canonical_note(one), "j", "v1")
    assert [score_from_labels(m, n) for m, n in [(0, 0), (0, 3), (1, 0), (2, 0), (4, 0)]] == [5, 4, 3, 2, 1]
    v = parse_verdict('{"claims": [{"field": "History of Patient", "claim": "No history to report (N/A)", '
                      '"label": "contradicted", "severity": "major"}], '
                      '"omissions": [{"fact": "denies HTN, DM, CAD, cancer", "severity": "major"}], "rationale": "x"}')
    s = summarize_verdict(v)
    assert v["claims"] == [] and s["faithfulness_score"] == 3 and s["faithful"] is False and s["hallucinated"] is False

    sample = pd.DataFrame({"example_id": [1, 2], "section_header": ["PMH", "CC"], "dialogue": ["d1", "d2"],
                           "reference": ["Symptoms: a\nDiagnosis: N/A\nHistory of Patient: N/A\nPlan of Action: N/A", "r2"]})
    model_notes = pd.DataFrame({"experiment": ["m_s1", "m_s2", "n_s1"], "family": ["m", "m", "n"], "seed": [1, 2, 1],
                                "example_id": [1, 1, 1], "note": [one, multi, "other"], "grounding_flag": [False, False, True]})
    items = build_items(sample, model_notes)
    assert len(items) == 5  # 2 references + 3 notes for audited example 1
    keyed, calls = plan_calls(items, ["j1", "j2"], cached_keys=set())
    assert len(keyed) == 10 and len(calls) == 8  # the two identical notes share one call per judge
    cached_key = calls["verdict_key"].iloc[0]
    _, calls2 = plan_calls(items, ["j1", "j2"], cached_keys={cached_key})
    assert len(calls2) == 7 and cached_key not in set(calls2["verdict_key"])
    verdicts = calls.assign(faithful=True, hallucinated=False, faithfulness_score=5.0, verdict_json="{}", error=None,
                            **{c: 0.0 for c in ["n_claims", "n_supported", "n_unsupported", "n_contradicted", "n_misplaced",
                                                "n_major_errors", "n_omissions", "n_major_omissions", "attempts",
                                                "prompt_tokens", "completion_tokens"]})
    out = assemble(keyed, verdicts, new_keys=set(calls["verdict_key"]) - {cached_key})
    assert out["faithful"].notna().all() and out["cached"].sum() == 1
    a, b = pd.Series([True, True, False, False]), pd.Series([True, False, False, False])
    assert agreement(a, b)["agreement"] == 0.75 and abs(cohen_kappa(a, b) - 0.5) < 1e-9
    print("judging ok: canonical notes, computed score, N/A rule, cache plan (dedupe + reuse), kappa")


def check_export_two_judges(silver):
    test_ids = get_split(silver, "test")["example_id"].head(4).tolist()
    preds = pd.DataFrame([{"experiment": f"m_s{s}", "family": "m", "seed": s, "split": "test", "example_id": e,
                           "prediction": "Symptoms: x\nDiagnosis: N/A\nHistory of Patient: N/A\nPlan of Action: N/A",
                           "rougeL": .5, "bertscore_f1": .9, "template_ok": True, "na_agreement": 1.0,
                           "unsupported_terms": [], "unsupported_numbers": 0, "grounding_flag": False,
                           "empty_prediction": False} for s in (1, 2) for e in test_ids])
    def judge_df(faithful):
        rows = [{"experiment": e, "example_id": x, "faithful": f, "hallucinated": False, "n_claims": 1, "n_unsupported": 0,
                 "n_contradicted": 0, "n_misplaced": 0, "n_major_errors": 0, "n_major_omissions": 0 if f else 1,
                 "faithfulness_score": 5 if f else 3, "verdict_json": '{"claims": [], "omissions": []}'}
                for e in ["m_s1", "m_s2", "reference"] for x, f in zip(test_ids, faithful)]
        return pd.DataFrame(rows)
    runs = pd.DataFrame({"experiment": ["m_s1", "m_s2"], "minutes": [1.0, 1.0], "best_epoch": [3, 3], "epochs_run": [5, 5],
                         "num_params_m": [100.0, 100.0], "model_name": ["m", "m"]})
    cons = pd.DataFrame({"endpoint": ["j1", "j1"], "faithful": [True, False], "faithful_repeat": [True, True],
                         "faithfulness_score": [5, 3], "score_repeat": [5, 5]})
    snap = build_snapshot(silver=silver, audit=pd.DataFrame({"example_id": test_ids, "audit_rank": range(4)}),
                          predictions=preds, judge=judge_df([True, True, False, False]),
                          judge2=judge_df([True, False, False, False]), consistency=cons,
                          reference_checks=reference_tier1(silver), runs=runs, training_jobs=None, meta={},
                          judge_names={"primary": "j1", "secondary": "j2"})
    j = snap["summary"]["judges"]
    assert j["cross_overall"]["agreement"] == 0.75 and j["self_consistency"][0]["faithful_agreement"] == 0.5
    assert "j2_faithful" in snap["notes"] and "ref_j2_faithful" in snap["examples"]
    print("export ok: second judge columns, cross-judge and self-consistency summaries")


if __name__ == "__main__":
    exps = check_configs()
    silver = check_data()
    check_audit(silver)
    check_judge()
    check_runner_and_ingest(exps, silver)
    check_remote(exps, silver)
    check_judging()
    check_export_two_judges(silver)
    print("SMOKE TEST PASSED")
