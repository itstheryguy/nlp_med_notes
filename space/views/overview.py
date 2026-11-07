import streamlit as st

from lib import chart_config, css, families_df, label, load, rate_chart

examples, notes, runs, summary = load()
css()
fam = families_df(summary)
ref = summary["reference"]

st.title("Where clinical-note models go wrong")
st.markdown(
    "Three fine-tuned summarizers turn a doctor-patient conversation into one section of a clinical note "
    "(**Symptoms / Diagnosis / History of Patient / Plan of Action**). Overlap scores like ROUGE say they're "
    "close to the clinician's note. An audit of what each note *claims* says otherwise. This Space shows "
    "both, and the specific traps behind the gap."
)

best = fam.sort_values("faithful_rate", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Faithful notes, best model ({label(best['family'])})", f"{best['faithful_rate']:.0%}",
          help="Share of audited test notes where the LLM judge found no major error and no major omission. "
               "Mean over 3 training seeds.")
c2.metric("Faithful notes, clinician reference", f"{ref['faithful_rate']:.0%}",
          help="The same audit run on the clinicians' own notes: the realistic ceiling.")
c3.metric("Training runs", f"{len(runs)}", help="Each model trained 3 times with different seeds; results are means.")
cost = summary.get("training_cost") or {}
if cost.get("estimated_usd") is not None:
    c4.metric("GPU cost, all runs", f"${cost['estimated_usd']:.2f}",
              help=f"{cost['billable_minutes']:.0f} billable minutes on {cost['instance_type']} (AWS SageMaker, on-demand).")

st.subheader("Fluent is not faithful")
st.caption("Dot = mean over 3 seeds · bar = ± 1 sd · faint rings = individual seeds · gray line = clinician reference. "
           "Hover for values; the table below has every number.")
left, right = st.columns(2)
with left:
    st.altair_chart(chart_config(rate_chart(fam, "rougeL", "ROUGE-L vs the clinician note (overlap)", seed_dots=runs,
                                            fmt=".2f")), width="stretch", theme=None)
    st.altair_chart(chart_config(rate_chart(fam, "template_rate", "Follows the 4-field template",
                                            reference=ref["template_rate"], seed_dots=runs)),
                    width="stretch", theme=None)
with right:
    st.altair_chart(chart_config(rate_chart(fam, "faithful_rate", "Faithful to the dialogue (LLM judge)",
                                            reference=ref["faithful_rate"], seed_dots=runs)),
                    width="stretch", theme=None)
    st.altair_chart(chart_config(rate_chart(fam, "hallucination_rate", "Contains a hallucinated claim (LLM judge)",
                                            reference=ref["hallucination_rate"], seed_dots=runs)),
                    width="stretch", theme=None)

st.subheader("What the numbers say")
by = fam.set_index("family")
lines = []
if {"t5_base", "biobart"} <= set(by.index):
    t, b = by.loc["t5_base"], by.loc["biobart"]
    lines.append(f"**T5-base and BioBART are effectively tied.** ROUGE-L {t['rougeL']:.3f} vs {b['rougeL']:.3f}; "
                 f"faithful {t['faithful_rate']:.0%} vs {b['faithful_rate']:.0%}, a gap about the size of the seed "
                 f"spread (± {max(t['faithful_rate_sd'], b['faithful_rate_sd']):.0%}).")
if "bart_large_cnn" in by.index:
    g = by.loc["bart_large_cnn"]
    lines.append(f"**BART-large-CNN looks safest and isn't.** Only {g['hallucination_rate']:.0%} of its notes "
                 f"hallucinate, because {g['empty_rate']:.0%} of its notes are empty. It also swings by "
                 f"± {g['rougeL_sd']:.2f} ROUGE-L between seeds. See **Model traps**.")
lines.append(f"**Even the clinicians' notes aren't 100%.** The judge rates {ref['faithful_rate']:.0%} faithful: "
             "references sometimes include facts the dialogue excerpt doesn't contain. Models are judged against "
             "that ceiling, not against perfection.")
if "biobart" in by.index:
    lines.append(f"**Seeds matter more than the class-project tweaks did.** BioBART's ROUGE-L varies by only "
                 f"± {by.loc['biobart', 'rougeL_sd']:.3f} across seeds, smaller than the learning-rate and warmup "
                 "differences the original project compared. Those differences were noise.")
st.markdown("\n".join(f"- {line}" for line in lines))

with st.expander("Table view: every metric, mean ± sd over seeds"):
    cols = ["rougeL", "bertscore_f1", "template_rate", "na_agreement", "empty_rate", "faithful_rate",
            "hallucination_rate", "major_errors_per_note", "major_omissions_per_note", "judge_score", "minutes"]
    table = fam[["family", "model_name", "seeds"]].assign(model=fam["family"].map(label))
    for c in cols:
        table[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(fam[c], fam[f"{c}_sd"])]
    st.dataframe(table.drop(columns="family").set_index("model"), width="stretch")
    st.caption(f"Reference (clinician notes): template {ref['template_rate']:.3f}, faithful {ref['faithful_rate']:.3f}, "
               f"hallucination {ref['hallucination_rate']:.3f}, judge score {ref['judge_score']:.2f}. "
               f"{summary['n_test_examples']} test examples; the judge audits a fixed {summary['n_audit_examples']} of them.")
