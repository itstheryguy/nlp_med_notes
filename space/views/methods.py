import altair as alt
import pandas as pd
import streamlit as st

from lib import INK, chart_config, color, css, label, load, mode

examples, notes, runs, summary = load()
css()


st.title("How it's measured")

st.markdown("""
### Data
[MTS-Dialog](https://github.com/abachaa/MTS-Dialog) (Ben Abacha et al., 2023, CC BY 4.0), via
[`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note):
1,301 short doctor-patient conversations, each paired with one note section. The split is frozen once
(80/10/10, stratified by section, repeated dialogues kept on one side): train 1,038 · val 131 · **test 132**.
Everything on this site is the test split.

### Training
Three models, one recipe each, **three seeds each**, so a model's result is a mean over repetitions rather than
one lucky run. Up to 10 epochs with early stopping on validation loss (keeping the best epoch), effective batch
16, label smoothing 0.1, bf16, section header as part of the input. Trained on AWS SageMaker (1× A10G),
orchestrated and tracked from Databricks (Delta tables, MLflow).

### Two-tier audit
ROUGE and BERTScore measure overlap with the clinician's wording. They can't tell a fluent, wrong note from a
faithful one, so every note is audited in two tiers:

| Tier | Runs on | Cost | Checks |
|---|---|---|---|
| **1. Deterministic** | every note | free | 4-field template · N/A agreement per field · grounding: words and numbers (including spoken numbers) that never appear in the dialogue |
| **2. LLM judge** | a fixed sample of 40 test examples, the same for every run | pay-per-token | every claim labeled supported / unsupported / contradicted / misplaced, with clinical severity; major omissions |

The judge is an open-weight model (`gpt-oss-120b`) served inside Databricks, so the audit data never leaves
the platform. The clinician notes go through both tiers too: that's the ceiling, and it shows how strict each
check is.
""")

st.subheader("Can the free check stand in for the judge?")
st.markdown(
    "In production you'd want the free check on every note and the judge on a sample. That only works if the free "
    "flag agrees with the judge. **Precision** is how often a flagged note really hallucinates; **recall** is how many "
    "of the judge's hallucinations the flag catches."
)
cal = pd.DataFrame(summary["calibration"])
cal = cal[cal["family"].isin(summary["family_order"])]
long = cal.melt(id_vars=["family"], value_vars=["precision", "recall"], var_name="measure", value_name="value")
long = long.assign(model=long["family"].map(label), c=long["family"].map(color))
ink = INK[mode()]
order = [label(f) for f in summary["family_order"]]
panels = []
for m in ("precision", "recall"):
    sub = long[long["measure"] == m]
    base = alt.Chart(sub).encode(y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)))
    panels.append(alt.layer(
        base.mark_bar(height=14, cornerRadiusEnd=4).encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5), title=None),
            color=alt.Color("c:N", scale=None), tooltip=[alt.Tooltip("model:N"), alt.Tooltip("value:Q", format=".0%")]),
        base.mark_text(align="left", dx=4, fontSize=11, color=ink["primary"]).encode(x="value:Q", text=alt.Text("value:Q", format=".0%")),
    ).properties(title=f"Grounding flag {m} vs the judge", height=110))
cols = st.columns(2, gap="large")
for col, panel in zip(cols, panels):
    col.altair_chart(chart_config(panel), width="stretch", theme=None)
st.markdown(
    "The answer depends on the model. The flag is a decent screen for BioBART and a weak one for the others, whose "
    "errors mostly reuse the dialogue's own words. **A free check has to be calibrated per model before it can "
    "replace a sampled audit.**"
)
with st.expander("Table view"):
    st.dataframe(cal.assign(model=cal["family"].map(label)).set_index("model").drop(columns="family"), width="stretch")

st.subheader("Seed spread")
st.markdown("The same recipe, three seeds. Spread this size is why single-run comparisons can mislead.")
show = runs.assign(model=runs["family"].map(label))[
    ["model", "seed", "rougeL", "faithful_rate", "hallucination_rate", "empty_rate", "best_epoch", "minutes"]]
st.dataframe(show.sort_values(["model", "seed"]).set_index(["model", "seed"]).style.format(
    {"rougeL": "{:.3f}", "faithful_rate": "{:.0%}", "hallucination_rate": "{:.0%}", "empty_rate": "{:.0%}",
     "best_epoch": "{:.0f}", "minutes": "{:.1f}"}), width="stretch")

st.subheader("Provenance")
cost = summary.get("training_cost") or {}
st.markdown(f"""
- Snapshot exported {summary['exported_at'][:10]} from Databricks job runs `{', '.join(summary['job_run_ids'])}`
  (`{summary['catalog']}.{summary['schema']}`).
- Judge: `{', '.join(summary['judge_endpoints'])}`, prompt version `{summary['judge_prompt_version']}`.
- Training: {cost.get('billable_minutes', 0):.0f} billable GPU-minutes on `{cost.get('instance_type')}`, about ${cost.get('estimated_usd', 0):.2f}.
- Origin: an MSDS 453 class project, rebuilt as a production pipeline.
""")
