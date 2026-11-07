import altair as alt
import pandas as pd
import streamlit as st

from lib import INK, chart_config, color, css, judge_name, label, load, mode

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
**Four models, one recipe each, three seeds each**, so a model's result is a mean over repetitions rather than one
lucky run: BioBART-base (166M), T5-base (223M), BioBART-large (406M), and PEGASUS-large (568M, the pretrained-only
checkpoint). Early stopping on validation loss (keeping the best epoch) with a 10-epoch ceiling; PEGASUS-large gets
25, because at 10 every seed was still improving (it converges around epoch 15). Effective batch 16, label smoothing
0.1, bf16, section header as part of the input. Trained on AWS SageMaker (1× A10G, on-demand), orchestrated and
tracked from Databricks (Delta tables, MLflow).

BART-large-CNN was tried first and archived: it's tuned to news summarization, and some training seeds collapsed
into repeating the start token (`<s><s><s>…`), which decodes to an empty note.

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
    ).properties(title=f"Grounding flag {m} vs the judge", height=34 * len(sub) + 30))
cols = st.columns(2, gap="large")
for col, panel in zip(cols, panels):
    col.altair_chart(chart_config(panel), width="stretch", theme=None)
ok = cal.dropna(subset=["precision", "recall"])
if len(ok):
    st.markdown(
        f"Precision runs {ok['precision'].min():.0%}–{ok['precision'].max():.0%} and recall "
        f"{ok['recall'].min():.0%}–{ok['recall'].max():.0%}: a flagged note is usually a real problem, but the free check "
        "misses many of the judge's findings. Those are mostly wrong claims made from the dialogue's own words, which "
        "a word-overlap check can't see. **A free check works as a screen, not as a replacement for a sampled audit.**"
    )
with st.expander("Table view"):
    st.dataframe(cal.assign(model=cal["family"].map(label)).set_index("model").drop(columns="family"), width="stretch")

judges = summary.get("judges") or {}
if judges.get("secondary"):
    j1, j2 = judge_name(judges.get("primary")), judge_name(judges["secondary"])
    st.subheader("Can the judge be trusted?")
    st.markdown(
        f"Every faithfulness number on this site comes from an LLM judge, so the judge needs auditing too. Two checks: "
        f"**{j1}** (the primary judge) re-judges a sample of notes it has already judged, and a second, independent "
        f"judge, **{j2}**, audits every note. Both judges only *label* claims and omissions; the score and the "
        "faithful/not decision are computed from those labels by a fixed rule, and identical notes are judged once "
        "and cached, so the same note can never get two different verdicts from the same judge."
    )

    def kappa_words(k):
        if k is None:
            return "n/a"
        return ("almost perfect" if k > .8 else "substantial" if k > .6 else "moderate" if k > .4
                else "fair" if k > .2 else "slight" if k > 0 else "no better than chance")

    c1, c2, c3 = st.columns(3)
    sc = {d["endpoint"]: d for d in judges.get("self_consistency", [])}
    if judges.get("primary") in sc:
        c1.metric(f"{j1} agrees with itself", f"{sc[judges['primary']]['faithful_agreement']:.0%}",
                  help=f"Faithful/not on {sc[judges['primary']]['n']} notes judged twice (cache bypassed for the repeat).")
    if judges["secondary"] in sc:
        c2.metric(f"{j2} agrees with itself", f"{sc[judges['secondary']]['faithful_agreement']:.0%}",
                  help=f"Faithful/not on {sc[judges['secondary']]['n']} notes judged twice.")
    co = judges.get("cross_overall") or {}
    if co.get("agreement") is not None:
        c3.metric("The two judges agree", f"{co['agreement']:.0%}",
                  help=f"Faithful/not on {co['n']} model notes. Cohen's kappa {co['kappa']:.2f} "
                       f"({kappa_words(co['kappa'])} agreement beyond chance).")

    if co.get("kappa") is not None:
        st.caption(f"Cohen's kappa between the judges: {co['kappa']:.2f}, {kappa_words(co['kappa'])} agreement beyond "
                   "chance. Raw agreement looks higher because most notes are judged not faithful, so two judges "
                   "would often agree by chance; kappa removes that.")

    by = pd.DataFrame(judges.get("cross_by_family", []))
    if len(by):  # same row order as the rest of the site
        by = by.set_index("family").loc[[f for f in summary["family_order"] if f in set(by["family"])]].reset_index()
    ref = judges.get("cross_reference") or {}
    if len(by):
        rows = by[["family", "primary_faithful", "secondary_faithful"]].copy()
        if ref.get("primary_faithful") is not None:
            rows = pd.concat([rows, pd.DataFrame([{"family": "reference", "primary_faithful": ref["primary_faithful"],
                                                   "secondary_faithful": ref["secondary_faithful"]}])], ignore_index=True)
        rows["model"] = [("Clinician reference" if f == "reference" else label(f)) for f in rows["family"]]
        rows["c"] = [(INK[mode()]["reference"] if f == "reference" else color(f)) for f in rows["family"]]
        long = rows.melt(id_vars=["family", "model", "c"], value_vars=["primary_faithful", "secondary_faithful"],
                         var_name="judge", value_name="rate")
        long["judge"] = long["judge"].map({"primary_faithful": j1, "secondary_faithful": j2})
        order = [m for m in rows["model"]]
        ink = INK[mode()]
        base = alt.Chart(long).encode(y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)))
        link = alt.Chart(rows).mark_rule(color=ink["grid"], strokeWidth=2).encode(
            y=alt.Y("model:N", sort=order), x="primary_faithful:Q", x2="secondary_faithful:Q")
        pts = base.mark_point(size=150, strokeWidth=2.5).encode(
            x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5), title=None),
            color=alt.Color("c:N", scale=None),
            shape=alt.Shape("judge:N", scale=alt.Scale(domain=[j1, j2], range=["circle", "diamond"]),
                            legend=alt.Legend(title=None, orient="top")),
            fill=alt.condition(alt.datum.judge == j1, alt.Color("c:N", scale=None), alt.value("transparent")),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("judge:N"), alt.Tooltip("rate:Q", format=".0%", title="Faithful")])
        st.altair_chart(chart_config(alt.layer(link, pts).properties(
            title=alt.TitleParams("Faithful rate under each judge", subtitle=f"filled circle: {j1} · hollow diamond: {j2}",
                                  subtitleColor=ink["secondary"], subtitleFontSize=11),
            height=46 * len(rows) + 20)), width="stretch", theme=None)

        models = by.sort_values("primary_faithful", ascending=False)["family"].tolist()
        models2 = by.sort_values("secondary_faithful", ascending=False)["family"].tolist()
        same_top = models[0] == models2[0]
        st.markdown(
            f"{'Both judges put **' + label(models[0]) + '** first' if same_top else f'The judges disagree on the top model ({label(models[0])} vs {label(models2[0])})'}"
            f"; the full ranking {'matches' if models == models2 else 'differs'} between them. "
            f"{j2} is {'stricter' if by['secondary_faithful'].mean() < by['primary_faithful'].mean() else 'more lenient'} overall. "
            "The absolute faithful rate depends on which judge you ask; comparisons made under one judge are what hold up."
        )
        with st.expander("Table view"):
            st.dataframe(by.assign(model=by["family"].map(label)).set_index("model").drop(columns="family"), width="stretch")

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
  (`{summary['catalog']}.{summary['schema']}`){f", code tag `{summary['code_version']}`" if summary.get('code_version') else ""}.
- Judge: `{', '.join(summary['judge_endpoints'])}`, prompt version `{summary['judge_prompt_version']}`.
- Training: {cost.get('billable_minutes', 0):.0f} billable GPU-minutes on `{cost.get('instance_type')}`, about ${cost.get('estimated_usd', 0):.2f}.
- Origin: an MSDS 453 class project, rebuilt as a production pipeline.
""")
