import altair as alt
import pandas as pd
import streamlit as st

from lib import INK, chart_config, color, css, dialogue_html, label, load, mode, note_html

examples, notes, runs, summary = load()
css()
ex_by_id = examples.set_index("example_id")

st.title("Model traps")
st.markdown(
    "Six ways these models fail that an overlap score hides. Each trap shows how often it happens per model "
    "(mean over 3 seeds), with real examples. Traps marked **judge** are measured on the "
    f"{summary['n_audit_examples']} audited test examples; the rest use all {summary['n_test_examples']}."
)


def has_label(verdict, wanted: set[str], major_only: bool = False) -> bool:
    if not verdict:
        return False
    return any(c.get("label") in wanted and (not major_only or c.get("severity") == "major")
               for c in verdict.get("claims", []))


def major_omission(verdict) -> bool:
    return bool(verdict) and any(o.get("severity") == "major" for o in verdict.get("omissions", []))


audited = notes["faithful"].notna()
TRAPS = [
    {
        "key": "numbers", "title": "Gets the numbers wrong", "source": "tier-1",
        "test": lambda n: n["unsupported_numbers"] > 0, "scope": None,
        "why": "Patients say numbers out loud (\"nineteen sixty eight\", \"one hundred and eighty three pounds\"). "
               "Models convert them to digits and often get them wrong: a wrong weight, dose, or year. Highest-risk "
               "error type, and the one a free check catches best.",
    },
    {
        "key": "filled_na", "title": "Fills a field the clinician left empty", "source": "tier-1",
        "test": lambda n: n["filled_na_fields"].apply(len) > 0, "scope": None,
        "why": "When the conversation has nothing for, say, Diagnosis, the clinician writes N/A. A model that writes "
               "*something* there is usually inventing or moving content. Most hallucinations enter this way.",
    },
    {
        "key": "empty", "title": "Writes nothing at all", "source": "tier-1",
        "test": lambda n: n["empty_prediction"], "scope": None,
        "why": "An empty note can't hallucinate, so it scores well on a hallucination rate while being useless. "
               "A metric you can win by staying silent needs a partner metric (omissions).",
    },
    {
        "key": "misplaced", "title": "Right fact, wrong field", "source": "judge",
        "test": lambda n: n["verdict"].apply(lambda v: has_label(v, {"misplaced"})), "scope": audited,
        "why": "A medication listed as a diagnosis, or past history as a current symptom. Every word is in the "
               "dialogue, so word-overlap checks pass it; only reading for meaning catches it.",
    },
    {
        "key": "invented", "title": "States something the dialogue doesn't support", "source": "judge",
        "test": lambda n: n["verdict"].apply(lambda v: has_label(v, {"unsupported", "contradicted"}, major_only=True)),
        "scope": audited,
        "why": "Major unsupported or contradicted claims, such as \"congested since age 11\" when the parent said she *is* "
               "eleven, or a symptom escalated (\"decreased vision\" becoming \"blindness\").",
    },
    {
        "key": "omits", "title": "Misses what matters", "source": "judge",
        "test": lambda n: n["verdict"].apply(major_omission), "scope": audited,
        "why": "Clinically important facts from the dialogue that belong in the section but are missing, such as a second complaint the patient raised.",
    },
]

rows = []
for t in TRAPS:
    pool = notes if t["scope"] is None else notes[t["scope"]]
    hit = t["test"](pool)
    per_run = pool.assign(hit=hit.astype(bool)).groupby(["family", "seed"])["hit"].mean().reset_index()
    for fam, g in per_run.groupby("family"):
        rows.append({"trap": t["key"], "family": fam, "rate": g["hit"].mean(), "sd": g["hit"].std(ddof=1) if len(g) > 1 else 0})
rates = pd.DataFrame(rows)

# Reference rates where the trap is defined for a clinician note.
ref_rates = {
    "numbers": float((examples["ref_unsupported_numbers"] > 0).mean()),
    "empty": 0.0,
    "misplaced": float(examples["ref_verdict"].apply(lambda v: has_label(v, {"misplaced"}))[examples["in_audit"]].mean()),
    "invented": float(examples["ref_verdict"].apply(lambda v: has_label(v, {"unsupported", "contradicted"}, True))[examples["in_audit"]].mean()),
    "omits": float(examples["ref_verdict"].apply(major_omission)[examples["in_audit"]].mean()),
}

# Overview: small multiples, one panel per trap, same x scale.
ink = INK[mode()]
d = rates.assign(model=rates["family"].map(label), c=rates["family"].map(color),
                 title=rates["trap"].map({t["key"]: f"{t['title']} ({t['source']})" for t in TRAPS}))
order = [label(f) for f in summary["family_order"]]
panels = []
for t in TRAPS:
    sub = d[d["trap"] == t["key"]]
    base = alt.Chart(sub).encode(y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)))
    bar = base.mark_bar(height=14, cornerRadiusEnd=4).encode(
        x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5), title=None),
        color=alt.Color("c:N", scale=None),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("rate:Q", format=".1%", title="Mean over seeds"),
                 alt.Tooltip("sd:Q", format=".1%", title="± sd")])
    txt = base.mark_text(align="left", dx=4, fontSize=11, color=ink["primary"]).encode(
        x="rate:Q", text=alt.Text("rate:Q", format=".0%"))
    layers = [bar, txt]
    if t["key"] in ref_rates:
        r = pd.DataFrame({"x": [ref_rates[t["key"]]]})
        layers.append(alt.Chart(r).mark_rule(color=ink["reference"], strokeWidth=1.5).encode(x="x:Q"))
    panels.append(alt.layer(*layers).properties(title=f"{t['title']} ({t['source']})", height=120))
grid = st.columns(2, gap="large")
for i, panel in enumerate(panels):
    grid[i % 2].altair_chart(chart_config(panel), width="stretch", theme=None)
st.caption("Bars = share of notes with the trap, mean over 3 seeds · gray line = the same check on clinician notes.")

st.divider()
st.subheader("Look at the examples")
t = st.radio("Trap", [x["title"] for x in TRAPS], horizontal=True, label_visibility="collapsed")
trap = next(x for x in TRAPS if x["title"] == t)
st.markdown(trap["why"])

c1, c2 = st.columns(2)
fam = c1.selectbox("Model", summary["family_order"], format_func=label)
seed = c2.selectbox("Seed", sorted(notes["seed"].unique()), index=1 if notes["seed"].nunique() > 1 else 0)
pool = notes if trap["scope"] is None else notes[trap["scope"]]
pool = pool[(pool["family"] == fam) & (pool["seed"] == seed)]
hits = pool[trap["test"](pool).astype(bool).values]
st.caption(f"{len(hits)} of {len(pool)} notes from {label(fam)} (seed {seed}) fall into this trap.")
if hits.empty:
    st.success("No examples of this trap for this model and seed.")
    st.stop()

pick = st.select_slider("Example", options=list(range(len(hits))), format_func=lambda i: f"{i + 1} / {len(hits)}")
n = hits.iloc[pick]
ex = ex_by_id.loc[n["example_id"]]
left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown(f"#### Dialogue · {ex['section_header']} · #{n['example_id']}")
    with st.container(height=480, border=True):
        st.markdown(dialogue_html(ex["dialogue"]), unsafe_allow_html=True)
with right:
    st.markdown(note_html("Clinician reference", None, ex["reference"], ex["ref_fields_json"], ex["ref_unsupported_terms"]),
                unsafe_allow_html=True)
    st.markdown(note_html(f"{label(fam)} · seed {seed}", fam, n["prediction"], n["fields_json"], n["unsupported_terms"],
                          verdict=n["verdict"], filled_na=n["filled_na_fields"]), unsafe_allow_html=True)
