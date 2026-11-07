import streamlit as st

from lib import css, dialogue_html, judge_name, label, load, note_html

examples, notes, runs, summary = load()
css()
judges = summary.get("judges") or {}
j2 = judge_name(judges.get("secondary")) if judges.get("secondary") else None

def second_for(row, prefix: str = "j2_", primary_col: str = "faithful"):
    """The second judge's one-line verdict for a note (or the reference, with prefix 'ref_j2_')."""
    if not j2 or f"{prefix}faithful" not in row:
        return None
    return {"name": j2, "faithful": row[f"{prefix}faithful"], "score": row.get(f"{prefix}faithfulness_score"),
            "primary_faithful": row.get(primary_col)}

st.title("Note explorer")
st.markdown(
    "Pick a test conversation and compare every model's note with the clinician's. "
    "<mark class='uns'>Highlighted</mark> words and numbers never appear in the dialogue (the free tier-1 check). "
    "On audited examples, the LLM judge's objections are listed under each note.",
    unsafe_allow_html=True,
)

# One filter row scopes everything below it.
f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])
sections = ["All sections"] + sorted(examples["section_header"].unique())
section = f1.selectbox("Section", sections)
audited_only = f2.toggle("Judge-audited only", value=True,
                         help=f"The LLM judge audits a fixed sample of {summary['n_audit_examples']} test examples.")
problem_only = f3.toggle("Hallucinations only", value=False,
                         help="Only conversations where the selected seed's notes contain a judge-flagged hallucination.")
seed = f4.selectbox("Training seed", sorted(notes["seed"].unique()), index=1 if notes["seed"].nunique() > 1 else 0,
                    help="Each model was trained 3 times; pick which run's notes to show.")

pool = examples
if section != "All sections":
    pool = pool[pool["section_header"] == section]
if audited_only:
    pool = pool[pool["in_audit"]]
if problem_only:
    bad_ids = set(notes[(notes["seed"] == seed) & (notes["hallucinated"] == True)]["example_id"])  # noqa: E712
    pool = pool[pool["example_id"].isin(bad_ids)]
if pool.empty:
    st.info("No examples match these filters.")
    st.stop()

def describe(r) -> str:
    first = r["dialogue"].splitlines()[0][:70] if r["dialogue"] else ""
    return f"#{r['example_id']} · {r['section_header']} · {first}…"

options = {describe(r): r["example_id"] for _, r in pool.sort_values("example_id").iterrows()}
choice = st.selectbox(f"Conversation ({len(options)} match)", list(options))
ex = examples.set_index("example_id").loc[options[choice]]
row_notes = notes[(notes["example_id"] == options[choice]) & (notes["seed"] == seed)]

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown(f"#### Dialogue · {ex['section_header']}")
    with st.container(height=560, border=True):
        st.markdown(dialogue_html(ex["dialogue"]), unsafe_allow_html=True)
with right:
    st.markdown("#### Notes")
    ref_meta = (f"judge score {ex['ref_faithfulness_score']:.0f}/5" if ex.get("ref_faithfulness_score") == ex.get("ref_faithfulness_score")
                and ex.get("ref_faithfulness_score") is not None else "")
    st.markdown(note_html("Clinician reference", None, ex["reference"], ex["ref_fields_json"], ex["ref_unsupported_terms"],
                          meta=ref_meta, verdict=ex["ref_verdict"],
                          second=second_for(ex, "ref_j2_", "ref_faithful")), unsafe_allow_html=True)
    for fam in summary["family_order"]:
        n = row_notes[row_notes["family"] == fam]
        if n.empty:
            continue
        n = n.iloc[0]
        bits = [f"ROUGE-L {n['rougeL']:.2f}"]
        if n["faithfulness_score"] == n["faithfulness_score"] and n["faithfulness_score"] is not None:
            bits.append(f"judge score {n['faithfulness_score']:.0f}/5")
        st.markdown(note_html(label(fam), fam, n["prediction"], n["fields_json"], n["unsupported_terms"],
                              meta=" · ".join(bits), verdict=n["verdict"], filled_na=n["filled_na_fields"],
                              second=second_for(n)),
                    unsafe_allow_html=True)
