"""Shared data, palette, and rendering for the Space. Reads only the exported snapshot in ./data.

Runs under regular Streamlit (local) and stlite (the static Hugging Face Space, in the browser)."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
FIELDS = ["Symptoms", "Diagnosis", "History of Patient", "Plan of Action"]

# Fixed order; a model keeps its color on every page, and rows are drawn in this order. Slots follow the
# reference palette's validated adjacent order (blue, orange, aqua, yellow); every chart also labels each row
# by name, so color is never the only cue. Archived BART-large shares a slot only if a snapshot includes it.
MODELS = {
    "biobart": {"label": "BioBART-base", "slot": 0, "params": "166M", "about": "BART pretrained on PubMed text"},
    "t5_base": {"label": "T5-base", "slot": 1, "params": "223M", "about": "text-to-text, general pretraining"},
    "biobart_large": {"label": "BioBART-large", "slot": 2, "params": "406M", "about": "larger biomedical BART"},
    "pegasus_large": {"label": "PEGASUS-large", "slot": 3, "params": "568M",
                      "about": "gap-sentence pretraining (pretrained-only checkpoint)"},
    "bart_large_cnn": {"label": "BART-large-CNN (archived)", "slot": 2, "params": "406M", "about": "news summarizer"},
    "bart_large_cnn_minlen": {"label": "BART-large-CNN, min_length=10 (archived)", "slot": 2, "params": "406M",
                              "about": "same model, minimum decode length"},
}
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "reference": "#52514e", "mark": "#fff3bf", "surface": "#fcfcfb"},
    "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "reference": "#c3c2b7", "mark": "#5c4a00", "surface": "#1a1a19"},
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
LABEL_STYLE = {  # judge label -> (status color, icon, text): status colors always ship with icon + text
    "supported": ("good", "✓", "supported"),
    "unsupported": ("critical", "✕", "not in dialogue"),
    "contradicted": ("critical", "✕", "contradicts dialogue"),
    "misplaced": ("warning", "↔", "wrong field"),
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def color(family: str) -> str:
    return SERIES[mode()][MODELS.get(family, {"slot": 0})["slot"]]


def label(family: str) -> str:
    return MODELS.get(family, {"label": family})["label"]


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # JSON, not parquet: the static Space runs in the browser (stlite/Pyodide), where pyarrow may be missing.
    examples = pd.read_json(DATA / "examples.json", orient="records")
    notes = pd.read_json(DATA / "notes.json", orient="records")
    runs = pd.read_json(DATA / "runs.json", orient="records")
    summary = json.loads((DATA / "summary.json").read_text())
    for df, col in ((notes, "verdict_json"), (examples, "ref_verdict_json")):
        df[col.replace("_json", "")] = [json.loads(v) if isinstance(v, str) else None for v in df[col]]
    for col in ("unsupported_terms", "filled_na_fields", "missed_fields"):
        notes[col] = notes[col].apply(lambda v: list(v) if v is not None else [])
    examples["ref_unsupported_terms"] = examples["ref_unsupported_terms"].apply(lambda v: list(v) if v is not None else [])
    notes["label"] = notes["family"].map(label)
    order = [f for f in MODELS if f in set(notes["family"])]
    return examples, notes, runs, {**summary, "family_order": order}


def css() -> None:
    ink = INK[mode()]
    st.markdown(f"""<style>
    .note {{ border-left: 3px solid var(--c); padding: .55rem .85rem; margin: .25rem 0 .9rem; border-radius: 4px;
             background: color-mix(in srgb, var(--c) 6%, transparent); }}
    .note h5 {{ margin: 0 0 .35rem; font-size: .95rem; }}
    .note .meta {{ color: {ink['secondary']}; font-size: .8rem; margin-bottom: .4rem; }}
    .field {{ margin: .15rem 0; line-height: 1.45; }}
    .field b {{ color: {ink['secondary']}; font-weight: 600; }}
    .na {{ color: {ink['muted']}; }}
    mark.uns {{ background: {ink['mark']}; color: inherit; padding: 0 .15em; border-radius: 3px;
               text-decoration: underline wavy {STATUS['critical']}; text-underline-offset: 3px; }}
    .claim {{ font-size: .85rem; margin: .1rem 0; }}
    .tag {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .05rem .4rem; border-radius: 999px;
            border: 1px solid var(--t); color: {ink['primary']}; margin-right: .35rem; white-space: nowrap; }}
    .turn {{ margin: .2rem 0; line-height: 1.45; }}
    .who {{ font-weight: 600; color: {ink['secondary']}; }}
    .empty {{ color: {ink['muted']}; font-style: italic; }}
    </style>""", unsafe_allow_html=True)


def _highlight(text: str, terms: list[str]) -> str:
    out = html.escape(text)
    for t in sorted(set(terms), key=len, reverse=True):
        out = re.sub(rf"(?<![\w-])({re.escape(html.escape(t))})(?![\w-])", r'<mark class="uns">\1</mark>', out,
                     flags=re.IGNORECASE)
    return out


JUDGE_NAMES = {"databricks-gpt-oss-120b": "gpt-oss-120b", "databricks-qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B"}


def judge_name(endpoint: str | None) -> str:
    return JUDGE_NAMES.get(endpoint or "", (endpoint or "").replace("databricks-", ""))


def _ok(v) -> bool:
    return v is not None and v == v  # not None / NaN


def note_html(title: str, family: str | None, text: str, fields_json: str | None, terms: list[str],
              meta: str = "", verdict: dict | None = None, filled_na: list[str] | None = None,
              second: dict | None = None) -> str:
    """second: {"name", "faithful", "score", "primary_faithful"} for the second judge's one-line verdict."""
    c = color(family) if family else INK[mode()]["reference"]
    if not (text or "").strip():
        body = '<div class="empty">(empty note: the model generated nothing)</div>'
    elif isinstance(fields_json, str):  # missing values arrive as NaN from JSON, which is truthy
        fields = json.loads(fields_json)
        rows = []
        for f in FIELDS:
            v = fields.get(f, "")
            is_na = v.strip().rstrip(".").lower() in ("n/a", "na", "none", "")
            flag = (' <span class="tag" style="--t:' + STATUS["warning"] + '">↔ clinician wrote N/A</span>'
                    if filled_na and f in filled_na else "")
            val = f'<span class="na">{html.escape(v)}</span>' if is_na else _highlight(v, terms)
            rows.append(f'<div class="field"><b>{f}:</b> {val}{flag}</div>')
        body = "".join(rows)
    else:
        # Shown as written, line by line, so invented or repeated field labels are visible.
        lines = "".join(f'<div class="field">{_highlight(line, terms)}</div>' for line in text.splitlines() if line.strip())
        body = (lines + f'<div class="meta"><span class="tag" style="--t:{STATUS["warning"]}">⚠ not in the 4-field '
                'template</span>a field label is missing, repeated, out of order, or invented</div>')
    claims = ""
    if verdict:
        bad = [cl for cl in verdict.get("claims", []) if cl.get("label") != "supported"]
        omit = [o for o in verdict.get("omissions", []) if o.get("severity") == "major"]
        items = []
        for cl in bad:
            key, icon, txt = LABEL_STYLE.get(cl["label"], ("warning", "?", cl["label"]))
            sev = "major" if cl.get("severity") == "major" else "minor"
            items.append(f'<div class="claim"><span class="tag" style="--t:{STATUS[key]}">{icon} {txt} · {sev}</span>'
                         f'{html.escape(cl.get("field", ""))}: {html.escape(cl.get("claim", ""))}</div>')
        for o in omit:
            items.append(f'<div class="claim"><span class="tag" style="--t:{STATUS["serious"]}">∅ missing · major</span>'
                         f'{html.escape(o.get("fact", ""))}</div>')
        if not items:
            items.append(f'<div class="claim"><span class="tag" style="--t:{STATUS["good"]}">✓ judge found no problems</span></div>')
        claims = '<div class="meta" style="margin-top:.5rem">LLM judge</div>' + "".join(items)
    if second and _ok(second.get("faithful")):
        key = "good" if second["faithful"] else "serious"
        mark = "✓ faithful" if second["faithful"] else "✕ not faithful"
        disagree = (_ok(second.get("primary_faithful")) and bool(second["primary_faithful"]) != bool(second["faithful"]))
        claims += (f'<div class="claim" style="margin-top:.35rem"><span class="meta">Second judge ({html.escape(second["name"])}):</span> '
                   f'<span class="tag" style="--t:{STATUS[key]}">{mark}</span>'
                   + (f'score {second["score"]:.0f}/5 ' if _ok(second.get("score")) else "")
                   + (f'<span class="tag" style="--t:{STATUS["warning"]}">⚠ judges disagree</span>' if disagree else "")
                   + "</div>")
    return (f'<div class="note" style="--c:{c}"><h5>{html.escape(title)}</h5>'
            + (f'<div class="meta">{meta}</div>' if meta else "") + body + claims + "</div>")


def dialogue_html(text: str) -> str:
    turns = []
    for line in text.splitlines():
        if not line.strip():
            continue
        who, sep, said = line.partition(":")
        if sep and len(who) < 25:
            turns.append(f'<div class="turn"><span class="who">{html.escape(who.strip())}:</span> {html.escape(said.strip())}</div>')
        else:
            turns.append(f'<div class="turn">{html.escape(line)}</div>')
    return "".join(turns)


def chart_config(chart: alt.Chart) -> alt.Chart:
    ink = INK[mode()]
    return (chart.configure(background="transparent", font="system-ui, -apple-system, 'Segoe UI', sans-serif")
            .configure_view(stroke=None)
            .configure_axis(gridColor=ink["grid"], domainColor=ink["axis"], tickColor=ink["axis"],
                            labelColor=ink["secondary"], titleColor=ink["secondary"], labelFontSize=12,
                            titleFontSize=12, titleFontWeight="normal")
            .configure_title(color=ink["primary"], fontSize=14, anchor="start", fontWeight=600)
            .configure_legend(labelColor=ink["secondary"], titleColor=ink["secondary"]))


def rate_chart(df: pd.DataFrame, value: str, title: str, reference: float | None = None, fmt: str = ".0%",
               domain=(0, 1), seed_dots: pd.DataFrame | None = None, height: int | None = None) -> alt.Chart:
    """Dot + seed-spread chart: one row per model, mean dot, ±sd bar, faint per-seed dots, gray reference rule."""
    order = [label(f) for f in df["family"]]
    height_px = height or 46 * len(df) + 20
    d = df.assign(model=df["family"].map(label), color=df["family"].map(color),
                  shape=["diamond" if f.endswith("_minlen") else "circle" for f in df["family"]],
                  lo=(df[value] - df[f"{value}_sd"]).clip(lower=domain[0]),
                  hi=(df[value] + df[f"{value}_sd"]).clip(upper=domain[1]))
    y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    x_scale = alt.Scale(domain=list(domain))
    tip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip(f"{value}:Q", title="Mean over seeds", format=fmt),
           alt.Tooltip(f"{value}_sd:Q", title="± sd", format=fmt)]
    bars = alt.Chart(d).mark_rule(strokeWidth=2, opacity=.55).encode(y=y, x=alt.X("lo:Q", scale=x_scale, title=None,
                                                                               axis=alt.Axis(format=fmt, tickCount=5)),
                                                                      x2="hi:Q", color=alt.Color("color:N", scale=None))
    dots = alt.Chart(d).mark_point(filled=True, size=140, opacity=1, strokeWidth=2).encode(
        y=y, x=f"{value}:Q", color=alt.Color("color:N", scale=None), tooltip=tip,
        shape=alt.Shape("shape:N", scale=None))
    text = alt.Chart(d).mark_text(align="left", dx=10, dy=-10, fontSize=12, color=INK[mode()]["primary"]).encode(
        y=y, x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    layers = [bars, dots, text]
    if seed_dots is not None and value in seed_dots:
        s = seed_dots.assign(model=seed_dots["family"].map(label), color=seed_dots["family"].map(color))
        layers.insert(0, alt.Chart(s).mark_point(size=40, opacity=.45, filled=False, strokeWidth=1.5).encode(
            y=y, x=f"{value}:Q", color=alt.Color("color:N", scale=None),
            tooltip=[alt.Tooltip("experiment:N", title="Run"), alt.Tooltip(f"{value}:Q", format=fmt)]))
    if reference is not None:
        r = pd.DataFrame({"x": [reference]})
        layers.append(alt.Chart(r).mark_rule(color=INK[mode()]["reference"], strokeWidth=1.5).encode(
            x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Clinician reference", format=fmt)]))
    subtitle = f"gray line: clinician reference, {format(reference, fmt)}" if reference is not None else " "
    ink = INK[mode()]
    return alt.layer(*layers).properties(
        title=alt.TitleParams(title, subtitle=subtitle, subtitleColor=ink["secondary"], subtitleFontSize=11),
        height=height_px)


def families_df(summary: dict) -> pd.DataFrame:
    df = pd.DataFrame(summary["families"])
    order = [f for f in summary["family_order"] if f in set(df["family"])]
    return df.set_index("family").loc[order].reset_index()
