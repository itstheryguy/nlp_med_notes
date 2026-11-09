---
title: Clinical Note Traps
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: cc-by-4.0
short_description: Where fine-tuned clinical-note summarizers go wrong
---

# Clinical Note Traps

Four fine-tuned seq2seq models (BioBART-base, T5-base, BioBART-large, PEGASUS-large) write one section of a clinical
note from a doctor-patient conversation ([MTS-Dialog](https://github.com/abachaa/MTS-Dialog), CC BY 4.0). Overlap
scores say they're close to the clinician's note; a two-tier faithfulness audit (free deterministic checks on every
note, two LLM judges on a fixed sample) shows where they actually go wrong.

- **Overview:** overlap vs faithfulness, mean ± sd over 3 training seeds per model
- **Note explorer:** any test conversation, every model's note, highlighted unsupported terms, both judges' verdicts
- **Model traps:** recurring failure patterns with rates and real examples
- **How it's measured:** data, training, the audit, and whether the judges can be trusted

The app runs entirely in your browser ([stlite](https://github.com/whitphx/stlite): Streamlit on Pyodide), so this
is a static Space with no server; the first load takes a few seconds. It reads a snapshot in `data/`, exported from
the Databricks pipeline that trained and audited the models (training on AWS SageMaker). No credentials, no live
inference.

Code: [github.com/itstheryguy/nlp_med_notes](https://github.com/itstheryguy/nlp_med_notes)
