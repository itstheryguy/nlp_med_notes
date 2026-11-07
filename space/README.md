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

Three fine-tuned seq2seq models (BioBART-base, T5-base, BART-large-CNN) write one section of a clinical note
from a doctor-patient conversation ([MTS-Dialog](https://github.com/abachaa/MTS-Dialog), CC BY 4.0). Overlap
scores say they're close to the clinician's note; a two-tier faithfulness audit (free deterministic checks on
every note, an LLM judge on a fixed sample) shows where they actually go wrong.

- **Overview:** overlap vs faithfulness, mean ± sd over 3 training seeds per model
- **Note explorer:** any test conversation, every model's note, highlighted unsupported terms, the judge's objections
- **Model traps:** six failure patterns with rates and real examples
- **How it's measured:** data, training, the audit, and how far the free check can stand in for the judge

The app runs entirely in your browser ([stlite](https://github.com/whitphx/stlite): Streamlit on Pyodide), so
this is a static Space with no server. It reads a snapshot in `data/`, exported from the Databricks pipeline that
trained and audited the models (training on AWS SageMaker). No credentials, no live inference.
