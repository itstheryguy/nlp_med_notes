"""Headless check of every page and interaction: `python test_app.py` from this folder."""
import os
import sys

from streamlit.testing.v1 import AppTest

APP = os.path.dirname(os.path.abspath(__file__))
pages = ["views/overview.py", "views/explorer.py", "views/traps.py", "views/methods.py"]
bad = 0
for p in pages:
    at = AppTest.from_file(f"{APP}/app.py", default_timeout=60)
    at.run()
    if p != pages[0]:
        at.switch_page(p); at.run()
    errs = [e.value for e in at.exception]
    print(f"{'OK ' if not errs else 'ERR'} {p}: {len(at.markdown)} markdown, {len(at.get('arrow_vega_lite_chart'))} charts, {len(at.metric)} metrics")
    for e in errs: print("   ", str(e)[:1500]); bad += 1
# Interactions: explorer with other filters, traps with each trap/model
at = AppTest.from_file(f"{APP}/app.py", default_timeout=60); at.run(); at.switch_page("views/explorer.py"); at.run()
at.toggle[0].set_value(False).run(); at.toggle[1].set_value(True).run()
print("explorer filters:", "OK" if not at.exception else at.exception[0].value)
at = AppTest.from_file(f"{APP}/app.py", default_timeout=60); at.run(); at.switch_page("views/traps.py"); at.run()
for i, opt in enumerate(at.radio[0].options):
    for fam in at.selectbox[0].options:
        at.radio[0].set_value(opt); at.selectbox[0].set_value(fam); at.run()
        if at.exception: print("traps ERR", opt, fam, str(at.exception[0].value)[:600]); bad += 1; break
print("traps all combos:", "OK" if not bad else "see errors")
sys.exit(1 if bad else 0)
