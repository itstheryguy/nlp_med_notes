import streamlit as st

st.set_page_config(page_title="Clinical Note Traps", page_icon=":material/stethoscope:", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/insights:", default=True),
    st.Page("views/explorer.py", title="Note explorer", url_path="explorer", icon=":material/compare:"),
    st.Page("views/traps.py", title="Model traps", url_path="traps", icon=":material/warning:"),
    st.Page("views/methods.py", title="How it's measured", url_path="methods", icon=":material/science:"),
]
st.navigation(pages).run()
