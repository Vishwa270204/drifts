import streamlit as st

_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)

CSS = """<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1100px}
[data-testid="stMetric"]{background:rgba(128,128,128,.07);border-radius:10px;padding:.6rem .9rem}
h1{font-size:1.8rem!important;padding-bottom:.2rem}
</style>"""


def wide(fn, *a, **kw):
    """Full-width widget across Streamlit versions."""
    kw["width" if _NEW_API else "use_container_width"] = "stretch" if _NEW_API else True
    return fn(*a, **kw)


def flash(level, text):
    st.session_state["flash"] = (level, text)


def show_flash():
    if st.session_state.get("flash"):
        lvl, txt = st.session_state.pop("flash")
        getattr(st, lvl)(txt)


def confirm_box(flag, message, yes_label):
    """Inline confirmation. Returns True only on the run where the user confirms."""
    if not st.session_state.get(flag):
        return False
    st.warning(message)
    a, b, _ = st.columns([1, 1, 4])
    if a.button(yes_label, key=flag + "_yes", type="primary"):
        st.session_state[flag] = False
        return True
    if b.button("Cancel", key=flag + "_no"):
        st.session_state[flag] = False
        st.rerun()
    return False
