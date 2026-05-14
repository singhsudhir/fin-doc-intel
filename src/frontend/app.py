"""Streamlit frontend for the Financial Document Intelligence Agent."""
from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
_TIMEOUT_SHORT = 15   # seconds — health, list, delete
_TIMEOUT_LONG  = 180  # seconds — ingest, query, compare

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FinDocIntel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
.citation-box {
    background: #f8f9fa;
    border-left: 3px solid #0068c9;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    color: #444;
}
</style>
""", unsafe_allow_html=True)


# ─── API helpers ────────────────────────────────────────────────

def _get(path: str, timeout: int = _TIMEOUT_SHORT) -> requests.Response | None:
    try:
        return requests.get(f"{API_URL}{path}", timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None


def _post(path: str, timeout: int = _TIMEOUT_LONG, **kwargs) -> requests.Response | None:
    try:
        return requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None


def _delete(path: str) -> requests.Response | None:
    try:
        return requests.delete(f"{API_URL}{path}", timeout=_TIMEOUT_SHORT)
    except requests.exceptions.ConnectionError:
        return None


@st.cache_data(ttl=5)
def _fetch_documents() -> list[dict]:
    resp = _get("/documents")
    return resp.json() if resp and resp.ok else []


def _bust_doc_cache() -> None:
    _fetch_documents.clear()


# ─── Sidebar ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 FinDocIntel")

    # Backend health indicator
    health = _get("/health")
    if health is None:
        st.error(
            "**Backend offline.**\n\n"
            "Start it with:\n```\nuvicorn src.api.main:app --reload\n```"
        )
        st.stop()

    h = health.json()
    qdrant_dot  = "🟢" if h.get("qdrant_connected")      else "🔴"
    model_dot   = "🟢" if h.get("embedding_model_loaded") else "🟡"
    st.caption(f"{qdrant_dot} Qdrant   {model_dot} Embedding model")

    st.divider()

    # ── Upload ──────────────────────────────────────────────────
    st.markdown("### Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        btn_label = f"Ingest  {uploaded.name[:24]}…" if len(uploaded.name) > 24 else f"Ingest  {uploaded.name}"
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("Ingesting document …"):
                resp = _post(
                    "/ingest",
                    files={
                        "file": (uploaded.name, uploaded.getvalue(), "application/pdf")
                    },
                )
            if resp and resp.ok:
                data = resp.json()
                if data.get("total_chunks", 0) > 0:
                    st.success(f"✅  {data['total_chunks']} chunks indexed")
                else:
                    st.info(data.get("message", "Already ingested"))
                _bust_doc_cache()
                st.rerun()
            else:
                detail = resp.json().get("detail", resp.text) if resp else "No response"
                st.error(f"Ingestion failed: {detail}")

    st.divider()

    # ── Document list ────────────────────────────────────────────
    st.markdown("### Indexed Documents")
    docs = _fetch_documents()

    if not docs:
        st.caption("No documents yet. Upload a PDF above.")
    else:
        for doc in docs:
            name = doc["document_name"]
            short = name[:26] + "…" if len(name) > 26 else name
            col_name, col_del = st.columns([5, 1])
            col_name.markdown(f"**{short}**")
            col_name.caption(f"{doc['chunk_count']} chunks · {doc['total_pages']}p")
            if col_del.button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                _delete(f"/documents/{name}")
                _bust_doc_cache()
                st.rerun()


# ─── Main area ──────────────────────────────────────────────────

st.title("Financial Document Intelligence")
st.caption("Grounded answers with page-level citations — powered by Gemini 2.5 Pro + Qdrant")

docs      = _fetch_documents()
doc_names = [d["document_name"] for d in docs]

tab_ask, tab_compare = st.tabs(["💬 Ask Questions", "📊 Compare Documents"])


# ══════════════════════════════════════════════════════════════
# TAB: Ask Questions
# ══════════════════════════════════════════════════════════════

with tab_ask:
    col_filter, col_k = st.columns([4, 1])
    with col_filter:
        scope_options = ["All documents"] + doc_names
        scope = st.selectbox("Scope:", scope_options, key="scope")
        doc_filter = None if scope == "All documents" else scope
    with col_k:
        top_k = st.number_input("Chunks", min_value=1, max_value=10, value=5, key="top_k")

    question = st.text_input(
        "Question",
        placeholder="What was ING's net profit in 2024?",
        label_visibility="collapsed",
        key="question",
    )

    if st.button("Ask ▶", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating …"):
            resp = _post(
                "/query",
                json={
                    "question": question,
                    "document_filter": doc_filter,
                    "top_k_final": top_k,
                },
            )

        if resp and resp.ok:
            result = resp.json()
            st.markdown("---")
            st.markdown("#### Answer")
            st.markdown(result["answer"])

            citations = result.get("citations", [])
            if citations:
                st.markdown(f"#### 📎 Sources ({len(citations)})")
                for c in citations:
                    label = f"📄 **{c['document_name']}** — page {c['page_number']}"
                    with st.expander(label):
                        st.markdown(
                            f"<div class='citation-box'>{c['source_text']}</div>",
                            unsafe_allow_html=True,
                        )

            st.caption(
                f"Model: `{result.get('model_used', 'n/a')}` · "
                f"Chunks used: {result.get('chunks_used', 0)}"
            )
        elif resp:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Error {resp.status_code}: {detail}")
        else:
            st.error("Backend did not respond. Is `uvicorn src.api.main:app` running?")


# ══════════════════════════════════════════════════════════════
# TAB: Compare Documents
# ══════════════════════════════════════════════════════════════

with tab_compare:
    if len(doc_names) < 2:
        st.info(
            "Upload at least **two documents** to use the comparison feature.",
            icon="ℹ️",
        )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            doc_a = st.selectbox("Document A", doc_names, key="cmp_a")
        with col_b:
            others = [d for d in doc_names if d != doc_a]
            doc_b  = st.selectbox("Document B", others, key="cmp_b")

        topics_raw = st.text_input(
            "Focus topics (comma-separated):",
            value="revenue, net income, risk factors, capital ratios, outlook",
            key="topics",
        )
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]

        if st.button("Compare ▶", type="primary"):
            with st.spinner(f"Comparing {doc_a}  vs  {doc_b} …"):
                resp = _post(
                    "/query/compare",
                    json={
                        "document_name_a": doc_a,
                        "document_name_b": doc_b,
                        "focus_topics":    topics,
                    },
                )

            if resp and resp.ok:
                result = resp.json()
                st.markdown("---")
                short_a = doc_a[:30] + "…" if len(doc_a) > 30 else doc_a
                short_b = doc_b[:30] + "…" if len(doc_b) > 30 else doc_b
                st.markdown(f"#### {short_a}  vs  {short_b}")
                st.markdown(result["comparison"])

                cit_a = result.get("citations_a", [])
                cit_b = result.get("citations_b", [])
                if cit_a or cit_b:
                    col_ca, col_cb = st.columns(2)
                    with col_ca:
                        if cit_a:
                            st.markdown(f"**Sources — {short_a}**")
                            for c in cit_a:
                                with st.expander(f"Page {c['page_number']}"):
                                    st.markdown(
                                        f"<div class='citation-box'>{c['source_text']}</div>",
                                        unsafe_allow_html=True,
                                    )
                    with col_cb:
                        if cit_b:
                            st.markdown(f"**Sources — {short_b}**")
                            for c in cit_b:
                                with st.expander(f"Page {c['page_number']}"):
                                    st.markdown(
                                        f"<div class='citation-box'>{c['source_text']}</div>",
                                        unsafe_allow_html=True,
                                    )
                st.caption(f"Model: `{result.get('model_used', 'n/a')}`")
            elif resp:
                detail = resp.json().get("detail", resp.text)
                st.error(f"Error {resp.status_code}: {detail}")
            else:
                st.error("Backend did not respond.")
