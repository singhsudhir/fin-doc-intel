"""Standalone Streamlit app for Financial Document Intelligence (Streamlit Cloud)."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import streamlit as st

# ── Inject Streamlit secrets into env before importing src modules ─────────
for _key in ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    _val = st.secrets.get(_key)
    if _val and _key not in os.environ:
        os.environ[_key] = str(_val)

from src.embedding.embedder import Embedder  # noqa: E402
from src.embedding.qdrant_store import QdrantStore  # noqa: E402
from src.generation.answer_generator import AnswerGenerator  # noqa: E402
from src.generation.comparator import Comparator  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinDocIntel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Sidebar: pull content up to top ── */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Main area: remove excessive top whitespace ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

/* ── Citation styling ── */
.citation-box {
    background: #f8f9fa;
    border-left: 3px solid #2BB3B3;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    color: #444;
}

/* ── Sidebar width ── */
[data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
</style>
""", unsafe_allow_html=True)


# ── Cached singletons (loaded once per session) ───────────────────────────

@st.cache_resource
def _get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def _get_store() -> QdrantStore:
    return QdrantStore()


@st.cache_resource
def _get_retriever() -> Retriever:
    return Retriever(embedder=_get_embedder(), store=_get_store())


@st.cache_resource
def _get_generator() -> AnswerGenerator:
    return AnswerGenerator()


@st.cache_resource
def _get_comparator() -> Comparator:
    return Comparator(embedder=_get_embedder(), store=_get_store())


@st.cache_resource
def _get_pipeline() -> IngestionPipeline:
    return IngestionPipeline(embedder=_get_embedder(), store=_get_store())


# ── Document listing ──────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def _fetch_documents() -> list[dict]:
    store = _get_store()
    try:
        if not store.client.collection_exists(store.collection_name):
            return []
    except Exception:
        return []

    docs: dict[str, dict] = {}
    offset = None
    while True:
        records, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            with_payload=["document_name", "doc_id", "page_number"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for record in records:
            p = record.payload or {}
            name = p.get("document_name")
            if not name:
                continue
            if name not in docs:
                docs[name] = {"document_name": name, "chunk_count": 0, "total_pages": 0}
            docs[name]["chunk_count"] += 1
            docs[name]["total_pages"] = max(
                docs[name]["total_pages"], p.get("page_number", 0)
            )
        if next_offset is None:
            break
        offset = next_offset

    return sorted(docs.values(), key=lambda x: x["document_name"])


def _bust_doc_cache() -> None:
    _fetch_documents.clear()


def _classify_exc(exc: Exception) -> str:
    err = str(exc)
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        return f"RATE_LIMIT:{exc}"
    if "503" in err or "UNAVAILABLE" in err:
        return f"UNAVAILABLE:{exc}"
    return f"ERROR:{exc}"


def _rate_limit_message(exc: Exception) -> str:
    m = re.search(r"retryDelay.*?(\d+)s", str(exc))
    wait = int(m.group(1)) if m else 30
    return (
        f"**Gemini quota reached.** Please wait ~{wait} seconds and try again. "
        "If this keeps happening, check that your API key has billing enabled at "
        "[Google AI Studio](https://aistudio.google.com)."
    )


def _unavailable_message() -> str:
    return (
        "**Gemini is temporarily overloaded.** The request was retried automatically "
        "but the service is still unavailable. Please wait a moment and try again."
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_query(question: str, doc_filter: str | None, top_k: int) -> dict | str:
    try:
        chunks = _get_retriever().retrieve(
            question, top_k_final=top_k, document_filter=doc_filter
        )
        result = _get_generator().generate(question, chunks)
        return {
            "answer": result.answer,
            "citations": [
                {"document_name": c.document_name, "page_number": c.page_number, "source_text": c.source_text}
                for c in result.citations
            ],
            "model_used": result.model_used,
            "chunks_used": result.chunks_used,
        }
    except Exception as exc:
        return _classify_exc(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_compare(doc_a: str, doc_b: str, topics: tuple[str, ...]) -> dict | str:
    try:
        result = _get_comparator().compare(doc_a, doc_b, list(topics))
        return {
            "comparison": result.comparison,
            "citations_a": [
                {"page_number": c.page_number, "source_text": c.source_text}
                for c in result.citations_a
            ],
            "citations_b": [
                {"page_number": c.page_number, "source_text": c.source_text}
                for c in result.citations_b
            ],
            "model_used": result.model_used,
        }
    except Exception as exc:
        return _classify_exc(exc)


# ── Session state ─────────────────────────────────────────────────────────

if "upload_rev" not in st.session_state:
    st.session_state.upload_rev = 0


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 FinDocIntel")
    st.divider()

    # Validate secrets (stop early with a clear message if any are missing)
    missing = [k for k in ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY") if not os.environ.get(k)]
    if missing:
        st.error(
            f"Missing secrets: **{', '.join(missing)}**\n\n"
            "Add them in the Streamlit Cloud dashboard under "
            "**Settings → Secrets**."
        )
        st.stop()

    # Validate Qdrant connection silently; only surface error if it fails
    try:
        _get_store().client.collection_exists(_get_store().collection_name)
    except Exception as _e:
        st.error(f"Cannot connect to Qdrant: {_e}")
        st.stop()

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("### Upload PDF")
    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.upload_rev}",
    )

    if uploaded:
        btn_label = (
            f"Ingest  {uploaded.name[:24]}…" if len(uploaded.name) > 24
            else f"Ingest  {uploaded.name}"
        )
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("Ingesting document …"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / uploaded.name
                    tmp_path.write_bytes(uploaded.getvalue())
                    try:
                        result = _get_pipeline().ingest(str(tmp_path), skip_if_exists=True)
                    except Exception as exc:
                        result = None
                        st.error(f"Ingestion failed: {exc}")

            if result is not None:
                if result.error:
                    st.error(f"Ingestion failed: {result.error}")
                elif result.skipped:
                    st.info(f"Already indexed: {uploaded.name}")
                else:
                    st.success(f"✅  {result.vectors_stored} chunks indexed")
                    st.session_state.upload_rev += 1
                    _bust_doc_cache()
                    st.rerun()

    st.divider()

    # ── Document list ───────────────────────────────────────────────────────
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
                _get_store().delete_document(name)
                _bust_doc_cache()
                st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────

st.title("Financial Document Intelligence")
st.caption("Grounded answers with page-level citations — powered by Gemini 2.5 Flash + Qdrant")

docs = _fetch_documents()
doc_names = [d["document_name"] for d in docs]

tab_ask, tab_compare = st.tabs(["💬 Ask Questions", "📊 Compare Documents"])


# ══════════════════════════════════════════════════════════════════════════
# TAB: Ask Questions
# ══════════════════════════════════════════════════════════════════════════

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
            raw = _cached_query(question.strip(), doc_filter, int(top_k))

        if isinstance(raw, str) and raw.startswith("RATE_LIMIT:"):
            st.warning(_rate_limit_message(raw[len("RATE_LIMIT:"):]))
        elif isinstance(raw, str) and raw.startswith("UNAVAILABLE:"):
            st.warning(_unavailable_message())
        elif isinstance(raw, str) and raw.startswith("ERROR:"):
            st.error(raw[len("ERROR:"):])
        else:
            st.markdown("---")
            st.markdown("#### Answer")
            st.markdown(raw["answer"])

            citations = raw.get("citations", [])
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
                f"Model: `{raw.get('model_used', 'n/a')}` · "
                f"Chunks used: {raw.get('chunks_used', 0)}"
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB: Compare Documents
# ══════════════════════════════════════════════════════════════════════════

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
            doc_b = st.selectbox("Document B", others, key="cmp_b")

        topics_raw = st.text_input(
            "Focus topics (comma-separated):",
            value="revenue, net income, risk factors, capital ratios, outlook",
            key="topics",
        )
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]

        if st.button("Compare ▶", type="primary"):
            with st.spinner(f"Comparing {doc_a}  vs  {doc_b} …"):
                raw = _cached_compare(doc_a, doc_b, tuple(topics))

            short_a = doc_a[:30] + "…" if len(doc_a) > 30 else doc_a
            short_b = doc_b[:30] + "…" if len(doc_b) > 30 else doc_b

            if isinstance(raw, str) and raw.startswith("RATE_LIMIT:"):
                st.warning(_rate_limit_message(raw[len("RATE_LIMIT:"):]))
            elif isinstance(raw, str) and raw.startswith("ERROR:"):
                st.error(raw[len("ERROR:"):])
            else:
                st.markdown("---")
                st.markdown(f"#### {short_a}  vs  {short_b}")
                st.markdown(raw["comparison"])

                cit_a = raw.get("citations_a", [])
                cit_b = raw.get("citations_b", [])
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
                st.caption(f"Model: `{raw.get('model_used', 'n/a')}`")
