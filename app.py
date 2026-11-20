"""Streamlit chat UI. Run with:  streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st

import config
from generator import NO_ANSWER, cited_numbers, stream_answer
from ingest import clear_index, find_documents, index_file
from retriever import retrieve, retrieval_query
from store import get_collection

st.set_page_config(page_title="Company Knowledge Assistant", layout="wide")


def indexed_documents() -> dict[str, int]:
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in metadatas:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return dict(sorted(counts.items()))


def render_sources(sources: list[dict], key_prefix: str) -> None:
    for s in sources:
        label = f"[{s['n']}] {s['source']}"
        if s["location"]:
            label += f" - {s['location']}"
        label += f"  (relevance {s['score']:.2f})"
        with st.expander(label):
            st.write(s["text"])
            path = config.DOCS_DIR / s["source"]
            if path.exists():
                st.download_button(
                    "Open original file",
                    data=path.read_bytes(),
                    file_name=s["source"],
                    key=f"{key_prefix}-{s['n']}",
                )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload documents",
        type=[ext.lstrip(".") for ext in sorted(config.SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", disabled=not uploads):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Indexing..."):
            for upload in uploads:
                destination = config.DOCS_DIR / Path(upload.name).name
                destination.write_bytes(upload.getbuffer())
                try:
                    count = index_file(destination)
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
                    continue
                if count:
                    st.success(f"{upload.name}: {count} chunks")
                else:
                    st.warning(f"{upload.name}: no text found (scanned PDF?)")

    if st.button("Re-index everything in data/docs"):
        with st.spinner("Re-indexing..."):
            for doc in find_documents(config.DOCS_DIR):
                index_file(doc)
        st.success("Done.")

    documents = indexed_documents()
    st.subheader(f"Indexed ({len(documents)})")
    for name, count in documents.items():
        st.caption(f"{name} - {count} chunks")
    selected = st.multiselect("Search only in", options=list(documents))
    sources_filter = selected or None

    if documents and st.button("Clear index"):
        clear_index()
        st.rerun()

    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("Company Knowledge Assistant")
if not os.getenv("GROQ_API_KEY"):
    st.warning("GROQ_API_KEY is not set. Add it to a .env file and restart the app.")
if not documents:
    st.info("No documents indexed yet. Upload files in the sidebar to get started.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"], f"history-{i}")

question = st.chat_input("Ask a question about your company documents")
if question:
    history = list(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks = retrieve(retrieval_query(question, history), sources=sources_filter)
        cited: list[dict] = []
        if not chunks:
            answer = NO_ANSWER
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(stream_answer(question, history, chunks))
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                st.stop()
            cited = [
                {
                    "n": n,
                    "source": chunks[n - 1].source,
                    "location": chunks[n - 1].location,
                    "score": chunks[n - 1].score,
                    "text": chunks[n - 1].text,
                }
                for n in cited_numbers(answer, len(chunks))
            ]
            if cited:
                render_sources(cited, f"new-{len(history)}")
            elif "couldn't find" not in answer.lower():
                st.caption("No sources were cited for this answer, so treat it with caution.")

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": cited})
