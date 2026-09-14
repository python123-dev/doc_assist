"""
Stage 5: Streamlit frontend.

This file is pure UI glue — it doesn't do any chunking/embedding/retrieval
itself. It just calls the functions we already built and tested in
ingest.py (Stages 1-2) and query.py (Stages 3-4), and displays the results.

Important Streamlit concept: every time you interact with the page (upload
a file, type a question, click a button), Streamlit reruns this ENTIRE
script top to bottom. `st.session_state` is how we remember things across
those reruns — here, we use it so we don't re-embed (and re-pay for) the
same PDF every time you ask a new question.
"""

import tempfile
import os

import streamlit as st

from ingest import load_and_split, embed_and_store
from query import answer_question
from agent import run_agent

st.title("Document Assistant")
st.caption("Upload a PDF, then ask questions about it.")

# session_state.ingested_file remembers the name of the file we've already
# processed, so re-running the script (e.g. after asking a question)
# doesn't trigger a second, wasted embedding pass.
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and uploaded_file.name != st.session_state.ingested_file:
    # PyPDFLoader (used inside load_and_split) needs a real file path, but
    # Streamlit gives us the upload as in-memory bytes — so we write it to
    # a temporary file on disk first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Reading and indexing your document..."):
        chunks = load_and_split(tmp_path)
        embed_and_store(chunks)

    os.remove(tmp_path)
    st.session_state.ingested_file = uploaded_file.name
    st.success(f"Indexed '{uploaded_file.name}' ({len(chunks)} chunks).")

elif uploaded_file is not None:
    st.info(f"'{uploaded_file.name}' is already indexed.")

question = st.text_input("Ask a question about the document")
use_agent = st.checkbox(
    "Allow web search (agent mode)",
    help="Off: answers only from your document. On: an agent decides whether "
         "to use the document or fall back to a live web search.",
)

if question:
    if st.session_state.ingested_file is None:
        st.warning("Upload a document first.")
    else:
        with st.spinner("Thinking..."):
            answer = run_agent(question) if use_agent else answer_question(question)
        st.write(answer)
