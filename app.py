import os
import io
import tempfile
import time
from typing import List, Tuple

import streamlit as st

# LangChain imports (stable, split packages)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.schema import Document


# -------------------------
# Secrets / configuration
# -------------------------
def get_openai_key() -> str:
    # Prefer Streamlit secrets, fall back to env var
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not key:
        st.error(
            "❌ OPENAI_API_KEY not set. In Streamlit Cloud, go to **Manage app → Settings → Secrets** "
            'and add: `OPENAI_API_KEY = "sk-..."`'
        )
        st.stop()
    os.environ["OPENAI_API_KEY"] = key
    return key


# -------------------------
# Helpers
# -------------------------
def save_uploaded_file_to_tmp(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp path and return that path."""
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_pdfs(paths: List[str]) -> List[Document]:
    """Load PDFs into LangChain Documents with source & page metadata."""
    docs: List[Document] = []
    for p in paths:
        loader = PyPDFLoader(p)
        pdf_docs = loader.load()  # page-wise Documents
        # Normalize metadata
        for d in pdf_docs:
            # Put a readable source (filename) in metadata
            d.metadata["source"] = os.path.basename(p)
        docs.extend(pdf_docs)
    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_faiss_index(chunks: List[Document]) -> FAISS:
    # Use modern OpenAI embeddings (compatible with openai>=1.x)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(chunks, embeddings)
    return db


def ensure_session_state():
    if "db" not in st.session_state:
        st.session_state.db = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Tuple[str, str]] = []  # (user, assistant)


def answer_question(query: str, k: int = 4) -> Tuple[str, List[Document]]:
    """Retrieve top-k chunks and answer with sources."""
    retriever = st.session_state.db.as_retriever(search_kwargs={"k": k})
    context_docs: List[Document] = retriever.get_relevant_documents(query)

    # Compose context
    context_text = "\n\n".join(
        [
            f"[{i+1}] ({d.metadata.get('source','?')} p.{d.metadata.get('page', '?')})\n{d.page_content}"
            for i, d in enumerate(context_docs)
        ]
    )

    # Use a small, fast model by default
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    system_prompt = (
        "You are a precise RAG assistant. Answer **only** from the provided context.\n"
        "If the answer is not in the context, say you don't have enough information.\n"
        "Cite sources like [1], [2] inline where relevant."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]

    resp = llm.invoke(messages)
    answer = resp.content

    return answer, context_docs


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="RAG Chatbot (PDF)", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot — PDF Q&A")

ensure_session_state()
get_openai_key()

with st.sidebar:
    st.header("📥 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drag & drop or select multiple PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload several PDFs about a person or a topic.",
    )

    chunk_size = st.slider("Chunk size", 400, 2000, 900, step=50)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, step=10)
    k = st.slider("Top-k passages to retrieve", 2, 10, 4)

    build_btn = st.button("⚙️ Build / Rebuild Index", type="primary")

    if st.session_state.db is not None:
        st.success("Vector index is ready ✅")
    else:
        st.info("No index yet. Upload PDFs and click **Build / Rebuild Index**.")

# Build index
if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    with st.spinner("Reading PDFs and building index…"):
        # Save to temp & load
        paths = [save_uploaded_file_to_tmp(f) for f in uploaded_files]
        docs = load_pdfs(paths)
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.write(f"Found **{len(docs)} pages**, created **{len(chunks)} chunks**.")

        # Build vector store
        st.session_state.db = build_faiss_index(chunks)
        st.success("Index built successfully ✅")

# Chat box
st.subheader("💬 Ask questions about your PDFs")
query = st.chat_input("Type your question and press Enter…")

# Render history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

if query:
    if st.session_state.db is None:
        st.warning("Please upload PDFs and click **Build / Rebuild Index** first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        resp_placeholder = st.empty()
        with st.spinner("Thinking…"):
            answer, src_docs = answer_question(query, k=k)

        # Show answer
        resp_placeholder.markdown(answer)

        # Show sources in an expander
        with st.expander("📚 Sources"):
            for i, d in enumerate(src_docs, start=1):
                meta = d.metadata or {}
                src = meta.get("source", "?")
                page = meta.get("page", "?")
                st.markdown(f"**[{i}]** `{src}`, page **{page}**")
                # Let user peek into snippet if needed
                with st.popover(f"Preview {i}"):
                    st.write(d.page_content[:1000] + ("…" if len(d.page_content) > 1000 else ""))

        # Save history
        st.session_state.chat_history.append((query, answer))

st.caption(
    "Tip: Keep questions specific (who/what/when/where). If the PDFs don’t contain the info, "
    "the bot will say it doesn’t know."
)



