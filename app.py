import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chatbot", layout="wide")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY in Secrets"

# Sidebar: PDF upload
st.sidebar.header("Upload PDFs")
files = st.sidebar.file_uploader("", type="pdf", accept_multiple_files=True)

db_path = "faiss_index"
if files:
    docs = []
    for f in files:
        loader = PyPDFLoader(f)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_path)
    st.sidebar.success("Index ready!")

st.title("RAG Chatbot")
if os.path.exists(db_path):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(db_path, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY, streaming=True, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    q = st.text_input("Your question:")
    if q:
        with st.spinner("Thinking…"):
            res = qa(q)
        st.markdown("**Answer:**")
        st.write(res["result"])
        st.markdown("---")
        st.markdown("**Sources:**")
        for src in res["source_documents"]:
            st.write(f"- Page {src.metadata.get('page_number','?')}")
else:
    st.info("Upload PDFs to begin.")
    

