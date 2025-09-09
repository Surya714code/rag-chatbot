import os
import streamlit as st
import tempfile
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chatbot", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY in Streamlit Secrets"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.sidebar.header("Upload PDFs (multiple allowed)")
uploaded_files = st.sidebar.file_uploader("", accept_multiple_files=True, type="pdf")

db_path = "faiss_index"

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = UnstructuredPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_path)
    st.sidebar.success("Index built/updated!")

st.title("📄 RAG Chatbot")

if os.path.exists(db_path):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Thinking..."):
            result = qa(query)

        st.markdown("**Answer:**")
        st.write(result["result"])
        st.markdown("---")
        st.markdown("**Sources:**")
        for doc in result["source_documents"]:
            page_no = doc.metadata.get("page_number", "?")
            st.write(f"- Page {page_no}")

else:
    st.info("Upload one or more PDFs to build the search index.")


