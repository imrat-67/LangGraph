
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import tempfile
import os
import re

st.set_page_config(page_title="Simple RAG Chatbot", page_icon="📄")
st.title("📄 Simple RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:

    is_new_file = st.session_state.get("file_name") != uploaded_file.name

    if "vector_store" not in st.session_state or is_new_file:
        with st.spinner("Reading and processing PDF... please wait"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            os.remove(tmp_path)
            total_text = "".join(doc.page_content for doc in docs).strip()

            if len(total_text) == 0:
                st.error(
                    "This PDF doesn't seem to have any readable text "
                    "(it might be a scanned document / images only). "
                    "Please try a different PDF that has selectable text."
                )
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            if len(chunks) == 0:
                st.error("Could not create any text chunks from this PDF. Please try another file.")
                st.stop()

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state.vector_store = vector_store
            st.session_state.file_name = uploaded_file.name

        st.success(f"PDF processed! Created {len(chunks)} chunks.")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):

            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            results = retriever.invoke(question)

            context = "\n\n".join(doc.page_content for doc in results)

            prompt = f"""/no_think
                    Answer the question using ONLY the context below.
                    If the answer is not in the context, say you don't know.
                    Keep the answer short and clear.

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""


            llm = ChatOllama(model="qwen3:0.6b", temperature=0)
            response = llm.invoke(prompt)

            answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

            st.markdown("### Answer")
            st.write(answer if answer else "(The model didn't return a clear answer. Try rephrasing your question.)")

            with st.expander("Show retrieved context (what the AI actually read)"):
                st.write(context)

else:
    st.info("👆 Please upload a PDF to get started.")
