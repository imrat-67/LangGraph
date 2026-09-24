# Simple RAG (Retrieval Augmented Generation) app
# No memory, no agent, no tools -> just: PDF -> chunks -> embeddings -> retrieve -> ask LLM

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
st.title("📄 Simple RAG Chatbot (Qwen3 0.6B)")

st.write(
    "Upload a PDF, then ask questions about it. "
    "The app finds the most relevant pieces of the PDF and gives them to the AI model to answer."
)

# ---------- STEP 1: Upload PDF ----------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:

    # Only re-process the PDF if it's a new file (so we don't redo work every time)
    is_new_file = st.session_state.get("file_name") != uploaded_file.name

    if "vector_store" not in st.session_state or is_new_file:
        with st.spinner("Reading and processing PDF... please wait"):

            # Save the uploaded file to a temporary location on disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # ---------- STEP 2: Load the PDF ----------
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            os.remove(tmp_path)

            # Check that the PDF actually has selectable/extractable text.
            # Scanned PDFs (photos of pages) have NO text layer, so PyPDFLoader
            # returns empty pages, and there would be nothing to search on.
            total_text = "".join(doc.page_content for doc in docs).strip()

            if len(total_text) == 0:
                st.error(
                    "This PDF doesn't seem to have any readable text "
                    "(it might be a scanned document / images only). "
                    "Please try a different PDF that has selectable text."
                )
                st.stop()

            # ---------- STEP 3: Split into small chunks ----------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            if len(chunks) == 0:
                st.error("Could not create any text chunks from this PDF. Please try another file.")
                st.stop()

            # ---------- STEP 4: Create embeddings + vector store ----------
            # Embeddings turn text into numbers so we can search by "meaning"
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Save in session_state so we don't rebuild it on every question
            st.session_state.vector_store = vector_store
            st.session_state.file_name = uploaded_file.name

        st.success(f"PDF processed! Created {len(chunks)} chunks.")

    # ---------- STEP 5: Ask a question ----------
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):

            # Find the most relevant chunks for this question
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            results = retriever.invoke(question)

            # Combine the chunks into one block of text
            context = "\n\n".join(doc.page_content for doc in results)

            # ---------- STEP 6: Build the prompt ----------
            # "/no_think" tells Qwen3 to skip its internal reasoning step and
            # just answer directly - simpler and faster for a small model.
            prompt = f"""/no_think
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.
Keep the answer short and clear.

Context:
{context}

Question: {question}

Answer:"""

            # ---------- STEP 7: Ask the local Qwen3 model (via Ollama) ----------
            # temperature=0 makes the model stick closely to the context
            # instead of getting creative / making things up.
            llm = ChatOllama(model="qwen3:0.6b", temperature=0)
            response = llm.invoke(prompt)

            # Qwen3 sometimes still adds a <think>...</think> block before the
            # real answer. We remove it so only the final answer is shown.
            answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

            # ---------- STEP 8: Show the answer ----------
            st.markdown("### Answer")
            st.write(answer if answer else "(The model didn't return a clear answer. Try rephrasing your question.)")

            with st.expander("Show retrieved context (what the AI actually read)"):
                st.write(context)

else:
    st.info("👆 Please upload a PDF to get started.")
