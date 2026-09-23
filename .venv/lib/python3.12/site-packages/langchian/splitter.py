from pathlib import Path
import re
import time
from pathlib import Path

RAG_APP_CODE = r'''
import os, json, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "{{URL}}"
DOC_FILE = "documents.txt"
MAX_TEXT_CHARS = 200_000

st.title("RAG Chat App")

st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("ready", False)
st.session_state.setdefault("messages", [])

if st.sidebar.button("Create vector DB"):
    st.write("Preparing data...")
    full_text = ""

    if os.path.exists(DOC_FILE):
        with open(DOC_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        response = requests.get(API_URL, timeout=60)
        response.raise_for_status()
        df = pd.DataFrame.from_dict(response.json(), orient="index").reset_index()

        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown").astype(str)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.lower().str.strip()

        for _, row in df.iterrows():
            full_text += " | ".join([f"{c}: {row[c]}" for c in df.columns]) + "\n"

        with open(DOC_FILE, "w", encoding="utf-8") as f:
            f.write(full_text)

    if len(full_text) > MAX_TEXT_CHARS:
        full_text = full_text[:MAX_TEXT_CHARS]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = splitter.create_documents([full_text])

    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    st.session_state.vector_db = FAISS.from_documents(documents, embeddings)
    st.session_state.ready = True
    st.success("Vector database is ready. You can now chat.")

# History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context"):
            with st.expander("Retrieved Context"):
                st.markdown(msg["context"])

# Chat
user_input = st.chat_input("Ask something about the data")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_context = ""

    if not st.session_state.ready:
        bot_reply = "Please create the vector database first."
    else:
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst assistant.\nUse the provided context.\nDo NOT use external knowledge."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

        results = st.session_state.vector_db.similarity_search(user_input, k=1)
        if results:
            retrieved_context = results[0].page_content.strip()
            if retrieved_context == "..." or len(retrieved_context) < 10:
                retrieved_context = "Retrieved context was too short or invalid."
        else:
            retrieved_context = "No relevant context found."

        response = llm.invoke(prompt.format(question=user_input, context=retrieved_context))
        bot_reply = response.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "context": retrieved_context
    })

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if retrieved_context:
            with st.expander("Retrieved Context"):
                st.markdown(retrieved_context)

'''


RAG_CODE_V2 = r'''
import streamlit as st
import os, json, requests, pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
SRC = "{{URL}}"
CACHE = "data.txt"

st.set_page_config("Raw RAG", layout="centered")
st.title("Raw Data QA")

if "db" not in st.session_state:
    st.session_state.db = None

def prepare_data():
    if os.path.exists(CACHE):
        return open(CACHE, encoding="utf-8").read()

    raw = requests.get(SRC, timeout=60).json()
    df = pd.DataFrame.from_dict(raw, orient="index").reset_index()
    df = df.head(300)

    for c in df.columns:
        df[c] = df[c].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    df = df.fillna("na").astype(str)

    lines = []
    for _, row in df.iterrows():
        lines.append(" | ".join(row.values))

    text = "\n".join(lines)
    text = text[:60000]
    open(CACHE, "w", encoding="utf-8").write(text)
    return text

def build_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    emb = OpenAIEmbeddings(api_key=KEY)
    vectors = []

    bar = st.progress(0)
    total = len(docs)

    for i, d in enumerate(docs):
        v = emb.embed_documents([d.page_content])[0]
        vectors.append((d.page_content, v))
        bar.progress((i + 1) / total)

    return FAISS.from_embeddings(vectors, emb)

if st.button("Load & Index"):
    if st.session_state.db:
        st.warning("Already built. Restart app to rebuild.")
    else:
        data = prepare_data()
        st.session_state.db = build_index(data)
        st.success("Index built")

query = st.text_input("Enter question")

if st.button("Ask"):
    if not st.session_state.db:
        st.text("Build index first.")
    elif not query.strip():
        st.text("Enter a question.")
    else:
        llm = ChatOpenAI(model="gpt-4o-mini")
        tmpl = ChatPromptTemplate.from_messages([
            ("system","Answer only from data."),
            ("human","Q: {q}\n\nDATA:\n{d}")
        ])
        hit = st.session_state.db.similarity_search(query, k=1)
        ctx = hit[0].page_content if hit else "No data."
        ans = llm.invoke(tmpl.format(q=query, d=ctx)).content
        st.text("Answer:")
        st.text(ans)
'''


def get_code_by_version(keep_first, url):
    if keep_first:
        base = RAG_APP_CODE
    else:
        base = RAG_CODE_V2   # default

    return base.replace("{{URL}}", url)

# ----------------- TYPE-WRITER FUNCTION -----------------

def split_document(url, filename="app.py", delay=0.025, keep_first=False):
    start_time = time.time()
    path = Path(filename)

    final_code = get_code_by_version(keep_first, url)

    with path.open("w", encoding="utf-8") as f:
        tokens = re.findall(r'\S+|\s+', final_code)

        for tok in tokens:
            if tok.isspace():
                f.write(tok)
                f.flush()
                time.sleep(delay * 0.8)
            else:
                for ch in tok:
                    f.write(ch)
                    f.flush()
                    if ch in "=:+-*/<>;,.(){}[]":
                        time.sleep(delay * 1.5)
                    elif ch.isdigit():
                        time.sleep(delay * 1.5)
                    else:
                        time.sleep(delay)






