
import re
import os
import sys
import json
import uuid
import asyncio
import tempfile
from pathlib import Path
import streamlit as st
import psycopg
from psycopg.rows import dict_row

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

st.set_page_config(page_title="All-in-One Chatbot", page_icon="🤖")


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
MODEL_NAME = "qwen3:0.6b"
USER_ID = "saikat"  # fixed user, since there is no login system
SESSIONS_NS = ("user", USER_ID, "sessions")  # where the list of chats is saved

# The tool server from day 3. It lives next to this project. We run it with ITS
# OWN venv python (that venv has fastmcp); if it is missing, we fall back to the
# python that is running Streamlit.
SERVER_FILE = (
    Path(__file__).resolve().parent.parent
    / "day3" / "custom-mcp-server" / "src" / "custom_mcp_server" / "server.py"
)
_server_venv_python = SERVER_FILE.parents[2] / ".venv" / "Scripts" / "python.exe"
SERVER_PYTHON = str(_server_venv_python) if _server_venv_python.exists() else sys.executable
SERVERS = {
    "custom-tools": {
        "transport": "stdio",
        "command": SERVER_PYTHON,
        "args": [str(SERVER_FILE)],
    }
}

llm = ChatOllama(model=MODEL_NAME, temperature=0)

# ----------------------------
# Tools (day 3): load them ONCE and keep them (cached by Streamlit)
# ----------------------------
@st.cache_resource
def load_tools():
    loop = asyncio.new_event_loop()
    client = MultiServerMCPClient(SERVERS)
    tools = loop.run_until_complete(client.get_tools())
    return loop, tools

loop, tools = load_tools()
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)  # this model is ALLOWED to ask for tools

# ----------------------------
# PDF box (day 5): a small box that stays alive between reruns.
# It holds the searchable PDF (vector store). Empty = no PDF uploaded.
# ----------------------------
@st.cache_resource
def get_pdf_box():
    return {"vector_store": None, "file_name": None}

pdf_box = get_pdf_box()

# ----------------------------
# Node 1: remember_node -> LONG-TERM MEMORY (day 4)
# Ask the model: "is there a fact worth remembering?" If yes, save it in Postgres.
# ----------------------------
def remember_node(state: MessagesState, config, *, store: BaseStore):
    ns = ("user", USER_ID, "facts")
    last_user_msg = state["messages"][-1].content

    prompt = (
        "Read the message below. If it contains a fact worth remembering "
        "about the user forever (name, job, likes, project, etc.), reply "
        "with ONLY that fact in one short sentence. If there is no such "
        "fact, reply with exactly: NONE\n\n"
        f"Message: {last_user_msg}\n/no_think"
    )
    fact = llm.invoke(prompt).content.strip()

    if fact and not fact.upper().startswith("NONE"):
        store.put(ns, str(uuid.uuid4()), {"data": fact})
    return {}


# ----------------------------
# Node 2: chat_node -> the model answers, using EVERYTHING it has:
#   - short-term memory : state["messages"] (loaded automatically by PostgresSaver)
#   - long-term memory  : facts read from PostgresStore
#   - RAG               : the best PDF chunks (only if a PDF was uploaded)
#   - tools             : the model may ask for a tool instead of answering
# ----------------------------
def chat_node(state: MessagesState, config, *, store: BaseStore):
    # long-term memory (day 4)
    saved_facts = store.search(("user", USER_ID, "facts"))
    facts_text = (
        "\n".join(f"- {item.value['data']}" for item in saved_facts)
        if saved_facts
        else "(nothing yet)"
    )

    # RAG (day 5): search the PDF using the user's latest question
    context = "(no PDF uploaded)"
    if pdf_box["vector_store"] is not None:
        question = [m for m in state["messages"] if m.type == "human"][-1].content
        retriever = pdf_box["vector_store"].as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        results = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in results)

    system_msg = SystemMessage(
        content=(
            "You are a friendly assistant.\n"
            "You have tools (calculate_age, random_number). Use a tool only when "
            "the request needs it. After a tool runs, give a short, direct final answer.\n\n"
            f"Here is what you remember about the user:\n{facts_text}\n"
            "Use this naturally if it is relevant.\n\n"
            "Text from the user's PDF (use it only if the question is about the PDF):\n"
            f"{context}\n/no_think"
        )
    )

    # Tools (day 3): if a tool just ran, answer now with the plain model.
    # Otherwise use the model that is allowed to ask for a tool.
    if state["messages"][-1].type == "tool":
        model = llm
    else:
        model = llm_with_tools

    response = model.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


# ----------------------------
# Node 3: tool_node -> runs the tool(s) the model asked for (day 3)
# ----------------------------
def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = loop.run_until_complete(tool.ainvoke(tool_call["args"]))
        results.append(
            ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(result))
        )
    return {"messages": results}


# ----------------------------
# Router: after the chat node, did the model ask for a tool?
# ----------------------------
def need_tool(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# ----------------------------
# Build the graph
# ----------------------------
builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_conditional_edges("chat", need_tool, ["tools", END])
builder.add_edge("tools", "chat")


# ----------------------------
# Open Postgres connections ONCE and reuse them (day 4)
# ----------------------------
@st.cache_resource
def load_graph():
    store_conn = psycopg.connect(
        DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    store = PostgresStore(store_conn)
    store.setup()  # creates LTM tables (safe to run every start)

    checkpointer_conn = psycopg.connect(
        DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    checkpointer = PostgresSaver(checkpointer_conn)
    checkpointer.setup()  # creates STM tables (safe to run every start)

    compiled = builder.compile(checkpointer=checkpointer, store=store)
    return compiled, store

graph, store = load_graph()


# ----------------------------
# Chat session helpers (day 4) -> power the sidebar chat list
# ----------------------------
def list_sessions():
    items = store.search(SESSIONS_NS)
    items = sorted(items, key=lambda item: item.created_at, reverse=True)
    return [(item.key, item.value["title"]) for item in items]


def create_session() -> str:
    thread_id = str(uuid.uuid4())
    store.put(SESSIONS_NS, thread_id, {"title": "New chat"})
    return thread_id


def rename_session(thread_id: str, title: str):
    short_title = title.strip().replace("\n", " ")[:40]
    store.put(SESSIONS_NS, thread_id, {"title": short_title or "New chat"})


def load_session_messages(thread_id: str):
    """Read one chat from Postgres and turn it into simple dicts for the screen."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    messages = []
    for msg in snapshot.values.get("messages", []):
        if msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        elif msg.type == "ai" and msg.tool_calls:
            for tool_call in msg.tool_calls:
                messages.append({"role": "tool", "content": tool_call["name"]})
        elif msg.type == "ai" and msg.content:
            messages.append({"role": "assistant", "content": msg.content})
        # (tool result messages are skipped: the answer already uses them)
    return messages


def clean(text: str) -> str:
    """Qwen3 sometimes adds a <think>...</think> block. Remove it (day 5)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ----------------------------
# PDF -> chunks -> embeddings -> vector store (day 5)
# ----------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    docs = PyPDFLoader(tmp_path).load()
    os.remove(tmp_path)

    if len("".join(doc.page_content for doc in docs).strip()) == 0:
        st.error("This PDF has no readable text (maybe scanned). Try another PDF.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pdf_box["vector_store"] = FAISS.from_documents(chunks, embeddings)
    pdf_box["file_name"] = uploaded_file.name
    return len(chunks)


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🤖 All-in-One Chatbot")
st.caption(f"Model: {MODEL_NAME} | Memory + Tools + PDF answers, all built in")

# ---- Sidebar: new chat button, PDF upload, list of past chats ----
with st.sidebar:
    st.header("📄 Your PDF (optional)")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is None:
        pdf_box["vector_store"] = None  # no PDF -> the chatbot just doesn't use one
        pdf_box["file_name"] = None
    elif pdf_box["file_name"] != uploaded_file.name:
        with st.spinner("Reading the PDF..."):
            chunk_count = process_pdf(uploaded_file)
        if chunk_count > 0:
            st.success(f"PDF ready! ({chunk_count} chunks)")
    else:
        st.success(f"Using: {pdf_box['file_name']}")

    st.divider()
    st.header("💬 Chats")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.thread_id = create_session()
        st.session_state.messages = []
        st.rerun()

    for thread_id, title in list_sessions():
        is_active = st.session_state.get("thread_id") == thread_id
        label = f"🟢 {title}" if is_active else title
        if st.button(label, key=f"session_{thread_id}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.session_state.messages = load_session_messages(thread_id)
            st.rerun()

# ---- First time this tab opens: resume the latest chat, or start one ----
if "thread_id" not in st.session_state:
    sessions = list_sessions()
    if sessions:
        st.session_state.thread_id = sessions[0][0]
        st.session_state.messages = load_session_messages(sessions[0][0])
    else:
        st.session_state.thread_id = create_session()
        st.session_state.messages = []

# ---- Show the messages of the ACTIVE chat ----
for msg in st.session_state.messages:
    if msg["role"] == "tool":
        st.success(f"✅ Ran tool: {msg['content']}")
    else:
        with st.chat_message(msg["role"]):
            st.write(clean(msg["content"]))

# ---- Chat box ----
if user_input := st.chat_input("Type a message..."):
    # the first message names the chat in the sidebar
    if len(st.session_state.messages) == 0:
        rename_session(st.session_state.thread_id, user_input)

    with st.chat_message("user"):
        st.write(user_input)

    # run the whole graph: remember -> chat -> (tools -> chat)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    # reload the chat from Postgres and redraw the page
    st.session_state.messages = load_session_messages(st.session_state.thread_id)
    st.rerun()
