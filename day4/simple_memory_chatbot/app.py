import uuid
import re
import streamlit as st
import psycopg
from psycopg.rows import dict_row

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
MODEL_NAME = "qwen3:1.7b"   # 0.6b was too small to follow instructions reliably (LTM was breaking)
USER_ID = "saikat"
SESSIONS_NS = ("user", USER_ID, "sessions")

llm = ChatOllama(model=MODEL_NAME)


def clean(text: str) -> str:
    """Remove Qwen's <think>...</think> reasoning block, keep only the real answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------- Step 1: save any fact worth remembering (LTM) ----------
def remember_node(state: MessagesState, store: BaseStore):
    ns = ("user", USER_ID, "facts")
    last_user_msg = state["messages"][-1].content

    prompt = (
        "/no_think\n"
        "Read the message below. If it contains a fact worth remembering "
        "about the user forever (name, job, likes, project, etc.), reply "
        "with ONLY that fact in one short sentence. If there is no such "
        "fact, reply with exactly: NONE\n\n"
        f"Message: {last_user_msg}"
    )

    fact = clean(llm.invoke(prompt).content)

    if fact and fact.upper() != "NONE":
        store.put(ns, str(uuid.uuid4()), {"data": fact})

    return {}


# ---------- Step 2: answer the user, using saved facts as memory (STM + LTM) ----------
def chat_node(state: MessagesState, store: BaseStore):
    ns = ("user", USER_ID, "facts")
    saved_facts = store.search(ns)
    facts_text = "\n".join(f"- {item.value['data']}" for item in saved_facts) or "(nothing yet)"

    system_msg = SystemMessage(
        content=(
            "/no_think\n"
            "You are a friendly assistant.\n"
            "Here is what you remember about the user:\n"
            f"{facts_text}\n"
            "Use this naturally if it is relevant."
        )
    )

    response = llm.invoke([system_msg] + state["messages"])
    response.content = clean(response.content)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")


@st.cache_resource
def load_graph():
    store_conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    store = PostgresStore(store_conn)
    store.setup()

    checkpointer_conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    checkpointer = PostgresSaver(checkpointer_conn)
    checkpointer.setup()

    return builder.compile(checkpointer=checkpointer, store=store), store


graph, store = load_graph()

st.title("🧠 Simple Memory Chatbot")
st.caption(f"Model: {MODEL_NAME} | STM + LTM via Postgres")


# ---------- Session helpers (each session = one saved chat thread) ----------
def list_sessions():
    items = sorted(store.search(SESSIONS_NS), key=lambda item: item.created_at, reverse=True)
    return [(item.key, item.value["title"]) for item in items]


def new_session() -> str:
    thread_id = str(uuid.uuid4())
    store.put(SESSIONS_NS, thread_id, {"title": "New chat"})
    return thread_id


def load_messages(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return [
        {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        for msg in snapshot.values.get("messages", [])
    ]


def switch_to(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_messages(thread_id)


# ---------- Sidebar: pick a chat or start a new one ----------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        switch_to(new_session())
        st.rerun()

    st.divider()
    for thread_id, title in list_sessions():
        active = st.session_state.get("thread_id") == thread_id
        if st.button(("🟢 " if active else "") + title, key=thread_id, use_container_width=True):
            switch_to(thread_id)
            st.rerun()

# start on the newest chat, or make one if there isn't any yet
if "thread_id" not in st.session_state:
    sessions = list_sessions()
    switch_to(sessions[0][0] if sessions else new_session())


# ---------- Show the conversation so far ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Handle a new message ----------
if user_input := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if len(st.session_state.messages) == 1:
        title = user_input.strip().replace("\n", " ")[:40] or "New chat"
        store.put(SESSIONS_NS, st.session_state.thread_id, {"title": title})

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
            reply = result["messages"][-1].content
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
