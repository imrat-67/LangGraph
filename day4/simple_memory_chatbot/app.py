# Simple Memory Chatbot (Streamlit + LangGraph + Postgres + Qwen3 0.6B)
# ------------------------------------------------------------------
# STM (Short-Term Memory) = the current conversation (via PostgresSaver)
#                            -> each chat "session" gets its OWN thread_id
# LTM (Long-Term Memory)  = facts remembered about the user forever (via PostgresStore)
#                            -> shared across ALL of the user's chat sessions
# No tools. No RAG. Just memory.

import uuid
import streamlit as st
import psycopg
from psycopg.rows import dict_row

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

# ----------------------------
# Settings
# ----------------------------
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
MODEL_NAME = "qwen3:0.6b"
USER_ID = "saikat"  # fixed user, since there is no login system

# The namespace where we keep the LIST of chat sessions itself.
# It's just another entry in the same LTM store, separate from the
# "facts" namespace used below -- so switching chats also survives
# an app restart, same as the remembered facts do.
SESSIONS_NS = ("user", USER_ID, "sessions")

llm = ChatOllama(model=MODEL_NAME)


# ----------------------------
# Node 1: remember_node -> this is the LONG-TERM MEMORY step
# After every user message, ask the model: "is there a fact worth
# remembering forever?" If yes, save it in Postgres (LTM store).
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

    if fact and fact.upper() != "NONE":
        store.put(ns, str(uuid.uuid4()), {"data": fact})

    return {}

# ----------------------------
# Node 2: chat_node -> this uses BOTH memories
# - STM: state["messages"] already holds the full thread history
#        (LangGraph loads it automatically from PostgresSaver, scoped
#        to whichever thread_id / session you're currently in)
# - LTM: we read saved facts from PostgresStore and put them in the
#        system prompt, so the model "remembers" the user in EVERY session
# ----------------------------
def chat_node(state: MessagesState, config, *, store: BaseStore):
    ns = ("user", USER_ID, "facts")

    saved_facts = store.search(ns)
    facts_text = (
        "\n".join(f"- {item.value['data']}" for item in saved_facts)
        if saved_facts
        else "(nothing yet)"
    )

    system_msg = SystemMessage(
        content=(
            "You are a friendly assistant.\n"
            "Here is what you remember about the user:\n"
            f"{facts_text}\n"
            "Use this naturally if it is relevant. /no_think"
        )
    )

    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# ----------------------------
# Build the graph: START -> remember -> chat
# ----------------------------
builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")


# ----------------------------
# Open Postgres connections ONCE and reuse them (cached by Streamlit)
#
# NOTE: PostgresStore.from_conn_string(...) / PostgresSaver.from_conn_string(...)
# are @contextmanager generators: internally they do `with Connection.connect(...) as conn:`.
# Calling `.__enter__()` on them without keeping a reference to the context-manager
# object itself is a trap -- that CM object has no other owner, so Python garbage
# collects it almost immediately, which closes the generator and therefore the
# underlying connection too. That's why store.setup() used to fail with
# "the connection is closed".
#
# Fix: open plain psycopg connections ourselves and hand them to PostgresStore /
# PostgresSaver directly (both accept a raw connection). st.cache_resource then
# keeps these connections alive for the lifetime of the app.
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
    # We return the store too, so the sidebar code further down can read/write
    # the session list directly, without running the whole graph for that.
    return compiled, store

graph, store = load_graph()


# ----------------------------
# Session helpers -- this is what powers the "ChatGPT-style" sidebar.
#
# Each "session" shown in the sidebar = one separate STM thread_id.
# We keep the LIST of sessions (id + title) in the LTM store, under
# SESSIONS_NS, so the sidebar survives app restarts and browser refreshes.
# ----------------------------
def list_sessions():
    """Return all chat sessions, newest first: [(thread_id, title), ...]"""
    items = store.search(SESSIONS_NS)
    items = sorted(items, key=lambda item: item.created_at, reverse=True)
    return [(item.key, item.value["title"]) for item in items]


def create_session() -> str:
    """Create a brand-new empty chat session and return its thread_id."""
    thread_id = str(uuid.uuid4())
    store.put(SESSIONS_NS, thread_id, {"title": "New chat"})
    return thread_id


def rename_session(thread_id: str, title: str):
    """Give a session a nicer title -- we reuse the user's first message,
    the same way ChatGPT names a new chat after your first question."""
    short_title = title.strip().replace("\n", " ")[:40]
    store.put(SESSIONS_NS, thread_id, {"title": short_title or "New chat"})

def load_session_messages(thread_id: str):
    """Read the STM (conversation history) for one session out of Postgres,
    and turn it into the simple {"role", "content"} dicts the UI displays."""
    config = {"configurable": {"thread_id": thread_id, "user_id": USER_ID}}
    snapshot = graph.get_state(config)

    messages = []
    for msg in snapshot.values.get("messages", []):
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
    return messages


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 Simple Memory Chatbot")
st.caption(f"Model: {MODEL_NAME} | STM + LTM via Postgres | No tools, no RAG")

# ---- Sidebar: "New chat" button + list of past sessions ----
with st.sidebar:
    st.header("💬 Chats")

    if st.button("➕ New chat", use_container_width=True):
        new_id = create_session()
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        st.rerun()

    st.divider()

    for thread_id, title in list_sessions():
        is_active = st.session_state.get("thread_id") == thread_id
        label = f"🟢 {title}" if is_active else title
        if st.button(label, key=f"session_{thread_id}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.session_state.messages = load_session_messages(thread_id)
            st.rerun()

# ---- First time this browser tab opens: resume last chat, or start one ----
if "thread_id" not in st.session_state:
    sessions = list_sessions()
    if sessions:
        latest_thread_id, _ = sessions[0]
        st.session_state.thread_id = latest_thread_id
        st.session_state.messages = load_session_messages(latest_thread_id)
    else:
        st.session_state.thread_id = create_session()
        st.session_state.messages = []

# show past messages of the ACTIVE session only
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])



# ----------------------------
# Chat input -- this is where you actually type to the bot.
# Everything above just set the stage; this box is the missing piece
# that makes the app usable.
# ----------------------------
if user_input := st.chat_input("Type a message..."):
    # 1) show the user's own message right away
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2) if this is the FIRST message of this session, use it to name
    #    the chat in the sidebar -- same trick ChatGPT uses
    if len(st.session_state.messages) == 1:
        rename_session(st.session_state.thread_id, user_input)

    # 3) run the graph for THIS session's thread_id.
    #    - STM: the checkpointer auto-loads this thread's past messages
    #    - LTM: the "remember" + "chat" nodes read/write facts in Postgres
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )
            reply = result["messages"][-1].content
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 4) refresh the page so the sidebar shows the new/renamed session
    st.rerun()
