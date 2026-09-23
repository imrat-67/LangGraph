# app.py — Simple Streamlit chat using local Ollama (qwen3:0.6b) + MCP tools
# No memory sent to the LLM — every question is answered fresh.
# We only keep a list to re-draw what happened on screen (including tool use).

import json
import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

SERVERS = {
    "custom-tools": {
        "transport": "stdio",
        "command": "/home/imtiaj-hossain-saikat/.local/bin/uv",
        "args": [
            "run",
            "--project",
            "/home/imtiaj-hossain-saikat/Documents/BJIT/langgraph/day3/custom-mcp-server",
            "python",
            "/home/imtiaj-hossain-saikat/Documents/BJIT/langgraph/day3/custom-mcp-server/src/custom_mcp_server/server.py",
        ],
    }
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to tools. "
    "Use a tool only when the user's request needs it. "
    "After a tool runs, give a short, direct final answer."
)

st.set_page_config(page_title="Local MCP Chat", page_icon="🤖")
st.title("🤖 Local MCP Chat (qwen3:0.6b)")

# ---- one-time setup ----
if "ready" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    st.session_state.llm = ChatOllama(model="qwen3:0.6b", temperature=0)

    client = MultiServerMCPClient(SERVERS)
    tools = st.session_state.loop.run_until_complete(client.get_tools())
    st.session_state.tools_by_name = {t.name: t for t in tools}
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    st.session_state.log = []  # everything we've shown on screen so far
    st.session_state.ready = True


# ---- ask the model, run a tool if it asks for one, return the answer ----
async def ask(user_text):
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
    reply = await st.session_state.llm_with_tools.ainvoke(messages)

    if not reply.tool_calls:
        return reply.content, None  # no tool used

    tool_call = reply.tool_calls[0]  # keep it simple: just handle the first tool call
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool = st.session_state.tools_by_name[tool_name]
    result = await tool.ainvoke(tool_args)

    messages.append(reply)
    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(result)))
    final_reply = await st.session_state.llm.ainvoke(messages)

    return final_reply.content, tool_name


# ---- redraw everything that happened so far ----
for entry in st.session_state.log:
    if entry["type"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["text"])
    elif entry["type"] == "tool":
        st.success(f"✅ Ran tool: {entry['text']}")
    elif entry["type"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(entry["text"])


# ---- handle new input ----
user_text = st.chat_input("Type a message…")
if user_text:
    st.session_state.log.append({"type": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    answer, tool_used = st.session_state.loop.run_until_complete(ask(user_text))

    if tool_used:
        st.session_state.log.append({"type": "tool", "text": tool_used})
        st.success(f"✅ Ran tool: {tool_used}")

    st.session_state.log.append({"type": "assistant", "text": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
