import json
import sys
import asyncio
from pathlib import Path
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

SERVER_FILE = (
    Path(__file__).resolve().parent.parent
    / "custom-mcp-server" / "src" / "custom_mcp_server" / "server.py"
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

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to tools. "
    "Use a tool only when the user's request needs it. "
    "After a tool runs, give a short, direct final answer."
)

st.set_page_config(page_title="Local MCP Chat", page_icon="🤖")
st.title("🤖 Local MCP Chat (qwen3:0.6b)")

if "ready" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    st.session_state.llm = ChatOllama(model="qwen3:0.6b", temperature=0)

    client = MultiServerMCPClient(SERVERS)
    tools = st.session_state.loop.run_until_complete(client.get_tools())
    st.session_state.tools_by_name = {t.name: t for t in tools}
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    st.session_state.log = []  # everything we've shown on screen so far
    st.session_state.ready = True


async def ask(user_text):
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
    reply = await st.session_state.llm_with_tools.ainvoke(messages)

    if not reply.tool_calls:
        return reply.content, None  

    tool_call = reply.tool_calls[0]  
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool = st.session_state.tools_by_name[tool_name]
    result = await tool.ainvoke(tool_args)

    messages.append(reply)
    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(result)))
    final_reply = await st.session_state.llm.ainvoke(messages)

    return final_reply.content, tool_name


for entry in st.session_state.log:
    if entry["type"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["text"])
    elif entry["type"] == "tool":
        st.success(f"✅ Ran tool: {entry['text']}")
    elif entry["type"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(entry["text"])


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
