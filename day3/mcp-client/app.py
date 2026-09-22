# app.py — Streamlit chat client that talks to multiple MCP servers at once
# (3 local stdio servers + 1 deployed remote server) through one Ollama LLM.
#
# Built directly on the `mcp` SDK (not langchain-mcp-adapters, whose latest
# release is incompatible with the mcp>=2.2 API installed in this project).
#
# Run with:  streamlit run app.py
# Prereqs:   `ollama serve` running locally, with a tool-calling model pulled
#            (default here: qwen3:8b — change OLLAMA_MODEL below if you like).

import os
import json
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack

import streamlit as st
from dotenv import load_dotenv
import httpx2

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# ─────────────────────────────
# Paths (this file lives in day3/mcp-client)
# ─────────────────────────────
DAY3_DIR = Path(__file__).resolve().parent.parent  # .../day3

CUSTOM_TOOLS_DIR = DAY3_DIR / "custom-mcp-server"
MANIM_DIR = DAY3_DIR / "manim-mcp-server"
WEATHER_DIR = DAY3_DIR / "mcp-weather"

load_dotenv(WEATHER_DIR / ".env")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "")

# Optional local .env for this client itself (OLLAMA_MODEL, FASTMCP_DEPLOYED_TOKEN, ...)
load_dotenv(Path(__file__).resolve().parent / ".env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEPLOYED_URL = "https://imrat-67-server.fastmcp.app/mcp"
# The deployed server (FastMCP Cloud / Prefect Horizon) requires OAuth by default.
# Get a static API token from its dashboard and put it in mcp-client/.env as:
#   FASTMCP_DEPLOYED_TOKEN=your-token-here
FASTMCP_DEPLOYED_TOKEN = os.getenv("FASTMCP_DEPLOYED_TOKEN", "")

# ─────────────────────────────
# MCP servers: 3 local (stdio, each in its own project .venv) + 1 remote (deployed)
# ─────────────────────────────
SERVERS = {
    "custom-tools": {
        "transport": "stdio",
        "command": str(CUSTOM_TOOLS_DIR / ".venv" / "Scripts" / "python.exe"),
        "args": [str(CUSTOM_TOOLS_DIR / "src" / "custom_mcp_server" / "server.py")],
    },
    "manim-server": {
        "transport": "stdio",
        "command": str(MANIM_DIR / ".venv" / "Scripts" / "python.exe"),
        "args": [str(MANIM_DIR / "src" / "manim_server.py")],
    },
    "weather": {
        "transport": "stdio",
        "command": str(WEATHER_DIR / ".venv" / "Scripts" / "python.exe"),
        "args": [str(WEATHER_DIR / "mcp_weather" / "weather.py")],
        "env": {**os.environ, "ACCUWEATHER_API_KEY": ACCUWEATHER_API_KEY},
    },
    "deployed": {
        "transport": "streamable_http",  # deployed via prefect.io/horizon/deploy
        "url": DEPLOYED_URL,
        "token": FASTMCP_DEPLOYED_TOKEN or None,
    },
}


class MCPManager:
    """Opens every configured MCP server once and keeps the sessions alive
    for the lifetime of the app, exposing a flat, LLM-ready tool list."""

    def __init__(self, servers: dict):
        self.servers = servers
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[dict] = []   # [{full_name, server, name, description, schema}]
        self.errors: dict[str, str] = {}

    async def connect_all(self):
        for key, cfg in self.servers.items():
            try:
                if cfg["transport"] == "stdio":
                    params = StdioServerParameters(
                        command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env")
                    )
                    read, write = await self.stack.enter_async_context(stdio_client(params))
                elif cfg["transport"] == "streamable_http":
                    http_client = None
                    if cfg.get("token"):
                        http_client = httpx2.AsyncClient(
                            headers={"Authorization": f"Bearer {cfg['token']}"}, timeout=30
                        )
                    read, write = await self.stack.enter_async_context(
                        streamable_http_client(cfg["url"], http_client=http_client)
                    )
                else:
                    raise ValueError(f"unknown transport {cfg['transport']}")

                session = await self.stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions[key] = session

                listed = await session.list_tools()
                for t in listed.tools:
                    self.tools.append(
                        {
                            "full_name": f"{key}__{t.name}",
                            "server": key,
                            "name": t.name,
                            "description": t.description or "",
                            "schema": t.input_schema or {"type": "object", "properties": {}},
                        }
                    )
            except Exception as e:
                self.errors[key] = str(e)

    def as_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["full_name"],
                    "description": t["description"],
                    "parameters": t["schema"],
                },
            }
            for t in self.tools
        ]

    async def call(self, full_name: str, args: dict):
        entry = next((t for t in self.tools if t["full_name"] == full_name), None)
        if entry is None:
            return f"Error: unknown tool '{full_name}'"
        session = self.sessions[entry["server"]]
        result = await session.call_tool(entry["name"], args)
        texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        if texts:
            return "\n".join(texts)
        if result.structured_content is not None:
            return json.dumps(result.structured_content, default=str)
        return str(result.content)


SYSTEM_PROMPT = (
    "You have access to tools from several connected MCP servers "
    "(age/random-number calculator, a Manim animation renderer, a weather "
    "forecaster, and a deployed remote copy of the calculator tools). "
    "When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

st.set_page_config(page_title="Multi-MCP Chat", page_icon="🧰", layout="centered")
st.title("🧰 Multi-MCP Chat")
st.caption(f"Model: {OLLAMA_MODEL} · Servers: {', '.join(SERVERS.keys())}")

# ─────────────────────────────
# One-time init (persistent event loop + MCP sessions + LLM)
# ─────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

    st.session_state.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    manager = MCPManager(SERVERS)
    st.session_state.loop.run_until_complete(manager.connect_all())
    st.session_state.manager = manager

    tool_defs = manager.as_openai_tools()
    st.session_state.llm_with_tools = (
        st.session_state.llm.bind_tools(tool_defs) if tool_defs else st.session_state.llm
    )

    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True

with st.sidebar:
    st.subheader("Connected servers")
    for key in SERVERS:
        if key in st.session_state.manager.sessions:
            n = sum(1 for t in st.session_state.manager.tools if t["server"] == key)
            st.markdown(f"✅ **{key}** — {n} tool(s)")
        else:
            st.markdown(f"❌ **{key}** — {st.session_state.manager.errors.get(key, 'failed')}")
    st.subheader("Tools")
    for t in st.session_state.manager.tools:
        st.markdown(f"- `{t['full_name']}`")

# ─────────────────────────────
# Render history (skip system + tool + intermediate tool-call messages)
# ─────────────────────────────
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ─────────────────────────────
# Run one full turn: loop tool calls until the model gives a final answer
# ─────────────────────────────
MAX_TOOL_ROUNDS = 5


async def run_turn():
    for _ in range(MAX_TOOL_ROUNDS):
        response = await st.session_state.llm_with_tools.ainvoke(st.session_state.history)
        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            return response

        st.session_state.history.append(response)

        with st.chat_message("assistant"):
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass

                with st.status(f"Running tool: {name}", expanded=True) as status:
                    st.write(f"Arguments: {args}")
                    try:
                        result = await st.session_state.manager.call(name, args)
                    except Exception as e:
                        result = f"Error running {name}: {e}"
                    status.update(label=f"✅ Ran tool: {name}", state="complete", expanded=False)

                st.session_state.history.append(
                    ToolMessage(tool_call_id=tc["id"], content=json.dumps(result, default=str))
                )

    return await st.session_state.llm.ainvoke(st.session_state.history)


# ─────────────────────────────
# Chat input
# ─────────────────────────────
user_text = st.chat_input("Type a message…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    final = st.session_state.loop.run_until_complete(run_turn())

    with st.chat_message("assistant"):
        st.markdown(final.content or "")
    st.session_state.history.append(AIMessage(content=final.content or ""))
