import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app import SERVERS, MCPManager, SYSTEM_PROMPT, OLLAMA_MODEL
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

async def main():
    mgr = MCPManager(SERVERS)
    await mgr.connect_all()
    tools = mgr.as_openai_tools()
    print("Bound tools:", [t["function"]["name"] for t in tools])

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    llm_t = llm.bind_tools(tools)

    history = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Give me a random number between 1 and 10, and also tell me the age of someone born 2000-05-15."),
    ]

    for round_ in range(5):
        resp = await llm_t.ainvoke(history)
        calls = getattr(resp, "tool_calls", None)
        print(f"--- round {round_}: tool_calls={calls}")
        if not calls:
            print("FINAL:", resp.content)
            break
        history.append(resp)
        for tc in calls:
            args = tc.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args)
            result = await mgr.call(tc["name"], args)
            print(f"  tool {tc['name']}({args}) -> {result}")
            history.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(result, default=str)))

asyncio.run(main())
