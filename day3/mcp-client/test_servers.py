import asyncio, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
from app import SERVERS, MCPManager

async def main():
    mgr = MCPManager(SERVERS)
    await mgr.connect_all()
    for key in SERVERS:
        if key in mgr.sessions:
            names = [t["full_name"] for t in mgr.tools if t["server"] == key]
            print(f"[OK] {key}: {names}")
        else:
            print(f"[FAIL] {key}: {mgr.errors.get(key)}")

asyncio.run(main())
