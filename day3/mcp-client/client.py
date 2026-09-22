from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = "https://imrat-67-server.fastmcp.app/mcp"


async def main():
    async with streamable_http_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
            except Exception as e:
                print("INIT ERROR:", repr(e))
                return

            tools = await session.list_tools()

            for tool in tools.tools:
                print(tool.name)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())