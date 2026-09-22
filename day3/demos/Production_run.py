import json
import random
from fastmcp import FastMCP
 
# Create a FastMCP server instance
mcp = FastMCP(name="Simple calculator server")
 
@mcp.tool
def roll_dice(n_dice: int = 1) -> list[int]:
    """Roll n_dice 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]
 
@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b
 
@mcp.tool
def random_number(min_value: int = 0, max_value: int = 100) -> int:
    """Generate a random integer between min_value and max_value (inclusive)."""
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")
    return random.randint(min_value, max_value)


# Resource: Server information
@mcp.resource("info://server")
def server_info() -> str:
    """Get information about this server."""
    info = {
        "name": "Simple Calculator Server",
        "version": "1.0.0",
        "description": "A basic MCP server with math tools",
        "tools": ["roll_dice","add", "random_number"],
        "author": "Your Name"
    }
    return json.dumps(info, indent=2)
 
# Start the server
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)