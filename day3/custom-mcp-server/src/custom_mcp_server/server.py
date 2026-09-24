import random
from datetime import date, datetime

from fastmcp import FastMCP


# Create the MCP server
mcp = FastMCP(name="Custom Tools Server")


@mcp.tool
def calculate_age(date_of_birth: str) -> int:
    """
    Calculate a person's current age from their date of birth.
    Expected date format: YYYY-MM-DD
    """
    birth_date = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
    today = date.today()

    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


@mcp.tool
def random_number(min_value: int, max_value: int) -> int:
    """
    Generate a random integer between min_value and max_value (inclusive).
    """
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")

    return random.randint(min_value, max_value)


if __name__ == "__main__":
    mcp.run()



