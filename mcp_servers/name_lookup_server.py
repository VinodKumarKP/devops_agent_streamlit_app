#!/usr/bin/env python3
"""
MCP Server for Name Lookup Tools
Run with: python name_lookup_server.py
"""

import logging

import nest_asyncio  # Added import

nest_asyncio.apply()  # Added call

from fastmcp import FastMCP

# from pydantic import BaseModel # Removed unused import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("name-lookup-tools")

# Mock database
NAMES_DB = {
    'smith': ['John Smith', 'Jane Smith', 'Robert Smith', 'Mary Smith', 'James Smith'],
    'johnson': ['Michael Johnson', 'Sarah Johnson', 'David Johnson', 'Lisa Johnson'],
    'williams': ['James Williams', 'Jennifer Williams', 'Christopher Williams', 'Amanda Williams'],
    'brown': ['William Brown', 'Elizabeth Brown', 'Joseph Brown', 'Michelle Brown'],
    'jones': ['Thomas Jones', 'Patricia Jones', 'Charles Jones', 'Linda Jones'],
    'garcia': ['Antonio Garcia', 'Maria Garcia', 'Jose Garcia', 'Carmen Garcia'],
    'miller': ['Richard Miller', 'Susan Miller', 'Paul Miller', 'Karen Miller'],
    'davis': ['Richard Davis', 'Nancy Davis', 'Kevin Davis', 'Donna Davis'],
    'rodriguez': ['Carlos Rodriguez', 'Ana Rodriguez', 'Luis Rodriguez', 'Elena Rodriguez'],
    'wilson': ['Daniel Wilson', 'Betty Wilson', 'Mark Wilson', 'Helen Wilson']
}


@mcp.tool()
async def get_names_by_surname(surname: str) -> str:
    """Get a list of common names for a given surname"""
    logger.info(f"Looking up names for surname: {surname}")

    surname_lower = surname.lower()
    names = NAMES_DB.get(surname_lower, [])

    if not names:
        return f"No names found for surname '{surname}'"

    return f"Found {len(names)} names for '{surname}': {', '.join(names)}"


@mcp.tool()
async def capitalize_name(string: str) -> str:
    """Capitalize the first letter of a name"""
    logger.info(f"Capitalizing: {string}")

    if not string:
        return "Empty string provided"

    result = string.capitalize()
    return f"'{string}' capitalized to '{result}'"


@mcp.tool()
async def format_names(names: str, format_type: str = "upper") -> str:
    """Format a list of names (upper, lower, title)"""
    logger.info(f"Formatting names with type: {format_type}")

    if not names:
        return "No names provided"

    name_list = [name.strip() for name in names.split(',')]

    if format_type == "upper":
        formatted = [name.upper() for name in name_list]
    elif format_type == "lower":
        formatted = [name.lower() for name in name_list]
    elif format_type == "title":
        formatted = [name.title() for name in name_list]
    else:
        return f"Unknown format type: {format_type}"

    return f"Formatted names: {', '.join(formatted)}"


if __name__ == "__main__":
    mcp.run()
