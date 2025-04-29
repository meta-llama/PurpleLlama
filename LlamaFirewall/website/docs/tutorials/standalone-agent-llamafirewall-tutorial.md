---
sidebar_position: 5
---

import styles from './styles.module.css';

# Notebook: Standalone Agent with LlamaFirewall

This demo showcases a standalone agent implementation that integrates LlamaFirewall for security scanning. The agent can interact with users through a console interface, perform web searches, and fetch URL content while ensuring security through LlamaFirewall's scanning capabilities.

NOTE: You can download this as a Jupyter Notebook from the [notebook](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall/notebook) directory in the repo.

## Overview

The Demo Standalone Agent is a Python application that:

- Connects to the [Llama API](https://llama.developer.meta.com/join_waitlist?utm_source=purple_llama&utm_medium=notebook&utm_campaign=agent_demo) for LLM capabilities
- Integrates LlamaFirewall for security scanning of messages
- Provides web search functionality using Tavily API
- Can fetch and process URL content
- Runs in an interactive console mode
- Demonstrates tool usage with LLM agents

## Prerequisites

- Python 3.10 or higher
- Conda (recommended for environment management)
- Access to [Llama API](https://llama.developer.meta.com/join_waitlist?utm_source=purple_llama&utm_medium=notebook&utm_campaign=agent_demo)
- Tavily API key (optional, for web search functionality)

## Setup

### 1. Create a Conda Environment

<div className={styles.regularCode}>
conda create -n llamafirewall python=3.10 -y
conda activate llamafirewall
</div>

### 2. Install Required Packages

<div className={styles.regularCode}>
pip install llamafirewall openai python-dotenv rich tavily-python
</div>

### 3. Configure Environment Variables

Create a `.env` file in the same directory as the script with the following variables:

<div className={styles.regularCode}>
# Llama API Configuration
LLAMA_API_KEY=your_llama_api_key
LLAMA_API_BASE_URL=https://api.llama.com/compat/v1/
LLAMA_API_MODEL=Llama-4-Maverick-17B-128E-Instruct-FP8

# Tavily API Configuration (optional, for web search)
TAVILY_API_KEY=your_tavily_api_key
</div>

## Features

### LlamaFirewall Integration

The demo uses LlamaFirewall to scan messages for security issues:

- Code Shield scanner for code-related security issues
- Llama Prompt Guard for user and tool input to the loop

### Available Tools

The agent has access to the following tools:

1. **websearch**: Performs web searches using the Tavily API
2. **fetch_url_contents**: Fetches HTML content from a specified URL

## How It Works

1. The agent initializes LlamaFirewall with appropriate scanners for different message roles
2. When a user sends a message, it's scanned by LlamaFirewall
3. If the message passes the scan, it's sent to the Llama API
4. The API may respond with tool calls, which are executed by the agent
5. Tool results are also scanned by LlamaFirewall before being sent back to the API
6. The process continues until the API provides a final response without tool calls
7. The response is displayed to the user

<!-- cells start here -->

<div className={styles.jupyterCodeCell}>
```python {showLineNumbers}
# use if the notebook is executing in a conda env without the packages installed
# !pip install llamafirewall openai python-dotenv rich tavily-python
```
</div>
<div className={styles.jupyterCodeCell}>
```python {showLineNumbers}
import os
from dotenv import load_dotenv

try:
  from google.colab import userdata
  in_colab = True
except ImportError:
  in_colab = False

if in_colab:
  # setup your keys in the colab secrets UI
  secrets_to_load = [ "TAVILY_SEARCH_API_KEY", "LLAMA_API_KEY", "LLAMA_API_BASE_URL", "LLAMA_API_MODEL"]

  for secret_name in secrets_to_load:
    secret_value = userdata.get(secret_name)
    if secret_value:
      os.environ[secret_name] = secret_value
    else:
      print(f"Warning: Secret '{secret_name}' not found in userdata.")
else:
    # Load environment variables from .env file
    load_dotenv()
```
</div>
<div className={styles.jupyterCodeCell}>
```python {showLineNumbers}
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import logging
import os
import sys
import traceback
import requests

from typing import Any, Callable, Dict, List
from rich import print

from openai import OpenAI

from llamafirewall import LlamaFirewall, ScanDecision, ScannerType
from llamafirewall.llamafirewall_data_types import (
    AssistantMessage,
    Role,
    ToolMessage,
    UserMessage,
)

# logging configs
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)  # Set the logging level
logger.propagate = False
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Note: To use the websearch and extract_url functions, install the tavily package:
# pip install tavily-python
try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning(
        "[bold yellow]Warning: Tavily package not found. Web search functionality will be limited.[/bold yellow]"
    )


client = OpenAI(
    api_key=os.getenv("LLAMA_API_KEY"), base_url=os.getenv("LLAMA_API_BASE_URL")
)

# Initialize LlamaFirewall
llama_firewall = LlamaFirewall()


def function_to_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    Convert a Python function to the tool schema format required by LlamaAPIClient.

    Args:
        func: The Python function to convert

    Returns:
        A dictionary representing the tool schema
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get function docstring for description
    description = inspect.getdoc(func) or f"Function to {func.__name__}"

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip self parameter for methods
        if param_name == "self":
            continue

        param_type = "string"  # Default type
        param_desc = ""

        # Try to get type annotation
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == str:
                param_type = "string"
            elif param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"

        # Add parameter to properties
        properties[param_name] = {
            "type": param_type,
            "description": param_desc
            or f"Parameter {param_name} for function {func.__name__}",
        }

        # Add to required list if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Build the complete tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    return tool_schema


def websearch(query: str) -> str:
    """
    Perform a web search for the given query using Tavily API.

    Args:
        query: The search query

    Returns:
        Search results from Tavily
    """
    if not TAVILY_AVAILABLE:
        return "Error: Tavily package not installed. Please install with 'pip install tavily-python'."

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Tavily API key not found in environment variables. Please add TAVILY_API_KEY to your .env file."

    try:
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=tavily_api_key)

        # Perform search
        result = tavily_client.search(query, search_depth="basic", max_results=5)

        # Format the results
        formatted_results = f"Search results for '{query}':\n\n"

        for i, item in enumerate(result.get("results", []), 1):
            title = item.get("title", "No title")
            content = item.get("content", "No content")
            url = item.get("url", "No URL")

            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   {content[:200]}...\n"
            formatted_results += f"   Source: {url}\n\n"

        if not result.get("results"):
            formatted_results += "No results found."

        return formatted_results

    except Exception as e:
        return f"Error performing web search: {str(e)}"


def fetch_url_content_pre_rendering(url):
    """
    Fetches the HTML content from the provided URL with additional logging for debugging.

    Args:
        url (str): The URL of the page to fetch.

    Returns:
        str: The HTML content of the page if successful.

    Raises:
        requests.RequestException: If the HTTP request fails.
    """

    try:
        logger.debug("Attempting to fetch URL: %s", url)

        response = requests.get(url)

        logger.debug("Received response with status code: %d", response.status_code)
        logger.debug("Sample content %s", response.text[:1000])

        # This will raise an HTTPError for 400, 500, etc.
        response.raise_for_status()

        html = response.text
        logger.debug(
            "Successfully fetched content; content length: %d characters", len(html)
        )
        return html

    except requests.RequestException as e:
        logger.error("Request failed with error: %s", e)

        # If we have a response object, log part of its text for debugging.
        if "response" in locals() and response is not None:
            content_preview = response.text[:500]  # print first 500 characters
            logger.error("Response content preview: %s", content_preview)
        raise


async def interact_with_ai(
    user_message: str,
    messages: List[Dict[str, Any]],
    model="Llama-4-Scout-17B-16E-Instruct-FP8",
) -> List[Dict[str, Any]]:
    """
    Interact with AI agent using OpenAI API with tools.

    Args:
        user_message: User's message
        history: Chat history as a list of Message objects

    Returns:
        Updated chat history as a list of Message objects
    """
    # Define available tools
    available_tools: Dict[str, Callable] = {
        "websearch": websearch,
        "fetch_url_content_pre_rendering": fetch_url_content_pre_rendering,
    }

    # Convert functions to tool schemas
    tools = [function_to_tool_schema(func) for func in available_tools.values()]

    # Add the current user message to our history
    user_msg = UserMessage(content=user_message)

    messages.append(
        {
            "role": Role.USER.value,
            "content": user_message,
        }
    )

    # Scan the user message
    result = await llama_firewall.scan_async(user_msg)

    if result.decision == ScanDecision.BLOCK:
        response = f"Blocked by LlamaFirewall, {result.reason}"
        messages.append(
            {
                "role": Role.ASSISTANT.value,
                "content": response,
            }
        )

        return messages

    try:
        # Start the Agent loop
        tool_call = True
        while tool_call:
            # debug last message
            logger.debug(f"messages: {messages[-1]}")

            # Call Llama API with tools enabled
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                tools=tools,
            )

            logger.debug(f"response: {response}")
            # Get the response message
            response_message = response.choices[0].message

            messages.append(
                {
                    "role": response_message.role,
                    "content": response_message.content,
                    "tool_calls": response_message.tool_calls,
                }
            )

            # Check if there are tool calls
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    # Execute the tool
                    tool_name = tool_call.function.name
                    if tool_name in available_tools:
                        try:
                            # Parse arguments
                            arguments = json.loads(tool_call.function.arguments)

                            # Call the function
                            tool_result = available_tools[tool_name](**arguments)

                            tool_msg = ToolMessage(content=str(tool_result))

                            # Scan the tool result
                            lf_result = await llama_firewall.scan_async(tool_msg)

                            if lf_result.decision == ScanDecision.BLOCK:
                                # for demonstration purposes, we show the tool output, but this should be avoided in production
                                blocked_response = f"Blocked by LlamaFirewall: {lf_result.reason} - Tool result: {tool_result}"
                                tool_result = blocked_response

                            # Add tool result to messages
                            messages.append(
                                {
                                    "role": Role.TOOL.value,
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result,
                                }
                            )

                        except Exception as e:
                            # Add error message if tool execution fails
                            error_msg = f"Error: {str(e)}"
                            tool_msg = AssistantMessage(content=error_msg)
                            messages.append(
                                {
                                    "role": Role.TOOL.value,
                                    "tool_call_id": tool_call.id,
                                    "content": error_msg,
                                }
                            )
                            logger.error(f"\nTool Error: {error_msg}")

                        logger.debug(f"Tool Call response: {messages[-1]}")

                # Continue the loop to get the next response
                continue
            else:
                # No tool calls, add the response to history and break the loop
                tool_call = False
                full_response = response_message.content
                messages.append(
                    {
                        "role": Role.ASSISTANT.value,
                        "content": full_response,
                    }
                )

    except Exception as e:
        error_message = f"Error calling API: {str(e)}"
        traceback.print_exc()
        messages.append(
            {
                "role": Role.ASSISTANT.value,
                "content": error_message,
            }
        )

    logger.debug(f"Assistant response: {messages[-1]}")

    return messages


async def run_demo():
    """
    Run the agent in console mode, taking input from the command line.
    """
    print(
        "[bold green]Starting AI Agent in console mode. Type 'exit' to quit.[/bold green]"
    )
    messages: List[Dict[str, Any]] = []

    model = os.getenv("LLAMA_API_MODEL")

    while True:
        user_message = input("\nYou: ")
        if user_message.lower() == "exit":
            break

        messages = await interact_with_ai(user_message, messages, model)

        logger.info(f"Full history: {messages}")

        # Print the assistant's response
        if messages:
            assistant_message = messages[-1]["content"]
            print(f"\n[bold green]Assistant:[/bold green] {assistant_message}")
```
</div>
<div className={styles.jupyterCodeCell}>
```python {showLineNumbers}
await run_demo()
```
</div>
