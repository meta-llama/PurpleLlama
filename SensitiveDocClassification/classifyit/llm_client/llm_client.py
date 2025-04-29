# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LlamaStack client handlers for content classification
"""

import json
import logging
import re
import uuid

from classifyit.structs import SessionConfig

from llama_stack_client import Agent, LlamaStackClient
from pydantic import BaseModel

# Get the package logger
logger = logging.getLogger("classifyit.llm_client")


class ClassificationOutput(BaseModel):
    """
    Pydantic model for the classification output
    """

    category_matched: str
    brief_explanation: str


def classify_content(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    llama_client: LlamaStackClient,
    session_config: SessionConfig,
    content: str,
    content_window_size: int = 5000,
    max_inference_iters: int = 5,
    temperature: float = 0.1,
    top_p: float = 0.5,
    max_tokens: int = 128000,
) -> ClassificationOutput:
    """
    Classify content using an LLM hosted using LlamaStack.

    Args:
        llama_client: The LlamaStack client
        session_config: The session configuration
        content: The text content to classify
        content_window_size: The size of the content window to classify
        max_inference_iters: The maximum number of inference iterations
        temperature: The temperature for the model
        top_p: The top_p for the model
        max_tokens: The maximum number of tokens for the model

    Returns:
        ClassificationOutput
    """
    agent_instructions = str(
        session_config.agent_settings.agent_instructions
        + "\n\nValid categories: "
        + json.dumps(session_config.agent_settings.response_categories)
        + "\n Remember to respond in JSON format (e.g. ```json "
        + "{'category_matched': '$category','brief_explanation': '$explanation'}"
        + "```)"
    )

    agent = Agent(
        llama_client,
        model=session_config.llama_server.model,
        instructions=agent_instructions,
        tools=[],
        input_shields=[],
        output_shields=[],
        max_infer_iters=max_inference_iters,
        sampling_params={
            "strategy": {"type": "top_p", "temperature": temperature, "top_p": top_p},
            "max_tokens": max_tokens,
        },
    )
    session_id = agent.create_session(str(uuid.uuid4()))

    response = agent.create_turn(
        messages=[{"role": "user", "content": session_config.agent_settings.prompt}],
        documents=[
            {
                "content": content[:content_window_size],
                "mime_type": "text/plain",
            }
        ],
        session_id=session_id,
        stream=False,
    )

    logger.debug("Raw model response: %s", response.output_message.content)

    try:
        result = extract_json_from_response(response.output_message.content)
        classification_output = ClassificationOutput.model_validate_json(
            json_data=json.dumps(result)
        )
        return classification_output

    except Exception as e:
        logger.error("Error extracting JSON from model response: %s", e)
        raise


def extract_json_from_response(response_text: str) -> dict:
    """
    Helper function to find the JSON markdown block in the model response.

    Args:
        response_text: The source response text to extract the JSON

    Returns:
        Dictionary containing the JSON data
    """
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response_text)

    if match:
        # Extract the JSON string
        json_str = match.group(1)
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.error("Error parsing JSON: %s", str(e))
            return None
    else:
        logger.error("No JSON block found in the response")
        return None
