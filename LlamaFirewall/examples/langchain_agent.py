# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

import os

from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # @manual
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent  # @manual

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Message,
    Role,
    ScanDecision,
    Trace,
)


LOG: logging.Logger = logging.getLogger(__name__)


Input = TypeVar("Input")
Output = TypeVar("Output")
Other = TypeVar("Other")


class LlamaFirewallRunnableClass(Runnable[Input, Output]):
    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Input,
    ) -> Input:
        """
        Process the input using the GR guardrail.
        """
        LOG.debug(f"Input to process: {getattr(input, 'content', '')[:300]}")
        llamafirewall = LlamaFirewall()
        lf_input = AssistantMessage(
            content=getattr(input, "content", ""),
        )
        result = llamafirewall.scan(lf_input)
        if result.decision == ScanDecision.BLOCK:
            input.content = [f"Blocked by LlamaFirewall, {result.reason}"]

        LOG.debug(f"Decision result: {result.decision}.")
        return input

    def __or__(
        self,
        other: Union[
            Runnable[Input, Other],
            Callable[[Input], Other],
            Callable[[Iterator[Input]], Iterator[Other]],
            Mapping[
                str, Union[Runnable[Input, Other], Callable[[Input], Other], Input]
            ],
        ],
    ) -> Runnable[Input, Output]:
        """
        Allow chaining with another runnable (or a LangChain chain) using the | operator.
        The output of this runnable will be passed as input to the next.
        """
        if (
            isinstance(other, Runnable)
            and hasattr(other, "invoke")
            and callable(other.invoke)
        ):
            # Define a composite runnable that calls self.invoke first, then passes the result to other.invoke.
            parent = self

            class CompositeRunnable(Runnable[Input, Output]):
                def invoke(
                    self,
                    input: Input,
                    config: Optional[RunnableConfig] = None,
                    **kwargs: Input,
                ) -> Output:
                    intermediate = parent.invoke(input, config=config, **kwargs)
                    return other.invoke(intermediate, config=config, **kwargs)

            return CompositeRunnable()
        else:
            raise ValueError(
                f"[LlamaFirewall] Langchain agent cannot chain with object of type: {type(other)}"
            )


class LangChainAgent:
    def __init__(self, base_llm: BaseChatModel) -> None:
        self.base_llm = base_llm
        memory = MemorySaver()
        search = TavilySearchResults(max_results=2)
        tools = [search]
        self.agent_executor: CompiledGraph = create_react_agent(
            base_llm, tools, checkpointer=memory
        )
        self.config: RunnableConfig = {"configurable": {"thread_id": "abc123"}}
        self.llama_firewall: LlamaFirewall = LlamaFirewall()
        self.stored_trace: Trace = []  # Store the trace of messages

    def _convert_to_llamafirewall_messages(self, action: Dict[str, Any]) -> Trace:
        """
        Convert langchain messages to LlamaFirewall Message format.
        """
        messages = []
        action_key = list(action.keys())[0]
        if "messages" not in action[action_key]:
            return messages
        # Handle different message types based on the action key
        scanning_role = Role.ASSISTANT
        scanning_message_type = AIMessage
        if action_key == "user":
            scanning_role = Role.USER
            scanning_message_type = HumanMessage
        elif action_key == "agent":
            scanning_role = Role.ASSISTANT
            scanning_message_type = AIMessage
        elif action_key == "tools":
            scanning_role = Role.TOOL
            scanning_message_type = ToolMessage

        for _, msg in enumerate(action[action_key]["messages"]):
            if isinstance(msg, scanning_message_type):
                content = str(msg.content)
                messages.append(Message(role=scanning_role, content=content))
        return messages

    def execute(self, prompt: str) -> Iterator[Dict[str, Any]]:
        user_input = HumanMessage(content=prompt)
        # Convert user input to LlamaFirewall Message format
        user_messages = self._convert_to_llamafirewall_messages(
            {"user": {"messages": [user_input]}}
        )

        # Process each user message and update the trace
        for user_message in user_messages:
            scan_result, updated_trace = self.llama_firewall.scan_replay_build_trace(
                user_message, self.stored_trace
            )
            # Update the stored trace if the message passes the scan
            self.stored_trace = updated_trace

            # If the message is blocked, yield the reason and return
            if scan_result.decision == ScanDecision.BLOCK:
                yield {"user": {"messages": [HumanMessage(content=scan_result.reason)]}}
                return

        # If user input passes the scan, proceed with the agent execution
        actions = self.agent_executor.stream({"messages": [user_input]}, self.config)
        for action in actions:
            # Convert langchain messages to LlamaFirewall Message format
            messages = self._convert_to_llamafirewall_messages(action)

            # Only scan if we have messages to scan
            if messages:
                action_key = list(action.keys())[0]

                # Process each message and update the trace
                for message in messages:
                    scan_result, updated_trace = (
                        self.llama_firewall.scan_replay_build_trace(
                            message, self.stored_trace
                        )
                    )
                    # Update the stored trace if the message passes the scan
                    self.stored_trace = updated_trace

                    # If the message is blocked, modify its content
                    if (
                        scan_result.decision == ScanDecision.BLOCK
                        and "messages" in action[action_key]
                    ):
                        for msg in action[action_key]["messages"]:
                            if hasattr(msg, "content"):
                                msg.content = (
                                    f"Blocked by LlamaFirewall: {scan_result.reason}"
                                )
                                break

            yield action

    def llm_chain(self, user_prompt: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are world class technical documentation writer."),
                ("user", "{input}"),
            ]
        )
        llama_firewall_runnable = LlamaFirewallRunnableClass()
        output_parser = StrOutputParser()

        chain = prompt | self.base_llm | llama_firewall_runnable | output_parser
        return chain.invoke({"input": user_prompt})


def get_langchain_openai_agent(
    model: str, openai_api_key: str, tavily_api_key: str
) -> LangChainAgent:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    base_llm = ChatOpenAI(model_name=model)
    return LangChainAgent(base_llm)


def get_langchain_azure_agent(
    openai_api_key: str,
    tavily_api_key: str,
    endpoint: str,
    deployment_name: str,
    api_version: str,
) -> LangChainAgent:
    os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
    os.environ["OPENAI_API_VERSION"] = api_version
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    base_llm = AzureChatOpenAI(deployment_name=deployment_name)
    return LangChainAgent(base_llm)
