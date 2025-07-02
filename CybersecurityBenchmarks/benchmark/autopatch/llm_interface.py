# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


LOGGER: logging.Logger = logging.getLogger(__name__)
LIONHEAD_BOT_FBID = 89002005279108
LIONHEAD_METAGEN_KEY = "mg-api-c40d0a29fcdf"


class LLMInterface:
    REASONING_START_TAG = "<|reasoning_thinking_start|>"
    REASONING_END_TAG = "<|reasoning_thinking_end|>"

    def __init__(self, system_prompt: str) -> None:
        """Init the LLM with the specified system prompt."""
        self._model: str = ""
        self._messages: List[BaseMessage] = []
        raise NotImplementedError()

    async def query(self, message: str, temperature: Optional[float] = None) -> str:
        """
        Query the LLM.

        Args:
            message: Message to send to the LLM.
            llm_max_retries: Number of retries to do if the query fails (e.g., due to rate limit).

        Returns:
            Response from the LLM.
        """
        raise NotImplementedError()

    def get_context(self) -> List[BaseMessage]:
        """
        Get the context of the LLM.

        Returns:
            Context of the LLM.
        """
        return self._messages

    def set_context(self, context: List[BaseMessage]) -> None:
        """
        Restore the context of the LLM.

        Args:
            context: Context to restore.
        """
        self._messages = context

    @classmethod
    def _strip_reasoning(cls, message: str) -> str:
        # Strip everything between <|reasoning_thinking_start|> and <|reasoning_thinking_end|>
        start_idx = message.find(cls.REASONING_START_TAG)
        if start_idx == -1:
            return message
        end_idx = message.find(cls.REASONING_END_TAG, start_idx)
        if end_idx == -1:
            return message

        return message[:start_idx] + message[end_idx + len(cls.REASONING_END_TAG) :]


def forget_middle_of_the_convo(llm: LLMInterface) -> None:
    """Forget the middle of the conversation.

    Suppose that the conversation was like this:
    0. System
    1. Human : Here's the crash. Explain.
    2. AI    : Explanation
    3. Human : Generate a patch.
    4. AI    : Fix 1 (forget)
    5. Human : That didn't work. Here's why. Try again. (forget)
    6. AI    : Fix 2

    At the point of sending another query to "try gain" at (5), we can "forget"
    about (2) and (3), and assume that the first answer returned from AI is (4) for
    efficiency.
    """
    context = llm.get_context()
    if (
        len(context) >= 7
        and isinstance(context[0], SystemMessage)
        and isinstance(context[1], HumanMessage)
        and isinstance(context[2], AIMessage)
        and isinstance(context[3], HumanMessage)
        and isinstance(context[4], AIMessage)
        and isinstance(context[-1], AIMessage)
    ):
        llm.set_context(context[:4] + [context[-1]])
    else:
        raise RuntimeError(
            "The context is not in the expected format of (system, human, ..., AI)."
        )
