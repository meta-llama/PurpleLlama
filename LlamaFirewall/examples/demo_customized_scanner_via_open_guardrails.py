# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio

from agents import (
    Agent,
    GuardrailFunctionOutput,
    input_guardrail,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)
from llamafirewall import (
    LlamaFirewall,
    Message,
    register_llamafirewall_scanner,
    Role,
    ScanDecision,
    Scanner,
    ScanResult,
    Trace,
    UserMessage,
)


@register_llamafirewall_scanner("demo_scanner")
class DemoScanner(Scanner):
    def __init__(self, block_threshold: float = 1.0) -> None:
        super().__init__("demo_scanner", block_threshold)

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        """
        Implement the custom scanner logic.
        """
        print(
            f'DemoScanner is invoked for input: "{message.content[:10]}...", nothing will happen.'
        )
        return ScanResult(
            decision=ScanDecision.ALLOW,
            reason="DemoScanner did not block anything.",
            score=0.0,
        )


@input_guardrail
async def llamafirewall_input_guardrail(
    # pyre-ignore[24]: openai doc uses None as the context parameter for input guardrail
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    lf = LlamaFirewall(
        scanners={
            Role.USER: ["demo_scanner"],
            Role.SYSTEM: ["demo_scanner"],
        }
    )
    lf_result: ScanResult = await lf.scan_async(
        UserMessage(content=input)  # pyre-ignore
    )

    return GuardrailFunctionOutput(
        output_info=lf_result.reason,
        tripwire_triggered=lf_result.decision != ScanDecision.ALLOW,
    )


# pyre-ignore
agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[llamafirewall_input_guardrail],
)


async def main():
    # This should not trip the guardrail
    print("Case 1 - Benign math problem:")
    try:
        agent_output = await Runner.run(
            agent, "Hello, can you help me solve for x: 2x + 3 = 11?"
        )
        print("Customized demo scanner didn't block anything - this is expected.")
        print(f"Agent output: {agent_output.new_items[0].raw_item.content[0].text}\n")
    except InputGuardrailTripwireTriggered:
        print(
            "LlamaFirewall custom guardrail tripped for the benign input - this is unexpected.\n"
        )


def invoke_main() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    invoke_main()  # pragma: no cover
