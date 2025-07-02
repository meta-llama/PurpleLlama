# AlignmentCheck

## Overview
AlignmentCheck is a pioneering, open-source guardrail that utilizes few-shot prompting to audit an agent's reasoning in real-time, detecting signs of goal hijacking or prompt-injection induced misalignment. This innovative approach allows for the examination of the full chain of thought behind an LLM decision or action, flagging contradictions, goal divergence, and other indicators of injection-induced misalignment.
As part of the LlamaFirewall suite, AlignmentCheck serves as a runtime reasoning auditor, providing a critical layer of defense against malicious behavior. Its ability to be layered with PromptGuard enables it to offer an additional level of protection, detecting misalignments around dangerous agentic behavior and ensuring the integrity of your system.

## Design of AlignmentCheck
As a defense mechanism that enhances the security of LlamaFirewall, this scanner continuously monitors an agent's actions and comparing them to the user's stated objective. This helps detect and prevent covert prompt injection, misleading tool output, and other forms of goal hijacking.
Unlike traditional content-based filters, AlignmentCheck reasons over the entire execution trace, flagging deviations that suggest malicious behavior. By leveraging language-model reasoning, it evaluates the consistency between planned and observed actions, detecting behavioral drift early and constraining the agent to its authorized task. This approach provides several benefits:
* Closes a critical gap in existing defenses by detecting indirect injections that appear benign in isolation
* Extends protection to threat classes that static rules and lexical filters routinely miss
* Provides a semantic-layer defense that complements traditional content-based filters

## Security Risks Covered
AlignmentCheck provides protection against a range of security risks, including:
* **Indirect Universal Jailbreak Prompt Injections:** AlignmentCheck detects when an agent's actions diverge from the user-defined goal and policy, preventing jailbreaks embedded in third-party content.
* **Agent Goal Hijacking Prompt Injections:** By monitoring an agent's actions and comparing them to the user-defined goal, AlignmentCheck can detect application-specific prompt injections that are harder to identify than universal jailbreaks.
* **Malicious Code via Prompt Injection:** AlignmentCheck provides a layered defense against code-oriented prompt injection, blocking malicious code from being executed through prompt injection.
