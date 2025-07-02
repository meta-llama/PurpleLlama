---
sidebar_position: 1
---

import Link from '@docusaurus/Link';

# About LlamaFirewall

LlamaFirewall is a framework designed to detect and mitigate AI centric security risks, supporting multiple layers of inputs and outputs, such as typical LLM chat and more advanced multi-step agentic operations. It consists of a set of scanners for different security risks.

Unlike traditional moderation tools that focus on filtering toxic content, LlamaFirewall provides system-level defenses tailored to modern agentic use cases—like code generation, tool orchestration, and autonomous decision-making. With modules that inspect user input, agent reasoning, and generated code, LlamaFirewall acts as a real-time guardrail framework purpose-built for production environments.

Feel free to read our [LlamaFirewall: An open source guardrail system for building secure AI agents](https://ai.facebook.com/research/publications/llamafirewall-an-open-source-guardrail-system-for-building-secure-ai-agents) paper to dive deeper into design and benchmark results.

Ready to get started? Check out the quick installation guide to learn how to set up LlamaFirewall in your environment.

<Link
  className="button button--primary button--lg"
  to="getting-started/how-to-use-llamafirewall"
>Go to guide</Link>


## Why Use LlamaFirewall?

LlamaFirewall stands out due to its unique combination of features and benefits:

- **Layered Defense Architecture**: Combines multiple scanners—PromptGuard 2, AlignmentCheck, CodeShield, and customizable regex filters—for comprehensive protection across the agent’s lifecycle.
- **Real-Time, Production-Ready**: Built for low-latency environments with support for high-throughput pipelines and real-world deployment constraints.
- **Open Source and Extensible**: Designed for transparency and community collaboration, allowing teams to build, audit, and extend defenses as threats evolve.

## Benefits of the Framework

- **Mitigates Prompt Injection Attacks**: Detects both direct and indirect prompt injection attempts, even when obfuscated or embedded in tool responses.
- **Monitors Agent Reasoning in Real Time**: Uses chain-of-thought auditing to catch goal hijacking or misaligned behavior before actions are taken.
- **Prevents Unsafe Code Generation**: Scans generated code for vulnerabilities, insecure API use, and policy violations with extensible static analysis support.

## Tutorials

To help you get started with LlamaFirewall, we have a series of tutorials available. These tutorials will guide you through the installation process, configuration, and best practices for using LlamaFirewall effectively. You can access the tutorials [here](/docs/tutorials/prompt-guard-scanner-tutorial).
