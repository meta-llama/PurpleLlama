# LlamaFirewall Architecture

LlamaFirewall is built to serve as a flexible, real-time guardrail framework for securing LLM-powered applications. Its architecture is modular, enabling security teams and developers to compose layered defenses that span from raw input ingestion to final output actions—across simple chat models and complex autonomous agents.

At its core, LlamaFirewall operates as a policy engine that orchestrates multiple security scanners, each tailored to detect a specific class of risks. These scanners can be plugged into various stages of an LLM agent’s workflow, ensuring broad and deep coverage.

## Core Architectural Components

LlamaFirewall is composed of the following primary components:

### PromptGuard 2
A fast, lightweight BERT-style classifier that detects direct prompt injection attempts. It operates on user inputs and untrusted content such as web data, providing high precision and low latency even in high-throughput environments. You can read more about it in the Scanners section or visiting its [README](https://github.com/meta-llama/PurpleLlama/blob/main/Prompt-Guard/README.md).

- **Use case**: Catching classic jailbreak patterns, social engineering prompts, and known injection attacks.
- **Strengths**: Fast, production-ready, easy to update with new patterns.

### AlignmentCheck
A chain-of-thought auditing module that inspects the reasoning process of an LLM agent in real time. It uses few-shot prompting and semantic analysis to detect goal hijacking, indirect prompt injections, and signs of agent misalignment.

- **Use case**: Verifying that agent decisions remain consistent with user intent.
- **Strengths**: Deep introspection, detects subtle misalignment, works with opaque or black-box models.

### Regex + Custom Scanners
A configurable scanning layer for applying regular expressions or simple LLM prompts to detect known patterns, keywords, or behaviors across inputs, plans, or outputs.

- **Use case**: Quick matching for known attack signatures, secrets, or unwanted phrases.
- **Strengths**: Easy to customize, flexible, language-agnostic.

### CodeShield
A static analysis engine that examines LLM-generated code for security issues in real time. Supports both Semgrep and regex-based rules across 8 programming languages. You can read more about it in the Scanners section or visiting its [README](https://github.com/meta-llama/PurpleLlama/blob/main/CodeShield/README.md).

- **Use case**: Preventing insecure or dangerous code from being committed or executed.
- **Strengths**: Syntax-aware, fast, customizable, extensible for different languages and org-specific rules.
