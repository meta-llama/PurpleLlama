# CodeShield

## Overview
CodeShield is an advanced online static-analysis engine designed to enhance the security of LLM-generated code. It supports both Semgrep and regex-based rules, providing syntax-aware pattern matching across eight programming languages. Initially released as part of the Llama 3 launch, CodeShield is now integrated into the LlamaFirewall framework, addressing the need for a syntax-aware, extensible static analysis pipeline that seamlessly integrates with LLM generation workflows. This tool is engineered to detect insecure coding patterns, offering coverage for over 50 Common Weakness Enumerations (CWEs), making it applicable across diverse software stacks.

Visit its [repository](https://github.com/meta-llama/PurpleLlama/blob/main/CodeShield/) to access the README and related source code.

## Design of Codeshield
CodeShield is built on a modular framework composed of rule-based analyzers tailored to assess code security across multiple programming languages., making it broadly applicable across diverse software stacks.

To support deployment in latency-sensitive production environments, CodeShield employs a two-tiered scanning architecture. The first tier utilizes lightweight pattern matching and static analysis, completing scans in under 100 milliseconds. When potential security concerns are flagged, inputs are escalated to a second, more comprehensive static analysis layer, which incurs an average latency of around 300 milliseconds.

The two-tiered architecture allows CodeShield to balance speed and accuracy, ensuring fast and scalable security screening in real-world, high-throughput environments. In internal production deployments, we observed that approximately 90% of inputs are fully resolved by the first layer, maintaining a typical end-to-end latency of under 70 milliseconds. For the remaining 10% of cases requiring deeper inspection, end-to-end latency can exceed 300 milliseconds, influenced by input complexity and system constraints.

## Security Risks Covered
CodeShield addresses several critical security risks, including:
* **Insecure Coding Practices:** As a static analysis engine, CodeShield detects insecure coding practices in LLM-generated code, providing a robust defense against potential vulnerabilities.
* **Malicious Code via Prompt Injection:** CodeShield, in conjunction with other scanners, offers a layered defense against code-oriented prompt injection, ensuring that malicious code is identified and mitigated before it can be executed.
