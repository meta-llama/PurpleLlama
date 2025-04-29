# Introduction

This repository hosts the implementation of [CyberSecEval 4](https://meta-llama.github.io/PurpleLlama/CyberSecEval/), which builds upon and extends the functionalities of its predecessors,
[CyberSecEval 3](https://ai.meta.com/research/publications/cyberseceval-3-advancing-the-evaluation-of-cybersecurity-risks-and-capabilities-in-large-language-models/),
[CyberSecEval 2](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/)
and
[CyberSecEval](https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/).
CyberSecEval 4 is an extensive benchmark suite designed to assess the
cybersecurity vulnerabilities and defensive capabilities of Large Language Models (LLMs). Building on its
predecessor, CyberSecEval 3, this latest version introduces a new benchmark, [AutoPatchBench](https://engineering.fb.com/2025/04/29/ai-research/autopatchbench-benchmark-ai-powered-security-fixes), which measures an LLM agent's capability to automatically patch security vulnerabilities in native code.

Created to measure and incentivize the development of highly capable LLM based defender tools, CyberSecEval 4 builds on the CyerSecEval code base to offer a comprehensive set of tools for evaluting both vulnerabilities and defensive capabilities in various security domains. It has been applied to well-known LLMs such as Meta's Llama4 as well as OpenAI, Google, and Anthropic models. The findings from the collection of CyberSecEval studies underscore both substantial cybersecurity threats, as well as promising oppottunities to leverage these models for uplifting defenders.

The repository includes several types of benchmarks:

1.  **MITRE and False Refusal Rate (FRR) Tests**: These tests evaluate an LLM's
    compliance when asked to assist in cyberattacks as well as how often an LLM
    incorrectly refuses a borderline but essentially benign query.

    1. **MITRE Tests**: These tests use the MITRE ATT&CK framework to evaluate
       an LLM's compliance when asked to assist in cyberattacks.

    2. **False Refusal Rate (FRR) Tests**: These tests measure how often an LLM
       incorrectly refuses a borderline but essentially benign query, due to
       misinterpreting the prompt as a malicious request.

2.  **Secure Code Generation Tests**: These tests assess an LLM's propensity to
    generate secure code across various contexts.

    1. **Instruct Tests**: These tests assess an LLM's propensity to generate
       insecure code when given a specific instruction.

    2. **Autocomplete Tests**: These tests measure how often an LLM suggests
       insecure coding practices in autocomplete contexts, where the LLM
       predicts subsequent code based on preceding code.

3.  **Prompt Injection Tests**: These tests assess an LLM’s susceptibility to
    “prompt injection attacks” - attacks in which a portion of the LLM prompt
    coming from untrusted user input contains malicious instructions intended to
    override the LLM’s original task.

    1. **Textual Prompt Injection Tests**: An english-only dataset of textual
       prompt injections.

    2. **Visual Prompt Injection Tests**: These tests assess an LLM’s
       susceptibility to “visual prompt injection attacks” - attacks in which
       untrusted multimodal user input (consisting of both text and images)
       contains malicious instructions intended to override the LLM’s original
       task.

4.  **Code Interpreter Tests**: These tests evaluate the security risks posed by
    integrating LLMs with code interpreters, specifically assessing how
    effectively an LLM can prevent malicious attempts to exploit the system or
    execute harmful code.

5.  **Vulnerability Exploitation Tests**: These tests measure the LLM's program
    exploitation capabilities by asking the LLMs to solve "capture the flag"
    style challenges.

6.  **Spear Phishing Capability Tests**: These tests are designed to evaluate
    the capabilities of LLMs in spear phishing scenarios. The focus is on
    assessing the LLM's persuasiveness and its effectiveness in convincing
    targeted victims to meet specific phishing objectives.

7.  **Autonomous Offensive Cyber Operations Tests**: These tests are designed to
    evaluate the capabilities of LLMs to function autonomously as a cyber attack
    agent.

8.  **AutoPatch Tests**: These tests are designed to evaluate the capabilities
    of LLMs to autonomously generate security patches for vulnerabilities, specifically
    the crashes discovered by fuzzer tests.
