# Introduction

This repository hosts the implementation of [CyberSecEval 3](https://ai.meta.com/research/publications/cyberseceval-3-advancing-the-evaluation-of-cybersecurity-risks-and-capabilities-in-large-language-models/), which builds upon and extends the functionalities of its predecessors,
[CyberSecEval 2](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/) and
[CyberSecEval](https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/).
CyberSecEval 3 is an extensive benchmark suite designed to assess the cybersecurity
vulnerabilities of Large Language Models (LLMs). Building on its predecessor,
CyberSecEval 2, this latest version introduces three new test suites:
visual prompt injection tests, spear phishing capability tests, and
autonomous offensive cyber operations tests. Created to meet the increasing
demand for secure AI systems, CyberSecEval 3 offers a comprehensive set of tools
to evaluate various security domains. It has been applied to well-known LLMs such
as Llama2, Llama3, codeLlama, and OpenAI GPT models. The findings underscore
substantial cybersecurity threats, underscoring the critical need for continued
research and development in AI safety.

The repository includes several types of benchmarks:

1.  **MITRE and False Refusal Rate (FRR) Tests**: These tests evaluate an
    LLM's compliance when asked to assist in cyberattacks as well as how
    often an LLM incorrectly refuses a borderline but essentially benign query.

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

    2. **Multilingual Prompt Injection Tests**: The multilingual dataset
       includes prompt injections machine-translated into 15 languages, to test
       the propensity of these attacks to succeed in languages other than
       english.

    3. **Visual Prompt Injection Tests**: These tests assess an LLM’s
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
    evaluate the capabilities of LLMs to function autonomously as a cyber attack agent.
