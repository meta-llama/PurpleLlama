<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/logo.png" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/purple-llama-open-trust-safety-generative-ai"> Blog</a>&nbsp |  <a href="https://ai.meta.com/llama/purple-llama">Website</a>&nbsp | <a href="https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/">CyberSec Eval Paper</a>&nbsp&nbsp | <a href="https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/">Llama Guard Paper</a>&nbsp
<br>

---

# Purple Llama

Purple Llama is an umbrella project that over time will bring together tools
and evals to help the community build responsibly with open generative AI
models. The initial release will include tools and evals for Cyber Security and
Input/Output safeguards but we plan to contribute more in the near future.

## Why purple?

Borrowing a [concept](https://www.youtube.com/watch?v=ab_Fdp6FVDI) from the
cybersecurity world, we believe that to truly mitigate the challenges which
generative AI presents, we need to take both attack (red team) and defensive
(blue team) postures. Purple teaming, composed of both red and blue team
responsibilities, is a collaborative approach to evaluating and mitigating
potential risks and the same ethos applies to generative AI and hence our
investment in Purple Llama will be comprehensive.

## License

Components within the Purple Llama project will be licensed permissively
enabling both research and commercial usage. We believe this is a major step
towards enabling community collaboration and standardizing the development and
usage of trust and safety tools for generative AI development. More concretely
evals and benchmarks are licensed under the MIT license while any models use the
Llama 2 Community license. See the table below:

| **Component Type** |            **Components**            |                                          **License**                                           |
| :----------------- | :----------------------------------: | :--------------------------------------------------------------------------------------------: |
| Evals/Benchmarks   | Cyber Security Eval (others to come) |                                              MIT                                               |
| Models             |             Llama Guard              | [Llama 2 Community License](https://github.com/facebookresearch/PurpleLlama/blob/main/LICENSE) |
| Models             |             Llama Guard 2            | Llama 3 Community License |
| Safeguard          |             Code Shield              | MIT |

## Evals & Benchmarks

### Cybersecurity

#### CyberSec Eval v1
CyberSec Eval v1 was what we believe was the first industry-wide set of cybersecurity safety evaluations for LLMs. These benchmarks are based on industry guidance and standards (e.g., CWE and MITRE ATT&CK) and built in collaboration with our security subject matter experts. We aim to provide tools that will help address some risks outlined in the [White House commitments on developing responsible AI](https://www.whitehouse.gov/briefing-room/statements-releases/2023/07/21/fact-sheet-biden-harris-administration-secures-voluntary-commitments-from-leading-artificial-intelligence-companies-to-manage-the-risks-posed-by-ai/), including:
* Metrics for quantifying LLM cybersecurity risks.
* Tools to evaluate the frequency of insecure code suggestions.
* Tools to evaluate LLMs to make it harder to generate malicious code or aid in carrying out cyberattacks.

We believe these tools will reduce the frequency of LLMs suggesting insecure AI-generated code and reduce their helpfulness to cyber adversaries. Our initial results show that there are meaningful cybersecurity risks for LLMs, both with recommending insecure code and for complying with malicious requests. See our [Cybersec Eval paper](https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/) for more details.

#### CyberSec Eval 2
CyberSec Eval 2 expands on its predecessor by measuring an LLM’s propensity to abuse a code interpreter, offensive cybersecurity capabilities, and susceptibility to prompt injection. You can read the paper [here](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/).

You can also check out the 🤗 leaderboard [here](https://huggingface.co/spaces/facebook/CyberSecEval).


## System-Level Safeguards

As we outlined in Llama 3’s
[Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/), we
recommend that all inputs and outputs to the LLM be checked and filtered in
accordance with content guidelines appropriate to the application.

### Llama Guard

To support this, and empower the community, we released Llama Guard, an openly-available model that performs competitively on common open benchmarks and provides developers with a pretrained model to help defend against generating potentially risky outputs. As part of our ongoing commitment to open and transparent science, we also released our methodology and an extended discussion of model performance in our [Llama Guard paper](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/).

We are happy to share an updated version, Meta Llama Guard 2. Llama Guard 2 was optimized to support the newly [announced](https://mlcommons.org/2024/04/mlc-aisafety-v0-5-poc/) policy published by MLCommons, expanding its coverage to a more comprehensive set of safety categories, out-of-the-box.

It also comes with better classification performance than Llama Guard 1 and improved zero-shot and few shot adaptability.

Ultimately, our vision is to enable developers to customize this model to support relevant use cases and to make it easier to adopt best practices and improve the open ecosystem.

### Code Shield

Code Shield adds support for inference-time filtering of insecure code produced by LLMs. Code Shield offers mitigation of insecure code suggestions risk, code interpreter abuse prevention, and secure command execution. [CodeShield Example Notebook](https://github.com/meta-llama/PurpleLlama/blob/main/CodeShield/notebook/CodeShieldUsageDemo.ipynb).



## Getting Started

To get started and learn how to use Purple Llama components with Llama models,
see the getting started guide [here](https://ai.meta.com/llama/get-started/).
The guide provides information and resources to help you set up Llama, including
how to access the model, hosting how-to information and integration guides. Additionally,
you will find supplemental materials to further assist you while responsibly
building with Llama. The guide will be updated as more Purple Llama components
get released.

## FAQ

For a running list of frequently asked questions, for not only Purple Llama
components but also generally for Llama models, see the FAQ
[here](https://ai.meta.com/llama/faq/).

## Join the Purple Llama community

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.
