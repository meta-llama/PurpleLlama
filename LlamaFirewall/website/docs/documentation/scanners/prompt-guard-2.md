# PromptGuard 2

## Overview
PromptGuard 2 is a fine-tuned BERT-style model designed to detect direct jailbreak attempts in real-time, with high accuracy and low latency. It operates on user prompts and untrusted data sources, providing an additional layer of defense when paired with other scanners. This model targets universal jailbreak attempts that may manifest as prompt injections originating from user inputs or tool outputs.

Visit its [repository](https://github.com/meta-llama/PurpleLlama/blob/main/Prompt-Guard/) to access the README and related source code.

## Design of PromptGuard 2
PromptGuard 2 is built using BERT-based architectures, specifically the DeBERTa series of models from Microsoft. This design allows for a more accurate and efficient detection of jailbreak attempts. The model has been improved through a refined model scope, which focuses solely on detecting explicit jailbreaking techniques. This means that the model is trained to recognize specific patterns and phrases that are commonly used in jailbreak attempts.

PromptGuard 2 has been trained on an expanded dataset featuring a diverse range of benign and malicious inputs, enabling the model to better distinguish between legitimate and malicious code and improve its jailbreak detection capabilities. The training objective has also been optimized with an energy-based loss function, enhancing the model's learning efficiency and ability to generalize to new data. Furthermore, a tokenization fix has been implemented to counter adversarial tokenization attacks, ensuring the model can detect and prevent input manipulation attempts. As a lightweight model, PromptGuard 2 is easily deployable on both CPU and GPU, making it ideal for real-time LLM input processing and facilitating rapid and accurate jailbreak detection.

## Security Risks Covered
By detecting and preventing jailbreak attempts, PromptGuard 2 helps protect against:
* **Direct Universal Jailbreak Prompt Injections:** PromptGuard 2 detects jailbreak input, preventing direct universal jailbreak prompt injections.
* **Indirect Universal Jailbreak Prompt Injections:** PromptGuard 2 detects jailbreak input, preventing indirect universal jailbreak prompt injections.
* **Malicious Code via Prompt Injection:** PromptGuard 2, along with other scanners, provides a layered defense against code-oriented prompt injection, ensuring that malicious code is identified and mitigated before it can be executed.
