# Prompt Guard 2

Prompt Guard 2 is a new model for guardrailing LLM inputs against prompt attacks and jailbreaking techniques. For more information, see our [Model card.](86M/MODEL_CARD.md)

The model is an improved version of the [original Prompt Guard model](../Prompt-Guard/README.md).

# Download

# Quick Start

PromptGuard is based on mDeBERTa, and can be loaded straightforwardly using the
huggingface `transformers` API with local weights or from huggingface:

```
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

We've added examples using Prompt Guard in the
[Llama recipes repository](https://github.com/facebookresearch/llama-recipes).
In particular, take a look at the
[Prompt Guard Tutorial](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/prompt_guard/prompt_guard_tutorial.ipynb)
and
[Prompt Guard Inference utilities](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/prompt_guard/inference.py).

# Issues

Please report any software bug, or other problems with the models through one of
the following means:

- Reporting issues with the Prompt Guard model:
  [github.com/meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama)
- Reporting issues with Llama in general:
  [github.com/meta-llama](https://github.com/meta-llama)
- Reporting bugs and security concerns:
  [facebook.com/whitehat/info](https://facebook.com/whitehat/info)

# License

Our model and weights are licensed for both researchers and commercial entities,
upholding the principles of openness. Our mission is to empower individuals, and
industry through this opportunity, while fostering an environment of discovery
and ethical AI advancements.

The same license as Llama 4 applies: see the [LICENSE](../LICENSE) file, as well
as our accompanying [Acceptable Use Policy](86M/USE_POLICY.md).

# References

- [LlamaFirewall Paper](https://ai.meta.com/research/publications/llamafirewall-an-open-source-guardrail-system-for-building-secure-ai-agents/)
