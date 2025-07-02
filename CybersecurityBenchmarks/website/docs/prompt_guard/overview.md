# Prompt Guard

![image](../../static/img/prompt_guard_visual.png)

Prompt Guard is a new model for guardrailing LLM inputs against prompt attacks -
in particular jailbreaking techniques and indirect injections embedded into
third party data. For more information, see our [Model card.](./model_card.md)

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
[Prompt Guard Tutorial](https://github.com/meta-llama/llama-recipes/blob/main/recipes/responsible_ai/prompt_guard/Prompt%20Guard%20Tutorial.ipynb)
and
[Prompt Guard Inference utilities](https://github.com/meta-llama/llama-recipes/blob/main/recipes/responsible_ai/prompt_guard/inference.py).

# Issues

Please report any software bug, or other problems with the models through one of
the following means:

- Reporting issues with the Llama Guard model:
  [github.com/meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama)
- Reporting issues with Llama in general:
  [github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- Reporting bugs and security concerns:
  [facebook.com/whitehat/info](https://facebook.com/whitehat/info)

# License

Our model and weights are licensed for both researchers and commercial entities,
upholding the principles of openness. Our mission is to empower individuals, and
industry through this opportunity, while fostering an environment of discovery
and ethical AI advancements.

The same license as Llama 3 applies: see the [LICENSE](../LICENSE.md) file, as
well as our accompanying Acceptable Use Policy.

# References

- [Llama 3 Paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [CyberSecEval 3](https://ai.meta.com/research/publications/cyberseceval-3-advancing-the-evaluation-of-cybersecurity-risks-and-capabilities-in-large-language-models/)
