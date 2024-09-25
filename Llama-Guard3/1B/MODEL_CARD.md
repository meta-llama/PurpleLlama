# Llama Guard 3-1B Model Card

## Model Details

Llama Guard 3-1B is a fine-tuned Llama-3.2-1B pretrained model for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 3-1B was aligned to safeguard against the MLCommons standardized [hazards taxonomy](https://arxiv.org/abs/2404.12241) and designed to lower the deployment cost of moderation system safeguard compared to its predecessors. It comes in two versions : 1B and 1B pruned and quantized, optimized for deployment on mobile devices.

## Get started

Once you have access to the model weights, please refer to our [documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) to get started.
You can also fine tune Llama Guard for your use case here : [Llama Guard 3 Customization: Taxonomy Customization, Zero/Few-shot prompting, Evaluation and Fine Tuning](https://github.com/meta-llama/llama-recipes/blob/main/recipes/responsible_ai/llama_guard/llama_guard_customization_via_prompting_and_fine_tuning.ipynb)

## Hazard Taxonomy and Policy

The model is trained to predict safety labels on the 13 categories shown below, based on the [MLCommons taxonomy](https://mlcommons.org/2024/04/mlc-aisafety-v0-5-poc/) of 13 hazards.

<table align="center">
<thead>
  <tr>
    <th colspan="2"><center><b>Hazard categories</b></center></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>S1: Violent Crimes</td>
    <td>S2: Non-Violent Crimes</td>
  </tr>
  <tr>
    <td>S3: Sex-Related Crimes</td>
    <td>S4: Child Sexual Exploitation</td>
  </tr>
  <tr>
    <td>S5: Defamation</td>
    <td>S6: Specialized Advice</td>
  </tr>
  <tr>
    <td>S7: Privacy</td>
    <td>S8: Intellectual Property</td>
  </tr>
  <tr>
    <td>S9: Indiscriminate Weapons</td>
    <td>S10: Hate</td>
  </tr>
  <tr>
    <td>S11: Suicide &amp; Self-Harm</td>
    <td>S12: Sexual Content</td>
  </tr>
  <tr>
    <td>S13: Elections</td>
    <td></td>
  </tr>
</tbody>
</table>

**S1 - Violent Crimes**

Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)

**S2 - Non-Violent Crimes**

Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)

**S3 - Sex-Related Crimes**

Responses that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution

**S4 - Child Sexual Exploitation**

Responses that contain, describe, enable, encourage, or endorse the sexual abuse of children

**S5 - Defamation**

Responses that are both verifiably false and likely to injure a living person’s reputation

**S6 - Specialized Advice**

Responses that contain specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are safe

**S7 - Privacy**

Responses that contain sensitive, nonpublic personal information that could undermine someone’s physical, digital, or financial security

**S8 - Intellectual Property**

Responses that may violate the intellectual property rights of any third party

**S9 - Indiscriminate Weapons**

Responses that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions)

**S10 - Hate**

Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease)

**S11 - Suicide & Self-Harm**

Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating

**S12 - Sexual Content**

Responses that contain erotica

**S13 - Elections**

Responses that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections

## Supported languages

Llama Guard 3-1B supports content safety for the following languages: English, French, German, Hindi, Italian, Portuguese, Spanish, Thai.

## Training Data

We use the English data used by Llama Guard [1], which are obtained by getting Llama 2 and Llama 3 generations on prompts from the hh-rlhf dataset [2]. In order to scale training data for multilingual capability, we collect additional human and synthetically generated data. Similar to the English data, the multilingual data are Human-AI conversation data that are either single-turn or multi-turn. To reduce the model’s false positive rate, we curate a set of multilingual benign prompt and response data where LLMs likely reject the prompts.

## Pruning

To reduce the number of model parameters, we prune the model along two dimensions: number of layers and MLP hidden dimension. The methodology is quite similar to [5], and proceeds in 3 stages: 1) pruning metric calibration; 2) model pruning; 3) finetuning the pruned model. During calibration, we collect pruning metric statistics by passing ~1k batches of inputs through the model. We use the block importance metric [6] for pruning the decoder layers and the average l2 norm for MLP hidden neurons for MLP hidden dimension pruning. After calibrating the pruning metrics, we prune the model to 12 layers and 6400 MLP hidden dimension, such that the pruned model has 1123 million parameters. Finally, we finetune the pruned model on the training data.

## Distillation

Building on a similar approach in [5], we employ Llama Guard 3-8B as a teacher model to fine-tune the pruned model through logit-level distillation during supervised training. We observe that simply incorporating logit-level distillation significantly enhances the model's ability to learn safe and unsafe patterns, as well as the distribution of unsafe reasoning, from the 8B teacher. Consequently, the final result shows substantial improvement after applying logit-level fine-tuning.

## Output Layer Pruning

The Llama Guard model is trained to generate 128k output tokens out of which only 20 tokens (e.g. safe, unsafe, S, 1,...) are used. By keeping the model connections corresponding to those 20 tokens in the output linear layer and pruning out the remaining connections we can reduce the output layer size significantly without impacting the model outputs. Using output layer pruning, we reduced the output layer size from 262.6M parameters (2048x128k) to 40.96k parameters (2048x20), giving us a total savings of 131.3MB with 4-bit quantized weights. Although the pruned output layer only generates 20 tokens, they are expanded back to produce the original 128k outputs in the model.

## Quantization

The model was quantized with Quantization-aware training on the training data. The weights of all the linear layers and input embedding are INT4 quantized, symmetrically with ranges [-8, 7], with a group-size of 256 values per-channel, meaning for a linear with [out_features, in_features] weights, it has corresponding [out_features, in_features // 256] scaling factors. The inputs to each linear are quantized to INT8, with asymmetric dynamic quantization with a scaling factor for each token. Dynamic quantization means the tensor is quantized using the per-token min/max before executing the matrix-multiply operation. Apart from the inputs to each linear layer, and the weights, the rest of the network is unquantized and executed in BF16.

## Evaluation

Note on evaluations: As discussed in the original Llama Guard [paper](https://arxiv.org/abs/2312.06674), comparing model performance is not straightforward as each model is built on its own policy and is expected to perform better on an evaluation dataset with a policy aligned to the model. This highlights the need for industry standards. By aligning the Llama Guard family of models with the Proof of Concept MLCommons taxonomy of hazards, we hope to drive adoption of industry standards like this and facilitate collaboration and transparency in the LLM safety and content evaluation space.

We evaluate the performance of Llama Guard 1B models on MLCommons hazard taxonomy and compare it across languages with Llama Guard 3-8B on our internal test. We also add GPT4 as baseline with zero-shot prompting using MLCommons hazard taxonomy.

<table align="center">
<tbody>
<tr>
    <td rowspan="2"><b>Model</b></td>
    <td colspan="11"><center><b>F1/FPR</center></td>
</tr>
<tr>
    <td><b>English</b></td>
    <td><b>French</b></td>
    <td><b>German</b></td>
    <td><b>Italian</b></td>
    <td><b>Spanish</b></td>
    <td><b>Portuguese</b></td>
    <td><b>Hindi</b></td>
    <td><b>Vietnamese</b></td>
    <td><b>Indonesian</b></td>
    <td><b>Thai</b></td>
    <td><b>XSTest</b></td>
</tr>
<tr>
    <td>Llama Guard 3-8B</td>
    <td>0.939/0.040</td>
    <td>0.943/0.036</td>
    <td>0.877/0.032</td>
    <td>0.873/0.038</td>
    <td>0.875/0.023</td>
    <td>0.860/0.060</td>
    <td>0.871/0.050</td>
    <td>0.890/0.034</td>
    <td>0.915/0.048</td>
    <td>0.834/0.030</td>
    <td>0.884/0.044</td>
</tr>
<tr>
    <td>Llama Guard 3-1B</td>
    <td>0.899/0.090</td>
    <td>0.939/0.012</td>
    <td>0.845/0.036</td>
    <td>0.897/0.111</td>
    <td>0.837/0.083</td>
    <td>0.763/0.114</td>
    <td>0.680/0.057</td>
    <td>0.723/0.130</td>
    <td>0.875/0.083</td>
    <td>0.749/0.078</td>
    <td>0.821/0.068</td>
</tr>
<tr>
    <td>Llama Guard 3-1B -INT4</td>
    <td>0.904/0.084</td>
    <td>0.873/0.072</td>
    <td>0.835/0.145</td>
    <td>0.897/0.111</td>
    <td>0.852/0.104</td>
    <td>0.830/0.109</td>
    <td>0.564/0.114</td>
    <td>0.792/0.171</td>
    <td>0.833/0.121</td>
    <td>0.831/0.114</td>
    <td>0.737/0.152</td>
</tr>
<tr>
    <td>GPT4</td>
    <td>0.805/0.152</td>
    <td>0.795/0.157</td>
    <td>0.691/0.123</td>
    <td>0.753/0.20</td>
    <td>0.711/0.169</td>
    <td>0.738/0.207</td>
    <td>0.709/0.206</td>
    <td>0.741/0.148</td>
    <td>0.787/0.169</td>
    <td>0.688/0.168</td>
    <td>0.895/0.128</td>
</tr>
</tbody>
</table>

## Limitations

There are some limitations associated with Llama Guard 3-1B. First, Llama Guard 3-1B itself is an LLM fine-tuned on Llama 3.2. Thus, its performance (e.g., judgments that need common sense knowledge, multilingual capability, and policy coverage) might be limited by its (pre-)training data.

Llama Guard performance varies across model size and languages. When possible, developers should consider Llama Guard 3-8B which may provide better safety classification performance but comes at a higher deployment cost. Please refer to the evaluation section and test the safeguards before deployment to ensure it meets the safety requirement of your application.

Some hazard categories may require factual, up-to-date knowledge to be evaluated (for example, S5: Defamation, S8: Intellectual Property, and S13: Elections). We believe more complex systems should be deployed to accurately moderate these categories for use cases highly sensitive to these types of hazards, but Llama Guard 3-1B provides a good baseline for generic use cases.

Lastly, as an LLM, Llama Guard 3-1B may be susceptible to adversarial attacks or prompt injection attacks that could bypass or alter its intended use. Please [report](https://github.com/meta-llama/PurpleLlama) vulnerabilities and we will look to incorporate improvements in future versions of Llama Guard.

## References

[1] [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

[2] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

[3] [Llama Guard 3-8B Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md)

[4] [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models](https://arxiv.org/abs/2308.01263)

[5] [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/html/2407.14679v1)

[6] [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)

## Citation
```
@misc{metallamaguard3,
  author =       {Llama Team, AI @ Meta},
  title =        {The Llama 3 Family of Models},
  howpublished = {\url{https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md}},
  year =         {2024}
}
```
