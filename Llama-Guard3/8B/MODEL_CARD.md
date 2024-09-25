# Llama Guard 3-8B Model Card

## Model Details

Llama Guard 3-8B is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 3 was aligned to safeguard against the MLCommons standardized hazards taxonomy and designed to support Llama 3.1 capabilities. Specifically, it provides content moderation in 8 languages, and was optimized to support safety and security for search and code interpreter tool calls.

Below is a response classification example for Llama Guard 3.

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/Llama-Guard3/8B/llama_guard_3_8B_figure.png" width="800"/>
</p>

In order to produce classifier scores, we look at the probability for the first token, and use that as the “unsafe” class probability. We can then apply score thresholding to make binary decisions.

## Get started

Once you have access to the model weights, please refer to our [documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) to get started.

## Hazard Taxonomy and Policy

The model is trained to predict safety labels on the 14 categories shown below, based on the [MLCommons taxonomy](https://mlcommons.org/2024/04/mlc-aisafety-v0-5-poc/) of 13 hazards, as well as an additional category for Code Interpreter Abuse for tool calls use cases.

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
    <td>S14: Code Interpreter Abuse</td>
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

**S11 - Suicide &amp; Self-Harm**

Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating

**S12 - Sexual Content**

Responses that contain erotica

**S13 - Elections**

Responses that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections

**S14 - Code Interpreter Abuse**

Responses that seek to abuse code interpreters, including those that enable denial of service attacks, container escapes or privilege escalation exploits

## Supported languages

Llama Guard 3 supports content safety for the following languages : English, French, German, Hindi, Italian, Portuguese, Spanish, Thai.

## Training Data

We use the English data used by Llama Guard [1], which are obtained by getting Llama 2 and Llama 3 generations on prompts from the hh-rlhf dataset [2]. In order to scale training data for new categories and new capabilities such as multilingual and tool use, we collect additional human and synthetically generated data. Similar to the English data, the multilingual data are Human-AI conversation data that are either single-turn or multi-turn. To reduce the model’s false positive rate, we curate a set of multilingual benign prompt and response data where LLMs likely reject the prompts.

For the tool use capability, we consider search tool calls and code interpreter abuse. To develop training data for search tool use, we use Llama3 to generate responses to a collected and synthetic set of prompts. The generations are based on the query results obtained from the Brave Search API. To develop synthetic training data to detect code interpreter attacks, we use an LLM to generate safe and unsafe prompts.  Then, we use a non-safety-tuned LLM to generate code interpreter completions that comply with these instructions.  For safe data, we focus on data close to the boundary of what would be considered unsafe, to minimize false positives on such borderline examples.

## Evaluation

**Note on evaluations:** As discussed in the original Llama Guard paper, comparing model performance is not straightforward as each model is built on its own policy and is expected to perform better on an evaluation dataset with a policy aligned to the model. This highlights the need for industry standards. By aligning the Llama Guard family of models with the Proof of Concept MLCommons taxonomy of hazards, we hope to drive adoption of industry standards like this and facilitate collaboration and transparency in the LLM safety and content evaluation space.

In this regard, we evaluate the performance of Llama Guard 3 on MLCommons hazard taxonomy and compare it across languages with Llama Guard 2 [3] on our internal test. We also add GPT4 as baseline with zero-shot prompting using MLCommons hazard taxonomy.

Tables 1, 2, and 3 show that Llama Guard 3 improves over Llama Guard 2 and outperforms GPT4 in English, multilingual, and tool use capabilities. Noteworthily,  Llama Guard 3 achieves better performance with much lower false positive rates. We also benchmark Llama Guard 3 in the OSS dataset XSTest [4] and observe that it achieves the same F1 score but a lower false positive rate compared to Llama Guard 2.

<div align="center">
<small> Table 1: Comparison of performance of various models measured on our internal English test set for MLCommons hazard taxonomy (response classification).</small>

|                | **F1 ↑** | **AUPRC ↑** | **False Positive<br>Rate ↓** |
|--------------------------|:--------:|:-----------:|:----------------------------:|
| Llama Guard 2            |  0.877 |   0.927   |          0.081          |
| Llama Guard 3            |  0.939 |   0.985   |          0.040          |
| GPT4                     |  0.805 |    N/A    |          0.152          |

</div>

<br>

<table align="center">
<small><center>Table 2: Comparison of multilingual performance of various models measured on our internal test set for MLCommons hazard taxonomy (prompt+response classification).</center></small>
<thead>
  <tr>
    <th colspan="8"><center>F1 ↑ / FPR ↓</center></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>French</center></td>
    <td><center>German</center></td>
    <td><center>Hindi</center></td>
    <td><center>Italian</center></td>
    <td><center>Portuguese</center></td>
    <td><center>Spanish</center></td>
    <td><center>Thai</center></td>
  </tr>
  <tr>
    <td>Llama Guard 2</td>
    <td><center>0.911/0.012</center></td>
    <td><center>0.795/0.062</center></td>
    <td><center>0.832/0.062</center></td>
    <td><center>0.681/0.039</center></td>
    <td><center>0.845/0.032</center></td>
    <td><center>0.876/0.001</center></td>
    <td><center>0.822/0.078</center></td>
  </tr>
  <tr>
    <td>Llama Guard 3</td>
    <td><center>0.943/0.036</center></td>
    <td><center>0.877/0.032</center></td>
    <td><center>0.871/0.050</center></td>
    <td><center>0.873/0.038</center></td>
    <td><center>0.860/0.060</center></td>
    <td><center>0.875/0.023</center></td>
    <td><center>0.834/0.030</center></td>
  </tr>
  <tr>
    <td>GPT4</td>
    <td><center>0.795/0.157</center></td>
    <td><center>0.691/0.123</center></td>
    <td><center>0.709/0.206</center></td>
    <td><center>0.753/0.204</center></td>
    <td><center>0.738/0.207</center></td>
    <td><center>0.711/0.169</center></td>
    <td><center>0.688/0.168</center></td>
  </tr>
</tbody>
</table>

<br>

<table align="center">
<small><center>Table 3: Comparison of performance of various models measured on our internal test set for other moderation capabilities (prompt+response classification).</center></small>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Search tool calls</th>
     <th colspan="3">Code interpreter abuse</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>F1 ↑</center></td>
    <td><center>AUPRC ↑</center></td>
    <td><center>FPR ↓</center></td>
    <td><center>F1 ↑</center></td>
    <td><center>AUPRC ↑</center></td>
    <td><center>FPR ↓</center></td>
  </tr>
  <tr>
    <td>Llama Guard 2</td>
    <td><center>0.749</center></td>
    <td><center>0.794</center></td>
    <td><center>0.284</center></td>
    <td><center>0.683</center></td>
    <td><center>0.677</center></td>
    <td><center>0.670</center></td>
  </tr>
  <tr>
    <td>Llama Guard 3</td>
    <td><center>0.856</center></td>
    <td><center>0.938</center></td>
    <td><center>0.174</center></td>
    <td><center>0.885</center></td>
    <td><center>0.967</center></td>
    <td><center>0.125</center></td>
  </tr>
  <tr>
    <td>GPT4</td>
    <td><center>0.732</center></td>
    <td><center>N/A</center></td>
    <td><center>0.525</center></td>
    <td><center>0.636</center></td>
    <td><center>N/A</center></td>
    <td><center>0.90</center></td>
  </tr>
</tbody>
</table>

## Application

As outlined in the Llama 3 paper, Llama Guard 3 provides industry leading system-level safety performance and is recommended to be deployed along with Llama 3.1. Note that, while deploying Llama Guard 3 will likely improve the safety of your system, it might increase refusals to benign prompts (False Positives). Violation rate improvement and impact on false positives as measured on internal benchmarks are provided in the Llama 3 paper.

## Quantization

We are committed to help the community deploy Llama systems responsibly. We provide a quantized version of Llama Guard 3 to lower the deployment cost. We used int 8 [implementation](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) integrated into the hugging face ecosystem, reducing the checkpoint size by about 40% with very small impact on model performance. In Table 5, we observe that the performance quantized model is comparable to the original model.

<table align="center">
<small><center>Table 5: Impact of quantization on Llama Guard 3 performance.</center></small>
<tbody>
<tr>
<td rowspan="2"><br />
<p><span><b>Task</b></span></p>
</td>
<td rowspan="2"><br />
<p><span><b>Capability</b></span></p>
</td>
<td colspan="4">
<p><center><span><b>Non-Quantized</b></span></center></p>
</td>
<td colspan="4">
<p><center><span><b>Quantized</b></span></center></p>
</td>
</tr>
<tr>
<td>
<p><span><b>Precision</b></span></p>
</td>
<td>
<p><span><b>Recall</b></span></p>
</td>
<td>
<p><span><b>F1</b></span></p>
</td>
<td>
<p><span><b>FPR</b></span></p>
</td>
<td>
<p><span><b>Precision</b></span></p>
</td>
<td>
<p><span><b>Recall</b></span></p>
</td>
<td>
<p><span><b>F1</b></span></p>
</td>
<td>
<p><span><b>FPR</b></span></p>
</td>
</tr>
<tr>
<td rowspan="3">
<p><span>Prompt Classification</span></p>
</td>
<td>
<p><span>English</span></p>
</td>
<td>
<p><span>0.952</span></p>
</td>
<td>
<p><span>0.943</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.057</span></p>
</td>
<td>
<p><span>0.961</span></p>
</td>
<td>
<p><span>0.939</span></p>
</td>
<td>
<p><span>0.950</span></p>
</td>
<td>
<p><span>0.045</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Multilingual</span></p>
</td>
<td>
<p><span>0.901</span></p>
</td>
<td>
<p><span>0.899</span></p>
</td>
<td>
<p><span>0.900</span></p>
</td>
<td>
<p><span>0.054</span></p>
</td>
<td>
<p><span>0.906</span></p>
</td>
<td>
<p><span>0.892</span></p>
</td>
<td>
<p><span>0.899</span></p>
</td>
<td>
<p><span>0.051</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Tool Use</span></p>
</td>
<td>
<p><span>0.884</span></p>
</td>
<td>
<p><span>0.958</span></p>
</td>
<td>
<p><span>0.920</span></p>
</td>
<td>
<p><span>0.126</span></p>
</td>
<td>
<p><span>0.876</span></p>
</td>
<td>
<p><span>0.946</span></p>
</td>
<td>
<p><span>0.909</span></p>
</td>
<td>
<p><span>0.134</span></p>
</td>
</tr>
<tr>
<td rowspan="3">
<p><span>Response Classification</span></p>
</td>
<td>
<p><span>English</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.931</span></p>
</td>
<td>
<p><span>0.939</span></p>
</td>
<td>
<p><span>0.040</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.925</span></p>
</td>
<td>
<p><span>0.936</span></p>
</td>
<td>
<p><span>0.040</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Multilingual</span></p>
</td>
<td>
<p><span>0.929</span></p>
</td>
<td>
<p><span>0.805</span></p>
</td>
<td>
<p><span>0.862</span></p>
</td>
<td>
<p><span>0.033</span></p>
</td>
<td>
<p><span>0.931</span></p>
</td>
<td>
<p><span>0.785</span></p>
</td>
<td>
<p><span>0.851</span></p>
</td>
<td>
<p><span>0.031</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Tool Use</span></p>
</td>
<td>
<p><span>0.774</span></p>
</td>
<td>
<p><span>0.884</span></p>
</td>
<td>
<p><span>0.825</span></p>
</td>
<td>
<p><span>0.176</span></p>
</td>
<td>
<p><span>0.793</span></p>
</td>
<td>
<p><span>0.865</span></p>
</td>
<td>
<p><span>0.827</span></p>
</td>
<td>
<p><span>0.155</span></p>
</td>
</tr>
</tbody>
</table>

## Get started

Llama Guard 3 is available by default in Llama 3.1 [reference implementations](https://github.com/meta-llama/llama-agentic-system). You can learn more about how to configure and customize using [Llama Recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai/) shared on our Github repository.

## Limitations
There are some limitations associated with Llama Guard 3. First, Llama Guard 3 itself is an LLM fine-tuned on Llama 3.1. Thus, its performance (e.g., judgments that need common sense knowledge, multilingual capability, and policy coverage) might be limited by its (pre-)training data.

Some hazard categories may require factual, up-to-date knowledge to be evaluated (for example, S5: Defamation, S8: Intellectual Property, and S13: Elections) . We believe more complex systems should be deployed to accurately moderate these categories for use cases highly sensitive to these types of hazards, but Llama Guard 3 provides a good baseline for generic use cases.

Lastly, as an LLM, Llama Guard 3 may be susceptible to adversarial attacks or prompt injection attacks that could bypass or alter its intended use. Please feel free to [report](https://github.com/meta-llama/PurpleLlama) vulnerabilities and we will look to incorporate improvements in future versions of Llama Guard.

## Citation
```
@misc{dubey2024llama3herdmodels,
  title =         {The Llama 3 Herd of Models},
  author =        {Llama Team, AI @ Meta},
  year =          {2024}
  eprint =        {2407.21783},
  archivePrefix = {arXiv},
  primaryClass =  {cs.AI},
  url =           {https://arxiv.org/abs/2407.21783}
}
```

## References

[1] [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

[2] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

[3] [Llama Guard 2 Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md)

[4] [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models](https://arxiv.org/abs/2308.01263)
