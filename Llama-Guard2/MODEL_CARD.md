# Model Details

Meta Llama Guard 2 is an 8B parameter Llama 3-based [1] LLM safeguard model. Similar to [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/), it can be used for classifying content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.
Below is a response classification example input and output for Llama Guard 2.

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/Llama-Guard2/llamaguard_example.png" width="800"/>
</p>

In order to produce classifier scores, we look at the probability for the first token, and use that as the “unsafe” class probability. We can then apply score thresholding to make binary decisions.

# Harm Taxonomy and Policy

The model is trained to predict safety labels on the 11 categories shown below, based on the [MLCommons taxonomy](https://mlcommons.org/2024/04/mlc-aisafety-v0-5-poc/) of hazards.
<table align="center">
<thead>
  <tr>
    <th colspan="2">Harm categories</th>
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
    <td>S5: Specialized Advice</td>
    <td>S6: Privacy</td>
  </tr>
  <tr>
    <td>S7: Intellectual Property</td>
    <td>S8: Indiscriminate Weapons </td>
  </tr>
  <tr>
    <td>S9: Hate</td>
    <td>S10: Suicide &amp; Self-Harm</td>
  </tr>
  <tr>
    <td>S11: Sexual Content</td>
    <td></td>
  </tr>
</tbody>
</table>

**S1: Violent Crimes**

Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)

**S2: Non-Violent Crimes**

Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)

**S3: Sex-Related Crimes**

Responses that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution

**S4: Child Sexual Exploitation**

Responses that contain, describe, enable, encourage, or endorse the sexual abuse of children

**S5: Specialized Advice**

Responses that contain specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are safe

**S6: Privacy**

Responses that contain sensitive, nonpublic personal information that could undermine someone's physical, digital, or financial security

**S7: Intellectual Property**

Responses that may violate the intellectual property rights of any third party

**S8: Indiscriminate Weapons**

Responses that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions)

**S9: Hate**

Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease)

**S10: Suicide & Self-Harm**

Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating

**S11: Sexual Content**

Responses that contain erotica


# Training and Evaluation Data

We start with the training set of Llama Guard (see [model card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard/MODEL_CARD.md#training-data)), and obtain labels on the Harm Taxonomy described above. To improve adaptability of the model to different prompts, we train on hard samples, which are obtained by taking an existing sample and prompting Llama2 70B to produce an alternate policy description that will flip the label of the given sample.  

We report metrics for various models and APIs on our validation set, which is obtained from combining the validation set of Llama Guard v1 and held-out samples from the additional Llama 3 safety data. 

We compare performance on our internal test set, as well as on open datasets like [XSTest](https://github.com/paul-rottger/exaggerated-safety?tab=readme-ov-file#license), [OpenAI moderation](https://github.com/openai/moderation-api-release), and [BeaverTails](https://github.com/PKU-Alignment/beavertails). 

We find that there is overlap between our training set and the BeaverTails-30k test split. Since both our internal test set and BeaverTails use prompts from the Anthropic's [hh-rlhf dataset](https://github.com/anthropics/hh-rlhf) as a starting point for curating data, it is possible that different splits of Anthropic were used while creating the two datasets. Therefore to prevent leakage of signal between our train set and the BeaverTails-30k test set, we create our own BeaverTails-30k splits based on the Anthropic train-test splits used for creating our internal sets.

*Note on evaluations*: As discussed in the Llama Guard [paper](https://arxiv.org/abs/2312.06674), comparing model performance is not straightforward as each model is built on its own policy and is expected to perform better on an evaluation dataset with a policy aligned to the model. This highlights the need for industry standards. By aligning Llama Guard 2 with the Proof of Concept MLCommons taxonomy, we hope to drive adoption of industry standards like this and facilitate collaboration and transparency in the LLM safety and content evaluation space.

# Model Performance

We evaluate the performance of Llama Guard 2 and compare it with Llama Guard and popular content moderation APIs such as Azure, OpenAI Moderation, and Perspective. We use the token probability of the first output token (i.e. safe/unsafe) as the score for classification. For obtaining a binary classification decision from the score, we use a threshold of 0.5.

Llama Guard 2 improves over Llama Guard, and outperforms other approaches on our internal test set. Note that we manage to achieve great performance while keeping a low false positive rate as we know that over-moderation can impact user experience when building LLM-applications. 
<div align="center">

| **Model**                | **F1 ↑** | **AUPRC ↑** | **False Positive<br>Rate ↓** |
|--------------------------|:------:|:---------:|:-----------------------:|
| Llama Guard\*             |  0.665 |   <ins>0.854</ins>   |          0.027          |
| Llama Guard 2            |  **0.915** |   **0.974**   |          0.040          |
| GPT4                     |  <ins>0.796</ins> |    N/A    |          0.151          |
| OpenAI Moderation API    |  0.347 |   0.669   |          0.030          |
| Azure Content Safety API |  0.519 |    N/A    |          0.245          |
| Perspective API          |  0.265 |   0.586   |          0.046          |


<small> Table 1: Comparison of performance of various approaches measured on our internal test set.
<br><small><small>
*The performance of Llama Guard is lower on our new test set due to expansion of the number of harm categories from 6 to 11, which is not aligned to what Llama Guard was trained on.
</small></small></small>
</div>

<br>

<div align="center">

| **Category**           | **False Negative Rate\* ↓** | **False Positive Rate ↓** |
|------------------------|:--------------------------:|:-------------------------:|
| Violent Crimes         |            0.042           |           0.002           |
| Privacy                |            0.057           |           0.004           |
| Non-Violent Crimes     |            0.082           |           0.009           |
| Intellectual Property  |            0.099           |           0.004           |
| Hate                   |            0.190           |           0.005           |
| Specialized Advice     |            0.192           |           0.009           |
| Sexual Content         |            0.229           |           0.004           |
| Indiscriminate Weapons |            0.263           |           0.001           |
| Child Exploitation     |            0.267           |           0.000           |
| Sex Crimes             |            0.275           |           0.002           |
| Self-Harm              |            0.277           |           0.002           |



<small> Table 2: Category-wise breakdown of false negative rate and false positive rate for Llama Guard 2 on our internal benchmark for response classification with safety labels from the ML Commons taxonomy.<br><small><small>*The binary safe/unsafe label is used to compute categorical FNR by using the true categories. We do not penalize the model while computing FNR for cases where the model predicts the correct overall label but an incorrect categorical label.</small></small></small>

</div>

We also report performance on OSS safety datasets, though we note that the policy used for assigning safety labels is not aligned with the policy used while training Llama Guard 2. Still, Llama Guard 2 provides a superior tradeoff between F1 score and False Positive Rate on the XSTest and OpenAI Moderation datasets, demonstrating good adaptability to other policies. 

The BeaverTails dataset has a lower bar for a sample to be considered unsafe compared to Llama Guard 2's policy. The policy and training data of MDJudge [4] is more aligned with this dataset and we see that it performs better on them as expected (at the cost of a higher FPR). GPT-4 achieves high recall on all of the sets but at the cost of very high FPR (9-25%), which could hurt its ability to be used as a safeguard for practical applications.

<table align="center">
<thead>
  <tr>
    <th></th>
    <th colspan="3">(F1 ↑ / False Positive Rate ↓)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>False Refusals<br>(XSTest)</center></td>
    <td><center>OpenAI policy<br>(OpenAI Mod)</center></td>
    <td><center>BeaverTails policy<br>(BeaverTails-30k)</center></td>
  </tr>
  <tr>
    <td>Llama Guard</td>
    <td>0.737 / 0.079</td>
    <td>0.737 / 0.079</td>
    <td>0.599 / 0.035</td>
  </tr>
  <tr>
    <td>Llama Guard 2</td>
    <td>0.884 / 0.084</td>
    <td>0.807 / 0.060</td>
    <td>0.736 / 0.059</td>
  </tr>
  <tr>
    <td>MDJudge</td>
    <td>0.856 / 0.172</td>
    <td>0.768 / 0.212</td>
    <td>0.849 / 0.098</td>
  </tr>
  <tr>
    <td>GPT4</td>
    <td>0.895 / 0.128</td>
    <td>0.842 / 0.092</td>
    <td>0.802 / 0.256</td>
  </tr>
  <tr>
    <td>OpenAI Mod API</td>
    <td>0.576 / 0.040</td>
    <td>0.788 / 0.156</td>
    <td>0.284 / 0.056</td>
  </tr>
</tbody>
</table>

<div align="center">
<small>Table 3: Comparison of performance of various approaches measured on our internal test set for response classification. <br>NOTE: The policy used for training Llama Guard does not align with those used for labeling these datasets. Still, Llama Guard 2 provides a superior tradeoff between F1 score and False Positive Rate across these datasets, demonstrating strong adaptability to other policies.</small>
</div>
<br>
We hope to provide developers with a high-performing moderation solution for most use cases by aligning Llama Guard 2 taxonomy with MLCommons standard. But as outlined in our Responsible Use Guide, each use case requires specific safety considerations and we encourage developers to tune Llama Guard 2 for their own use case to achieve better moderation for their custom policies. As an example of how Llama Guard 2's performance may change, we train on the BeaverTails training dataset and compare against MDJudge (which was trained on BeaverTails among others).


<div align="center">
<br>

|          **Model**          | **F1 ↑** | **False Positive Rate ↓** |
|:---------------------------:|:--------:|:-------------------------:|
| Llama Guard 2               |   0.736  |           0.059           |
| MDJudge                     |   <ins>0.849</ins>  |           0.098           |
| Llama Guard 2 + BeaverTails |   **0.852**  |           0.101           |


<small>Table 4: Comparison of performance on BeaverTails-30k.</small>
</div>

# Limitations

There are some limitations associated with Llama Guard 2. First, Llama Guard 2 itself is an LLM fine-tuned on Llama 3. Thus, its performance (e.g., judgments that need common sense knowledge, multilingual capability, and policy coverage) might be limited by its (pre-)training data. 

Second, Llama Guard 2 is finetuned for safety classification only (i.e. to generate "safe" or "unsafe"), and is not designed for chat use cases. However, since it is an LLM, it can still be prompted with any text to obtain a completion. 

Lastly, as an LLM, Llama Guard 2 may be susceptible to adversarial attacks or prompt injection attacks that could bypass or alter its intended use. However, with the help of external components (e.g., KNN, perplexity filter), recent work (e.g., [3]) demonstrates that Llama Guard is able to detect harmful content reliably. 

**Note on Llama Guard 2's policy**

Llama Guard 2 supports 11 out of the 13 categories included in the [MLCommons AI Safety](https://mlcommons.org/working-groups/ai-safety/ai-safety/) taxonomy. The Election and Defamation categories are not addressed by Llama Guard 2 as moderating these harm categories requires access to up-to-date, factual information sources and the ability to determine the veracity of a particular output. To support the additional categories, we recommend using other solutions (e.g. Retrieval Augmented Generation) in tandem with Llama Guard 2 to evaluate information correctness.

# Citation

```
@misc{metallamaguard2,
  author =       {Llama Team},
  title =        {Meta Llama Guard 2},
  howpublished = {\url{https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md}},
  year =         {2024}
}
```

# References

[1] [Llama 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

[2] [Llama Guard Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard/MODEL_CARD.md)

[3] [RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content](https://arxiv.org/pdf/2403.13031.pdf)

[4] [MDJudge for Salad-Bench](https://huggingface.co/OpenSafetyLab/MD-Judge-v0.1)
