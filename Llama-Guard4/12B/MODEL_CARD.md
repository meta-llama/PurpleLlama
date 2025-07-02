# Llama Guard 4 Model Card

## Model Details

Llama Guard 4 is a natively multimodal safety classifier with 12 billion parameters trained jointly on text and multiple images. It is a dense architecture pruned from the Llama 4 Scout pre-trained model and fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). Llama Guard 4 itself acts as an LLM: it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 4 was aligned to safeguard against the standardized MLCommons [hazards taxonomy](https://arxiv.org/abs/2503.05731) and designed to support multimodal Llama 4 capabilities within a single safety classifier. Specifically, it combines the capabilities of the previous Llama Guard 3-8B and Llama Guard 3-11B-vision models by supporting English and multilingual text prompts (on the languages [supported by Llama Guard 3](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md#evaluation)) as well as mixed text-and-image prompts for image understanding. Unlike Llama Guard 3-11B-vision, Llama Guard 4 now supports safety classification when multiple images are given in the prompt as input. Llama Guard 4 is also integrated into the Llama Moderations API for text and images.

## Getting Started

Once you have access to the model weights, please refer to our [documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) to get started.

## Hazard Taxonomy and Policy

The model is trained to predict safety labels on the categories shown below, based on the MLCommons safety taxonomy. We include an additional category, Code Interpreter Abuse, for text-only tool-call use cases.

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
    <td>S11: Suicide & Self-Harm</td>
    <td>S12: Sexual Content</td>
  </tr>
  <tr>
    <td>S13: Elections</td>
    <td>S14: Code Interpreter Abuse (text only)</td>
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

**S14 - Code Interpreter Abuse (text-only)**

Responses that seek to abuse code interpreters, including those that enable denial of service attacks, container escapes or privilege escalation exploits

## Model Architecture

Llama Guard 4 is a natively multimodal safeguard model. The model has 12 billion parameters in total and uses an early fusion transformer architecture with dense layers to keep the overall size small. The model can be run on a single GPU. Llama Guard 4 shares the same tokenizer and vision encoder as Llama 4 Scout and Maverick.

## Model Training

### Pretraining and Pruning

Llama Guard 4 employs a dense feedforward early-fusion architecture, and it differs from Llama 4 Scout, which employs Mixture-of-Experts (MoE) layers. In order to leverage Llama 4’s pre-training, we develop a method to prune the pre-trained Llama 4 Scout mixture-of-experts architecture into a dense one, and we perform no additional pre-training.

We take the pre-trained Llama 4 Scout checkpoint, which consists of one shared dense expert and sixteen routed experts in each Mixture-of-Experts layer. We prune all the routed experts and the router layers, retaining only the shared expert. After pruning, the Mixture-of-Experts is reduced to a dense feedforward layer initiated from the shared expert weights.

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/Llama-Guard4/12B/llama_guard_4_12b_before_pruning.png" width="800"/>
  <figcaption>Before pruning: Llama 4 Scout pre-trained checkpoint</figcaption>
</p>

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/Llama-Guard4/12B/llama_guard_4_12b_after_pruning.png" width="800"/>
  <figcaption>After pruning and post-training: Llama Guard 4</figcaption>
</p>

### Post-Training for Safety Classification

We post-trained the model after pruning with a blend of data from the [Llama Guard 3-8B](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/README.md) and [Llama Guard 3-11B-vision](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/11B-vision/README.md) models, with the following additional data:
- Multi-image training data, with most samples containing from 2 to 5 images
- Multilingual data, both written by expert human annotators and translated from English

We blend the training data from both modalities, with a ratio of roughly 3:1 text-only data to multimodal data containing one or more images.

## Evaluation

### System-level safety

Llama Guard 4 is designed to be used in an integrated system with a generative language model, reducing the overall rate of safety violations exposed to the user. Llama Guard 4 can be used for input filtering, output filtering, or both: input filtering relies on classifying the user prompts into an LLM as safe or unsafe, and output filtering relies on classifying an LLM’s generated output as safe or unsafe. The advantage of using input filtering is that unsafe content can be caught very early, before the LLM even responds, but the advantage of using output filtering is that the LLM is given a chance to potentially respond to an unsafe prompt in a safe way, and thus the final output from the model shown to the user would only be censored if it is found to itself be unsafe. Using both filtering types gives additional security.

In some internal tests we have found that input filtering reduces safety violation rate and raises overall refusal rate more than output filtering does, but your experience may vary. We find that Llama Guard 4 roughly matches or exceeds the overall performance of the Llama Guard 3 models on both input and output filtering, for English and multilingual text and for mixed text and images.

### Classifier performance

The table below demonstrates how Llama Guard 4 matches or exceeds the overall performance of Llama Guard 3-8B (LG3) on English and multilingual text, as well as Llama Guard 3-11B-vision (LG3v) on prompts with single or multiple images, using an in-house test set:

<br>

<table align="center">
<thead>
  <tr>
    <th></th>
    <th colspan="3">Absolute values</th>
    <th colspan="3">vs. Llama Guard 3</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>R</center></td>
    <td><center>FPR</center></td>
    <td><center>F1</center></td>
    <td><center>Δ R</center></td>
    <td><center>Δ FPR</center></td>
    <td><center>Δ F1</center></td>
  </tr>
  <tr>
    <td><left>English</left></td>
    <td>69%</td>
    <td>11%</td>
    <td>61%</td>
    <td>4%</td>
    <td>-3%</td>
    <td>8%</td>
  </tr>
  <tr>
    <td><left>Multilingual</left></td>
    <td>43%</td>
    <td>3%</td>
    <td>51%</td>
    <td>-2%</td>
    <td>-1%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td><left>Single-image</left></td>
    <td>41%</td>
    <td>9%</td>
    <td>38%</td>
    <td>10%</td>
    <td>0%</td>
    <td>8%</td>
  </tr>
  <tr>
    <td><left>Multi-image</left></td>
    <td>61%</td>
    <td>9%</td>
    <td>52%</td>
    <td>20%</td>
    <td>-1%</td>
    <td>17%</td>
  </tr>
</tbody>
</table>

<br>

R: recall, FPR: false positive rate. Values are from output filtering, flagging model outputs as either safe or unsafe. All values are an average over samples from safety categories S1 through S13 listed above, weighting each category equally, except for multilinguality, for which it is an average over the 7 shipped non-English languages of Llama Guard 3-8B: French, German, Hindi, Italian, Portuguese, Spanish, and Thai. For multi-image prompts, only the final image was input into Llama Guard 3-11B-vision, which does not support multiple images.

We omit evals against competitor models, which are typically not aligned with the specific safety policy that this classifier was trained on, limiting our ability to make direct comparisons.

## Limitations

There are some limitations associated with Llama Guard 4. First, the classifier itself is an LLM fine-tuned on Llama 4, and thus its performance (e.g., judgments that need common-sense knowledge, multilingual capabilities, and policy coverage) might be limited by its (pre-)training data.

Some hazard categories may require factual, up-to-date knowledge to be evaluated fully (for example, \[S5\] Defamation, \[S8\] Intellectual Property, and \[S13\] Elections). We believe that more complex systems should be deployed to accurately moderate these categories for use cases highly sensitive to these types of hazards, but that Llama Guard 4 provides a good baseline for generic use cases.

Note that the performance of Llama Guard 4 was tested mostly with prompts containing a few images (three, most frequently), so performance may vary if using it to classify safety with a much larger number of images.

Lastly, as an LLM, Llama Guard 4 may be susceptible to adversarial attacks or prompt injection attacks that could bypass or alter its intended use: see [Prompt Guard 2](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md) for detecting prompt attacks. Please feel free to [report](https://github.com/meta-llama/PurpleLlama) vulnerabilities, and we will look into incorporating improvements into future versions of Llama Guard.

Please refer to the [Developer Use Guide](https://www.llama.com/developer-use-guide/) for additional best practices and safety considerations.
