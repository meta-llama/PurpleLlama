# Llama Guard 3-11B-vision Model Card

## Model Details

Llama Guard 3 Vision is a Llama-3.2-11B pretrained model, fine-tuned for content safety classification. Similar to previous versions [1-3], it can be used to safeguard content for both LLM inputs (prompt classification) and LLM responses (response classification).

Llama Guard 3 Vision was specifically designed to support image reasoning use cases and was optimized to detect harmful multimodal (text and image) prompts and text responses to these prompts.

Llama Guard 3 Vision acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated. Below is a response classification example input and output for Llama Guard 3 Vision.

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/Llama-Guard3/11B-vision/llama_guard_3_11B_vision_figure.png" width="800"/>
</p>

## Get started

Once you have access to the model weights, please refer to our [documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) to get started.

## Hazard Taxonomy and Policy

The model is trained to predict safety labels on the 13 categories shown below, based on the [MLCommons taxonomy](https://mlcommons.org/2025/01/mlc-aisafety-v0-5-poc/) of 13 hazards.


Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)

**S2 - Non-Violent Crimes**

Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation,), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)

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

**Note on violation for image understanding.** Compared to text-only Llama Guards, specific attention was paid to risks emerging from potential prompts to identify people in images (e.g., recognizing a real world person from the image through any visual clue). Llama Guard 3 Vision was trained to classify the response as unsafe.

To train the Llama Guard 3 Vision, we employed a hybrid dataset comprising both human-generated and synthetically generated data. Our approach involved collecting human-created prompts paired with corresponding images, as well as generating benign and violating model responses using our in-house Llama models. We utilized jailbreaking techniques to elicit vio
nses from these models. The resulting dataset includes samples labeled either by humans or the Llama 3.1 405B model. To ensure comprehensive coverage, we carefully curated the dataset to encompass a diverse range of prompt-image pairs, spanning all hazard categories listed above. For the image data we use, our vision encoder will rescale it into 4 chunks, each of 560x560.

## Evaluation

We evaluate the performance of Llama Guard 3 vision on our internal test following MLCommons hazard taxonomy. To the best of our knowledge, Llama Guard 3 Vision is the first safety classifier for the LLM image understanding task. In this regard, we use GPT-4o and GPT-4o mini with zero-shot prompting using MLCommons hazard taxonomy as a baseline.
**Result Analysis.** Table 1 shows that Llama Guard 3 Vision outperforms GPT-4o and GPT-4o mini in terms of F1, specifically in response classification. Noteworthily,  Llama Guard 3 Vision achieves better performance with much lower false positive rates. We also observe that the recall (false positive rates) of prompt classification are much lower (higher) than in general, this is because of the ambiguity of prompt (text+image) in the image understanding tasks making prompt classification more challenging than response classification. For example, given a text prompt “how do I buy this” with an image of multiple objects, whether the prompt is safe or not depends on which objects the user is referring to, and it is hard to decide in some cases. In such cases, we recommend using Llama Guard 3 Vision in the response classification task. Table 2 also shows the category breakdown in our internal test set. We observe that Llama Guard 3 Vision performs well in Indiscriminate Weapon and Elections categories, while showing > 0.69 F1 scores across all categories.



There are some limitations associated with Llama Guard 3 Vision. First, Llama Guard 3 Vision itself is an LLM fine-tuned on Llama 3.2-vision. Thus, its performance (e.g., judgments that need common sense knowledge, multilingual capability, and policy coverage) might be limited by its (pre-)training data.

Llama Guard 3 Vision is not meant to be used as an image safety classifier nor a text-only safety classifier. Its task is to classify the multimodal prompt or the multimodal prompt along with the text response. It was optimized for English language and only supports one image at the moment. Images will be rescaled into 4 chunks each of 560x560, so the classification performance may vary depending on the actual image size. For text-only mitigation, we recommend using other safeguards in the Llama Guard family of models, such as Llama Guard 3-8B or Llama Guard 3-1B depending on your use case.

Some hazard categories may require factual, up-to-date knowledge to be evaluated (for example, S5: Defamation, S8: Intellectual Property, and S13: Elections) . We believe more complex systems should be deployed to accurately moderate these categories for use cases highly sensitive to these types of hazards, but Llama Guard 3 Vision provides a good baseline for generic use cases.

Lastly, as an LLM, Llama Guard 3 Vision may be susceptible to adversarial attacks [4, 5] that could bypass or alter its intended use. Please [report](https://github.com/meta-llama/PurpleLlama) vulnerabilities and we will look to incorporate improvements in future versions of Llama Guard.



[1] [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

[2] [Llama Guard 2 Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md)

[3] [Llama Guard 3-8B Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md)

[4] [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)

[5] [Are aligned neural networks adversarially aligned?](https://arxiv.org/abs/2306.15447)

## Citation
```
@misc{chi2024llamaguard3vision,
      title={Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations}, 
      author={Jianfeng Chi and Ujjwal Karn and Hongyuan Zhan and Eric Smith and Javier Rando and Yiming Zhang and Kate Plawiak and Zacharie Delpierre Coudert and Kartikeya Upasani and Mahesh Pasupuleti},
      year={2024}Vqs
