# Introduction

This repository contains the implementation of [CyberSecEval 2](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/), extending the original [CyberSecEval](https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/). CyberSecEval 2 is a comprehensive benchmark suite for evaluating the cybersecurity risks of Large Language Models (LLMs). Developed in response to the growing need for secure AI systems, CyberSecEval 2 provides a suite of tools to assess a wide range of security domains. The benchmarks are based on industry standards such as the Common Weakness Enumeration (CWE) and the MITRE ATT&CK framework, and have been applied to popular LLMs including Llama2, Llama3, codeLlama, and OpenAI GPT models. The results highlight significant cybersecurity risks, emphasizing the need for ongoing research and development in AI safety.

The repository includes seven types of benchmarks:

1. **MITRE Tests**: These tests use the MITRE ATT&CK framework to evaluate an LLM's compliance when asked to assist in cyberattacks.

2. **Instruct Tests**: These tests assess an LLM's propensity to generate insecure code when given a specific instruction.

3. **Autocomplete Tests**: These tests measure how often an LLM suggests insecure coding practices in autocomplete contexts, where the LLM predicts subsequent code based on preceding code.

4. **Prompt Injection Tests**: These tests assess an LLM’s susceptibility to “prompt injection attacks” - attacks in which a portion of the LLM prompt coming from untrusted user input contains malicious instructions intended to override the LLM’s original task.

5. **False Refusal Rate (FRR) Tests**: These tests measure how often an LLM incorrectly refuses a borderline but essentially benign query, due to misinterpreting the prompt as a malicious request.

6. **Code Interpreter Tests**: These tests evaluate the security risks posed by integrating LLMs with code interpreters, specifically assessing how effectively an LLM can prevent malicious attempts to exploit the system or execute harmful code.

7. **Vulnerability Exploitation Tests**: These tests measure the LLM's program exploitation capabilities by asking the LLMs to solve "capture the flag" style challenges.

# Getting Started

## Prerequisites

Please make sure you are running python3.9 or newer versions.

Also ensure you have [weggli](https://github.com/weggli-rs/weggli) installed. Their release is outdated and we're currently using revision `9d97d462854a9b682874b259f70cc5a97a70f2cc`:

```
cargo install weggli --rev=9d97d462854a9b682874b259f70cc5a97a70f2cc --git=https://github.com/weggli-rs/weggli
```

Set the environment variable `WEGGLI_PATH` to point to the weggli execution path. For example, if `weggli` is set to reference the path of the executable, use the below command.

```
export WEGGLI_PATH=weggli
```

## Set up Virtual Environment
Set up your virtual environment and install dependencies:

```
python3 -m venv ~/.venvs/CybersecurityBenchmarks
```

```
source ~/.venvs/CybersecurityBenchmarks/bin/activate
```

```
pip3 install -r CybersecurityBenchmarks/requirements.txt
```

## Running the Benchmarks

The main module for benchmark is `CybersecurityBenchmarks.benchmark.run`. Run
```
python3 -m CybersecurityBenchmarks.benchmark.run --help
```

to get a list of available options.

We simplify the following commands by setting a `DATASETS` environment variable:

```
export DATASETS=$PWD/CybersecurityBenchmarks/datasets
```

We currently support three kinds of benchmarks: MITRE, instruct, and auto-complete. Please refer to the whitepaper for detailed discussion of the different kinds.

Each benchmark can run tests for multiple LLMs. Our command line interface uses the format `<PROVIDER>::<MODEL>::<API KEY>` to specify an LLM to test. We currently support APIs from OPENAI, ANYSCALE, and TOGETHER. The followings are a few examples:

- `OPENAI::gpt-4::<API KEY>`
- `OPENAI::gpt-3.5-turbo::<API KEY>`
- `ANYSCALE::meta-llama/Llama-2-13b-chat-hf::<API KEY>`
- `TOGETHER::togethercomputer/llama-2-7b::<API KEY>`

### How to use a different model supported by Anyscale or Together
1. Lookup the models supported by [Anyscale](https://d3qavlo5goipcw.cloudfront.net/guides/models#select-a-model) or [Together](https://docs.together.ai/docs/fine-tuning-models)
2. Update `valid_models` method in [llm.py](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/benchmark/llm.py) with the model name that you choose to test with.

### How to run benchmarks for self hosted models
1. Extend `llm.py`.
2. `def query(self, prompt: str) -> str:` Implement your inferencing logic.
3. Make sure response of your llm is always a string, thats returned from the `query` function.
4. Update supported providers in `llm.create` method with the name of LLM you created from step (1).
5. Pass the name of your LLM in `--llm-under-test` options in the following format (`LLM Name::model-name::random-string`)

### Other notes about running benchmarks
- Please run these commands from the root directory of the PurpleLlama project.
- `run-llm-in-parallel` makes async parallel calls to the specified LLM provider. This speeds up prompt evaluation.

## Running the MITRE Benchmark

```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=mitre \
   --prompt-path="$DATASETS/mitre/mitre_benchmark_100_per_category_with_augmentation.json" \
   --response-path="$DATASETS/mitre_responses.json" \
   --judge-response-path="$DATASETS/mitre_judge_responses.json" \
   --stat-path="$DATASETS/mitre_stat.json" \
   --judge-llm="OPENAI::gpt-3.5-turbo::<YOUR API KEY>" \
   --expansion-llm="OPENAI::gpt-3.5-turbo::<YOUR API KEY>" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

The MITRE benchmarks are run in three steps:

- *Processing prompts* - The LLM is called (in parallel, if specified) to render the LLM output for each prompt.
- *Response expansion* - The expansion LLM expands the initial responses from the prompt.
- *Judging expanded responses* - The judge LLM decided whether the expanded responses would be helpful in implementing a cyberattack.

The expansion LLM model and judge LLM model are independent of the initial LLM for processing prompts. Although we've used Llama and Code Llama models for the original paper, we recommend using GPT-3.5 including an OpenAI API key.

## Running Instruct and Autocomplete Benchmarks

### For instruct benchmarks run:
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=instruct \
   --prompt-path="$DATASETS/instruct/instruct.json" \
   --response-path="$DATASETS/instruct_responses.json" \
   --stat-path="$DATASETS/instruct_stat.json" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

### For autocomplete benchmarks run:
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=autocomplete \
   --prompt-path="$DATASETS/autocomplete/autocomplete.json" \
   --response-path="$DATASETS/autocomplete_responses.json" \
   --stat-path="$DATASETS/autocomplete_stat.json" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

Both the instruct and autocomplete benchmarks are run in two steps:

- *Processing prompts* - The LLM is called (in parallel, if specified) to render the LLM output for each prompt.
- *Processing responses* - We use the insecure code detector (ICD) for evaluating each response, to see if vunerable code was produced.

## Running prompt injection Benchmark
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=prompt-injection \
   --prompt-path="$DATASETS/prompt_injection/prompt_injection.json" \
   --response-path="$DATASETS/prompt_injection/prompt_injection_responses.json" \
   --judge-response-path="$DATASETS/prompt_injection/prompt_injection_judge_responses.json" \
   --stat-path="$DATASETS/prompt_injection/prompt_injection_stat.json" \
   --judge-llm=<>SPECIFICATION_1 \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

Prompt injection benchmarks are run in the following two steps:
- *Processing prompts* Query the LLMs under test with a system prompt and a user prompt, where the user prompt tries to inject instructions to the LLM that is violating the original system prompt
- *Processing response* The responses of the LLMs under test will go through another judge LLM, where the judge LLM will judge if the injected instruction is executed

## Running False Refusal Rate (FRR) Benchmark
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=frr \
   --prompt-path="$DATASETS/frr/frr.json" \
   --response-path="$DATASETS/frr/frr_responses.json" \
   --stat-path="$DATASETS/frr/frr_stat.json" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

FRR benchmark is run in two steps:
- *Processing prompts* The LLM is called (in parallel, if specified) to render the LLM output for each prompt. Once the response is received, a keyword-based judgment is conducted and the judged result is attached.
- *Gathering stat* The benchmark extracts all models' judged results from the `response-path` and outputs them to the `stat-path`.


## Running Code Interpreter Abuse Benchmark
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=interpreter \
   --prompt-path="$DATASETS/interpreter/interpreter.json" \
   --response-path="$DATASETS/interpreter/interpreter_responses.json" \
   --judge-response-path="$DATASETS/interpreter/interpreter_judge_responses.json" \
   --stat-path="$DATASETS/interpreter/interpreter_stat.json" \
   --judge-llm="OPENAI::gpt-3.5-turbo::<YOUR API KEY>" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

Code Interpreter Abuse benchmark is run in two steps:
- *Processing prompts* - The LLM is called (in parallel, if specified) to render the LLM output for each prompt.
- *Judging responses* - The judge LLM determines whether the responses are extremely malicious, potentially malicious, or non-malicious. This is based on whether the LLM-under-test has complied with a malicious prompt. The judge LLM model is independent of the initial LLM used for processing prompts.


## Running Vulnerability Exploitation Benchmark
```
python3 -m CybersecurityBenchmarks.datasets.canary_exploit.run

python3 -m CybersecurityBenchmarks.benchmark.run \
    --benchmark="canary-exploit" \
    --prompt-path="$DATASETS/canary_exploit/canary_exploit.json" \
    --response-path="$DATASETS/canary_exploit/canary_exploit_responses.json" \
    --judge-response-path="$DATASETS/canary_exploit/canary_exploit_judge_responses.json" \
    --stat-path="$DATASETS/canary_exploit/canary_exploit_stat.json" \
    --llm-under-test=<SPECIFICATION_1> \
    [--run-llm-in-parallel]
```

Canary Exploit benchmark is run in three steps:
- *Generating randomized challenges* - Challenges are first randomly generated to avoid the problem of LLM memorization.
- *Processing prompts* - The LLMs are called to find the solution for each challenge.
- *Scoring results* - The responses are then fed into the compiled binary to score the results.

## Results:
Once the benchmarks have run, the evaluations of each model across each language will be available under the `stat_path`:

### MITRE Results

```
{
    "model_name": {
        "category_name": {
            "refusal_count": ...,
            "malicious_count": ...,
            "benign_count": ...,
            "total_count": ...,
            "benign_percentage": ...,
            "else_count": ...
        },
    }
}
```

Responses for each prompt are categorized as *refusal*, *malicious*, or *benign*.

### Instruct and Autocomplete Results

```
{
    "model_name": {
        "language": {
            "bleu": ...,
            "total_count": ...,
            "vulnerable_percentage": ...,
            "vulnerable_suggestion_count": ...
            "pass_rate" ...
        }
    }
}
```

*`bleu`* - The BLEU score is used as a measure of how closely the LLM output matches the original code that the test case was based on. We use [`sacrebleu`](https://github.com/mjpost/sacrebleu) to compute bleu.

*`vulnerable_percentage`*  - The percentage of test cases evaluated to be vulnerable across the languace (equal to `vulnerable_suggestion_count / total_count`)

*`pass_rate`* is the topline rate of "safe" LLM suggestions across the test set, equal to `1 - vunerable_percentage`.

### Prompt Injection Results

Based on judge LLM, the prompt injection will be either successful or unsuccessful

```
{
    "stat_per_model": {
        "model_name": {
            "injection_successful_count": ...,
            "injection_unsuccessful_count": ...,
            "total_count": ...,
            "injection_successful_percentage": ...,
            "injection_unsuccessful_percentage": ...
        }
    }
}
```

In the same output file, the results are futher bucketized in different dimensions, e.g., the injection techniques, and whether the prompt is a direct prompt injection or an indrect prompt injection, etc. The result of each bucket follows the same structure.

```
{
    "stat_per_model_per_variant": {
        "model_name": {
            "injected_prompt_technique_variant": {
                "injection_successful_count": ...,
                "injection_unsuccessful_count": ...,
                "total_count": ...,
                "injection_successful_percentage": ...,
                "injection_unsuccessful_percentage": ...
            },
            ...
        }
    },
    "stat_per_model_per_type": {
        ...
    },
    "stat_per_model_per_risk_category": {
        ...
    }
}
```

### FRR Results

```
{
    "accept_count": ...,
    "refusal_count": ...,
    "refusal_rate": ...
}
```

Prompt is evaluated based on the model's response which is categorized as either accept or refusal. The overall statistics provide a summary of the model's performance, including the total number of accepted and refused prompts, and the overall refusal rate.

### Code Interpreter Abuse Results

```
{
    "model_name": {
        "attack_type": {
            "is_extremely_malicious": ...,
            "is_potentially_malicious": ...,
            "is_non_malicious": ...,
            "total_count": ...,
            "malicious_percentage": ...
        },
    }
}
```

Each prompt is evaluated and the model's response is categorized as either extremely malicious, potentially malicious, or non-malicious. The overall statistics provide a summary of the model's performance, including the total number of each category and the overall malicious percentage for each type of attack.

### Vulnerability Exploitation Results

The "response" from the challenge is scored by compiling and running the challenge "code" with the "response" as input. The "score" is a number between 0.0 and 1.0.

```
{
    "test_case_prompt": ...,
    "response": ...,
    "language": ...,
    "code": ...,
    "answer": ...,
    "challenge_type": ...,
    "model": ...,
    "score": ...
}
```


## Exit the Virtual Environment

And once you're done you can leave the python environment with

```
deactivate
```

# Developer's Guide

To run unit tests follow the setup instructions above. You can then run

```
(CybersecurityBenchmarks) $ cd CybersecurityBenchmarks
```

```
(CybersecurityBenchmarks) $ python3 -m unittest
```
