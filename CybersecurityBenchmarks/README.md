# Getting Started

## Prerequisites
Ensure you have [weggli](https://github.com/weggli-rs/weggli) installed. Their release is outdated and we're currently using revision `9d97d462854a9b682874b259f70cc5a97a70f2cc`:

```
cargo install weggli --rev=9d97d462854a9b682874b259f70cc5a97a70f2cc --git=https://github.com/weggli-rs/weggli
```

Set the environment variable `WEGGLI_PATH` to point to the weggli execution path. For example, if `weggli` is set to reference the path of the executable, use the below command.
```
export WEGGLI_PATH=weggli
```

## Setup Virtual Environment
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

## Runing the benchmarks
The main module for benchmarsk is `CybersecurityBenchmarks.benchmark.run`. Run
```
python3 -m CybersecurityBenchmarks.benchmark.run --help
```
to get a list of available options.

We simplify the following commands by setting a `DATASETS` environment variable:
```
export DATASETS=$PWD/CybersecurityBenchmarks/datasets
```
We currently support three kinds of benchmarks: MITRE, instruct, and auto-complete. Please refer to the whitepaper for detailed discussion of the different kinds.

Each benchmark can run tests for multiple LLMs. Our command line interface uses the format `<PROVIDER>::<MODEL>::<API KEY>` to specify an LLM to test. We currently support the following LLMs:

- `OPENAI::gpt-4::<API KEY>`
- `OPENAI::gpt-3.5-turbo::<API KEY>`
- `ANYSCALE::meta-llama/Llama-2-13b-chat-hf::<API KEY>`
- `TOGETHER::togethercomputer/llama-2-7b::<API KEY>`

### How to use a different model supported by Anyscale or Together?
1. Lookup the models supported by [Anyscale](https://d3qavlo5goipcw.cloudfront.net/guides/models#select-a-model) or [Together](https://docs.together.ai/docs/fine-tuning-models)
2. Update ```valid_models``` method in [llm.py](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/benchmark/llm.py) with the model name that you choose to test with.

### How to run benchmarks for self hosted models?
1. Extend `llm.py`.
2. `def query(self, prompt: str) -> str:` Implement your inferencing logic.
3. Make sure response of your llm is always a string, thats returned from the `query` function.
4. Update supported providers in `llm.create` method with the name of LLM you created from step (1).
5. Pass the name of your LLM in `--llm-under-test` options in the following format (`LLM Name::model-name::random-string`)

## Running MITRE benchmark

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
Notes:
- Please run this command from the root directory of the PurpleLlama project.
- `run-llm-in-parallel` makes async parallel calls to the specified LLM provider. This speeds up prompt evaluation.


The MITRE benchmarks are run in three steps:

- *Processing prompts* - The LLM is called (in parallel, if specified) to render the LLM output for each prompt.
- *Response expansion* - The expansion LLM expands the initial responses from the prompt.
- *Judging expanded responses* - The judge LLM decided whether the expanded responses would be helpful in implementing a cyberattack.

The expansion LLM model and judge LLM model are independent of the initial LLM for processing prompts. Although we've used Llama and Code Llama models for the original paper, we recommend using GPT-3.5 including an OpenAI API key.

## Running Instruct and Autocomplete benchmarks

For instruct benchmarks run:
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=instruct \
   --prompt-path="$DATASETS/instruct/instruct.json" \
   --response-path="$DATASETS/instruct_responses.json" \
   --stat-path="$DATASETS/instruct_stat.json \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```
Notes:
- Please run this command from the root directory of the PurpleLlama project.

### For autocomplete benchmarks run:
```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=autocomplete \
   --prompt-path="$DATASETS/autocomplete/autocomplete.json" \
   --response-path="$DATASETS/autocomplete_responses.json" \
   --stat-path="$DATASETS/instruct_stat.json \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```
Notes:
- Please run this command from the root directory of the PurpleLlama project.

Both the instruct and autocomplete benchmarks are run in two steps:

- *Processing prompts* - The LLM is called (in parallel, if specified) to render the LLM output for each prompt.
- *Processing responses* - We use the insecure code detector for evaluating each response, to see if vunerable code was produced.

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

### Instruct and Autocomplete Results

```
{
    "model_name": {
        "language": {
            "bleu": ...,
            "total_count": ...,
            "vulnerable_percentage": ...,
            "vulnerable_suggestion_count": ...
        }
    }
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
