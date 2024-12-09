# Getting Started

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

Each benchmark can run tests for multiple LLMs. Our command line interface uses
the format `<PROVIDER>::<MODEL>::<API KEY>` to specify an LLM to test. We
currently support APIs from OPENAI, ANYSCALE, and TOGETHER. The followings are a
few examples:

- `OPENAI::gpt-4::<API KEY>`
- `OPENAI::gpt-3.5-turbo::<API KEY>`
- `ANYSCALE::meta-llama/Llama-2-13b-chat-hf::<API KEY>`
- `TOGETHER::togethercomputer/llama-2-7b::<API KEY>`

### How to use a different model supported by Anyscale or Together

1. Lookup the models supported by
   [Anyscale](https://d3qavlo5goipcw.cloudfront.net/guides/models#select-a-model)
   or [Together](https://docs.together.ai/docs/fine-tuning-models)
2. Update `valid_models` method in
   [llm.py](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/benchmark/llm.py)
   with the model name that you choose to test with.

### How to run benchmarks for self hosted models

1. Extend `llm.py`.
2. `def query(self, prompt: str) -> str:` Implement your inferencing logic.
3. Make sure response of your llm is always a string, thats returned from the
   `query` function.
4. Update supported providers in `llm.create` method with the name of LLM you
   created from step (1).
5. Pass the name of your LLM in `--llm-under-test` options in the following
   format (`LLM Name::model-name::random-string`)

### Other notes about running benchmarks

- Please run these commands from the root directory of the PurpleLlama project.
- `run-llm-in-parallel` makes async parallel calls to the specified LLM
  provider. This speeds up prompt evaluation.
