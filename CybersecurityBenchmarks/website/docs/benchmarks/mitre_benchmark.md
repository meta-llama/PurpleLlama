# MITRE and MITRE FRR Benchmarks
## Running the MITRE and MITRE False Refusal Rate (FRR) Benchmarks

### Running MITRE Benchmark

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

- _Processing prompts_ - The LLM is called (in parallel, if specified) to render
  the LLM output for each prompt.
- _Response expansion_ - The expansion LLM expands the initial responses from
  the prompt.
- _Judging expanded responses_ - The judge LLM decided whether the expanded
  responses would be helpful in implementing a cyberattack.

The expansion LLM model and judge LLM model are independent of the initial LLM
for processing prompts. Although we've used Llama and Code Llama models for the
original paper, we recommend using GPT-3.5 including an OpenAI API key.

### Running MITRE False Refusal Rate (FRR) Benchmark

```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=mitre-frr \
   --prompt-path="$DATASETS/mitre_frr/mitre_frr.json" \
   --response-path="$DATASETS/mitre_frr/mitre_frr_responses.json" \
   --stat-path="$DATASETS/mitre_frr/mitre_frr_stat.json" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

MITRE FRR benchmark is run in two steps:

- _Processing prompts_ The LLM is called (in parallel, if specified) to render
  the LLM output for each prompt. Once the response is received, a keyword-based
  judgment is conducted and the judged result is attached.
- _Gathering stat_ The benchmark extracts all models' judged results from the
  `response-path` and outputs them to the `stat-path`.

## Results:

Once the benchmarks have run, the evaluations of each model across each language
will be available under the `stat_path`:

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

Responses for each prompt are categorized as _refusal_, _malicious_, or
_benign_.

### MITRE False Refusal Rate (FRR) Results

```
{
    "accept_count": ...,
    "refusal_count": ...,
    "refusal_rate": ...
}
```
