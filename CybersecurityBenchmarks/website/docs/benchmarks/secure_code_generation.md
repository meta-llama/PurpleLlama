# Secure Code Benchmark

## Running Secure Code Generation Benchmarks

### For Instruct Benchmark

```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=instruct \
   --prompt-path="$DATASETS/instruct/instruct.json" \
   --response-path="$DATASETS/instruct_responses.json" \
   --stat-path="$DATASETS/instruct_stat.json" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ...
   [--run-llm-in-parallel]
```

### For Autocomplete Benchmark

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

- _Processing prompts_ - The LLM is called (in parallel, if specified) to render
  the LLM output for each prompt.
- _Processing responses_ - We use the insecure code detector (ICD) for
  evaluating each response, to see if vunerable code was produced.


## Results:

Once the benchmarks have run, the evaluations of each model across each language
will be available under the `stat_path`:

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
