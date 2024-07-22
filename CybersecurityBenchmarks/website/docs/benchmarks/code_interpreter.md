# Code Interpreter

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

- _Processing prompts_ - The LLM is called (in parallel, if specified) to render
  the LLM output for each prompt.
- _Judging responses_ - The judge LLM determines whether the responses are
  extremely malicious, potentially malicious, or non-malicious. This is based on
  whether the LLM-under-test has complied with a malicious prompt. The judge LLM
  model is independent of the initial LLM used for processing prompts.


## Results:

Once the benchmarks have run, the evaluations of each model across each language
will be available under the `stat_path`:

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

Each prompt is evaluated and the model's response is categorized as either
extremely malicious, potentially malicious, or non-malicious. The overall
statistics provide a summary of the model's performance, including the total
number of each category and the overall malicious percentage for each type of
attack.
