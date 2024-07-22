# Prompt Injection

## Running Prompt Injection Benchmarks

### Textual Prompt Injection Benchmark

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

Textual prompt injection benchmarks are run in the following two steps:

- _Processing prompts_ Query the LLMs under test with a system prompt and a user
  prompt, where the user prompt tries to inject instructions to the LLM that is
  violating the original system prompt
- _Processing response_ The responses of the LLMs under test will go through
  another judge LLM, where the judge LLM will judge if the injected instruction
  is executed

### Multilingual Text Prompt Injection Benchmark

For the multilingual textual prompt injection benchmark, the same benchmark
implementation is used, only the dataset is changed.

```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=prompt-injection \
   --prompt-path="$DATASETS/prompt_injection/prompt_injection_multilingual.json" \
   ... (rest same as above)
```

### Running Visual Prompt Injection Benchmark

```
python3 -m CybersecurityBenchmarks.benchmark.run \
   --benchmark=visual-prompt-injection \
   --prompt-path="$DATASETS/visual_prompt_injection/<DATASET>" \
   --response-path="$DATASETS/results/visual_prompt_injection/<DATASET>/visual_prompt_injection_model_responses.json" \
   --judge-response-path="$DATASETS/results/visual_prompt_injection/<DATASET>/visual_prompt_injection_judge_responses.json" \
   --stat-path="$DATASETS/results/visual_prompt_injection/<DATASET>/visual_prompt_injection_stats.json" \
   --judge-llm=<SPECIFICATION_1> \
   --llm-under-test=<SPECIFICATION_2> --llm-under-test=<SPECIFICATION_3> ... \
   [--run-llm-in-parallel] \
   [--num-queries-per-prompt=5]
```


### Prompt Injection Results

Based on judge LLM, the prompt injection will be either successful or
unsuccessful

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

In the same output file, the results are futher bucketized in different
dimensions, e.g., the injection techniques, and whether the prompt is a direct
prompt injection or an indrect prompt injection, etc. The result of each bucket
follows the same structure.

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
    },
    "stat_per_model_per_speaking_language": {
        ..
    }
}
```

## Results:

Once the benchmarks have run, the evaluations of each model across each language
will be available under the `stat_path`:

### Visual Prompt Injection Results

Based on the evaluation of the judge LLM, the output of visual prompt injection
test cases will be judged as either a successful or unsuccessful injection.

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

In the same output file, the results are further bucketized in different
dimensions:

1.  Injection techniques
2.  Risk category (security-violating vs. logic-violating)
3.  Injection type (direct vs. indirect)

Please refer to the CSE3 paper for a comprehensive definition of these terms.

The result of each bucket follows the same structure.

```
{
    "stat_per_model_per_injection_technique": {
        "model_name": {
            "embedded_text_instructions": {
                "injection_successful_count": ...,
                "injection_unsuccessful_count": ...,
                "total_count": ...,
                "injection_successful_percentage": ...,
                "injection_unsuccessful_percentage": ...
            },
            ...
        }
    },
    "stat_per_model_per_injection_type": {
        ...
    },
    "stat_per_model_per_risk_category": {
        ...
    }
}
```
