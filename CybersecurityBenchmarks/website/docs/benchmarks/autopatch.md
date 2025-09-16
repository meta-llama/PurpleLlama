# AutoPatch

## Running the AutoPatch Benchmark

### Hardware Requirements

The AutoPatch benchmark is compute-intensive and requires a significant amount of storage space. While the benchmark can run on machines with any number of CPU cores, we recommend using machines with a large number of cores to maximize parallelism and minimize evaluation runtime. The benchmark has been tested on an AWS EC2 instance type c7a.48xlarge running Ubuntu 24.04. For optimal performance, consider the following setup:

* A large number of CPUs (e.g., >= 80)
* Ample storage space (e.g., >= 8TB)

Please note that the benchmark does not currently support MacBooks running on Apple Silicon.

### Setup

In addition to the standard Python dependencies, the AutoPatch benchmark requires you to install Podman on your machine. Podman is used to run the benchmark in containers. If you're using an Ubuntu machine, you can install Podman with the following command:
```
sudo apt install podman
```

### Running the benchmark

The AutoPatch benchmark evaluates the ability of AI patch generation agents to resolve C/C++ vulnerabilities found by fuzzing. Running the benchmark requires creating two Linux containers for each test case, which demands significant disk space. To accommodate different needs, we provide three datasets:

 * [autopatch_bench.json](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/datasets/autopatch/autopatch_bench.json): The full dataset with 142 test cases. To run this dataset, we recommend having about 3TB of storage.
 * [autopatch_lite.json](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/datasets/autopatch/autopatch_lite.json): The lite dataset with 120 test cases. All crash samples in this dataset were fixable by one hunk of code change. To run this dataset, we recommend having about 2TB of storage.
 * [autopatch_samples.json](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/datasets/autopatch/autopatch_samples.json):  The sample dataset with 20 test cases. To run this dataset, we recommend having about 500GB of storage.

If you're unsure where to begin, we recommend starting with [autopatch_samples.json](https://github.com/facebookresearch/PurpleLlama/blob/main/CybersecurityBenchmarks/datasets/autopatch/autopatch_samples.json).

**Important Note**: The initial run of the AutoPatch benchmark involves building over 200 container images, which may take a few hours. This is a one-time setup, as these images will be cached locally for future runs. Once the containers are built, the evaluation process itself can take several hours, potentially up to 8 hours. We recommend planning for the initial run to complete within 24 hours. Subsequent runs will be faster, though they will still require a few hours to complete.

To run:
```
python3 -m CybersecurityBenchmarks.benchmark.run \
  --benchmark "autopatch" \
   --llm-under-test=<SPECIFICATION_1> --llm-under-test=<SPECIFICATION_2> ... \
  --prompt-path "${DATASETS}/autopatch/autopatch_samples.json" \
  --response-path "${DATASETS}/autopatch/output/generated_patches.json" \
  --stat-path "${DATASETS}/autopatch/output/autopatch_stat.json" \
  --judge-response-path="${DATASETS}/autopatch/output/autopatch_results.json" \
  [--run-llm-in-parallel]
```

After AutoPatch-Bench finishes running, you can find the following outputs:
* **Generated Patches and Binaries**: Located in the JSON file specified by the `--response-path`` argument.
* **Summarized Evaluation Statistics**: Available in the JSON file specified by the `--stat-path`` argument.
* **Detailed Evaluation Information**: Found in the JSON file specified by the `--judge-response-path`` argument.

## Results:

Once the benchmarks have run, the evaluations of each model across each language
will be available under the `stat_path`:

### AutoPatch Results

Output stats will appear in a json according to the following structure:

```
 "model_name":  {
        "num_passed_qa_checks": ...,
        "num_generated_patched_function_name": ...,
        "num_passed_fuzzing": ...,
        "num_fuzzing_decode_errors": ...,
        "num_passed_fuzzing_and_differential_debugging": ...,
        "num_differential_debugging_errors": ...,
        "shr_patches_generated": ...,
        "shr_passing_fuzzing": ...,
        "shr_correct_patches": ...,
    }
```

where:
* num_passed_qa_checks contains the number of test cases that passed the QA checks
* num_generated_patched_function_name contains the number of test cases that generated a patched function name
* num_passed_fuzzing contains the number of test cases that generated a patched function name and also passed an extensive period of fuzzing tests
* num_fuzzing_decode_errors contains the number of test cases that generated a patched function name but failed the fuzzing tests with decode errors
* num_passed_fuzzing_and_differential_debugging contains the number of test cases that generated a patched function name, passed an extensive period of fuzzing tests, and also passed the differential debugging tests
* num_differential_debugging_errors contains the number of test cases that generated a patched function name, passed an extensive period of fuzzing tests, but failed the differential debugging tests
* shr_patches_generated contains the share of test cases that generated a patched function name
* shr_passing_fuzzing contains the share of test cases that generated a patched function name and also passed an extensive period of fuzzing tests
* shr_correct_patches contains the share of test cases that generated a patched function name, passed an extensive period of fuzzing tests, and also passed the differential debugging tests
