# Community Contributions

This directory contains code contributed by the community to the CybersecurityBenchmarks project. Please note that our team is not accountable for the content of these files or responsible for thorough testing of the code. Users should perform their own assessment before implementation.

## Integration Guide

To integrate your own LLM implementation:

1. Implement your LLM class by extending the `LLM` abstract class from `llm_base.py`. Also update the `__init__.py` file to include your LLM class in the `__all__` list. See `ftgooglegenai.py` for a reference implementation.
2. Add any necessary dependencies required by your LLM class to requirements.txt.
3. Update the `create` function in `llm.py` to include your LLM class.

## Example: Use Fine-tuned Gemini Models

### Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r CybersecurityBenchmarks/benchmark/llms/community_contribution/requirements.txt
   ```

2. Configure authentication by populating the `google_adc_key.json` file located at:
   ```
   CybersecurityBenchmarks/benchmark/llms/community_contribution/google_adc_key.json
   ```
   with your Google Application Default Credentials. Note that this file is only used for the class `FTGOOGLEGENAI` and is not used for any other LLM classes.

3. Specify the LLM to test using the following format:
   ```
   FTGOOGLEGENAI::ft-gemini-2.0-flash::projects/PROJECT_ID/locations/TUNING_JOB_REGION/endpoints/ENDPOINT_ID/buckets/BUCKET_NAME
   ```

   Where:
   - `FTGOOGLEGENAI` is the provider name
   - `ft-gemini-2.0-flash` is the model name
   - The API key contains your project configuration in the format:
     `projects/PROJECT_ID/locations/TUNING_JOB_REGION/endpoints/ENDPOINT_ID/buckets/BUCKET_NAME`
