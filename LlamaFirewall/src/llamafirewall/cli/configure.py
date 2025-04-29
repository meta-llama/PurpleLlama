# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Configuration module for LlamaFirewall.

This module provides a simple command-line interface for configuring LlamaFirewall
with the necessary tokens and API keys.
"""

import os

import typer
from huggingface_hub import HfFolder, login

# Default model name for PromptGuard
DEFAULT_MODEL_NAME = "meta-llama/Llama-Prompt-Guard-2-86M"


def set_key_in_environment(key_name: str, key_value: str) -> bool:
    """
    Set an environment variable in the current process and provide a command to export it.

    Args:
        key_name: The name of the environment variable.
        key_value: The value of the environment variable.

    Returns:
        bool: True if the key was set successfully, False otherwise.
    """
    try:
        # Set the key in the current process environment
        os.environ[key_name] = key_value

        # Print the export command that can be copied and pasted
        print(
            "\n⚠️  IMPORTANT: The key is set in this Python process but NOT in your shell."
        )
        print("   To make it available in your shell, copy and paste this command:")
        print(f'\n   export {key_name}="{key_value}"')
        print(f"\n   After running this command, verify with: echo ${{{key_name}}}")

        return True
    except Exception as e:
        print(f"Error setting {key_name}: {e}")
        return False


def check_model_availability(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """
    Check if the model is available locally.

    Args:
        model_name: The name of the model to check.

    Returns:
        bool: True if the model is available locally, False otherwise.
    """
    # Set the local path for the model if not already set
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

    model_path = os.path.expanduser(
        os.path.join(os.environ["HF_HOME"], model_name.replace("/", "--"))
    )

    return os.path.exists(model_path)


def download_model(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """
    Download a model from Hugging Face.

    Args:
        model_name: The name of the model to download.

    Returns:
        bool: True if the model was downloaded successfully, False otherwise.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Set the local path for the model
        if not os.environ.get("HF_HOME"):
            os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

        model_path = os.path.expanduser(
            os.path.join(os.environ["HF_HOME"], model_name.replace("/", "--"))
        )

        # Check if the user is logged in to Hugging Face
        if not HfFolder.get_token():
            print("You need to log in to Hugging Face to download the model.")
            login()

        # Download and save the model and tokenizer
        print(f"Downloading model {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save the model and tokenizer locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print(f"Model downloaded and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def check_together_key() -> bool:
    """
    Check if the Together API key is set in the environment.

    Returns:
        bool: True if the key is set, False otherwise.
    """
    # Check both TOGETHER_API_KEY and TOGETHER_API_TOKEN (some users might use either)
    key = os.environ.get("TOGETHER_API_KEY")
    token = os.environ.get("TOGETHER_API_TOKEN")

    # If either is set, consider it valid
    return bool(key or token)


def check_tokenizers_parallelism() -> bool:
    """
    Check if the TOKENIZERS_PARALLELISM environment variable is set to true.

    Returns:
        bool: True if the variable is set to true, False otherwise.
    """
    parallelism = os.environ.get("TOKENIZERS_PARALLELISM", "").lower()
    return parallelism == "true"


def run_configure() -> None:
    """
    Configure LlamaFirewall with the necessary tokens and API keys.

    This function checks for the availability of models and API keys,
    and helps the user set them up if they are not available.
    """
    print("=== LlamaFirewall Configuration ===\n")

    # Check for model availability
    print("Checking for model availability...")
    model_available = check_model_availability()

    if model_available:
        print(f"✅ Model {DEFAULT_MODEL_NAME} is available locally.")
    else:
        print(f"❌ Model {DEFAULT_MODEL_NAME} is not available locally.")
        print("NOTE: The Prompt Guard Scanner requires this model to function.")
        download = typer.confirm(
            "Would you like to download it now? (You can skip if you don't plan to use Prompt Guard)",
            default=True,
        )

        if download:
            success = download_model()
            if success:
                print("✅ Model downloaded successfully.")
            else:
                print("❌ Failed to download model.")
                print("You can try again later by running 'llamafirewall configure'.")
        else:
            print(
                "Skipping model download. You can download it later by running 'llamafirewall configure'."
            )

    # Check for TOKENIZERS_PARALLELISM environment variable
    print("\nChecking for TOKENIZERS_PARALLELISM environment variable...")
    tokenizers_parallelism_set = check_tokenizers_parallelism()

    if tokenizers_parallelism_set:
        print("✅ TOKENIZERS_PARALLELISM is set to true in the environment.")
    else:
        print("❌ TOKENIZERS_PARALLELISM is not set to true in the environment.")
        print(
            "NOTE: Setting this variable is preferred if you plan to use Prompt Guard in parallel."
        )
        print("      This is not mandatory but can improve tokenization performance.")
        print("\nTo set the variable, run this command in your terminal:")
        print('export TOKENIZERS_PARALLELISM="true"')

    # Check for Together API key
    print("\nChecking for Together API key...")
    together_key_set = check_together_key()

    if together_key_set:
        print("✅ Together API key is detected in the environment.")
    else:
        print("❌ Together API key is not set in the environment.")
        print("NOTE: The Alignment Check Scanner requires this key to function.")
        print("\nTo set the key, run this command in your terminal:")
        print('export TOGETHER_API_KEY="your_together_api_key_here"')
        print("\nYou can get a Together API key from https://together.ai")

    print("\n=== Configuration Summary ===")
    print(f"Model: {'Available' if check_model_availability() else 'Not Available'}")
    print(f"Together API Key: {'Set' if check_together_key() else 'Not Set'}")
    print(
        f"TOKENIZERS_PARALLELISM: {'Set' if check_tokenizers_parallelism() else 'Not Set'}"
    )

    print("\nConfiguration complete!")
    print(
        "You can run 'llamafirewall configure' again at any time to update your configuration."
    )


# For backward compatibility
configure = run_configure

if __name__ == "__main__":
    run_configure()
