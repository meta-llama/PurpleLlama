# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Command-line interface for LlamaFirewall.
"""

import typer

app = typer.Typer(name="llamafirewall", help="LlamaFirewall CLI")


@app.callback()
def callback() -> None:
    """
    LlamaFirewall - A security guardrail for AI applications.

    This CLI helps you set up and use LlamaFirewall to secure your LLM applications.
    """
    pass


@app.command()
def configure() -> None:
    """
    Configure LlamaFirewall with necessary tokens and API keys.

    This command helps you set up LlamaFirewall by:
    1. Checking if required models are available locally
    2. Helping you download models if needed
    3. Setting up your OpenAI API key for LangChain integration
    4. Setting up your Tavily Search API key for examples
    """
    from llamafirewall.cli.configure import run_configure

    run_configure()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
