#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


set -e
set -x

# Switch to repository root
SCRIPTS_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPTS_DIRECTORY/../.."

# Set up Python
VENV_ROOT=$( mktemp -d )
function cleanup {
  rm -rf "$VENV_ROOT"
}
trap cleanup EXIT

python3 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
pip3 install -r CybersecurityBenchmarks/requirements.txt

# Command help runs properly
python3 -m CybersecurityBenchmarks.benchmark.run --help

# Tests pass
cd CybersecurityBenchmarks
python3 -m unittest discover benchmark.tests
