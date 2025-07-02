# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Setup script for the ClassifyIt package
"""

from setuptools import find_packages, setup

setup(
    name="classifyit",
    version="0.1.1",
    description="A tool for classifying content using LLMs",
    author="Meta Platforms, Inc.",
    author_email="opensource@meta.com",
    packages=find_packages(),
    install_requires=[
        "llama-stack-client",
        "pyyaml",
        "tika",
        "peewee",
        "google-api-python-client",
    ],
    py_modules=["scanner"],
    entry_points={
        "console_scripts": [
            "classifyit=scanner:main",
        ],
    },
    python_requires=">=3.8",
)
