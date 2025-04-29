# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import enum

from typing import Dict


class Language(enum.Enum):
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    HACK = "hack"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    KOTLIN = "kotlin"
    OBJECTIVE_C = "objective_c"
    OBJECTIVE_CPP = "objective_cpp"
    PHP = "php"
    PYTHON = "python"
    RUBY = "ruby"
    RUST = "rust"
    SWIFT = "swift"
    XML = "xml"
    LANGUAGE_AGNOSTIC = "language_agnostic"

    def __str__(self) -> str:
        return self.name.lower()


EXTENSION_MAP: Dict[Language, str] = {
    Language.C: ".c",
    Language.CPP: ".cpp",
    Language.CSHARP: ".cs",
    Language.HACK: ".php",
    Language.JAVA: ".java",
    Language.JAVASCRIPT: ".js",
    Language.KOTLIN: ".kt",
    Language.OBJECTIVE_C: ".m",
    Language.OBJECTIVE_CPP: ".mm",
    Language.PHP: ".php",
    Language.PYTHON: ".py",
    Language.RUBY: ".rb",
    Language.RUST: ".rs",
    Language.SWIFT: ".swift",
    Language.XML: ".xml",
}


def get_file_extension(language: Language) -> str:
    if language in EXTENSION_MAP:
        return EXTENSION_MAP[language]
    else:
        raise ValueError("Unknown language")


def get_supported_languages() -> list[Language]:
    supported_languages = [
        Language.C,
        Language.CPP,
        Language.CSHARP,
        Language.JAVA,
        Language.JAVASCRIPT,
        Language.PHP,
        Language.PYTHON,
        Language.RUST,
    ]
    return supported_languages
