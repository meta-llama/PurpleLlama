# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from llamafirewall.scanners.promptguard_utils import PromptGuard


def main() -> None:
    prompt_guard = PromptGuard()

    score = prompt_guard.get_jailbreak_score("test")
    print(f"Test score from PG for jailbreak: {score}")


if __name__ == "__main__":
    main()
