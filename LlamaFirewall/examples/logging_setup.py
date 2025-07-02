# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL: int = logging.INFO


def init_logger(name: str = __name__, verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else DEFAULT_LOG_LEVEL,
        format=LOG_FORMAT,
    )
    return logging.getLogger(name)
