# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ClassifyIt - A Google Drive document classification tool.
"""

import logging

from classifyit.classifier import ClassifyIt
from classifyit.google_drive_client.drive_client import DriveClient

logger = logging.getLogger("classifyit")
logger.propagate = False

__version__ = "0.1.1"
__all__ = ["ClassifyIt", "DriveClient"]
