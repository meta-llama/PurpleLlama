# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract content from Google Workspace files.
"""

import json
import logging
from typing import Any, Optional

from classifyit.google_drive_client.mime_types import EXPORT_MIME_TYPES
from tika import config, parser

logger = logging.getLogger("classifyit.google_drive_client.content_extractor")

mimeTypesJson = json.loads(config.getMimeTypes())
TIKA_MIME_TYPES = list(mimeTypesJson.keys())


class DownloadError(Exception):
    """Base exception for download errors."""

    pass  # pylint: disable=W0107


class ContentExtractionError(Exception):
    """Base exception for content extraction errors."""

    pass  # pylint: disable=W0107


class UnsupportedMimeTypeError(ContentExtractionError):
    """Exception raised when the file mime type is not supported for extraction."""

    def __init__(self, mime_type: str):
        self.mime_type = mime_type
        super().__init__(f"Unsupported MIME type: {mime_type}")


class EmptyContentError(ContentExtractionError):
    """Exception raised when no content could be extracted from the file."""

    def __init__(self, file_id: str):
        self.file_id = file_id
        super().__init__(f"No content extracted from file {file_id}")


def extract_content(
    drive_client: Any, file_id: str, mime_type: str, file_size: int
) -> Optional[str]:
    """
    Extract plaintext content (up to 1MB)from a Google Workspace file.

    Args:
        drive_client: The Drive client instance
        file_id: The ID of the file to extract content from
        mime_type: The MIME type of the file
        file_size: The size of the file

    Returns:
        The extracted content as a string, or None if extraction failed
    """
    logger.debug("Extracting content from file %s, mime type: %s", file_id, mime_type)

    # Check if this is a Google Workspace file type; if so extract it
    # and return the plaintext content
    if mime_type in EXPORT_MIME_TYPES:
        # Get the appropriate export MIME type
        export_mime_type = EXPORT_MIME_TYPES[mime_type]

        try:
            # Export the file content
            content = drive_client.export_file(file_id, export_mime_type)

            if not content:
                logger.warning("No content extracted from file %s", file_id)
                raise EmptyContentError(file_id)

            # Limit content size to 1 million characters
            content = content[:1_000_000]
            if len(content) == 1_000_000:
                logger.debug("Content from file %s truncated to 1MB", file_id)

            return content
        except Exception as e:  # pylint: disable=W0718
            logger.error("Failed to extract content from file %s: %s", file_id, e)
            raise ContentExtractionError(  # pylint: disable=W0707
                f"Failed to extract content from file {file_id}: {e}"
            )
    # If the file is a supported Tika MIME type, download the file content
    # and use Tika to extract the content
    elif mime_type in TIKA_MIME_TYPES:
        # Download the file content using the Drive client
        try:
            if file_size > 10_000_000:  # 10MB
                logger.error("File %s is too large to download", file_id)
                raise DownloadError(f"File {file_id} is too large to download")

            content = drive_client.get_file(file_id)

        except Exception as e:
            logger.error("Failed to download file %s: %s", file_id, e)
            raise DownloadError(  # pylint: disable=W0707
                f"Failed to download file {file_id}: {e}"
            )

        # Use Tika to extract content and return plaintext (up to 1MB)
        try:
            parsed = parser.from_buffer(content)
            plaintext = parsed["content"][:1_000_000]
            return plaintext

        except Exception as e:
            logger.error("Failed to parse file %s: %s", file_id, e)
            raise ContentExtractionError(  # pylint: disable=W0707
                f"Failed to parse file {file_id}: {e}"
            )

    # If the file is not a supported MIME type, raise an UnsupportedMimeTypeError error
    else:
        logger.info("Unsupported MIME type: %s", mime_type)
        raise UnsupportedMimeTypeError(mime_type)  # pylint: disable=W0707
