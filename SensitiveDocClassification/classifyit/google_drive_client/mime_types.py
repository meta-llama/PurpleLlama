# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MIME type constants for Google Drive file operations.
"""

# Google Workspace MIME types
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDES_MIME_TYPE = "application/vnd.google-apps.presentation"

# Export MIME types
EXPORT_MIME_TYPES = {
    GOOGLE_DOC_MIME_TYPE: "text/plain",
    GOOGLE_SHEET_MIME_TYPE: "text/csv",
    GOOGLE_SLIDES_MIME_TYPE: "text/plain",
}
