# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for the content extractor module
"""

import unittest
from unittest.mock import MagicMock, patch

from classifyit.google_drive_client.content_extractor import (
    ContentExtractionError,
    DownloadError,
    extract_content,
    UnsupportedMimeTypeError,
)
from classifyit.google_drive_client.mime_types import (
    EXPORT_MIME_TYPES,
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    GOOGLE_SLIDES_MIME_TYPE,
)


class TestContentExtractor(unittest.TestCase):
    """Test suite for the content extractor module"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.drive_client = MagicMock()
        self.file_id = "test_file_id"

    def test_extract_content_google_doc(self):
        """Test extracting content from a Google Doc"""
        # Mock the export_file method to return content
        self.drive_client.export_file.return_value = "Google Doc content"

        content = extract_content(
            self.drive_client, self.file_id, GOOGLE_DOC_MIME_TYPE, 1000
        )

        # Verify the correct method was called with right parameters
        self.drive_client.export_file.assert_called_once_with(
            self.file_id, EXPORT_MIME_TYPES[GOOGLE_DOC_MIME_TYPE]
        )

        # Check that the content was returned correctly
        self.assertEqual(content, "Google Doc content")

    def test_extract_content_google_sheet(self):
        """Test extracting content from a Google Sheet"""
        # Mock the export_file method to return content
        self.drive_client.export_file.return_value = "col1,col2\nval1,val2"

        content = extract_content(
            self.drive_client,
            self.file_id,
            GOOGLE_SHEET_MIME_TYPE,
            1000,  # Small file size
        )

        # Verify the correct method was called with right parameters
        self.drive_client.export_file.assert_called_once_with(
            self.file_id, EXPORT_MIME_TYPES[GOOGLE_SHEET_MIME_TYPE]
        )

        # Check that the content was returned correctly
        self.assertEqual(content, "col1,col2\nval1,val2")

    def test_extract_content_google_slides(self):
        """Test extracting content from Google Slides"""
        # Mock the export_file method to return content
        self.drive_client.export_file.return_value = "Slide content"

        content = extract_content(
            self.drive_client,
            self.file_id,
            GOOGLE_SLIDES_MIME_TYPE,
            1000,  # Small file size
        )

        # Verify the correct method was called with right parameters
        self.drive_client.export_file.assert_called_once_with(
            self.file_id, EXPORT_MIME_TYPES[GOOGLE_SLIDES_MIME_TYPE]
        )

        # Check that the content was returned correctly
        self.assertEqual(content, "Slide content")

    def test_extract_content_truncates_large_content(self):
        """Test that content is truncated if it's too large"""
        # Create content that exceeds the limit
        large_content = "x" * 2_000_000  # 2MB of content

        # Mock the export_file method to return large content
        self.drive_client.export_file.return_value = large_content

        content = extract_content(
            self.drive_client,
            self.file_id,
            GOOGLE_DOC_MIME_TYPE,
            2_000_000,  # File size
        )

        # Check that the content was truncated
        self.assertEqual(len(content), 1_000_000)
        self.assertEqual(content, "x" * 1_000_000)

    def test_extract_content_export_error(self):
        """Test handling export error"""
        # Mock the export_file method to raise an exception
        self.drive_client.export_file.side_effect = Exception("Export failed")

        # Check that ContentExtractionError is raised
        with self.assertRaises(ContentExtractionError) as context:
            extract_content(
                self.drive_client,
                self.file_id,
                GOOGLE_DOC_MIME_TYPE,
                1000,  # File size
            )

        # Verify error message contains the original error
        self.assertIn("Export failed", str(context.exception))

    @patch("classifyit.google_drive_client.content_extractor.parser")
    def test_extract_content_pdf_file(self, mock_parser):
        """Test extracting content from a PDF file"""
        # Mock the get_file method to return binary content
        pdf_content = b"%PDF-1.5\nThis is a fake PDF"
        self.drive_client.get_file.return_value = pdf_content

        # Mock Tika parser to return parsed content
        mock_parser.from_buffer.return_value = {"content": "Parsed PDF content"}

        content = extract_content(
            self.drive_client, self.file_id, "application/pdf", 1000
        )

        # Verify the drive client was called to get the file
        self.drive_client.get_file.assert_called_once_with(self.file_id)

        # Verify Tika parser was called with the file content
        mock_parser.from_buffer.assert_called_once_with(pdf_content)

        # Check that the parsed content was returned
        self.assertEqual(content, "Parsed PDF content")

    def test_extract_content_file_too_large(self):
        """Test that large files (>10MB) trigger a download error"""
        # Check that DownloadError is raised for files >10MB
        with self.assertRaises(DownloadError) as context:
            extract_content(
                self.drive_client, self.file_id, "application/pdf", 15_000_000
            )

        # Verify error message
        self.assertIn("too large to download", str(context.exception))

        # Verify get_file was not called
        self.drive_client.get_file.assert_not_called()

    def test_extract_content_download_error(self):
        """Test handling download error"""
        # Mock the get_file method to raise an exception
        self.drive_client.get_file.side_effect = Exception("Download failed")

        # Check that DownloadError is raised
        with self.assertRaises(DownloadError) as context:
            extract_content(self.drive_client, self.file_id, "application/pdf", 1000)

        # Verify error message contains the original error
        self.assertIn("Download failed", str(context.exception))

    @patch("classifyit.google_drive_client.content_extractor.parser")
    def test_extract_content_tika_parsing_error(self, mock_parser):
        """Test handling Tika parsing error"""
        # Mock the get_file method to return content
        self.drive_client.get_file.return_value = b"file content"

        # Mock Tika parser to raise an exception
        mock_parser.from_buffer.side_effect = Exception("Parsing failed")

        # Check that ContentExtractionError is raised
        with self.assertRaises(ContentExtractionError) as context:
            extract_content(self.drive_client, self.file_id, "application/pdf", 1000)

        # Verify error message contains the original error
        self.assertIn("Parsing failed", str(context.exception))

    def test_extract_content_unsupported_mime_type(self):
        """Test handling unsupported MIME types"""
        unsupported_mime_type = "unsupported/mimetype"

        # Check that UnsupportedMimeTypeError is raised
        with self.assertRaises(UnsupportedMimeTypeError) as context:
            extract_content(
                self.drive_client, self.file_id, unsupported_mime_type, 1000
            )

        # Verify error message
        self.assertEqual(
            str(context.exception), f"Unsupported MIME type: {unsupported_mime_type}"
        )

    @patch("classifyit.google_drive_client.content_extractor.parser")
    def test_extract_content_truncates_large_tika_content(self, mock_parser):
        """Test that Tika parsed content is truncated if it's too large"""
        # Mock the get_file method to return content
        self.drive_client.get_file.return_value = b"file content"

        # Create large content that exceeds the limit
        large_content = "x" * 2_000_000  # 2MB of content

        # Mock Tika parser to return large content
        mock_parser.from_buffer.return_value = {"content": large_content}

        content = extract_content(
            self.drive_client, self.file_id, "application/pdf", 1000
        )

        # Check that the content was truncated
        self.assertEqual(len(content), 1_000_000)
        self.assertEqual(content, "x" * 1_000_000)


if __name__ == "__main__":
    unittest.main()
