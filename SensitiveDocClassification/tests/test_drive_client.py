# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for the Google Drive client module
"""

import unittest
from unittest.mock import call, MagicMock, patch

from classifyit.google_drive_client.drive_client import DriveClient
from classifyit.structs import GoogleDriveLabelSettings


REQUEST_FIELDS = (
    "nextPageToken, files(id, name, mimeType, size, "
    + "createdTime, modifiedTime, md5Checksum, "
    + "fileExtension, labelInfo)"
)


class TestDriveClient(unittest.TestCase):
    """Test suite for the DriveClient class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.drive_owner = "test@example.com"
        self.service_account_file = "/path/to/service_account.json"

        self.mock_delegated_service = MagicMock()

        patcher = patch.object(DriveClient, "get_delegated_service")
        self.mock_get_delegated_service = patcher.start()
        self.mock_get_delegated_service.return_value = self.mock_delegated_service
        self.addCleanup(patcher.stop)

        self.drive_client = DriveClient(self.drive_owner, self.service_account_file)

    def test_list_files_my_drive(self):
        """Test listing files from My Drive"""
        mock_files = [
            {
                "id": "file1",
                "name": "Test File 1",
                "mimeType": "application/pdf",
                "size": "1000",
                "createdTime": "2023-01-01T00:00:00.000Z",
                "modifiedTime": "2023-01-02T00:00:00.000Z",
                "md5Checksum": "abcdef123456",
                "fileExtension": "pdf",
                "labelInfo": {"labels": [{"labelId": "label1"}]},
            }
        ]
        mock_response = {"files": mock_files}

        mock_files_list = MagicMock()
        mock_files_list.list.return_value.execute.return_value = mock_response
        self.mock_delegated_service.files.return_value = mock_files_list

        result = self.drive_client.list_files("my_drive", "label1")

        # Verify the correct API call was made
        mock_files_list.list.assert_called_once_with(
            spaces="drive",
            fields=REQUEST_FIELDS,
            includeLabels="label1",
            pageToken=None,
        )

        # Verify the result was processed correctly
        expected_result = [
            {
                "id": "file1",
                "name": "Test File 1",
                "mimeType": "application/pdf",
                "size": "1000",
                "createdTime": "2023-01-01T00:00:00.000Z",
                "modifiedTime": "2023-01-02T00:00:00.000Z",
                "md5Checksum": "abcdef123456",
                "fileExtension": "pdf",
                "labelInfo": {"labels": [{"labelId": "label1"}]},
            }
        ]
        self.assertEqual(result, expected_result)

    def test_list_files_shared_drive(self):
        """Test listing files from a Shared Drive"""
        mock_files = [
            {
                "id": "file2",
                "name": "Test File 2",
                "mimeType": "application/pdf",
                "size": "2000",
                "createdTime": "2023-02-01T00:00:00.000Z",
                "modifiedTime": "2023-02-02T00:00:00.000Z",
                "md5Checksum": "abcdef789012",
                "fileExtension": "pdf",
                "labelInfo": {"labels": [{"labelId": "label1"}]},
            }
        ]
        mock_response = {"files": mock_files}

        mock_files_list = MagicMock()
        mock_files_list.list.return_value.execute.return_value = mock_response
        self.mock_delegated_service.files.return_value = mock_files_list

        shared_drive_id = "shared_drive_id_1"
        result = self.drive_client.list_files("shared_drive", "label1", shared_drive_id)

        # Verify the correct API call was made
        mock_files_list.list.assert_called_once_with(
            driveId=shared_drive_id,
            spaces="drive",
            corpora="drive",
            fields=REQUEST_FIELDS,
            includeLabels="label1",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=None,
        )

        # Verify the result was processed correctly
        expected_result = [
            {
                "id": "file2",
                "name": "Test File 2",
                "mimeType": "application/pdf",
                "size": "2000",
                "createdTime": "2023-02-01T00:00:00.000Z",
                "modifiedTime": "2023-02-02T00:00:00.000Z",
                "md5Checksum": "abcdef789012",
                "fileExtension": "pdf",
                "labelInfo": {"labels": [{"labelId": "label1"}]},
            }
        ]
        self.assertEqual(result, expected_result)

    def test_list_files_pagination(self):
        """Test pagination when listing files"""
        # Setup mock responses for first and second pages
        mock_files_page1 = [
            {"id": "file1", "name": "Test File 1", "mimeType": "application/pdf"}
        ]
        mock_files_page2 = [
            {"id": "file2", "name": "Test File 2", "mimeType": "application/pdf"}
        ]

        # First response includes nextPageToken
        mock_response1 = {"files": mock_files_page1, "nextPageToken": "page1_token"}
        # Second response has no nextPageToken
        mock_response2 = {"files": mock_files_page2}

        # Setup mock for files().list() that returns different responses for different calls
        mock_files_list = MagicMock()
        mock_list = MagicMock()
        mock_files_list.list.return_value = mock_list

        # Configure execute() to return different values on subsequent calls
        mock_list.execute.side_effect = [mock_response1, mock_response2]

        self.mock_delegated_service.files.return_value = mock_files_list

        result = self.drive_client.list_files("my_drive", "label1")

        # Verify list was called twice with the correct parameters
        expected_calls = [
            call(
                spaces="drive",
                fields=REQUEST_FIELDS,
                includeLabels="label1",
                pageToken=None,
            ),
            call(
                spaces="drive",
                fields=REQUEST_FIELDS,
                includeLabels="label1",
                pageToken="page1_token",
            ),
        ]
        self.assertEqual(mock_files_list.list.call_args_list, expected_calls)

        # Verify the result combined both pages
        expected_result = [
            {
                "id": "file1",
                "name": "Test File 1",
                "mimeType": "application/pdf",
                "size": "0",
                "createdTime": "",
                "modifiedTime": "",
                "md5Checksum": "",
                "fileExtension": "",
                "labelInfo": {},
            },
            {
                "id": "file2",
                "name": "Test File 2",
                "mimeType": "application/pdf",
                "size": "0",
                "createdTime": "",
                "modifiedTime": "",
                "md5Checksum": "",
                "fileExtension": "",
                "labelInfo": {},
            },
        ]
        self.assertEqual(result, expected_result)

    def test_list_files_shared_drive_missing_id(self):
        """Test that an error is raised when shared_drive_id is missing for shared drive"""
        with self.assertRaises(ValueError) as context:
            self.drive_client.list_files("shared_drive", "label1")

        self.assertIn("Shared drive ID is required", str(context.exception))

    def test_list_files_error(self):
        """Test handling errors when listing files"""
        # Setup mock to raise an exception
        mock_files_list = MagicMock()
        mock_files_list.list.return_value.execute.side_effect = Exception("API error")
        self.mock_delegated_service.files.return_value = mock_files_list

        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            self.drive_client.list_files("my_drive", "label1")

        self.assertIn("API error", str(context.exception))

    def test_export_file(self):
        """Test exporting a Google Workspace file"""
        mock_response = b"Exported file content"

        # Set up the mock service to return our response
        mock_files_export = MagicMock()
        mock_files_export.export.return_value.execute.return_value = mock_response
        self.mock_delegated_service.files.return_value = mock_files_export

        file_id = "test_file_id"
        mime_type = "text/plain"
        result = self.drive_client.export_file(file_id, mime_type)

        mock_files_export.export.assert_called_once_with(
            fileId=file_id, mimeType=mime_type
        )

        self.assertEqual(result, "Exported file content")

    def test_export_file_error(self):
        """Test handling errors when exporting a file"""
        mock_files_export = MagicMock()
        mock_files_export.export.return_value.execute.side_effect = Exception(
            "Export error"
        )
        self.mock_delegated_service.files.return_value = mock_files_export

        with self.assertRaises(Exception) as context:
            self.drive_client.export_file("test_file_id", "text/plain")

        self.assertIn("Export error", str(context.exception))

    def test_get_file(self):
        """Test getting a file's content"""
        mock_response = b"File content"

        mock_files_get = MagicMock()
        mock_files_get.get_media.return_value.execute.return_value = mock_response
        self.mock_delegated_service.files.return_value = mock_files_get

        file_id = "test_file_id"
        result = self.drive_client.get_file(file_id)

        mock_files_get.get_media.assert_called_once_with(
            fileId=file_id, supportsAllDrives=True
        )

        self.assertEqual(result, mock_response)

    def test_get_file_error(self):
        """Test handling errors when getting a file"""
        mock_files_get = MagicMock()
        mock_files_get.get_media.return_value.execute.side_effect = Exception(
            "Get file error"
        )
        self.mock_delegated_service.files.return_value = mock_files_get

        with self.assertRaises(Exception) as context:
            self.drive_client.get_file("test_file_id")

        self.assertIn("Get file error", str(context.exception))

    @patch("classifyit.google_drive_client.drive_client.extract_content")
    def test_extract_file_content(self, mock_extract_content):
        """Test extracting content from a file"""
        mock_extract_content.return_value = "Extracted content"

        file_id = "test_file_id"
        mime_type = "application/pdf"
        file_size = 1000
        result = self.drive_client.extract_file_content(file_id, mime_type, file_size)

        mock_extract_content.assert_called_once_with(
            self.drive_client, file_id, mime_type, file_size
        )

        # Verify result was returned correctly
        self.assertEqual(result, "Extracted content")

    @patch("classifyit.google_drive_client.drive_client.extract_content")
    def test_extract_file_content_error(self, mock_extract_content):
        """Test handling errors when extracting content"""
        mock_extract_content.side_effect = Exception("Extraction error")

        with self.assertRaises(Exception) as context:
            self.drive_client.extract_file_content(
                "test_file_id", "application/pdf", 1000
            )

        self.assertIn("Extraction error", str(context.exception))

    def test_apply_label(self):
        """Test applying a label to a file"""
        mock_files_modify = MagicMock()
        mock_files_modify.modifyLabels.return_value.execute.return_value = {}
        self.mock_delegated_service.files.return_value = mock_files_modify

        file_id = "test_file_id"
        label_id = "test_label_id"
        field_id = "test_field_id"
        selection_ids = ["selection_id_1"]
        result = self.drive_client.apply_label(
            file_id, label_id, field_id, selection_ids
        )

        expected_body = {
            "labelModifications": [
                {
                    "labelId": label_id,
                    "fieldModifications": [
                        {
                            "fieldId": field_id,
                            "setSelectionValues": [selection_ids],
                        }
                    ],
                }
            ]
        }
        mock_files_modify.modifyLabels.assert_called_once_with(
            fileId=file_id, body=expected_body
        )

        self.assertTrue(result)

    def test_apply_label_error(self):
        """Test handling errors when applying a label"""
        mock_files_modify = MagicMock()
        mock_files_modify.modifyLabels.return_value.execute.side_effect = Exception(
            "Label error"
        )
        self.mock_delegated_service.files.return_value = mock_files_modify

        with self.assertRaises(Exception) as context:
            self.drive_client.apply_label(
                "test_file_id", "test_label_id", "test_field_id", ["selection_id_1"]
            )

        self.assertIn("Label error", str(context.exception))

    def test_apply_label_from_classification(self):
        """Test applying a label from classification output"""
        mock_db = MagicMock()

        mock_classification = MagicMock()
        mock_classification.category_matched = "category1"
        mock_classification.brief_explanation = "This is a test explanation"

        label_settings = GoogleDriveLabelSettings(
            label_id="test_label_id",
            field_id="test_field_id",
            selection_ids={
                "category1": ["selection_id_1"],
                "category2": ["selection_id_2"],
            },
        )

        self.drive_client.apply_label = MagicMock(return_value=True)

        file_id = "test_file_id"
        result = self.drive_client.apply_label_from_classification(
            file_id, mock_classification, label_settings, mock_db
        )

        self.drive_client.apply_label.assert_called_once_with(
            file_id=file_id,
            label_id=label_settings.label_id,
            field_id=label_settings.field_id,
            selection_ids=["selection_id_1"],
        )

        mock_db.update_classification.assert_called_once_with(
            file_id=file_id,
            label_id=label_settings.label_id,
            selection_id=["selection_id_1"],
            brief_explanation="This is a test explanation",
        )

        expected_result = {"action_taken": "Applied label", "error": ""}
        self.assertEqual(result, expected_result)

    def test_apply_label_from_classification_no_matching_selection(self):
        """Test applying a label when no matching selection ID is found"""
        mock_db = MagicMock()

        mock_classification = MagicMock()
        mock_classification.category_matched = "unknown_category"

        label_settings = GoogleDriveLabelSettings(
            label_id="test_label_id",
            field_id="test_field_id",
            selection_ids={
                "category1": ["selection_id_1"],
                "category2": ["selection_id_2"],
            },
        )

        file_id = "test_file_id"
        result = self.drive_client.apply_label_from_classification(
            file_id, mock_classification, label_settings, mock_db
        )

        # Verify apply_label was not called
        self.assertTrue(not hasattr(self.drive_client, "apply_label.assert_not_called"))

        # Verify database update was not called
        mock_db.update_classification.assert_not_called()

        # Verify result contains error
        expected_result = {
            "action_taken": "",
            "error": "No matching Google Drive selection id found",
        }
        self.assertEqual(result, expected_result)

    def test_apply_label_from_classification_db_error(self):
        """Test handling database errors when applying a label from classification"""
        mock_db = MagicMock()
        mock_db.update_classification.side_effect = Exception("Database error")

        mock_classification = MagicMock()
        mock_classification.category_matched = "category1"
        mock_classification.brief_explanation = "This is a test explanation"

        # Setup label settings
        label_settings = GoogleDriveLabelSettings(
            label_id="test_label_id",
            field_id="test_field_id",
            selection_ids={
                "category1": ["selection_id_1"],
                "category2": ["selection_id_2"],
            },
        )

        # Mock apply_label method
        self.drive_client.apply_label = MagicMock(return_value=True)

        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            self.drive_client.apply_label_from_classification(
                "test_file_id", mock_classification, label_settings, mock_db
            )

        self.assertIn("Database error", str(context.exception))


if __name__ == "__main__":
    unittest.main()
