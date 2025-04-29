# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for the Database module
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from classifyit.database.database import Database, File, Label

from peewee import DatabaseError


class TestDatabase(unittest.TestCase):
    """Test suite for the Database class"""

    def setUp(self):
        """Set up a test database before each test"""
        self.db_path = (
            tempfile.NamedTemporaryFile().name  # pylint: disable=consider-using-with
        )
        self.db = Database(db_path=self.db_path)
        self.db.connect()
        # Ensure tables are created by binding the models
        with self.db.db.bind_ctx([File, Label]):
            self.db.db.create_tables([File, Label], safe=True)

    def tearDown(self):
        """Clean up after each test"""
        if not self.db.db.is_closed():
            self.db.close()
        os.remove(self.db_path)

    def test_connect_creates_tables(self):
        """Test that connect creates tables if they don't exist"""
        # Close and reconnect to ensure tables exist
        self.db.close()
        self.db.connect()

        # Check that the tables exist
        with self.db.db.bind_ctx([File, Label]):
            self.assertTrue(File.table_exists())
            self.assertTrue(Label.table_exists())

    def test_connect_handles_existing_tables(self):
        """Test that connect handles the case where tables already exist"""
        # Close the connection first
        self.db.close()
        # Tables should already exist from setUp, this should not raise an exception
        self.db.connect()

    @patch("classifyit.database.database.SqliteDatabase")
    def test_connect_error_handling(self, mock_sqlite):
        """Test error handling during connect"""
        # Setup mock to raise an exception
        mock_db = MagicMock()
        mock_db.connect.side_effect = DatabaseError("Test error")
        mock_sqlite.return_value = mock_db

        # Create a new database instance with the mock
        db = Database("test_path")

        # Verify that the exception is propagated
        with self.assertRaises(Exception):  # noqa: B017
            db.connect()

    def test_close(self):
        """Test that close properly closes the connection"""
        self.db.close()
        self.assertTrue(self.db.db.is_closed())

    def test_iso_to_unix(self):
        """Test conversion from ISO time to Unix timestamp"""
        # Test with a valid ISO time
        iso_time = "2023-01-01T12:00:00Z"
        expected = int(datetime.fromisoformat("2023-01-01T12:00:00+00:00").timestamp())
        self.assertEqual(self.db.iso_to_unix(iso_time), expected)

        # Test with None
        self.assertEqual(self.db.iso_to_unix(None), 0)

        # Test with invalid format
        self.assertEqual(self.db.iso_to_unix("invalid"), 0)

    def test_upsert_file_new(self):
        """Test upserting a new file"""
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }

        self.db.upsert_file(file_data)

        # Verify the file was inserted
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)
        self.assertEqual(file.name, "test.txt")
        self.assertEqual(file.lastSeen, 0)  # Should be 0 for new files

    def test_upsert_file_update(self):
        """Test updating an existing file"""
        # First insert a file
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }
        self.db.upsert_file(file_data)

        # Now update it
        updated_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "updated.txt",  # Changed name
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-03T12:00:00Z",  # Updated time
        }
        self.db.upsert_file(updated_data)

        # Verify the file was updated
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)
        self.assertEqual(file.name, "updated.txt")
        # lastSeen should remain 0
        self.assertEqual(file.lastSeen, 0)

    def test_upsert_file_with_labels(self):
        """Test upserting a file with labels"""
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
            "labelInfo": {
                "labels": [
                    {
                        "id": "label1",
                        "fields": {
                            "field1": {
                                "valueType": "selection",
                                "selection": ["selection1"],
                            }
                        },
                    }
                ]
            },
        }

        self.db.upsert_file(file_data)

        # Verify the file was inserted
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)

        # Verify the label was inserted
        label = Label.get_or_none(Label.fileId == "file1", Label.labelId == "label1")
        self.assertIsNotNone(label)
        self.assertEqual(label.selectionId, "selection1")

    def test_upsert_file_with_empty_labels(self):
        """Test upserting a file with empty labels dictionary"""
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
            "labelInfo": {},  # Empty dict
        }

        # This should not raise an exception
        self.db.upsert_file(file_data)

        # Verify the file was inserted
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)

    def test_extract_selection_id(self):
        """Test extracting selection ID from label data"""
        # Test with valid selection
        label_data = {
            "id": "label1",
            "fields": {
                "field1": {"valueType": "selection", "selection": ["selection1"]}
            },
        }
        self.assertEqual(self.db.extract_selection_id(label_data), "selection1")

        # Test with no selection field
        label_data = {"id": "label1", "fields": {"field1": {"valueType": "text"}}}
        self.assertEqual(self.db.extract_selection_id(label_data), "")

        # Test with empty fields
        label_data = {"id": "label1", "fields": {}}
        self.assertEqual(self.db.extract_selection_id(label_data), "")

        # Test with no fields
        label_data = {"id": "label1"}
        self.assertEqual(self.db.extract_selection_id(label_data), "")

    def test_get_file_exists(self):
        """Test getting a file that exists"""
        # Insert a file
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }
        self.db.upsert_file(file_data)

        # Get the file
        file = self.db.get_file("file1")
        self.assertIsNotNone(file)
        self.assertEqual(file["name"], "test.txt")

    def test_get_file_not_exists(self):
        """Test getting a file that does not exist"""
        file = self.db.get_file("nonexistent")
        self.assertIsNone(file)

    def test_update_classification_new_label(self):
        """Test updating classification for a new label"""
        # Insert a file
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }
        self.db.upsert_file(file_data)

        # Update classification
        self.db.update_classification(
            "file1", "label1", "selection1", "Test explanation"
        )

        # Verify the label was created
        label = Label.get_or_none(Label.fileId == "file1", Label.labelId == "label1")
        self.assertIsNotNone(label)
        self.assertEqual(label.selectionId, "selection1")
        self.assertEqual(label.lastClassificationExplanation, "Test explanation")
        self.assertIsNotNone(label.lastClassifiedTime)

        # Verify the file lastSeen was updated
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)
        self.assertNotEqual(file.lastSeen, 0)

    def test_update_classification_existing_label(self):
        """Test updating classification for an existing label"""
        # Insert a file with a label
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
            "labelInfo": {
                "labels": [
                    {
                        "id": "label1",
                        "fields": {
                            "field1": {
                                "valueType": "selection",
                                "selection": ["selection1"],
                            }
                        },
                    }
                ]
            },
        }
        self.db.upsert_file(file_data)

        # Update classification
        self.db.update_classification(
            "file1", "label1", "selection2", "Updated explanation"
        )

        # Verify the label was updated
        label = Label.get_or_none(Label.fileId == "file1", Label.labelId == "label1")
        self.assertIsNotNone(label)
        self.assertEqual(label.lastClassificationExplanation, "Updated explanation")

    def test_get_files_to_classify(self):
        """Test getting files that need classification"""
        # Insert two files
        file1 = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test1.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }
        file2 = {
            "id": "file2",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test2.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-03T12:00:00Z",
        }
        self.db.upsert_file(file1)
        self.db.upsert_file(file2)

        # Set file1 as seen with an older timestamp
        current_time = int(datetime.now(timezone.utc).timestamp())
        File.update(lastSeen=current_time).where(File.fileId == "file1").execute()

        # Update file1's modified time to be newer than lastSeen
        modified_time = current_time + 100
        File.update(modifiedTime=modified_time).where(File.fileId == "file1").execute()

        # Get files to classify
        files = self.db.get_files_to_classify()

        # Both files should be returned since both have modifiedTime > lastSeen
        self.assertEqual(len(files), 2)

        # Test dry run mode
        files = self.db.get_files_to_classify(dry_run=True)
        self.assertEqual(len(files), 2)

    def test_flag_file_as_seen(self):
        """Test flagging a file as seen"""
        # Insert a file
        file_data = {
            "id": "file1",
            "driveId": "drive1",
            "mimeType": "text/plain",
            "name": "test.txt",
            "createdTime": "2023-01-01T12:00:00Z",
            "modifiedTime": "2023-01-02T12:00:00Z",
        }
        self.db.upsert_file(file_data)

        # Flag as seen
        self.db.flag_file_as_seen("file1")

        # Verify lastSeen was updated
        file = File.get_or_none(File.fileId == "file1")
        self.assertIsNotNone(file)
        self.assertNotEqual(file.lastSeen, 0)

        # Get files to classify should return nothing
        files = self.db.get_files_to_classify()
        self.assertEqual(len(files), 0)


if __name__ == "__main__":
    unittest.main()
