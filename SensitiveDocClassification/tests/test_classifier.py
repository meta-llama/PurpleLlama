# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for the main program logic
"""

# pylint: disable=protected-access
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from classifyit.classifier import ClassifyIt
from classifyit.database.database import Database
from classifyit.google_drive_client.drive_client import DriveClient

from classifyit.structs import (
    AgentSettings,
    ClassificationResult,
    GoogleDriveLabelSettings,
    LlamaServerConfig,
    SessionConfig,
)

from llama_stack_client import LlamaStackClient


class TestClassifyIt(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test suite for the ClassifyIt class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.mock_drive_client = MagicMock(spec=DriveClient)
        self.mock_llama_client = MagicMock(spec=LlamaStackClient)
        self.mock_db = MagicMock(spec=Database)

        # Mock the constructors for our dependencies
        self.drive_client_patcher = patch("classifyit.classifier.DriveClient")
        self.mock_drive_client_constructor = self.drive_client_patcher.start()
        self.mock_drive_client_constructor.return_value = self.mock_drive_client

        self.llama_client_patcher = patch("classifyit.classifier.LlamaStackClient")
        self.mock_llama_client_constructor = self.llama_client_patcher.start()
        self.mock_llama_client_constructor.return_value = self.mock_llama_client

        self.db_patcher = patch("classifyit.classifier.Database")
        self.mock_db_constructor = self.db_patcher.start()
        self.mock_db_constructor.return_value = self.mock_db

        # Test configuration
        self.drive_owner = "test@example.com"
        self.service_account_file = "/path/to/service_account.json"

        # Dict version for initialization
        self.session_config_dict = {
            "llama_server": {"host": "localhost", "port": 8000, "model": "llama3"},
            "agent_settings": {
                "agent_instructions": "test instructions",
                "prompt": "test prompt",
                "response_categories": {"category1": {}, "category2": {}},
                "model": "llama3",
                "temperature": 0.7,
            },
            "google_drive_label_settings": {
                "label_id": "label1",
                "field_id": "field1",
                "selection_ids": {"category1": "id1", "category2": "id2"},
            },
        }
        self.output_dir = tempfile.mkdtemp()

        # Create the expected session config instance that should result from conversion
        self.expected_session_config = SessionConfig(
            llama_server=LlamaServerConfig(**self.session_config_dict["llama_server"]),
            agent_settings=AgentSettings(**self.session_config_dict["agent_settings"]),
            google_drive_label_settings=GoogleDriveLabelSettings(
                **self.session_config_dict["google_drive_label_settings"]
            ),
        )

        self.classifier = ClassifyIt(
            drive_owner=self.drive_owner,
            service_account_file=self.service_account_file,
            session_config=self.session_config_dict,
            output_dir=self.output_dir,
            verbose=False,
            dry_run=False,
        )

        self.classifier._db = self.mock_db

    def tearDown(self):
        """Clean up after each test"""
        self.drive_client_patcher.stop()
        self.llama_client_patcher.stop()
        self.db_patcher.stop()

        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

    def test_init(self):
        """Test the initialization of ClassifyIt"""
        # Verify that the correct dependencies were initialized
        self.mock_drive_client_constructor.assert_called_once_with(
            drive_owner=self.drive_owner, service_account_file=self.service_account_file
        )

        base_url = (
            f"http://{self.session_config_dict['llama_server']['host']}:"
            + f"{self.session_config_dict['llama_server']['port']}"
        )
        self.mock_llama_client_constructor.assert_called_once_with(base_url=base_url)

        # Verify that the database was constructed with the correct path
        self.assertIn("sqlite", str(self.mock_db_constructor.call_args))

        # Verify internal state
        self.assertEqual(self.classifier._drive_owner, self.drive_owner)

        # Check that the session_config was converted to a SessionConfig
        self.assertIsInstance(self.classifier._session_config, SessionConfig)

        # Verify session config fields were correctly set
        self.assertEqual(
            self.classifier._session_config.llama_server.host,
            self.expected_session_config.llama_server.host,
        )
        self.assertEqual(
            self.classifier._session_config.llama_server.port,
            self.expected_session_config.llama_server.port,
        )
        self.assertEqual(
            self.classifier._session_config.agent_settings.agent_instructions,
            self.expected_session_config.agent_settings.agent_instructions,
        )
        self.assertEqual(
            self.classifier._session_config.google_drive_label_settings.label_id,
            self.expected_session_config.google_drive_label_settings.label_id,
        )

        self.assertEqual(self.classifier._output_dir, self.output_dir)
        self.assertEqual(self.classifier._scanned_files, [])
        self.assertEqual(self.classifier._processed_count, 0)

    def test_scan_drive_my_drive(self):
        """Test scanning My Drive"""
        mock_files = [{"fileId": "file1", "name": "test.txt", "mimeType": "text/plain"}]
        self.mock_db.get_files_to_classify.return_value = mock_files

        with patch.object(
            self.classifier, "process_classification_queue"
        ) as mock_process:
            self.classifier.scan_drive(drive_type="my_drive")

            # Verify that get_files_to_classify was called
            self.mock_db.get_files_to_classify.assert_called_once_with(dry_run=False)

            # Verify that process_classification_queue was called
            mock_process.assert_called_once()

            # Verify that files_to_classify queue has the correct files
            self.assertEqual(self.classifier._files_to_classify.qsize(), 1)
            self.assertEqual(self.classifier._files_to_classify.get(), mock_files[0])

    @patch("classifyit.classifier.time.sleep")
    def test_classify_file_success(self, mock_sleep):  # pylint: disable=unused-argument
        """Test successful file classification"""
        mock_classification_output = MagicMock()
        mock_classification_output.brief_explanation = "Confidential content detected"
        mock_classification_output.confidence = "high"

        with patch("classifyit.classifier.classify_content") as mock_classify:
            mock_classify.return_value = mock_classification_output

            self.mock_drive_client.apply_label_from_classification.return_value = {
                "action_taken": "Applied label",
                "error": "",
            }

            result = self.classifier.classify_file(
                file_id="file1",
                file_name="test.txt",
                mime_type="text/plain",
                content="This is confidential information",
            )

            # Check the arguments that were passed to classify_content
            mock_classify.assert_called_once()
            _, kwargs = mock_classify.call_args
            self.assertEqual(kwargs["llama_client"], self.mock_llama_client)
            self.assertEqual(kwargs["content"], "This is confidential information")
            self.assertIsInstance(kwargs["session_config"], SessionConfig)

            # Check that we passed the GoogleDriveLabelSettings to apply_label_from_classification
            self.mock_drive_client.apply_label_from_classification.assert_called_once()
            _, kwargs = self.mock_drive_client.apply_label_from_classification.call_args
            self.assertEqual(kwargs["file_id"], "file1")
            self.assertEqual(
                kwargs["classification_output"], mock_classification_output
            )
            self.assertEqual(kwargs["db"], self.mock_db)
            self.assertIsInstance(kwargs["label_settings"], GoogleDriveLabelSettings)

            self.assertEqual(result.file_id, "file1")
            self.assertEqual(result.model_response, "Confidential content detected")
            self.assertEqual(result.action_taken, "Applied label")
            self.assertEqual(result.error, "")

    @patch("classifyit.classifier.time.sleep")
    def test_classify_file_retry_success(self, mock_sleep):
        """Test classification retry that succeeds"""
        mock_classification_output = MagicMock()
        mock_classification_output.brief_explanation = "Confidential content detected"

        with patch("classifyit.classifier.classify_content") as mock_classify:
            # First call raises exception, second call succeeds
            mock_classify.side_effect = [
                Exception("Connection error"),
                mock_classification_output,
            ]

            self.mock_drive_client.apply_label_from_classification.return_value = {
                "action_taken": "Applied label",
                "error": "",
            }

            result = self.classifier.classify_file(
                file_id="file1",
                file_name="test.txt",
                mime_type="text/plain",
                content="This is confidential information",
            )

            # Verify that classify_content was called twice
            self.assertEqual(mock_classify.call_count, 2)
            mock_sleep.assert_called_once_with(1)
            self.mock_drive_client.apply_label_from_classification.assert_called_once()

            self.assertEqual(result.file_id, "file1")
            self.assertEqual(result.model_response, "Confidential content detected")
            self.assertEqual(result.action_taken, "Applied label")
            self.assertEqual(result.error, "")

    @patch("classifyit.classifier.time.sleep")
    def test_classify_file_all_retries_fail(
        self,
        mock_sleep,  # pylint: disable=unused-argument
    ):
        """Test classification when all retries fail"""
        error_message = "Connection error"

        with patch("classifyit.classifier.classify_content") as mock_classify:
            # All retries raise exceptions
            mock_classify.side_effect = [
                Exception(error_message),
                Exception(error_message),
                Exception(error_message),
            ]

            result = self.classifier.classify_file(
                file_id="file1",
                file_name="test.txt",
                mime_type="text/plain",
                content="This is confidential information",
            )

            # Verify that classify_content was called the maximum number of times
            self.assertEqual(mock_classify.call_count, 3)

            # Verify that apply_label_from_classification was not called
            self.mock_drive_client.apply_label_from_classification.assert_not_called()

            self.assertEqual(result.file_id, "file1")
            self.assertEqual(result.model_response, "")
            self.assertEqual(result.action_taken, "")
            self.assertIn(error_message, result.error)

    def test_classify_file_dry_run(self):
        """Test classification in dry run mode"""
        self.classifier._dry_run = True

        mock_classification_output = MagicMock()
        mock_classification_output.brief_explanation = "Confidential content detected"

        with patch("classifyit.classifier.classify_content") as mock_classify:
            mock_classify.return_value = mock_classification_output

            result = self.classifier.classify_file(
                file_id="file1",
                file_name="test.txt",
                mime_type="text/plain",
                content="This is confidential information",
            )

            # Verify that apply_label_from_classification was not called
            self.mock_drive_client.apply_label_from_classification.assert_not_called()

            self.assertEqual(result.file_id, "file1")
            self.assertEqual(result.model_response, "Confidential content detected")
            self.assertEqual(result.action_taken, "Dry run")
            self.assertEqual(result.error, "")

    def test_export_to_csv(self):
        """Test exporting results to CSV"""
        self.classifier._scanned_files = [
            ClassificationResult(
                file_id="file1",
                file_name="test1.txt",
                mime_type="text/plain",
                model_response="Confidential",
                action_taken="Applied label",
                error="",
            ),
            ClassificationResult(
                file_id="file2",
                file_name="test2.txt",
                mime_type="text/plain",
                model_response="No match",
                action_taken="No action",
                error="",
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
            output_path = tmp_file.name

            self.classifier.export_to_csv(output_path)

            # Verify that the CSV file exists and has the expected content
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Check that headers are present
                self.assertIn("File ID", content)
                self.assertIn("File Name", content)
                self.assertIn("MIME Type", content)
                self.assertIn("Model Response", content)
                self.assertIn("Action Taken", content)
                self.assertIn("Error", content)

                # Check that file data is present
                self.assertIn("file1", content)
                self.assertIn("test1.txt", content)
                self.assertIn("Confidential", content)
                self.assertIn("Applied label", content)

                self.assertIn("file2", content)
                self.assertIn("test2.txt", content)
                self.assertIn("No match", content)
                self.assertIn("No action", content)

    def test_snapshot_drive_to_db(self):
        """Test scanning a drive and updating the database"""
        mock_files = [
            {
                "id": "file1",
                "name": "test.txt",
                "mimeType": "text/plain",
                "createdTime": "2023-01-01T12:00:00Z",
                "modifiedTime": "2023-01-02T12:00:00Z",
                "size": "1000",
                "md5Checksum": "abc123",
                "fileExtension": "txt",
                "labelInfo": {"labels": []},
            }
        ]

        self.mock_drive_client.list_files.return_value = mock_files

        self.classifier.snapshot_drive_to_db("my_drive")

        # Verify list_files was called with the label_id from our SessionConfig
        self.mock_drive_client.list_files.assert_called_once_with(
            drive_type="my_drive",
            shared_drive_id=None,
            label_id=self.expected_session_config.google_drive_label_settings.label_id,
        )

        # Verify that upsert_file was called for each file
        self.mock_db.upsert_file.assert_called_once()
        file_data = self.mock_db.upsert_file.call_args[0][0]
        self.assertEqual(file_data["id"], "file1")
        self.assertEqual(file_data["name"], "test.txt")
        self.assertEqual(file_data["driveId"], "my_drive")

    def testprocess_classification_queue(self):
        """Test processing the classification queue"""
        mock_file = {
            "fileId": "file1",
            "name": "test.txt",
            "mimeType": "text/plain",
            "size": "1000",
        }
        self.classifier._files_to_classify.put(mock_file)

        self.mock_drive_client.extract_file_content.return_value = "Sample content"

        mock_result = ClassificationResult(
            file_id="file1",
            file_name="test.txt",
            mime_type="text/plain",
            model_response="Confidential",
            action_taken="Applied label",
            error="",
        )

        with patch.object(self.classifier, "classify_file") as mock_classify_file:
            mock_classify_file.return_value = mock_result

            self.classifier.process_classification_queue()

            self.mock_drive_client.extract_file_content.assert_called_once_with(
                file_id="file1", mime_type="text/plain", file_size=1000
            )

            mock_classify_file.assert_called_once_with(
                file_id="file1",
                file_name="test.txt",
                mime_type="text/plain",
                content="Sample content",
            )

            self.assertEqual(len(self.classifier._scanned_files), 1)
            self.assertEqual(self.classifier._scanned_files[0], mock_result)

    def testprocess_classification_queue_extraction_error(self):
        """Test handling extraction errors in process_classification_queue"""
        mock_file = {"fileId": "file1", "name": "test.txt", "mimeType": "text/plain"}
        self.classifier._files_to_classify.put(mock_file)

        error_message = "Failed to extract content"
        self.mock_drive_client.extract_file_content.side_effect = Exception(
            error_message
        )

        self.classifier.process_classification_queue()

        # Verify that flag_file_as_seen was called
        self.mock_db.flag_file_as_seen.assert_called_once_with("file1")

        # Verify that a result with an error was added to scanned_files
        self.assertEqual(len(self.classifier._scanned_files), 1)
        result = self.classifier._scanned_files[0]
        self.assertEqual(result.file_id, "file1")
        self.assertEqual(result.action_taken, "Skipped")
        self.assertIn(error_message, result.error)

    def testprocess_classification_queue_empty_content(self):
        """Test handling empty content in process_classification_queue"""
        mock_file = {"fileId": "file1", "name": "test.txt", "mimeType": "text/plain"}
        self.classifier._files_to_classify.put(mock_file)

        # Mock the drive client to return empty content
        self.mock_drive_client.extract_file_content.return_value = ""

        self.classifier.process_classification_queue()

        # Verify that flag_file_as_seen was called
        self.mock_db.flag_file_as_seen.assert_called_once_with("file1")

        # Verify that a result was added to scanned_files
        self.assertEqual(len(self.classifier._scanned_files), 1)
        result = self.classifier._scanned_files[0]
        self.assertEqual(result.file_id, "file1")
        self.assertEqual(result.action_taken, "Skipped")
        self.assertIn("Could not extract content from file", result.error)


if __name__ == "__main__":
    unittest.main()
