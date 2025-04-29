# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ClassifyIt - Main class for content classification
"""

import csv
import logging
import os
import time
from datetime import datetime
from queue import Queue
from typing import Any, List, Mapping, Optional

from classifyit.database.database import Database
from classifyit.google_drive_client.drive_client import DriveClient
from classifyit.llm_client.llm_client import classify_content
from classifyit.structs import ClassificationResult, SessionConfig

from llama_stack_client import LlamaStackClient

logger = logging.getLogger("classifyit.classifier")


class ClassifyIt:  # pylint: disable=too-many-instance-attributes,too-many-positional-arguments
    """
    ClassifyIt - A tool for classifying content stored in Google Drive using LLMs
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        drive_owner: str,
        service_account_file: str,
        session_config: Mapping[str, Any],
        dry_run: bool = False,
        output_dir: str = ".",
        verbose: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the ClassifyIt tool.

        Args:
            drive_owner: Drive Organizer email
            service_account_file: Path to the Google Cloud Platform service
                account credentials file (e.g. 'credentials.json')
            session_config: Dictionary containing LLM and cloud session
                configuration
            output_dir: Directory to save output report files
            verbose: Enable verbose logging
            dry_run: Scan entire drive without applying Google Workspace
                labels on classification match
            db_path: Path to the SQLite database file (optional)
        """
        # Convert input dictionary to SessionConfig
        self._session_config = SessionConfig.from_dict(dict(session_config))

        self._output_dir = output_dir
        self._verbose = verbose
        self._dry_run = dry_run
        self._processed_count = 0
        self._drive_owner = drive_owner
        self._shared_drive_id = None
        self._drive_type = None

        self._scanned_files: List[ClassificationResult] = []

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(console_handler)

        # Initialize Google Drive client
        self._drive_client = DriveClient(
            drive_owner=drive_owner,
            service_account_file=service_account_file,
        )

        # Initialize LlamaStack client
        base_url = (
            f"http://{self._session_config.llama_server.host}:"
            + f"{self._session_config.llama_server.port}"
        )
        self._llama_client = LlamaStackClient(base_url=base_url)

        # Initialize database
        if db_path is None:
            # Default database path based on drive ID (shared or my_drive)
            # and label ID
            if self._shared_drive_id:
                drive_id = self._shared_drive_id.split("@")[0]
                label_id = self._session_config.google_drive_label_settings.label_id
                self._db_path = os.path.join(
                    "/tmp",
                    f"shareddrive_{drive_id}_label_{label_id}.sqlite",
                )
            else:
                drive_id = self._drive_owner.split("@")[0]
                label_id = self._session_config.google_drive_label_settings.label_id
                self._db_path = os.path.join(
                    "/tmp",
                    f"mydrive_{drive_id}_label_{label_id}.sqlite",
                )
        else:
            self._db_path = db_path

        self._db = Database(self._db_path)
        self._files_to_classify = Queue()

    def scan_drive(
        self,
        drive_type: str,
        shared_drive_id: str | None = None,
    ) -> None:
        """
        Scan a Google Drive for files to classify. Output a CSV report of the results.

        Args:
            drive_type: Type of drive to scan ('my_drive' or 'shared_drive')
            shared_drive_id: Shared drive ID to scan (required for shared_drive)
        """

        self._db.connect()

        try:
            logger.info("Starting classification process")

            # Scan the drive and update the database with the file metadata
            self.snapshot_drive_to_db(drive_type, shared_drive_id)

            # Get list of files to classify and add to queue
            files_to_classify = self._db.get_files_to_classify(dry_run=self._dry_run)
            logger.info("Classifying %d files", len(files_to_classify))
            for file in files_to_classify:
                self._files_to_classify.put(file)

            # Process files in the classification queue
            self.process_classification_queue()

            # Export list of changed files to CSV
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(self._output_dir, filename)
            if self._scanned_files:
                self.export_to_csv(output_path)

            logger.info(
                "Classification process completed. Processed %d new or modified files.",
                self._processed_count,
            )

        finally:
            self._db.close()

    def snapshot_drive_to_db(
        self,
        drive_type: str,
        shared_drive_id: Optional[str] = None,
    ) -> None:
        """
        Scan a Google Drive and update the database with the file metadata.

        Args:
            drive_type: Type of drive to scan ('my_drive' or 'shared_drive')
            shared_drive_id: Shared drive ID to scan (required for shared_drive)
        """
        self._drive_type = drive_type
        self._shared_drive_id = shared_drive_id

        # List all files in selected Google Drive
        files = self._drive_client.list_files(
            drive_type=drive_type,
            shared_drive_id=shared_drive_id,
            label_id=self._session_config.google_drive_label_settings.label_id,
        )

        # Process each file and write to database
        for file in files:
            file_data = {
                "createdTime": file.get("createdTime", ""),
                "driveId": shared_drive_id if shared_drive_id else "my_drive",
                "fileExtension": file.get("fileExtension", ""),
                "id": file.get("id"),
                "labelInfo": file.get("labelInfo", {"labels": []}),
                "md5Checksum": file.get("md5Checksum", ""),
                "mimeType": file.get("mimeType"),
                "modifiedTime": file.get("modifiedTime", ""),
                "name": file.get("name"),
                "size": file.get("size", "0"),
            }

            # Upsert file in database
            self._db.upsert_file(file_data)

    def process_classification_queue(self) -> None:
        """
        Process files in the classification queue
        """
        while not self._files_to_classify.empty():
            file_data = self._files_to_classify.get()
            file_id = file_data["fileId"]
            file_name = file_data["name"]
            mime_type = file_data.get("mimeType", "")
            file_size = int(file_data.get("size", "0"))

            # Extract content based on file type.
            # If the file is not supported or not parseable, mark it as seen
            # to skip it until it is updated in the future
            try:
                content = self._drive_client.extract_file_content(
                    file_id=file_id,
                    mime_type=mime_type,
                    file_size=file_size,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error extracting content from %s: %s", file_name, e)
                self._db.flag_file_as_seen(file_id)
                self._scanned_files.append(
                    ClassificationResult(
                        file_id=file_id,
                        file_name=file_name,
                        mime_type=mime_type,
                        model_response="",
                        action_taken="Skipped",
                        error=str(e),
                    )
                )
                continue

            if not content:
                logger.warning(
                    "Could not extract content from %s. Skipping file.", file_name
                )
                self._db.flag_file_as_seen(file_id)
                self._scanned_files.append(
                    ClassificationResult(
                        file_id=file_id,
                        file_name=file_name,
                        mime_type=mime_type,
                        model_response="",
                        action_taken="Skipped",
                        error="Could not extract content from file",
                    )
                )
                continue

            # Classify the file content
            result = self.classify_file(
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
                content=content,
            )
            self._scanned_files.append(result)

    def classify_file(
        self, file_id: str, file_name: str, mime_type: str, content: str
    ) -> ClassificationResult:
        """
        Classify file content against the LLM prompt criteria. Label the file if a match is found.

        Args:
            file_id: ID of the file
            file_name: Name of the file
            mime_type: MIME type of the file
            content: Content of the file

        Returns:
            ClassificationResult object containing the classification result
        """
        logger.info("Processing file: %s (%s)", file_name, file_id)
        self._processed_count += 1

        result = ClassificationResult(
            file_id=file_id,
            file_name=file_name,
            mime_type=mime_type,
            model_response="",
            action_taken="",
            error="",
        )

        max_retries = 3
        retry_delay = 1
        success = False

        for attempt in range(max_retries):
            try:
                # Classify content
                classification_output = classify_content(
                    llama_client=self._llama_client,
                    session_config=self._session_config,
                    content=content,
                )
                success = True
                break

            except Exception as e:  # pylint: disable=broad-exception-caught
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(
                        "Error classifying file %s after %d attempts: %s",
                        file_id,
                        max_retries,
                        e,
                    )
                    result.error = str(e)
                    return result

                logger.warning(
                    "Attempt %d failed for file %s: %s. Retrying in %d seconds...",
                    attempt + 1,
                    file_id,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= 2

        if not success:
            logger.error(
                "Failed to classify file %s after %d attempts", file_id, max_retries
            )
            result.error = "Failed to classify file"
            return result

        # If a match is found, apply the label
        logger.info("Model classification match detected in %s", file_name)
        result.model_response = classification_output.brief_explanation

        if self._dry_run:
            result.action_taken = "Dry run"
            logger.info("Dry run: Would apply label to %s", file_name)
            return result

        label_result = self._drive_client.apply_label_from_classification(
            file_id=file_id,
            classification_output=classification_output,
            label_settings=self._session_config.google_drive_label_settings,
            db=self._db,
        )
        result.action_taken = label_result["action_taken"]
        result.error = label_result["error"]

        return result

    def export_to_csv(self, output_path: str) -> None:
        """
        Export the list of classified files to a CSV file.

        Args:
            output_path: Path to the output CSV file
        """
        logger.info(
            "Exporting %d classified file results to CSV. Output path: %s",
            len(self._scanned_files),
            output_path,
        )
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            # Write header
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "File ID",
                    "File Name",
                    "MIME Type",
                    "Model Response",
                    "Action Taken",
                    "Error",
                ]
            )

            # Write data
            for result in self._scanned_files:
                writer.writerow(
                    [
                        result.file_id,
                        result.file_name,
                        result.mime_type,
                        result.model_response,
                        result.action_taken,
                        result.error,
                    ]
                )
