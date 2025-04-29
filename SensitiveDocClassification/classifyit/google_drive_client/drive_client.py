# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google Drive client for file operations and integration.

This module provides a client for interacting with the Google Drive Files API.
It handles authentication, file operations, content extraction, and label management.

There are a number of ways to authenticate with Google APIs,
including OAuth 2.0, API keys, and service accounts.

For this application, we use a service account to authenticate as the drive owner using DWD.

See https://developers.google.com/identity/protocols/oauth2/service-account for more information.
"""

import logging
from typing import Any, Dict, List, Optional

from classifyit.database.database import Database

from classifyit.google_drive_client.content_extractor import extract_content
from classifyit.google_drive_client.mime_types import (
    EXPORT_MIME_TYPES,
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    GOOGLE_SLIDES_MIME_TYPE,
)
from classifyit.structs import GoogleDriveLabelSettings

from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource

logger = logging.getLogger("classifyit.google_drive_client")

SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveClient:
    """Client for interacting with Google Drive Files v3 API.

    This class provides a client for interacting with the Google Drive Files API.
    It allows for listing files, exporting files, applying labels, and managing content extraction.

    Attributes:
        drive_owner: Email of the drive owner
        service_account_file: Path to the service account file
    """

    # Google Workspace MIME types
    GOOGLE_DOC_MIME_TYPE = GOOGLE_DOC_MIME_TYPE
    GOOGLE_SHEET_MIME_TYPE = GOOGLE_SHEET_MIME_TYPE
    GOOGLE_SLIDES_MIME_TYPE = GOOGLE_SLIDES_MIME_TYPE

    # Export MIME types
    EXPORT_MIME_TYPES = EXPORT_MIME_TYPES

    def __init__(self, drive_owner: str, service_account_file: str):
        """
        Initialize the Drive client.
        Obtain Service Account credentials from the credentials file
        and build a delegated service for the drive owner.

        Args:
            drive_owner: Email of the drive owner
            service_account_file: Path to the service account file (JSON format)
        """
        self.drive_owner = drive_owner
        self.service_account_file = service_account_file
        self.delegated_service = self.get_delegated_service()

        logger.info("Google Drive client initialized")

    def get_delegated_service(self) -> Resource:
        """Build and return an authorized Drive API service instance,
        delegated to the drive owner."""

        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=SCOPES
        )

        delegated_credentials = credentials.with_subject(self.drive_owner)
        return build("drive", "v3", credentials=delegated_credentials)

    def list_files(
        self, drive_type: str, label_id: str, shared_drive_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List files from the specified drive type and drive ID.

        Args:
            drive_type: Either 'my_drive' or 'shared_drive'
            shared_drive_id: ID of the shared drive (required for sharedDrive)

        Returns:
            List of file metadata dictionaries
        """
        if drive_type == "shared_drive" and not shared_drive_id:
            raise ValueError("Shared drive ID is required when using shared drive")

        drive_type_enum = "myDrive" if drive_type == "my_drive" else "sharedDrive"
        drive_fields = str(
            "nextPageToken, files(id, name, mimeType, size, createdTime, "
            + "modifiedTime, md5Checksum, fileExtension, labelInfo)"
        )
        try:
            next_page_token = None
            files = []

            while True:
                # Call the Drive API to list files from the shared drive
                if drive_type_enum == "sharedDrive":
                    logger.info(
                        "Listing files from shared drive ID: %s, for %s",
                        shared_drive_id,
                        self.drive_owner,
                    )
                    response = (
                        self.delegated_service.files()  # pylint: disable=E1101
                        .list(
                            driveId=shared_drive_id,
                            spaces="drive",
                            corpora="drive",
                            fields=drive_fields,
                            includeLabels=label_id,
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True,
                            pageToken=next_page_token,
                        )
                        .execute()
                    )
                else:
                    # Call the Drive API to list files from the user's MyDrive
                    logger.info("Listing files from MyDrive for %s", self.drive_owner)
                    response = (
                        self.delegated_service.files()  # pylint: disable=E1101
                        .list(
                            spaces="drive",
                            fields=drive_fields,
                            includeLabels=label_id,
                            pageToken=next_page_token,
                        )
                        .execute()
                    )

                # Process the response
                files.extend(response.get("files", []))
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            files = [
                {
                    "id": file["id"],
                    "name": file["name"],
                    "mimeType": file["mimeType"],
                    "size": file.get("size", "0"),
                    "createdTime": file.get("createdTime", ""),
                    "modifiedTime": file.get("modifiedTime", ""),
                    "md5Checksum": file.get("md5Checksum", ""),
                    "fileExtension": file.get("fileExtension", ""),
                    "labelInfo": file.get("labelInfo", {}),
                }
                for file in files
            ]

            logger.info("Found %d files to scan", len(files))
            return files

        except Exception as e:
            logger.error("Error listing files: %s", e)
            raise

    def export_file(self, file_id: str, mime_type: str) -> Optional[str]:
        """
        Export a Google Workspace file to the specified MIME type.

        Args:
            file_id: The ID of the file to export
            mime_type: The MIME type to export to

        Returns:
            The exported file content as a string, or None if export failed
        """
        logger.info("Exporting file %s as %s", file_id, mime_type)
        try:
            response = (
                self.delegated_service.files()  # pylint: disable=E1101
                .export(fileId=file_id, mimeType=mime_type)
                .execute()
            )
            return response.decode("utf-8")
        except Exception as e:
            logger.error("Error exporting file %s: %s", file_id, e)
            raise

    def get_file(self, file_id: str) -> Optional[str]:
        """
        Get the content of a file as a buffer.

        Args:
            file_id: The ID of the file to get

        Returns:
            The content of the file as a buffer, or None if the file is not found
        """
        try:
            response = (
                self.delegated_service.files()  # pylint: disable=E1101
                .get_media(fileId=file_id, supportsAllDrives=True)
                .execute()
            )
            return response
        except Exception as e:
            logger.error("Error getting file %s: %s", file_id, e)
            raise

    def extract_file_content(
        self,
        file_id: str,
        mime_type: str,
        file_size: int,
    ) -> Optional[str]:
        """
        Extract content from a file.

        Args:
            file_id: ID of the file
            mime_type: MIME type of the file
            file_size: Size of the file in bytes

        Returns:
            Extracted content as string, or None if extraction failed
        """
        try:
            content = extract_content(
                self,
                file_id,
                mime_type,
                file_size,
            )
            return content
        except Exception as e:
            logger.error("Error extracting content from file %s: %s", file_id, e)
            raise

    def apply_label(
        self, file_id: str, label_id: str, field_id: str, selection_ids: List[str]
    ) -> Optional[bool]:
        """
        Apply a label to a file.

        Args:
            file_id: The ID of the file to label
            label_id: The ID of the label to apply
            field_id: The ID of the field within the label
            selection_ids: List of selection IDs to apply

        Returns:
            True if the label was applied successfully, False otherwise
        """
        logger.info(
            "Applying label %s, field %s, selections %s to file %s",
            label_id,
            field_id,
            selection_ids,
            file_id,
        )

        try:
            field_modification = {
                "fieldId": field_id,
                "setSelectionValues": [selection_ids],
            }
            label_modification = {
                "labelId": label_id,
                "fieldModifications": [field_modification],
            }

            self.delegated_service.files().modifyLabels(  # pylint: disable=E1101
                fileId=file_id, body={"labelModifications": [label_modification]}
            ).execute()

            return True
        except Exception as e:
            logger.error("Error applying label %s to file %s: %s", label_id, file_id, e)
            raise

    def apply_label_from_classification(
        self,
        file_id: str,
        classification_output: Dict[str, str],
        label_settings: GoogleDriveLabelSettings,
        db: Database,
    ) -> Dict[str, str]:
        """
        Apply a label to a file based on classification output.

        Args:
            file_id: ID of the file
            classification_output: Classification output containing category_matched
            label_settings: GoogleDriveLabelSettings containing label configuration
            db: Database for recording label application

        Returns:
            Dictionary containing action_taken and error (if any)
        """
        result = {"action_taken": "", "error": ""}

        # Get matching selection id
        selection_id = None
        for key, value in label_settings.selection_ids.items():
            if key == classification_output.category_matched:
                selection_id = value
                break

        if not selection_id:
            result["error"] = "No matching Google Drive selection id found"
            return result

        # Apply label
        try:
            self.apply_label(
                file_id=file_id,
                label_id=label_settings.label_id,
                field_id=label_settings.field_id,
                selection_ids=selection_id,
            )
            result["action_taken"] = "Applied label"
        except Exception as e:
            logger.error(
                "Error applying label %s to file %s: %s",
                label_settings.label_id,
                file_id,
                e,
            )
            result["error"] = str(e)

        # Success; update classification in database
        try:
            db.update_classification(
                file_id=file_id,
                label_id=label_settings.label_id,
                selection_id=selection_id,
                brief_explanation=classification_output.brief_explanation,
            )
        except Exception as e:
            logger.error("Error updating classification for file %s: %s", file_id, e)
            raise

        return result
