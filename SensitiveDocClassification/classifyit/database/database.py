# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Database handler for ClassifyIt
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from peewee import (
    CharField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# Get the package logger
logger = logging.getLogger("classifyit.database")


class BaseModel(Model):
    """
    Base model for all models
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """
        Meta class for all models
        """

        database = None  # Will be set when database is initialized


class File(BaseModel):
    """
    Peewee model for storing file metadata
    """

    fileId = CharField(primary_key=True)
    driveId = CharField()
    fileExtension = CharField(null=True)
    md5Checksum = CharField(null=True)
    size = CharField(null=True)
    mimeType = CharField()
    name = CharField()
    createdTime = IntegerField()  # Unix timestamp
    modifiedTime = IntegerField()  # Unix timestamp
    lastSeen = IntegerField(null=True)  # Unix timestamp


class Label(BaseModel):
    """
    Peewee model for storing label metadata
    Many labels can be associated with a file
    """

    fileId = ForeignKeyField(File, backref="labels")
    labelId = CharField()
    selectionId = CharField()
    lastClassifiedTime = IntegerField(null=True)  # Unix timestamp
    lastClassificationExplanation = TextField(null=True)


class Database:
    """
    Peewee database handler for ClassifyIt
    """

    def __init__(self, db_path: str):
        """
        Initialize the database connection

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db = SqliteDatabase(db_path)

        # Initialize models with the database
        File._meta.database = self.db  # pylint: disable=no-member
        Label._meta.database = self.db  # pylint: disable=no-member

    def connect(self) -> None:
        """Connect to the database and create tables"""
        try:
            self.db.connect()

            # Create tables if they don't exist
            self.db.create_tables([File, Label], safe=True)
            logger.debug("Connected to database at %s", self.db_path)
        except Exception as e:
            logger.error("Error connecting to database: %s", str(e))
            raise

    def close(self) -> None:
        """Close the database connection"""
        if not self.db.is_closed():
            self.db.close()
            logger.debug("Database connection closed")

    def iso_to_unix(self, iso_time: str) -> int:
        """
        Convert ISO format time string to Unix timestamp

        Args:
            iso_time: ISO format time string

        Returns:
            Unix timestamp as integer
        """
        if not iso_time:
            return 0

        try:
            dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            logger.warning("Invalid ISO time format: %s", iso_time)
            return 0

    def upsert_file(self, file_data: Dict[str, Any]) -> None:
        """
        Insert or update a file record in the database

        Args:
            file_data: Dictionary containing file metadata
        """
        try:
            data = file_data.copy()

            if "id" in data:
                data["fileId"] = data["id"]
                del data["id"]

            # Convert ISO time strings to Unix timestamps
            if "createdTime" in data:
                data["createdTime"] = self.iso_to_unix(data["createdTime"])
            if "modifiedTime" in data:
                data["modifiedTime"] = self.iso_to_unix(data["modifiedTime"])

            # If this is the first time we've seen the file, initialize lastSeen to 0
            previous_record = self.get_file(file_id=data["fileId"])
            if not previous_record:
                logger.debug("First time seeing file %s", data["fileId"])
                data["lastSeen"] = 0
            # Else- copy the lastSeen timestamp to the new record
            else:
                last_seen_timestamp = previous_record.get("lastSeen", 0)
                data["lastSeen"] = last_seen_timestamp

            # Remove the labels from the file data and write them to the database later
            label_data = None
            if "labelInfo" in data:
                label_data = data["labelInfo"]
                del data["labelInfo"]

            # Write file metadata to the database
            File.replace(**data).execute()

            # Write each label from the nested labelInfo list to the database
            if label_data:
                for label_list_entry in label_data.values():
                    for label in label_list_entry:
                        self.upsert_label(label, data["fileId"])

        except Exception as e:
            logger.error(
                "Error upserting file %s: %s", file_data.get("id", "unknown"), str(e)
            )
            raise

    def upsert_label(self, label_data: Dict[str, Any], file_id: str) -> None:
        """
        Insert or update a label record in the database

        Args:
            label_data: Dictionary containing the label object from the Google Drive API
            file_id: The file ID to associate the label with
        """
        try:
            # Try to find the existing label
            label = Label.get_or_none(
                (Label.fileId == file_id) & (Label.labelId == label_data["id"])
            )

            selection_id = self.extract_selection_id(label_data)

            label_info = {
                "labelId": label_data["id"],
                "fileId": file_id,
                "selectionId": selection_id,
            }

            if label:
                Label.update(**label_info).where(
                    (Label.fileId == file_id) & (Label.labelId == label_data["id"])
                ).execute()
            else:
                Label.create(**label_info)

        except Exception as e:
            logger.error(
                "Error upserting label %s: %s", label_data.get("id", "unknown"), str(e)
            )
            raise

    def extract_selection_id(self, label_data: Dict[str, Any]) -> str:
        """
        Extract the selection ID from a Google Drive API label response

        Args:
            label_data: Dictionary containing the label object from the Google Drive API

        Returns:
            The selection ID as a string, or an empty string if not found
        """
        try:
            if "fields" not in label_data:
                logger.warning("No fields found in label data: %s", label_data)
                return ""

            fields = label_data["fields"]

            for _, field_data in fields.items():
                # Check if this is a selection field
                if field_data.get("valueType") == "selection":
                    # Get the selection array
                    selection = field_data.get("selection", [])

                    # Return the first & onlyselection ID if available
                    if selection and len(selection) > 0:
                        return selection[0]

            logger.warning("No selection found in label data: %s", label_data)
            return ""

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error extracting selection ID from label data: %s", str(e))
            return ""

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a file record by ID

        Args:
            file_id: The file ID to look up

        Returns:
            Dictionary containing file metadata or None if not found
        """
        try:
            file = File.get_or_none(File.fileId == file_id)
            if file:
                file_data = file.__data__
                return file_data
            return None

        except Exception as e:
            logger.error("Error getting file %s: %s", file_id, str(e))
            raise

    def update_classification(
        self,
        file_id: str,
        label_id: str,
        selection_id: str,
        brief_explanation: str,
    ) -> None:
        """
        Update a label with fresh classification information

        Args:
            file_id: The file ID to update
            label_id: The label ID
            selection_id: The selection ID
            brief_explanation: The brief explanation of the classification
        """
        try:
            current_time = int(datetime.now(timezone.utc).timestamp())

            # First try to find the existing label
            label = Label.get_or_none(
                (Label.fileId == file_id) & (Label.labelId == label_id)
            )

            if label:
                Label.update(
                    lastClassifiedTime=current_time,
                    lastClassificationExplanation=brief_explanation,
                ).where(
                    (Label.fileId == file_id) & (Label.labelId == label_id)
                ).execute()
            else:
                # Create new label if it doesn't exist
                Label.create(
                    fileId=file_id,
                    labelId=label_id,
                    selectionId=selection_id,
                    lastClassifiedTime=current_time,
                    lastClassificationExplanation=brief_explanation,
                )

            current_time_plus_one_second = current_time + 1
            File.update(lastSeen=current_time_plus_one_second).where(
                File.fileId == file_id
            ).execute()

        except Exception as e:
            logger.error(
                "Error updating classification for file %s: %s", file_id, str(e)
            )
            raise

    def get_files_to_classify(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Get a list of files that need to be classified

        Args:
            dry_run: If True, ignore lastSeen timestamp and return all files

        Returns:
            List of file dictionaries that need classification
        """
        try:
            if dry_run:
                query = File.select().order_by(File.modifiedTime.desc())
            else:
                query = (
                    File.select()
                    .where(File.modifiedTime > File.lastSeen)
                    .order_by(File.modifiedTime.desc())
                )

            return [file.__data__ for file in query]

        except Exception as e:
            logger.error("Error getting files to classify: %s", str(e))
            raise

    def flag_file_as_seen(self, file_id: str) -> None:
        """
        Flag a file as seen to prevent it from being classified again

        Args:
            file_id: The file ID to flag as seen
        """
        try:
            File.update(lastSeen=int(datetime.now(timezone.utc).timestamp())).where(
                File.fileId == file_id
            ).execute()
        except Exception as e:
            logger.error("Error flagging file %s as seen: %s", file_id, str(e))
            raise
