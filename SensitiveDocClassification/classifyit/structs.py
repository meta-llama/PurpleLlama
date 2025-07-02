# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Structs for ClassifyIt
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ClassificationResult:
    """
    Struct representing the result of a classification request
    """

    file_id: str
    file_name: str
    mime_type: str
    model_response: str
    action_taken: str
    error: str


@dataclass
class LlamaServerConfig:
    """Configuration settings for Llama server"""

    host: str
    port: int
    model: str


@dataclass
class AgentSettings:
    """Configuration settings for the agent"""

    agent_instructions: str
    prompt: str
    response_categories: Dict[str, Dict[str, str]]
    model: str = ""
    temperature: float = 0.5


@dataclass
class GoogleDriveLabelSettings:
    """Configuration settings for Google Drive labels"""

    label_id: str
    field_id: str
    selection_ids: Dict[str, str]


@dataclass
class SessionConfig:
    """
    Configuration settings for a ClassifyIt session
    """

    llama_server: LlamaServerConfig
    agent_settings: AgentSettings
    google_drive_label_settings: GoogleDriveLabelSettings

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SessionConfig":
        """Create a SessionConfig from a dictionary"""
        # Validate the required keys are present
        required_keys = [
            "llama_server",
            "agent_settings",
            "google_drive_label_settings",
        ]
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )

        # Create the dataclass
        return cls(
            llama_server=LlamaServerConfig(**config_dict["llama_server"]),
            agent_settings=AgentSettings(**config_dict["agent_settings"]),
            google_drive_label_settings=GoogleDriveLabelSettings(
                **config_dict["google_drive_label_settings"]
            ),
        )
