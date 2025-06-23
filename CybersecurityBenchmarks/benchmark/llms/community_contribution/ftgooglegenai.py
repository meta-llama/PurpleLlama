# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from google.auth.transport.requests import Request
from google.cloud import storage
from google.oauth2 import service_account
from typing_extensions import override

from ...benchmark_utils import get_mime_type, image_to_b64
from ..llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
    UnretriableQueryException,
)


class FTGOOGLEGENAI(LLM):
    """Accessing fine-tuned Google Generative AI models"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # Use the google_adc_key.json file in the community_contribution directory
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
            os.path.dirname(__file__), "google_adc_key.json"
        )
        credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        credentials.refresh(Request())
        self.access_token = credentials.token

        # Extract tuning job parameters from api_key string
        # Assuming api_key is in the form of
        # projects/12345/locations/us-central1/endpoints/67890/buckets/bucket_name
        pattern = (
            r"projects/(?P<PROJECT_ID>[^/]+)/"
            r"locations/(?P<TUNING_JOB_REGION>[^/]+)/"
            r"endpoints/(?P<ENDPOINT_ID>[^/]+)/"
            r"buckets/(?P<BUCKET_NAME>[^/]+)"
        )
        # Ensure api_key is a string before passing to re.match
        api_key_str = "" if self.api_key is None else self.api_key
        match = re.match(pattern, api_key_str)
        if not match:
            raise ValueError(
                "Invalid API key format. Expected format: projects/<PROJECT_ID>/locations/<REGION>/endpoints/<ENDPOINT_ID>/buckets/<BUCKET_NAME>"
            )

        PROJECT_ID = match.group("PROJECT_ID")
        TUNING_JOB_REGION = match.group("TUNING_JOB_REGION")
        ENDPOINT_ID = match.group("ENDPOINT_ID")
        self.bucket_name: str = match.group("BUCKET_NAME")
        self.BASE_API_URL = (
            f"https://{TUNING_JOB_REGION}-aiplatform.googleapis.com/v1/projects/"
            f"{PROJECT_ID}/locations/{TUNING_JOB_REGION}/endpoints/{ENDPOINT_ID}:generateContent"
        )

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        data = {"contents": [{"parts": [{"text": prompt}], "role": "user"}]}
        response = requests.post(
            self.BASE_API_URL, headers=headers, data=json.dumps(data)
        )
        json_response = response.json()
        response_str = json.dumps(json_response)
        if '[{"finishReason": "OTHER", "index": 0}]' in response_str:
            return "FAILED TO QUERY: no response sent"  # GoogleGenAI didn't return a response
        text_value = (
            json_response.get("candidates", [])[0]
            .get("content", {})
            .get("parts", [])[0]
            .get("text", None)
        )
        return text_value

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for GOOGLEGENAI."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        data = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
                "role": "model",
            },
            "generation_config": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(
            self.BASE_API_URL, headers=headers, data=json.dumps(data)
        )
        json_response = response.json()
        response_str = json.dumps(json_response)
        if '[{"finishReason": "OTHER", "index": 0}]' in response_str:
            return "FAILED TO QUERY: no response sent"  # GoogleGenAI didn't return a response
        text_value = (
            json_response.get("candidates", [])[0]
            .get("content", {})
            .get("parts", [])[0]
            .get("text", None)
        )
        return text_value

    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for GOOGLEGENAI."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_to_b64(image_path),
                            },
                        },
                    ],
                    "role": "user",
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_prompt,
                    }
                ],
                "role": "model",
            },
            "generation_config": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(
            self.BASE_API_URL, headers=headers, data=json.dumps(data)
        )
        json_response = response.json()
        candidates = json_response.get("candidates", [])
        if not candidates:
            return "FAILED TO QUERY: no candidates in response"  # GoogleGenAI didn't return a response
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "FAILED TO QUERY: no content parts in candidate"  # GoogleGenAI didn't return a response
        text_value = parts[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY: no text in content parts"  # GoogleGenAI didn't return a response
        return text_value

    def _upload_file(
        self,
        file_path: str,
    ) -> str:
        """Uploads a file to GCS and returns the gs:// URI."""
        bucket_name = self.bucket_name
        destination_blob_name = os.path.basename(file_path)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)

        return f"gs://{bucket_name}/{destination_blob_name}"

    def _delete_file(self, file_uri: str) -> None:
        """Deletes a GCS object given its gs:// URI."""
        match = re.match(r"gs://([^/]+)/(.+)", file_uri)
        if not match:
            print(f"[WARNING] Not a valid GCS URI: {file_uri}")
            return
        bucket_name, blob_name = match.groups()

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        content_parts = []
        if text_prompt:
            content_parts.append({"text": text_prompt})

        file_uris = []

        content_parts = []
        if text_prompt:
            content_parts.append({"text": text_prompt})

        file_uris = []
        if image_paths:
            for image_path in image_paths:
                file_uri = self._upload_file(image_path)
                file_uris.append(file_uri)
                image_mimetype = get_mime_type(image_path)

                content_parts.append(
                    {"file_data": {"mime_type": image_mimetype, "file_uri": file_uri}}
                )

        if audio_paths:
            for audio_path in audio_paths:
                file_uri = self._upload_file(audio_path)
                file_uris.append(file_uri)
                audio_mimetype = get_mime_type(audio_path)
                content_parts.append(
                    {"file_data": {"mime_type": audio_mimetype, "file_uri": file_uri}}
                )

        data: Dict[str, Any] = {
            "contents": [{"parts": content_parts, "role": "user"}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            data["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
                "role": "model",
            }

        response = requests.post(
            self.BASE_API_URL, headers=headers, data=json.dumps(data)
        )

        # Delete files after query
        for file_uri in file_uris:
            self._delete_file(file_uri)

        if not response.ok:
            raise Exception(
                f"Got non-200 status code when querying model: {response.status_code}"
            )

        json_response = response.json()

        candidates = json_response.get("candidates", [])
        if not candidates:
            raise UnretriableQueryException(
                f"Failed to query model, got no candidates: {json_response}"
            )
        content = candidates[0].get("content", {})
        if not content:
            raise UnretriableQueryException(
                f"Failed to query model, got no content from first candidate: {candidates[0]}"
            )

        finish_reason = candidates[0].get("finishReason")
        if finish_reason != "STOP":
            raise UnretriableQueryException(
                f"Failed to query model, likely due to safety violation. Got unexpected finish reason: {finish_reason}"
            )

        parts = content.get("parts", [])
        if not parts:
            raise UnretriableQueryException(
                f"Failed to query model, got no content parts in first candidate content: {content}"
            )
        text_value = parts[0].get("text", None)
        if not text_value:
            raise UnretriableQueryException(
                f"Failed to query model, candidate content parts did not return text value: {parts[0]}"
            )

        return text_value

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Chat API not implemented yet.")

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return super().chat_with_system_prompt(
            system_prompt,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

    @override
    def valid_models(self) -> list[str]:
        return [
            "ft-gemini-2.0-flash",
            "ft-gemini-2.0-flash-lite",
            "ft-gemini-1.5-pro",
            "ft-gemini-1.5-flash",
        ]
