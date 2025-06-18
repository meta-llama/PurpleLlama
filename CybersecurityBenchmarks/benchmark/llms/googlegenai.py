# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from typing_extensions import override

from ..benchmark_utils import get_mime_type, image_to_b64
from .llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
    UnretriableQueryException,
)


class GOOGLEGENAI(LLM):
    """Accessing Google Generative AI"""

    BASE_API_URL: str = "https://generativelanguage.googleapis.com"

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}], "role": "user"}]}
        response = requests.post(url, headers=headers, data=json.dumps(data))
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

        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
                "role": "model",
            },
            "generationConfig": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
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

        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
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
            "generationConfig": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
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
        mime_type = get_mime_type(file_path)
        num_bytes = os.path.getsize(file_path)
        display_name = "MEDIA_FILE"
        # Initial resumable request defining metadata
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        data = json.dumps({"file": {"display_name": display_name}})

        response = requests.post(
            f"{self.BASE_API_URL}/upload/v1beta/files?key={self.api_key}",
            headers=headers,
            data=data,
        )
        response.raise_for_status()

        upload_url = response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise Exception(f"Failed to upload file ({file_path})")

        # Upload the actual bytes
        with open(file_path, "rb") as f:
            file_data = f.read()
        headers = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        response = requests.post(upload_url, headers=headers, data=file_data)
        file_info = response.json()
        file_uri = file_info.get("file", {}).get("uri")

        if not file_uri:
            raise Exception("Failed to upload file.")

        return file_uri

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
        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        headers = {"Content-Type": "application/json"}

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

        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Delete files after query
        for file_uri in file_uris:
            delete_url = f"{file_uri}?key={self.api_key}"
            delete_response = requests.delete(delete_url)
            if not delete_response.ok:
                print(f"[WARNING] Failed to delete file: {delete_response.text}")

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
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
