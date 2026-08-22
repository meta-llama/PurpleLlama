# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from unittest.mock import MagicMock, patch

from llamafirewall.cli.configure import download_model


class TestConfigureModelDownload(unittest.TestCase):
    @patch("llamafirewall.cli.configure.login")
    @patch(
        "llamafirewall.cli.configure.get_token",
        return_value="existing-huggingface-token",
    )
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch.dict(os.environ, {"HF_HOME": "test-hf-home"})
    def test_download_model_uses_existing_huggingface_token(
        self,
        mock_model_from_pretrained: MagicMock,
        mock_tokenizer_from_pretrained: MagicMock,
        mock_get_token: MagicMock,
        mock_login: MagicMock,
    ) -> None:
        model_name = "meta-llama/test-model"

        result = download_model(model_name)

        self.assertTrue(result)
        mock_get_token.assert_called_once_with()
        mock_login.assert_not_called()
        mock_model_from_pretrained.assert_called_once_with(model_name)
        mock_tokenizer_from_pretrained.assert_called_once_with(
            model_name,
            fix_mistral_regex=True,
        )

    @patch("llamafirewall.cli.configure.login")
    @patch("llamafirewall.cli.configure.get_token", return_value=None)
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch.dict(os.environ, {"HF_HOME": "test-hf-home"})
    def test_download_model_logs_in_when_token_is_missing(
        self,
        mock_model_from_pretrained: MagicMock,
        mock_tokenizer_from_pretrained: MagicMock,
        mock_get_token: MagicMock,
        mock_login: MagicMock,
    ) -> None:
        model_name = "meta-llama/test-model"

        result = download_model(model_name)

        self.assertTrue(result)
        mock_get_token.assert_called_once_with()
        mock_login.assert_called_once_with()
        mock_model_from_pretrained.assert_called_once_with(model_name)
        mock_tokenizer_from_pretrained.assert_called_once_with(
            model_name,
            fix_mistral_regex=True,
        )


if __name__ == "__main__":
    unittest.main()