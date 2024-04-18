#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e
read -p "Enter the URL from email: " PRESIGNED_URL
echo ""
TARGET_FOLDER="."             # where all files should end up
mkdir -p ${TARGET_FOLDER}

echo "Downloading LICENSE and Acceptable Usage Policy"
wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE"
wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md"


echo "Downloading tokenizer"
wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/consolidated.00.pth"} -O ${TARGET_FOLDER}"/consolidated.00.pth"
wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/params.json"} -O ${TARGET_FOLDER}"/params.json"
