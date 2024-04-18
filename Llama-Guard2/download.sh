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

read -p "Enter the version to download: (llama-guard-2,llama-guard)" VERSION

if [[ $VERSION == "" ]]; then
    VERSION="llama-guard,llama-guard-2"
fi

echo "Downloading LICENSE and Acceptable Usage Policy"
wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE"
wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md"

for v in ${VERSION//,/ }
do
    if [[ $v == "llama-guard" ]]; then
        MODEL_PATH="llama-guard"
        mkdir -p ${MODEL_PATH}

        echo "Downloading tokenizer"
        wget --continue ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}/${MODEL_PATH}"/tokenizer.model"
        mkdir -p ${TARGET_FOLDER}"/llama-guard"
        wget --continue ${PRESIGNED_URL/'*'/"consolidated.00.pth"} -O ${TARGET_FOLDER}/${MODEL_PATH}"/consolidated.00.pth"
        wget --continue ${PRESIGNED_URL/'*'/"params.json"} -O ${TARGET_FOLDER}/${MODEL_PATH}/"/params.json"

    elif [[ $v == "llama-guard-2" ]]; then
        MODEL_PATH="llama-guard-2"
        mkdir -p ${MODEL_PATH}

        echo "Downloading tokenizer"
        wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/tokenizer.model"} -O ${TARGET_FOLDER}/${MODEL_PATH}"/tokenizer.model"
        mkdir -p ${TARGET_FOLDER}"/llama-guard"
        wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/consolidated.00.pth"} -O ${TARGET_FOLDER}/${MODEL_PATH}"/consolidated.00.pth"
        wget --continue ${PRESIGNED_URL/'*'/"llama-guard-2/params.json"} -O ${TARGET_FOLDER}/${MODEL_PATH}/"/params.json"
    fi
done
