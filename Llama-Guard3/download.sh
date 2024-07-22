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

if [ $# -eq 0 ]; then
    echo "No model name provided. Please provide a model name."
    exit 1
fi

model_name="$1"

# Access the first argument
# If-else statement to check if the input is either "llama-guard-3" or "llama-guard-3-int8-hf"
if [ "$model_name" == "llama-guard-3" ]; then
echo "Downloading Llama Guard 3"
    model_name_folder=Meta-Llama-Guard-3-8B
    mkdir -p ${TARGET_FOLDER}/${model_name_folder}
    wget --continue ${PRESIGNED_URL/'*'/"Meta-Llama-Guard-3-8B/tokenizer.model"} -O ${TARGET_FOLDER}/${model_name_folder}/tokenizer.model
    wget --continue ${PRESIGNED_URL/'*'/"Meta-Llama-Guard-3-8B/params.json"} -O ${TARGET_FOLDER}/${model_name_folder}/params.json
    wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}/${model_name_folder}/LICENSE
    wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}/${model_name_folder}/USE_POLICY.md
    wget --continue ${PRESIGNED_URL/'*'/"Meta-Llama-Guard-3-8B/consolidated.00.pth"} -O ${TARGET_FOLDER}/${model_name_folder}/consolidated.00.pth
elif [ "$model_name" == "llama-guard-3-int8-hf" ]; then
    echo "Downloading Llama Guard 3 int8 hf"
    model_name_folder=Meta-Llama-Guard-3-8B-INT8-HF
    mkdir -p ${TARGET_FOLDER}/${model_name_folder}
    wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}/${model_name_folder}/LICENSE
    wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}/${model_name_folder}/USE_POLICY.md
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"config.json"} -O ${TARGET_FOLDER}/${model_name_folder}/config.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"generation_config.json"} -O ${TARGET_FOLDER}/${model_name_folder}/generation_config.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"model.safetensors.index.json"} -O ${TARGET_FOLDER}/${model_name_folder}/model.safetensors.index.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"special_tokens_map.json"} -O ${TARGET_FOLDER}/${model_name_folder}/special_tokens_map.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"tokenizer.json"} -O ${TARGET_FOLDER}/${model_name_folder}/tokenizer.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"tokenizer_config.json"} -O ${TARGET_FOLDER}/${model_name_folder}/tokenizer_config.json
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"model-00001-of-00002.safetensors"} -O ${TARGET_FOLDER}/${model_name_folder}/model-00001-of-00002.safetensors
    wget --continue ${PRESIGNED_URL/'*'/'Meta-Llama-Guard-3-8B-INT8-HF'/"model-00002-of-00002.safetensors"} -O ${TARGET_FOLDER}/${model_name_folder}/model-00002-of-00002.safetensors
else
    echo "Invalid model name. Please enter either 'llama-guard-3' or 'llama-guard-3-int8-hf'."
    exit 1
fi
