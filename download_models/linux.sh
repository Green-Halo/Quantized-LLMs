#!/bin/bash

# Define the model folder path
modelFolder="tasks/onnx/onnx_models"

# Create the model folder if it doesn't exist
if [ ! -d "$modelFolder" ]; then
    mkdir -p "$modelFolder"
fi

# Model names and folders
declare -A modelNamesAndFolders=(
    ["GreenHalo/Llama-3.1-8B-Instruct-32bit-ONNX"]="$modelFolder/Llama-3.1-8B-Instruct-32bit-ONNX"
    ["GreenHalo/Llama-3.1-8B-Instruct-16bit-ONNX"]="$modelFolder/Llama-3.1-8B-Instruct-16bit-ONNX"
    ["GreenHalo/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX"]="$modelFolder/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX"
    ["GreenHalo/Llama-3.1-8B-Instruct-AWQ-4bit-ONNX"]="$modelFolder/Llama-3.1-8B-Instruct-AWQ-4bit-ONNX"
)

# Download models
for modelName in "${!modelNamesAndFolders[@]}"; do
    folderPath="${modelNamesAndFolders[$modelName]}"
    if [ ! -d "$folderPath" ]; then
        mkdir -p "$folderPath"
    fi
    huggingface-cli download --resume-download $modelName --local-dir "$folderPath"
done
