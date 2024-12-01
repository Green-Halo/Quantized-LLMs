# Define the model folder path
$modelFolder = "tasks/onnx/onnx_models"

# Create the model folder if it doesn't exist
if (-Not (Test-Path $modelFolder)) {
    New-Item -ItemType Directory -Path $modelFolder
}

# Model names and folders
$modelNamesAndFolders = @{
    "GreenHalo/Llama-3.1-8B-Instruct-32bit-ONNX" = "$modelFolder/Llama-3.1-8B-Instruct-32bit-ONNX";
    "GreenHalo/Llama-3.1-8B-Instruct-16bit-ONNX" = "$modelFolder/Llama-3.1-8B-Instruct-16bit-ONNX";
    "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX" = "$modelFolder/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX";
    "GreenHalo/Llama-3.1-8B-Instruct-AWQ-4bit-ONNX" = "$modelFolder/Llama-3.1-8B-Instruct-AWQ-4bit-ONNX";
    "GreenHalo/Qwen2.5-7B-Instruct-32bit-ONNX" = "$modelFolder/Qwen2.5-7B-Instruct-32bit-ONNX";
    "GreenHalo/Qwen2.5-7B-Instruct-16bit-ONNX" = "$modelFolder/Qwen2.5-7B-Instruct-16bit-ONNX";
    "GreenHalo/Qwen2.5-7B-Instruct-GPTQ-4bit-ONNX" = "$modelFolder/Qwen2.5-7B-Instruct-GPTQ-4bit-ONNX";
    "GreenHalo/Qwen2.5-7B-Instruct-AWQ-4bit-ONNX" = "$modelFolder/Qwen2.5-7B-Instruct-AWQ-4bit-ONNX";
}

# Download models
foreach ($modelName in $modelNamesAndFolders.Keys) {
    $folderPath = $modelNamesAndFolders[$modelName]
    if (-Not (Test-Path $folderPath)) {
        New-Item -ItemType Directory -Path $folderPath
    }
    & huggingface-cli download --resume-download $modelName --local-dir $folderPath
}
