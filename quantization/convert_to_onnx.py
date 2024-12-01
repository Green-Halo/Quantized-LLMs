from onnxruntime_genai.models.builder import create_model

# Create ONNX model
# model_name = "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-8bit-glue_base"
model_name = None
input_folder = "/path/to/quantized_model"
output_folder = "/path/to/onnx_model"
precision = "fp32" # "fp32", "fp16", "int4"
execution_provider = "cuda" # "cpu", "cuda", "rocm"
cache_dir = "/path/to/cache_dir"
# cache_dir = os.path.join(".", "cache_dir")

extra_options = {
    "use_qdq": True,
    "use_8bits_moe": False,
    # "enable_cuda_graph": True if execution_provider == "cuda" else False,
    "tunable_op_enable": True if execution_provider == "rocm" else False,
    "tunable_op_tuning_enable": True if execution_provider == "rocm" else False,
}

create_model(model_name, input_folder, output_folder, precision, execution_provider, cache_dir, **extra_options)