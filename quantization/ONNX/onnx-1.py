from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_dir = "/mnt/2T/Codes/models/quantized_model/llama3.1-8B-instruct-onnx"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "llama3.1-8B-instruct.onnx")
quantized_model_path = os.path.join(save_dir, "llama3.1-8B-instruct-quantized.onnx")

# 确保模型处于评估模式
model.eval()

# 创建示例输入
input_text = "This is a sample input."
inputs = tokenizer(input_text, return_tensors="pt")

# 获取模型输入的名称和形状
input_names = ["input_ids", "attention_mask"]
output_names = ["logits", "past_key_values"]

# 转换为ONNX格式
torch.onnx.export(
    model, 
    (inputs["input_ids"], inputs["attention_mask"]), 
    model_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
        "past_key_values": {0: "batch_size", 1: "sequence_length"}
    }
)

# onnx_model = torch.onnx.dynamo_export(model, (inputs["input_ids"], inputs["attention_mask"]))
# onnx.save(onnx_model, model_path)

# 加载ONNX模型
onnx_model = onnx.load(model_path)

# 动态量化模型
# session_options = ort.SessionOptions()
# session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# session = ort.InferenceSession(model_path, session_options, providers=['MIGraphXExecutionProvider'])

quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8, execution_providers=['MIGraphXExecutionProvider'])

print("量化完成，模型已保存为", quantized_model_path)