import torch, optimum
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建示例输入
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")

# 使用 dynamo_export 导出
torch.onnx.dynamo_export(
    model,
    args=(inputs["input_ids"],),
    f="Llama-3.1-8B-Instruct-ONNX/model.onnx",
    opset_version=22
)
