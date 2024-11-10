import time, os, torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

save_dir = "/mnt/2T/Codes/models/quantized_model"
os.makedirs(save_dir, exist_ok=True)

model_path = 'meta-llama/Llama-3.1-8B-Instruct'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# List of w_bit values
w_bits = [8, 4]
for w_bit in w_bits:
    quant_config["w_bit"] = w_bit
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path
        , trust_remote_code=True
        , device_map="auto"
        # , low_cpu_mem_usage=True
        # , use_cache=False
    )
    # Quantize
    start_time = time.time()
    model.quantize(tokenizer, quant_config=quant_config
            #    , export_compatible=True
               )
    end_time = time.time()
    
    # Save quantized model
    quantized_model_dir = f"{save_dir}/Llama-3.1-8B-Instruct-AWQ-{w_bit}bit"
    # model.save_quantized(quantized_model_dir)
    model.save_quantized(quantized_model_dir, safetensors=False)
    tokenizer.save_pretrained(quantized_model_dir)
    # model.pack() # makes the model CUDA compat
    # model.save_quantized(save_dir + "/Llama-3.1-8B-Instruct-AWQ-4bit", safetensors=False)
    # tokenizer.save_pretrained(save_dir + "/Llama-3.1-8B-Instruct-AWQ-4bit")

    # Export to ONNX
    # onnx_path = save_dir + "/Llama-3.1-8B-Instruct-AWQ-4bit/model.onnx"
    # convert_pytorch_to_onnx(model, onnx_path, opset_version=17)
    
    print(f'Model with w_bit={w_bit} is quantized and saved at "{quantized_model_dir}", time: {end_time - start_time:.2f} seconds')
