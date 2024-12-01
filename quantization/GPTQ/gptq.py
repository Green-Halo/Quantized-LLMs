import time, os, gc, torch
from transformers import AutoTokenizer, onnx
from datasets import load_dataset, DownloadConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# config hyperparameters
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "glue"
dataset_split = "train"
data_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]
selected_bits = [8, 4]
batch_size = 12000
tokenized_length = 64

# set save path
save_dir = "/path/to/save/quantized_model"
os.makedirs(save_dir, exist_ok=True)
datasets_dir = "/path/to/datasets"
os.makedirs(datasets_dir, exist_ok=True)

# load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

examples = []

datasets = {}
download_config = DownloadConfig(num_proc=8, max_retries=5)
for data_task in data_tasks:
    try:
        dataset = load_dataset(dataset_name, data_task, split=dataset_split, cache_dir=datasets_dir, download_config=download_config)
        datasets[data_task] = dataset
    except KeyError:
        print(f"Error: Task {data_task} not found or the task does not have {dataset_split} split")
        continue

# get the maximum length of each dataset
max_length = max(len(dataset) for dataset in datasets.values())

# process each dataset
for i in range(max_length):
    for data_task, dataset in datasets.items():
        if i < len(dataset):
            example = dataset[i]
            if data_task == "mrpc":
                examples.append(f"{example['sentence1']} {example['sentence2']}")
            elif data_task == "qqp":
                examples.append(f"{example['question1']} {example['question2']}")
            elif data_task == "mnli":
                examples.append(f"{example['premise']} {example['hypothesis']}")
            elif data_task == "qnli":
                examples.append(f"{example['question']} {example['sentence']}")
            elif data_task in ["rte", "wnli"]:
                examples.append(f"{example['sentence1']} {example['sentence2']}")
            elif data_task == "sst2":
                examples.append(example["sentence"])

# Tokenization
examples_tokenized = [
    tokenizer(example, return_tensors="pt", truncation=True, 
              max_length=tokenized_length, padding=True)
    for example in examples[:batch_size]  
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for bits in selected_bits:
    quantized_model_name = f"Llama-3.1-8B-Instruct-GPTQ-{bits}bit"

    # configure quantization
    quantize_config = BaseQuantizeConfig(bits=bits, desc_act=True)
    
    # load model and quantize
    model = AutoGPTQForCausalLM.from_pretrained(base_model_name, quantize_config, device_map="auto", torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.quantize(examples_tokenized)

    # save as safetensors
    save_path = os.path.join(save_dir, quantized_model_name)
    model.save_quantized(save_path, use_safetensors = True)
    tokenizer.save_pretrained(save_path)
    # save as torchscript
    model.save_quantized(save_path, use_safetensors = False)
    model.config.save_pretrained(save_path)
    
    # release memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
