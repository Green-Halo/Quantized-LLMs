import time, os
import torch
from transformers import default_data_collator, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, DownloadConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

selected_bits = [8, 4]  # 从较高比特位开始

# 设置下载配置，避免发生连接超时
download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 创建一个目录来保存预下载的数据集
datasets_dir = "/mnt/2T/Codes/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 加载数据集
datasets = {
    "wikitext2": load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=datasets_dir, download_config=download_config),
    "ptb": load_dataset("datasets/ptb.py", "penn_treebank", split="train", cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True),
    "commonsense_qa": load_dataset("commonsense_qa", split="train", cache_dir=datasets_dir, download_config=download_config),
    "mmlu": load_dataset("cais/mmlu", "all", split="test", cache_dir=datasets_dir, download_config=download_config),
    "gsm8k": load_dataset("gsm8k", "main", split="test", cache_dir=datasets_dir, download_config=download_config),
    "humaneval": load_dataset("openai_humaneval", split="test", cache_dir=datasets_dir, download_config=download_config)
}

# 数据处理
examples = []
batch_size = 4096
tokenized_length = 32

# 获取每个数据集的最大长度
max_length = max(len(dataset) for dataset in datasets.values())

# 遍历并处理每个数据集
for i in range(max_length):
    for name, dataset in datasets.items():
        if i < len(dataset):
            example = dataset[i]
            if name == "wikitext2":
                examples.append(example["text"])
            elif name == "ptb":
                examples.append(example["sentence"])
            elif name == "commonsense_qa":
                question = example["question"]
                choices = example["choices"]["text"]
                answer_key = example["answerKey"]
                input_text = f"Question: {question}\nOptions: {', '.join(choices)}\nAnswer: {answer_key}"
                examples.append(input_text)
            elif name == "mmlu":
                question = example["question"]
                choices = example["choices"]
                input_text = f"Question: {question}\nOptions: {', '.join(choices)}"
                examples.append(input_text)
            elif name == "gsm8k":
                question = example["question"]
                answer = example["answer"]
                examples.append(f"{question} {answer}")
            elif name == "humaneval":
                prompt = example["prompt"]
                solution = example["canonical_solution"]
                test = example["test"]
                examples.append(f"{prompt} {solution} {test}")

# Tokenize the examples
examples_tokenized = [
    tokenizer(example, return_tensors="pt", truncation=True, max_length=tokenized_length, padding=True) 
    for example in examples[:batch_size]
]
# print(examples[:5])
# print("************************************************************************************************")
# print(examples_tokenized[:5])
# print(type(examples_tokenized))
# print(type(examples_tokenized[0]))
# #暂停
# input("Press Enter to continue...")

# 将列表转换为包含字典的列表
# examples_tokenized_dict = [{"input_ids": tensor} for tensor in examples_tokenized]

# # 将张量移动到指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# examples_tokenized_dict = [
#     {"input_ids": tensor.to(device)} for tensor in examples_tokenized
# ]

print("数据集编码完成.")


for bits in selected_bits:
    start_time = time.time()

    quantized_model_name = f"Llama-3.1-8B-Instruct-GPTQ-{bits}bit"
    quantize_config = BaseQuantizeConfig(bits=bits, 
                                        #  group_size=None, 
                                         desc_act=True)

    model = AutoGPTQForCausalLM.from_pretrained(base_model_name, quantize_config, device_map="auto", torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    print(f"量化模型为{bits}-bit...")
    model.quantize(examples_tokenized)
    save_dir = "/mnt/2T/Codes/models/quantized_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, quantized_model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"模型已保存到{save_path}.")

    time_taken = time.time() - start_time
    print(f"{bits}-bit quantization: {time_taken:.2f} seconds.")
    #with open(f"{quantized_model_name}_quantization_time.txt", "w") as file:
    #   file.write(f"{bits}-bit: {time_taken:.2f} seconds.\n")