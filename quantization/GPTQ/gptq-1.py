import time, os, gc, torch
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

selected_bits = [8, 4]  # 从较高比特位开始

# 设置下载配置，避免连接超时
download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 创建目录以保存数据集
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
tokenized_length = 64

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

# 将数据集编码为 token
examples_tokenized = [
    tokenizer(example, return_tensors="pt", truncation=True, max_length=tokenized_length, padding=True) 
    for example in examples[:batch_size]
]

print("数据集编码完成.")

# 清理不必要的部分，确保可序列化
# examples_serializable = [{'input_ids': example['input_ids'][0].tolist()} for example in examples_tokenized]


# 执行量化并保存模型
for bits in selected_bits:
    start_time = time.time()
    
    quantized_model_name = f"Llama-3.1-8B-Instruct-GPTQ-{bits}bit"
    gptq_config = GPTQConfig(
        bits=bits,  
        dataset=examples_tokenized,
        tokenizer=tokenizer,
        # group_size=128,
        desc_act=True  
    )

    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",  
            quantization_config=gptq_config,
            torch_dtype=torch.bfloat16
        )

    # 移除 dataset 避免序列化问题
    model.quantize_config.dataset = None
    
    save_dir = "/mnt/2T/Codes/models/quantized_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, quantized_model_name)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    time_taken = time.time() - start_time
    print(f"{bits}-bit quantization: {time_taken:.2f} seconds.")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
