import time, os
import torch
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

selected_bits = [
    2, 
    3, 
    4,  
    8   
]

# 设置下载配置，避免发生连接超时
download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 创建一个目录来保存预下载的数据集
datasets_dir = "/mnt/2T/Codes/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 加载 Wikitext-2 数据集
print("加载 Wikitext-2 数据集...")
wikitext2 = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=datasets_dir, download_config=download_config)

# 加载 PTB (Penn Treebank) 数据集
print("加载 PTB (Penn Treebank) 数据集...")
ptb = load_dataset("datasets/ptb.py", "penn_treebank", cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True)

# 加载 CommonSenseQA 数据集
print("加载 CommonSenseQA 数据集...")
commonsense_qa = load_dataset("commonsense_qa", cache_dir=datasets_dir, download_config=download_config)

# 加载 MMLU 数据集
print("加载 MMLU 数据集...")
mmlu = load_dataset("cais/mmlu", "all", split="test", cache_dir=datasets_dir, download_config=download_config)

# 加载 GSM8K 数据集
print("加载 GSM8K 数据集...")
gsm8k = load_dataset("gsm8k", "main", split="test", cache_dir=datasets_dir, download_config=download_config)

# 加载 HumanEval 数据集
print("加载 HumanEval 数据集...")
humaneval = load_dataset("openai_humaneval", cache_dir=datasets_dir, download_config=download_config)
humaneval_dataset = humaneval['test']

# 对每个数据集进行处理
examples = []

# 处理 Wikitext-2 数据集
for example in wikitext2['train']:
    examples.append(example["text"])

# 处理 PTB 数据集
for example in ptb['train']:
    examples.append(example["sentence"])

# 处理 CommonSenseQA 数据集
for example in commonsense_qa['train']:
    question = example["question"]
    choices = example["choices"]["text"]
    answer_key = example["answerKey"]
    answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    label = answer_mapping[answer_key]

    input_text = f"Please answer the following questions and select the correct option. \n\nQuestion: {question}\n\nOptions:\n"
    options = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        input_text += f"{options[i]}. {choice}\n"
    input_text += "\nAnswer: "
    input_text += answer_key
    examples.append(input_text)

# 处理 MMLU 数据集
for example in mmlu:
    question = example["question"]
    choices = example["choices"]
    label = example["answer"]  # 直接使用整数标签

    input_text = f"Please choose the correct answer based on the following questions. \n\nQuestion: {question}\n\nOptions:\n"
    options = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        input_text += f"{options[i]}. {choice}\n"
    input_text += "\nAnswer："
    examples.append(input_text)

# 处理 GSM8K 数据集
for example in gsm8k:
    question = example["question"]
    answer = example["answer"]
    examples.append(f"{question} {answer}")

# 处理 HumanEval 数据集
for example in humaneval_dataset:
    prompt = example["prompt"]
    solution = example["canonical_solution"]
    test = example["test"]
    examples.append(f"{prompt} {solution} {test}")

examples.append("auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.")

# 数据集Tokenization
examples_tokenized = [tokenizer(example, return_tensors="pt", truncation=True, max_length=512) for example in examples]
print("数据集编码完成.")

def batch_quantize(model, examples_tokenized, batch_size, device):
    total_batches = len(examples_tokenized) // batch_size + (1 if len(examples_tokenized) % batch_size != 0 else 0)
    # 将 examples_tokenized 分成多个批次
    for i in range(0, len(examples_tokenized), batch_size):
        current_batch = i // batch_size + 1
        print(f"Processing batch {current_batch}/{total_batches}")
        
        batch = examples_tokenized[i:i + batch_size]
        # batch = [{k: v.to(device) for k, v in example.items()} for example in batch]
        batch = [example.to(device) for example in batch]
        # 对当前批次进行量化
        model.quantize(batch)
        
        # 将该批次的数据从显存中删除
        del batch
        torch.cuda.empty_cache()
    
    # 保存模型到文件
    model.save_pretrained('quantized_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for bits in selected_bits:
    start_time = time.time()

    quantized_model_name = f"Llama-3.1-8B-Instruct-GPTQ-{bits}bit-gs128"
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)

    model = AutoGPTQForCausalLM.from_pretrained(base_model_name, quantize_config, device_map="auto", torch_dtype=torch.bfloat16)
    # model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    print(f"量化模型为{bits}-bit...")
    save_dir = "/mnt/2T/Codes/models/quantized_model/2"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, quantized_model_name)
    batch_size = 8
    batch_quantize(model, examples_tokenized, batch_size, device)
    tokenizer.save_pretrained(save_path)
    print(f"模型已保存到{save_path}.")

    time_taken = time.time() - start_time
    print(f"{bits}-bit quantization: {time_taken:.2f} seconds.")
    #with open(f"{quantized_model_name}_quantization_time.txt", "w") as file:
    #   file.write(f"{bits}-bit: {time_taken:.2f} seconds.\n")