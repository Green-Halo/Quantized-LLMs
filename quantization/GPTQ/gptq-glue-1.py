import time, os, gc, torch
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM, onnx
from datasets import load_dataset, DownloadConfig
import LlamaOnnxConfig

# 配置超参数
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "glue"
dataset_split = "train"
data_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]
selected_bits = [
    # 8,
    4
    ]
batch_size = 32768
tokenized_length = 64

# 设置保存路径
save_dir = "/mnt/2T/Codes/models/quantized_model"
os.makedirs(save_dir, exist_ok=True)
datasets_dir = "/mnt/2T/Codes/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("一、数据准备阶段")
# 创建合并样本列表
examples = []

# 加载并处理每个任务的数据集
print("1) 加载" + dataset_name.upper() + "数据集")
datasets = {}
download_config = DownloadConfig(num_proc = 8, max_retries = 5)
start_time = time.time()
for data_task in data_tasks:
    print(f"数据集子集：{data_task}")
    try:
        dataset = load_dataset(dataset_name, data_task, split=dataset_split, cache_dir=datasets_dir, download_config=download_config)
        datasets[data_task] = dataset
        print(f"从{data_task}取列: {dataset.column_names}")
    except KeyError:
        print(f"未找到任务 {data_task} 或该任务没有 'train' 分割")
        continue
end_time = time.time()
print(f"错误：数据集加载完成. 耗时: {end_time - start_time:.2f} 秒")

print("2) 开始合并数据集")
start_time = time.time()
# 获取每个数据集的最大长度
max_length = max(len(dataset) for dataset in datasets.values())

# 遍历并处理每个数据集
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
end_time = time.time()
print(f"数据集合并完成. 耗时: {end_time - start_time:.2f} 秒")

print("3) 开始编码数据集")
start_time = time.time()

# 对合并的数据进行Tokenization
tokenized_examples = [
    tokenizer(example, return_tensors="pt", truncation=True, 
              max_length=tokenized_length, padding=True)
    for example in examples[:batch_size]  
    # for example in examples
]
end_time = time.time()
print(f"数据集编码完成. 耗时: {end_time - start_time:.2f} 秒")

print("二、量化阶段")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, bits in enumerate(selected_bits, start=4):
    print(f"4) 开始{bits}-bit量化")
    quantized_model_name = f"Llama-3.1-8B-Instruct-GPTQ-{bits}bit-glue_base"
    start_time_all = start_time = time.time()
    print(f"{i}.1) 执行量化")
    # 配置GPTQ
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=tokenized_examples,
        tokenizer=tokenizer
    )
    # 加载模型并进行量化
    with device:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            # device_map="auto",  
            # trust_remote_code=True,
            # low_cpu_mem_usage=True,
            quantization_config=gptq_config,
        )
    end_time = time.time()
    print(f"量化结束. 耗时: {end_time - start_time:.2f} 秒")
    
    print(f"{i}.2) 保存模型")
    start_time = time.time()
    # 移除 dataset 避免序列化问题
    model.config.quantization_config.dataset = None
    # 保存为SafeTensors
    save_path = os.path.join(save_dir, quantized_model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # 保存为PyTorch
    torch_model_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(model.state_dict(), torch_model_path)
    model.config.save_pretrained(save_path)
        # torch.save(model.config, os.path.join(save_path, "config.json"))
        
    # 保存为ONNX
    try:
        onnx_config = LlamaOnnxConfig(model.config)
        onnx_model_path = os.path.join(save_path, "model.onnx")
        onnx.export(preprocessor=tokenizer, model=model, config=onnx_config, opset=17, output=onnx_model_path, device=device)
    except Exception as e:
        print(f"保存ONNX时出错: {e}")
        
    end_time = time.time()
    print(f"模型保存完成. 耗时: {end_time - start_time:.2f} 秒")

    print(f"{bits}-bit 量化完成. 耗时: {time.time() - start_time_all:.2f} 秒")
    
    # 清理模型并释放CUDA内存
    del model
    torch.cuda.empty_cache()
    gc.collect()