import os
import torch
import logging
import argparse
import numpy as np
import onnxruntime as ort
from datasets import load_dataset, Dataset
import onnxruntime as ort
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import LlamaConfig, AutoTokenizer
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model_input',
    type=str,
    help="Folder path of onnx model"
)
parser.add_argument(
    '--benchmark',
    action='store_true', \
    default=False,
    help="whether benchmark the model"
)
parser.add_argument(
    '--quantize',
    action='store_true', \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    '--model_output',
    type=str,
    default=None,
    help="Folder path of quantized onnx model "
)
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
)
parser.add_argument(
    '--bitwidth',
    default=8,
    type=int,
)
# parser.add_argument(
#     '--workspace',
#     type=str,
#     help="workspace to save intermediate files",
#     default="nc_workspace"
# )
parser.add_argument(
    '--quant_format',
    type=str,
    default='QOperator', 
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
parser.add_argument(
    '--pad_max',
    default=196,
    type=int,
)
parser.add_argument(
    "--tasks",
    nargs='+',
    default=["winogrande", "copa", "piqa", "rte", "hellaswag", "openbookqa", \
             "lambada_openai", "lambada_standard", "wikitext"],
    type=str,
    help="tasks list for accuracy validation"
)
parser.add_argument(
    "--dataset",
    nargs="?",
    default="NeelNanda/pile-10k",
    const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--smooth_quant_alpha",
    type=float,
    default=0.6
)
parser.add_argument(
    "--sampling_size",
    type=int, 
    default=8,
    help="sampling size of calibration"
)
parser.add_argument(
    '--dynamic',
    action='store_true', \
    default=False,
    help="whether to use dynamic quantization"
)
args = parser.parse_args()

# load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained(args.model_input)
config = LlamaConfig.from_pretrained(args.model_input)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # 创建一个空列表来存储文本
    texts = []

    # 遍历每个示例
    for example in examples['text']:
        texts.append(example)

    # 使用 tokenizer 对文本进行编码
    return tokenizer(texts, padding=True, truncation=True, max_length=512)


def eval_func(model):
    logger.info("开始评估 ONNX 模型...")
    # 加载模型
    session = ort.InferenceSession(model)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 加载并处理 GLUE 数据集（可以选择一个任务进行评估）
    dataset = load_dataset("glue", "sst2", split="validation")
    examples = dataset['sentence']
    labels = dataset['label']

    # 标记化
    inputs = tokenizer(examples, padding=True, truncation=True, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)

    # 创建 DataLoader
    eval_dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # 开始评估
    correct = 0
    total = 0
    for batch in eval_loader:
        input_ids, attention_mask, labels = batch
        ort_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
        }
        logits = session.run(None, ort_inputs)[0]
        predictions = np.argmax(logits, axis=-1)
        correct += (predictions == labels.numpy()).sum()
        total += labels.size(0)

    accuracy = correct / total
    print(f"SST-2 任务的准确率：{accuracy:.4f}")

class CalibDataloader(CalibrationDataReader):
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder='train', sampling_size=8):
        self.pad_max = pad_max
        self.batch_size = batch_size

        # 加载和处理 GLUE 数据集
        dataset_name = "glue"
        dataset_split = sub_folder
        data_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]

        examples = []
        datasets_dict = {}

        for data_task in data_tasks:
            print(f"数据集子集：{data_task}")
            try:
                dataset = load_dataset(dataset_name, data_task, split=dataset_split)
                datasets_dict[data_task] = dataset
                print(f"从 {data_task} 取列: {dataset.column_names}")
            except KeyError:
                print(f"未找到任务 {data_task} 或该任务没有 '{sub_folder}' 分割")
                continue

        # 获取每个数据集的最大长度
        max_length = max(len(dataset) for dataset in datasets_dict.values())

        # 遍历并处理每个数据集
        for i in range(max_length):
            for data_task, dataset in datasets_dict.items():
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

        # 将示例转换为 Dataset
        dataset = Dataset.from_dict({'text': examples})

        # 对数据集进行标记化
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.select(range(sampling_size))

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        # 准备校准所需的变量
        session = ort.InferenceSession(model_path)
        inputs_names = [input.name for input in session.get_inputs()]
        self.key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

        self.processed_data = iter(self.process_data(dataloader))

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)
    
    def process_data(self, dataloader):
        processed_data = []
        for (input_ids, attention_mask) in dataloader:
            ort_input = {}
            if not self.use_cache:
                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype('int64')
            else:
                num_attention_heads = config.num_key_value_heads
                embed_size_per_head = config.hidden_size // config.num_attention_heads
                shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                key_or_value = np.zeros(shape, dtype=np.float32)

                for key_value_input_name in self.key_value_input_names:
                    ort_input[key_value_input_name] = key_or_value

                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

            input_shape = ort_input["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
            ort_input["position_ids"] = position_ids.numpy()
            processed_data.append(ort_input)
        return processed_data


    def get_next(self) -> dict:
        return next(self.processed_data, None)
        # res = next(self.dataloader, None)
        # if res is not None:
        #     input_ids, attention_mask = res
        #     ort_input = {}
        #     if not self.use_cache:
        #         ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
        #         ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype('int64')
        #     else:
        #         num_attention_heads = config.num_key_value_heads
        #         embed_size_per_head = config.hidden_size // config.num_attention_heads
        #         shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
        #         key_or_value = np.zeros(shape, dtype=np.float32)

        #         for key_value_input_name in self.key_value_input_names:
        #             ort_input[key_value_input_name] = key_or_value

        #         ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
        #         ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

        #     input_shape = ort_input["input_ids"].shape
        #     position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        #     ort_input["position_ids"] = position_ids.numpy()
        #     return ort_input
        # else:
        #     return None


if __name__ == "__main__":
    # from neural_compressor import set_workspace
    # set_workspace(args.workspace)

    if args.benchmark:
        eval_func(args.model_input)

    if args.bitwidth == 8:
        bitwidthtype = QuantType.QUInt8
    elif args.bitwidth == 4:
        bitwidthtype = QuantType.QUInt4
    
    if args.quantize:
        model_name = "model.onnx"
        model_file = os.path.join(args.model_input, model_name)
        output_model_file = os.path.join(args.model_output, model_name)

        data_reader = CalibDataloader(model_file, pad_max=args.pad_max, batch_size=1)
        if args.dynamic:
            quantize_dynamic(model_file, 
                             output_model_file,
                             weight_type=bitwidthtype,
                            use_external_data_format=True,
                            # extra_options={"SmoothQuant": True,
                            #             "SmoothQuantAlpha": args.smooth_quant_alpha,
                            #             "OpTypesToExcludeOutputQuantization": ["MatMul"]}
                            )
        else:
            quantize_static(model_file,
                            output_model_file, 
                            calibration_data_reader=data_reader,
                            quant_format=args.quant_format,
                            activation_type=bitwidthtype,
                            weight_type=bitwidthtype,
                            op_types_to_quantize=["MatMul"],
                            use_external_data_format=True,
                            extra_options={"SmoothQuant": True,
                                        "SmoothQuantAlpha": args.smooth_quant_alpha,
                                        "OpTypesToExcludeOutputQuantization": ["MatMul"]})
        tokenizer.save_pretrained(args.model_output)
        config.save_pretrained(args.model_output)
