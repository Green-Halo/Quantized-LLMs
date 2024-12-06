from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import shutil
import time
import warnings
import psutil
import onnxruntime_genai as og
import numpy as np
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict


warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pyJoules.energy_meter import measure_energy
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyContext
import pynvml
import pyRAPL

pyRAPL.setup()
meter = pyRAPL.Measurement("Model_inference")


# Init pynvml
def init_pynvml():
    try:
        pynvml.nvmlInit()
        output.console_log("pynvml Successfully Initialized")
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except pynvml.NVMLError as e:
        output.console_log(f"Failed to initialize pynvml: {str(e)}")
        return False


# Init pyJoules
def init_gpu_meter():
    try:
        gpu_meter = NvidiaGPUDomain(0)
        output.console_log("pyJoules gpu_meter Successfully Initialized")
        return gpu_meter
    except Exception as e:
        output.console_log(f"Failed to initialize pyJoules gpu_meter: {str(e)}")
        return None


# Init gpu_available
gpu_available = init_pynvml()
gpu_meter = init_gpu_meter() if gpu_available else None


# Init psutil
def init_psutil():
    try:
        psutil.cpu_percent(interval=None)
        output.console_log("psutil Successfully Initialized")
        return True
    except Exception as e:
        output.console_log(f"Failed to initialize psutil: {str(e)}")
        return False


psutil_available = init_psutil()


class LLAMA:
    def __init__(self, model_path, max_length):
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.max_length = max_length
        self.batch_size_for_cuda_graph = 1
        self.params = og.GeneratorParams(self.model)
        self.search_options = {
            name: getattr(self, name)
            for name in [
                "do_sample",
                "max_length",
                "min_length",
                "top_p",
                "top_k",
                "temperature",
                "repetition_penalty",
            ]
            if hasattr(self, name)
        }
        self.params.set_search_options(**self.search_options)

    def run(self, chat_template, prompt):
        if chat_template.count("{") != 1 or chat_template.count("}") != 1:
            print(
                "Error, chat template must have exactly one pair of curly braces, e.g., 'Prompt: {input}'"
            )
            exit(1)
        prompt_formatted = chat_template.format(input=prompt)

        input_tokens = self.tokenizer.encode(prompt_formatted)

        # Set the batch size for the CUDA graph to the number of prompts if the user didn't specify a batch size
        self.params.try_graph_capture_with_max_batch_size(1)
        self.params.input_ids = input_tokens

        # output_tokens = model.generate(params)

        # for i in range(len(prompts)):
        #     # print(f'Prompt #{i}: {prompts[i]}')
        #     # print()
        #     print(tokenizer.decode(output_tokens[i]))
        #     print()

        generator = og.Generator(self.model, self.params)
        output = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output += self.tokenizer.decode(new_token)
        return output.strip()
        # file.write(output+'\n')


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name: str = "new_runner_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path: Path = ROOT_DIR / "experiments"

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type: OperationType = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms: int = 5000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria

    def __init__(self):
        print("Initializing the RunnerConfig...")
        output.console_log("Initializing the RunnerConfig...")
        EventSubscriptionController.subscribe_to_multiple_events(
            [
                (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
                (RunnerEvents.BEFORE_RUN, self.before_run),
                (RunnerEvents.START_RUN, self.start_run),
                (RunnerEvents.START_MEASUREMENT, self.start_measurement),
                (RunnerEvents.INTERACT, self.interact),
                (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
                (RunnerEvents.STOP_RUN, self.stop_run),
                (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
                (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
            ]
        )
        self.run_table_model = None
        self.task_name = ["SA", "SPS", "NLI"]
        self.quantization_type = ["awq-4-bit", "gptq-4-bit", "16-bit", "32-bit"]
        self.tasks = {}
        self.sample_size = 300
        self.template = {}
        self.datas = defaultdict(list)

    def create_run_table_model(self) -> RunTableModel:
        output.console_log("Creating run table model...")
        factor1 = FactorModel("quantization_type", self.quantization_type)
        factor2 = FactorModel("task_name", self.task_name)

        self.run_table_model = RunTableModel(
            factors=[factor1, factor2],
            repetitions=5,
            data_columns=[
                "Inference Time",
                "GPU Energy",
                "CPU Energy",
                "Memory Energy",
                "Accuracy",
                "GPU Busy Time",
                "CPU Busy Time",
                "Memory Usage",
            ],
        )
        output.console_log("Run table model created with factors.")
        # if there is a folder named experiments in the directory with this file, delete this folder recursively
        if self.results_output_path.exists():
            output.console_log(
                f"Deleting the existing folder: {self.results_output_path}"
            )
            shutil.rmtree(self.results_output_path)
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("Starting the experiment and loading the dataset...")

        # Load datasets
        imdb = (
            load_dataset("imdb", split="test")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )
        sst2 = (
            load_dataset("glue", "sst2", split="validation")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )
        mrpc = (
            load_dataset("glue", "mrpc", split="validation")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )
        qqp = (
            load_dataset("glue", "qqp", split="validation")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )
        wnli = (
            load_dataset("glue", "wnli", split="train")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )
        rte = (
            load_dataset("glue", "rte", split="train")
            .shuffle(seed=42)
            .select(range(self.sample_size))
        )

        # Define self.datasets
        self.datasets = {
            "imdb": imdb,
            "sst2": sst2,
            "mrpc": mrpc,
            "qqp": qqp,
            "wnli": wnli,
            "rte": rte,
        }

        # Group dataset names under tasks
        self.tasks = {
            "SA": ["imdb", "sst2"],
            "SPS": ["mrpc", "qqp"],
            "NLI": ["wnli", "rte"],
        }

        def format_example(data_task, example):
            if data_task == "mrpc":
                return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
            elif data_task == "qqp":
                return f"sentence1: '{example['question1']}' & sentence2: '{example['question2']}'"
            elif data_task == "rte":
                return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
            elif data_task == "wnli":
                return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
            elif data_task == "sst2":
                return f"{example['sentence']}"
            elif data_task == "imdb":
                return f"{example['text']}"
            else:
                return None

        # Initialize data storage
        from collections import defaultdict

        self.datas = defaultdict(list)

        # Process and format each dataset
        for task_name, dataset_names in self.tasks.items():
            for data_task in dataset_names:
                dataset = self.datasets[data_task]
                for example in dataset:
                    formatted_text = format_example(data_task, example)
                    if formatted_text:
                        self.datas[task_name].append(
                            {"text": formatted_text, "label": example["label"]}
                        )

        # Initialize template storage
        self.template = {}

        # Set up templates for each task
        for task_name in self.tasks.keys():
            if task_name == "SA":
                self.template[
                    task_name
                ] = """
                Prompt: "{input}"
                Instruct: Answer as briefly as possible. Please determine the sentiment of the above sentence in Prompt. The options are: 0 if the sentence is negative, 1 if the sentence is positive. No analyses or explanations. Only respond with 0 or 1.
                """
            elif task_name == "SPS":
                self.template[
                    task_name
                ] = """
                Prompt: "{input}"
                Instruct: Answer as briefly as possible. Please determine whether the two sentences above in Prompt are equivalent, and return 1 if they are, or 0 if they are not. No analyses or explanations. Only respond with 0 or 1.
                """
            elif task_name == "NLI":
                self.template[
                    task_name
                ] = """
                Prompt: "{input}"
                Instruct: Answer as briefly as possible. From the above two sentences in Prompt, please determine whether the sentences are entailments. The options are: 0 if the sentences are entailments, 1 if they are not. No analyses or explanations. Only respond with 0 or 1.
                """

        # Output the total number of samples loaded
        total_samples = sum(len(dataset) for dataset in self.datasets.values())
        output.console_log(f"Datasets loaded: {total_samples} samples.")

    def before_run(self) -> None:
        output.console_log("Preparing for the next run...")

    def start_run(self, context: RunnerContext) -> None:
        quantization_type = context.run_variation["quantization_type"]
        task_name = context.run_variation["task_name"]
        model = self.load_model(quantization_type)
        data = self.datas[task_name]
        chat_template = self.template[task_name]

        if data is not None:
            results = self.run_experiment(
                model, data, chat_template, quantization_type, task_name
            )
            if results is not None:
                (
                    inference_time,
                    gpu_energy,
                    cpu_energy,
                    accuracy,
                    gpu_busy_time,
                    cpu_busy_time,
                    memory_usage,
                ) = results
            else:
                (
                    inference_time,
                    gpu_energy,
                    cpu_energy,
                    accuracy,
                    gpu_busy_time,
                    cpu_busy_time,
                    memory_usage,
                ) = (0, 0, 0, 0, 0, 0, 0)
        else:
            (
                inference_time,
                gpu_energy,
                cpu_energy,
                accuracy,
                gpu_busy_time,
                cpu_busy_time,
                memory_usage,
            ) = (0, 0, 0, 0, 0, 0, 0)

        context.run_data = {
            "Inference Time": inference_time,
            "GPU Energy": gpu_energy,
            "CPU Energy": cpu_energy,
            "Accuracy": accuracy,
            "GPU Busy Time": gpu_busy_time,
            "CPU Busy Time": cpu_busy_time,
            "Memory Usage": memory_usage,
        }
        output.console_log(
            f"Run completed for {quantization_type} with task: {task_name} with data: {context.run_data}"
        )

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log(
            f"Starting measurements for {context.run_variation['quantization_type']} model."
        )

    def interact(self, context: RunnerContext) -> None:
        output.console_log("Interacting with the running system...")

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Stopping measurements...")

    def stop_run(self, context: RunnerContext) -> None:
        output.console_log("Run completed, cooling down...")
        time.sleep(self.time_between_runs_in_ms / 1000)
        output.console_log("Cool down completed.")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        output.console_log(
            f"Populating run data for {context.run_variation['quantization_type']}."
        )
        return context.run_data

    def after_experiment(self) -> None:
        output.console_log("Experiment finished.")
        if gpu_available:
            try:
                pynvml.nvmlShutdown()
                output.console_log("pynvml shutdown successfully.")
            except pynvml.NVMLError as e:
                output.console_log(f"Failed to shutdown pynvml: {str(e)}")

    def load_model(self, quantization_type):
        if not hasattr(self, "model_cache"):
            self.model_cache = {}
        output.console_log(f"Loading model for quantization type: {quantization_type}")

        if quantization_type == "32-bit":
            model = LLAMA(
                "tasks/onnx/onnx_models/Llama-3.1-8B-Instruct-32bit-ONNX", 512
            )
        elif quantization_type == "16-bit":
            model = LLAMA(
                "tasks/onnx/onnx_models/Llama-3.1-8B-Instruct-16bit-ONNX", 512
            )
        elif quantization_type == "awq-4-bit":
            model = LLAMA(
                "tasks/onnx/onnx_models/Llama-3.1-8B-Instruct-AWQ-4bit-ONNX", 512
            )
        elif quantization_type == "gptq-4-bit":
            model = LLAMA(
                "tasks/onnx/onnx_models/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX",
                512,
            )

        return model

    def run_experiment(
        self, model, data, chat_template, quantization_type, task_name, batch_size=1
    ):
        total_gpu_energy = 0
        total_cpu_energy = 0
        total_inference_time = 0
        correct_predictions = 0

        total_gpu_busy_time = 0
        total_cpu_busy_time = 0
        total_memory_usage = 0
        num_batches = 0

        if gpu_available and gpu_meter:
            device = NvidiaGPUDomain(0)
            labels = []
            predictions = []
            with EnergyContext(domains=[device], start_tag="start") as ctx:
                for item in data:
                    prompt = item["text"]
                    label = item["label"]
                    labels.append(label)

                    meter.begin()
                    start_time = time.perf_counter()
                    cpu_start_time = time.process_time()

                    # run model
                    output = model.run(chat_template, prompt)

                    end_time = time.perf_counter()
                    cpu_end_time = time.process_time()

                    meter.end()

                    inference_time = end_time - start_time
                    total_inference_time += inference_time
                    cpu_busy_time = cpu_end_time - cpu_start_time
                    total_cpu_busy_time += cpu_busy_time

                    cpu_energy = meter.result.pkg[0]
                    total_cpu_energy += cpu_energy

                    ctx.record(tag="inference_step")
                    output = output[:200]
                    pred_label = -1
                    for ch in output:
                        if ch == "0":
                            pred_label = 0
                            break
                        elif ch == "1":
                            pred_label = 1
                            break
                        elif ch == "2":
                            pred_label = 2
                            break
                        elif ch in [" ", "\n", "\t"]:
                            continue
                        else:
                            continue
                    predictions.append(pred_label)

                labels = labels[: len(predictions)]
                labels = np.array(labels)
                predictions = np.array(predictions)
                correct_predictions = predictions == labels
                accuracy = np.mean(correct_predictions)

                gpu_busy_time = inference_time
                total_gpu_busy_time += gpu_busy_time

                memory_usage = self.get_memory_usage()
                total_memory_usage += memory_usage
                num_batches += 1

            energy_data = ctx.get_trace()
            for measurement in energy_data:
                if measurement.tag == "inference_step":
                    energy = measurement.energy
                    if "nvidia_gpu_0" in energy:
                        total_gpu_energy += energy["nvidia_gpu_0"] / 1_000_000

        average_memory_usage = (
            total_memory_usage / num_batches if num_batches > 0 else 0
        )

        return (
            total_inference_time,  # / valid_text_count,
            total_gpu_energy,
            total_cpu_energy,
            accuracy,
            total_gpu_busy_time,  # / valid_text_count,
            total_cpu_busy_time,  # / valid_text_count,
            average_memory_usage,
        )

    def get_memory_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / (1024**2)

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None
