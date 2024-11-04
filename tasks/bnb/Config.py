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

import subprocess
import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from auto_gptq import exllama_set_max_input_length
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pyJoules.energy_meter import measure_energy
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyContext
import pynvml


# Init pynvml
def init_pynvml():
    try:
        pynvml.nvmlInit()
        output.console_log("pynvml 初始化成功")
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError as e:
        output.console_log(f"Failed to initialize pynvml: {str(e)}")
        raise RuntimeError(f"pynvml 初始化失败: {str(e)}")

# Init pyJoules
def init_gpu_meter():
    try:
        gpu_meter = NvidiaGPUDomain(0)
        output.console_log("pyJoules gpu_meter 初始化成功")
        return gpu_meter
    except Exception as e:
        output.console_log(f"Failed to initialize pyJoules gpu_meter: {str(e)}")
        raise RuntimeError(f"pyJoules gpu_meter 初始化失败: {str(e)}")



init_pynvml() 
gpu_meter = init_gpu_meter() 
class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "new_runner_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path            = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 1000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
 
    def __init__(self):
        print("Initializing the RunnerConfig...")
        output.console_log("Initializing the RunnerConfig...")
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        self.run_table_model = None


    def create_run_table_model(self) -> RunTableModel:
        output.console_log("Creating run table model...")
        

        factor1 = FactorModel(
            "quantization_type",  
            [
                '32-bit', 
                '8-bit',  
                '4-bit'    
            ]
        )
        

        factor2 = FactorModel(
            "task_name", 
            [
                "SA",  
                "SPS",  
                "NLI"   
            ]
        )

 
        self.run_table_model = RunTableModel(
            factors=[
                factor1,  
                factor2   
            ],
            repetitions=10,  # Number of repetitions ****
            data_columns=[
                'Inference Time',
                'GPU Energy',
                'Accuracy',
                'GPU Utilization',
                'Memory Usage'
            ]
        )

        output.console_log("Run table model created with factors.")
        return self.run_table_model


    def before_experiment(self) -> None:
        output.console_log("Starting the experiment and loading the dataset...")

        imdb = load_dataset("imdb", split='test')
        qqp= load_dataset("glue", "qqp", split='validation')
        sst2 = load_dataset("glue", "sst2", split='train')
        mrpc = load_dataset("glue", "mrpc", split='train')
        wnli = load_dataset("glue", "wnli", split='train')
        rte = load_dataset("glue", "rte", split='train') 

        imdb_subset = imdb.train_test_split(test_size=0.1)['test']
        sst2_subset = sst2.train_test_split(test_size=0.1)['test'] 

        mrpc_subset = mrpc.train_test_split(test_size=0.1)['test']
        qqp_subset = qqp.train_test_split(test_size=0.01)['test']   
         
        wnli_subset = wnli.train_test_split(test_size=0.1)['test']
        rte_subset = rte.train_test_split(test_size=0.1)['test']


        self.datasets = {
            # Sentiment Analysis 
            "SA": [
                imdb_subset,
                sst2_subset,
                #None  
            ],
    
            # Sentence Pair Semantic Similarity 
            "SPS": [
                mrpc_subset,
                qqp_subset
                #None  
            ],
            # Natural Language Inference 
            "NLI": [
                wnli_subset,
                rte_subset
                #None 
            ]
        }

        # Check samples
        for task_name, datasets in self.datasets.items():
            for idx, dataset in enumerate(datasets):
                if dataset is None:
                    continue  
                output.console_log(f"Checking dataset {task_name} - part {idx + 1}")
                for sample_idx, sample in enumerate(dataset):
                    if sample_idx >= 2:  
                        break
                    output.console_log(f"Sample {sample_idx}: {sample}")
                    if not isinstance(sample, dict):
                        output.console_log(f"Error: Expected dict, got {type(sample)}. Sample content: {sample}")

        output.console_log(f"Datasets loaded: {sum(len(dataset) for task_datasets in self.datasets.values() for dataset in task_datasets if dataset is not None)} samples.")


    def before_run(self) -> None:
        output.console_log("Preparing for the next run...")
        
    def start_run(self, context: RunnerContext) -> None:
        quantization_type = context.run_variation['quantization_type']
        task_name = context.run_variation['task_name']
        output.console_log(f"Starting run for quantization type: {quantization_type} with task: {task_name}")
        model, tokenizer = self.load_model(quantization_type)
        output.console_log(f"Model and tokenizer loaded for quantization type: {quantization_type}")
        
        dataset1, dataset2 = self.datasets[task_name]
        
        # Run first dataset
        if dataset1 is not None:
            output.console_log(f"Running first dataset for task: {task_name}")
            results1 = self.run_experiment(model, tokenizer, dataset1, quantization_type, task_name)
            if results1 is not None:
                inference_time1, gpu_energy1, accuracy1, gpu_utilization1, memory_usage1 = results1
            else:
                inference_time1, gpu_energy1, accuracy1, gpu_utilization1, memory_usage1 = 0, 0, 0, 0, 0
        else:
            output.console_log(f"Dataset1 for task {task_name} is None, skipping...")
            inference_time1, gpu_energy1, accuracy1, gpu_utilization1, memory_usage1 = 0, 0, 0, 0, 0
            
        # Run second dataset
        if dataset2 is not None:
            output.console_log(f"Running second dataset for task: {task_name}")
            results2 = self.run_experiment(model, tokenizer, dataset2, quantization_type, task_name)
            if results2 is not None:
                inference_time2, gpu_energy2, accuracy2, gpu_utilization2, memory_usage2 = results2
            else:
                inference_time2, gpu_energy2, accuracy2, gpu_utilization2, memory_usage2 = 0, 0, 0, 0, 0
        else:
            output.console_log(f"Dataset2 for task {task_name} is None, skipping...")
            inference_time2, gpu_energy2, accuracy2, gpu_utilization2, memory_usage2 = 0, 0, 0, 0, 0
        
        # Record averages
        context.run_data = {
            'Inference Time': (inference_time1 + inference_time2) / 2,
            'GPU Energy': (gpu_energy1 + gpu_energy2) / 2,
            'Accuracy': (accuracy1 + accuracy2) / 2,
            'GPU Utilization': (gpu_utilization1 + gpu_utilization2) / 2,
            'Memory Usage': (memory_usage1 + memory_usage2) / 2
        }
        output.console_log(f"Run completed for {quantization_type} with task: {task_name} with data: {context.run_data}")

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log(f"Starting measurements for {context.run_variation['quantization_type']} model.")

    def interact(self, context: RunnerContext) -> None:
        output.console_log("Interacting with the running system...")

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Stopping measurements...")

    def stop_run(self, context: RunnerContext) -> None:
        output.console_log("Run completed.")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        output.console_log(f"Populating run data for {context.run_variation['quantization_type']}.")
        return context.run_data

    def after_experiment(self) -> None:
        output.console_log("Experiment finished.")
        # Shut down PyNVML to release GPU resources
        try:
            pynvml.nvmlShutdown()
            output.console_log("pynvml shutdown successfully.")
        except pynvml.NVMLError as e:
            output.console_log(f"Failed to shutdown pynvml: {str(e)}")

    def load_model(self, quantization_type):
        if not hasattr(self, 'model_cache'):
            self.model_cache = {}
        output.console_log(f"Loading model for quantization type: {quantization_type}")
    
        if quantization_type == "32-bit":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side='left')
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",device_map="auto") #torch_dtype = torch.float16

        elif quantization_type == "8-bit":
            tokenizer = AutoTokenizer.from_pretrained("GreenHalo/8bit", padding_side='left')
            model = AutoModelForCausalLM.from_pretrained("GreenHalo/8bit",device_map="auto",trust_remote_code=True)

        elif quantization_type == "4-bit":
            tokenizer = AutoTokenizer.from_pretrained("GreenHalo/4bit", padding_side='left')
            model = AutoModelForCausalLM.from_pretrained("GreenHalo/4bit",device_map="auto")

        if "exllama" in str(type(model)).lower():
            model = exllama_set_max_input_length(model, max_input_length=2082)   
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        #self.model_cache[quantization_type] = (model, tokenizer)
        return model, tokenizer

    def process_input(self, data, task_name, tokenizer, max_length=256):
        if task_name == "SA":  
            # Truncate sentence
            sentence = data.get('sentence', data.get('text'))
            truncated_input = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors="pt")
            truncated_text = tokenizer.decode(truncated_input["input_ids"][0], skip_special_tokens=True)
        
            return f"Question: Is this text positive or negative?  Text: \"{truncated_text}\". Answer: The text is", None

        elif task_name == "SPS": 
            sentence1 = data['sentence1'] if 'sentence1' in data else data['question1']
            sentence2 = data['sentence2'] if 'sentence2' in data else data['question2']
            
            truncated_input1 = tokenizer(sentence1, truncation=True, max_length=max_length, return_tensors="pt")
            truncated_input2 = tokenizer(sentence2, truncation=True, max_length=max_length, return_tensors="pt")
            
            truncated_text1 = tokenizer.decode(truncated_input1["input_ids"][0], skip_special_tokens=True)
            truncated_text2 = tokenizer.decode(truncated_input2["input_ids"][0], skip_special_tokens=True)

            return f"Question: Are these sentences semantically equivalent? Sentence 1: \"{truncated_text1}\" Sentence 2: \"{truncated_text2}\". Answer: These sentences are", None

        elif task_name == "NLI": 
            sentence1 = data['sentence1']
            sentence2 = data['sentence2']
            
            truncated_input1 = tokenizer(sentence1, truncation=True, max_length=max_length, return_tensors="pt")
            truncated_input2 = tokenizer(sentence2, truncation=True, max_length=max_length, return_tensors="pt")
            
            truncated_text1 = tokenizer.decode(truncated_input1["input_ids"][0], skip_special_tokens=True)
            truncated_text2 = tokenizer.decode(truncated_input2["input_ids"][0], skip_special_tokens=True)


            return f"Question: Are these sentences logically entailed? Sentence 1: \"{truncated_text1}\" and Sentence 2: \"{truncated_text2}\". Answer: These sentences are", None

        return None





    def batch_process_input(self, dataset, task_name, batch_size, tokenizer):
        inputs_list = []
        labels = []
        total = len(dataset) 

        for idx, data in enumerate(dataset):
            if idx % batch_size == 0 and idx > 0:
                output.console_log(f"Processing data {idx} / {total} for task: {task_name}...")

                yield inputs_list, labels
                inputs_list, labels = [], []


            inputs = self.process_input(data, task_name, tokenizer) 
            correct_choice = data.get('label')

            if not inputs or correct_choice is None:
                continue

            # Save as dictionary structure
            if task_name in ["SPS", "NLI"] and inputs[1] is not None:
                inputs_list.append({"sentence1": inputs[0], "sentence2": inputs[1]})
            else:
                inputs_list.append({"text": inputs[0]})

            labels.append(correct_choice)
        
        # Return remaining data
        if inputs_list:
            yield inputs_list, labels


    def run_experiment(self, model, tokenizer, dataset, quantization_type, task_name, batch_size=1):
        device = NvidiaGPUDomain(0)  # GPU index
        total_gpu_energy = 0
        total_inference_time = 0  
        correct_predictions = 0  
        valid_text_count = 0  
        
        total_gpu_utilization = 0
        total_memory_usage = 0
        num_batches = 0  
    
        # Init EnergyContext for energy measurement
        with EnergyContext(domains=[device], start_tag='start') as ctx:
            for inputs_list, labels in self.batch_process_input(dataset, task_name, batch_size, tokenizer):
                inputs_texts = [input_data["text"] for input_data in inputs_list]
    
                # Move inputs to CUDA
                inputs = tokenizer(
                    inputs_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to('cuda')
    
                torch.cuda.synchronize()  
                start_time = time.perf_counter()
    
                # Inference
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs['attention_mask'], 
                        max_new_tokens=5,
                        num_beams=1,
                        do_sample=False
                    )
    
                torch.cuda.synchronize()  
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                total_inference_time += inference_time
    
                # Record energy
                ctx.record(tag='inference_step')
    
                # Decode and evaluate
                generated_texts = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
    
                # Evaluate predictions
                for generated_text, correct_choice in zip(generated_texts, labels):
                    predicted_choice = self.map_output_to_label(generated_text)
                    if predicted_choice is not None:
                        valid_text_count += 1
                        if predicted_choice == correct_choice:
                            correct_predictions += 1
    
                # Measure performance
                gpu_utilization = self.get_gpu_utilization()
                memory_usage = self.get_memory_usage()
    
                # Accumulate metrics
                total_gpu_utilization += gpu_utilization
                total_memory_usage += memory_usage
                num_batches += 1  
    
        # Get energy data
        energy_data = ctx.get_trace()
    
        # Accumulate energy
        for measurement in energy_data:
            #print(vars(measurement)) 
            if measurement.tag == 'inference_step':
                energy = measurement.energy
                if 'nvidia_gpu_0' in energy:
                    total_gpu_energy += energy['nvidia_gpu_0'] / 1_000_000  # Convert to Joules
    
        if valid_text_count == 0:
            output.console_log("No valid samples processed")
            return None
    
        # Calculate averages
        average_gpu_utilization = total_gpu_utilization / num_batches if num_batches > 0 else 0
        average_memory_usage = total_memory_usage / num_batches if num_batches > 0 else 0
    
        # Calculate accuracy
        accuracy = correct_predictions / valid_text_count
    

        return (
            total_inference_time / valid_text_count,
            total_gpu_energy,  
            accuracy,
            average_gpu_utilization,  
            average_memory_usage 
        )


    def map_output_to_label(self, generated_text):

        generated_text = generated_text.strip().lower()
    
        if "no" in generated_text or "not" in generated_text or "negative" in generated_text:
            return 0
    
        if "yes" in generated_text or "equivalent" in generated_text or "same" in generated_text or "positive" in generated_text or "logically" in generated_text:
            return 1

        return None



    def get_gpu_utilization(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_utilization = min(max(utilization.gpu, 0), 100)
        
        return utilization.gpu 
 
    def get_memory_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / (1024**2)

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
