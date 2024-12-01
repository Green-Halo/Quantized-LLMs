# LLaMA-3.1-8B-Instruct Quantization: Energy Efficiency and Performance Trade-offs

## Overview
This study aims to evaluate the **LLaMA-3.1-8B-Instruct** model under 4-bit Post-Training Quantization (PTQ). Our focus is on **energy efficiency** and **performance** across different natural language processing (NLP) tasks, quantifying the trade-offs between reduced model precision and its impact on accuracy, energy consumption, and resource utilization.

Our analysis explores whether quantization, while improving energy efficiency, affects model accuracy across NLP tasks. We assess three primary NLP task types:
1. **Sentiment Analysis (SA)**
2. **Sentence Pair Semantic Similarity (SPS)**
3. **Natural Language Inference (NLI)**

## Requirements

### Prerequisites
- **Miniconda 3**: Ensure Miniconda 3 is installed on your system. [Download Miniconda](https://docs.anaconda.com/miniconda/)
- Install required Python libraries by running:
  ```bash
   conda env create -f environment.yml
   conda activate py310
  ```
  ```bash
  pip install -r requirements.txt
  ```

### Download models 
1. **Windows**  
   ```
   .\download_models\windows.ps1
   ```

2. **Linux**  
   Install `pynvml` to monitor GPU utilization. Use the following command:
   ```bash
   bash ./download_models/linux.sh
   ```

## Running the project

To run the main experiment, execute the following command from the root directory:
   ```bash
   python experiment-runner/ tasks/onnx/RunnerConfig.py
   ```
This command initiates the quantization experiments on the LLaMA3-8B model using 4-bit and 8-bit precision levels. The experiment assesses the impact of quantization on energy efficiency, accuracy, and resource utilization across various NLP tasks, including those in the GLUE and IMDB datasets.

The results table is available at the folder `run_tables`, which contains detailed experiment results of different models for further analysis.

### Quantization
The quantization code can be found in the `quantization` folder.

### Data Analysis
Statistical tests and visualizations are performed using R scripts to interpret the impact of quantization effectively. All data analysis scripts are stored in the `data-analysis` folder.
