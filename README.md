# LongMab-PO
Long-Context LLM Preference Optimization
Source code for paper: Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization
## Overview
LongMab-PO is a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training.

![](fig/main.png)

## Requirements

### 1. Requirement.
**Install the following packages using Pip or Conda under this environment**

```
Python==3.10.15
torch==2.4.0
transformers==4.45.1
tqdm
trl==0.8.6
vllm==0.6.2
sentence-transformers==3.4.1
accelerate==1.0.0
deepspeed==0.16.5
peft==0.12.0
```

### 2. Install LLaMA-Factory.
Refer to [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed instructions.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Training LongMab-PO

### 1. Prepare the Training Data
You can follow [SeaLong](https://github.com/SihengLi99/SEALONG/tree/main) to synthesize the training data, or download the file from [here](https://drive.google.com/drive/folders/1Mgqx54ZarGC5UR4CL4uRaoMBTq3VQKOf) and place them in the `data/raw_data/` directory.
 **Each sample must contain the following four required fields:**

```json
{
  "id": "A unique identifier for the sample (int)",
  "question": "The input question (str)",
  "answer": "The ground truth answer to the question (str)",
  "context": "The synthesized long context (str)"
}     
```

### 2. Run the LongMab-PO Pipeline
**(1) Generate Probe CoT:**
```
cd scripts
bash gen_probe.sh
```
**(2) Running the Multi-Armed Bandit Rollout Process:**
```
bash rollout.sh
```
**(3) Construct Preference Data Pairs:**
```
bash construct_po_data.sh
```

### 3. Train the Model
You can train the model by utilizing LLaMA-Factory framework quickly, we provide the yaml files. Please refer to LLaMA-Factory for relevant environment installation and configuration.
```
cd scripts
bash llama3_dpo.sh
bash qwen2_dpo.sh
```
You can also download the checkpoint of [Llama-3.1-8B-Instruct](https://huggingface.co/rocketduan/Llama-3.1-8B-Instruct-LongMab-PO) and [Qwen-2.5-7B-Instruct](https://huggingface.co/rocketduan/Qwen-2.5-7B-Instruct-LongMab-PO) directly.

## Evaluation
```
cd scripts
bash eval.sh
```

## Acknowledgement
We gratefully acknowledge the following projects that LongMab-PO builds upon:
- [**MuSiQue**](https://github.com/StonyBrookNLP/musique)
- [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory)
- [**vLLM**](https://github.com/vllm-project/vllm)
