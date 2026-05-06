<div align="center">

# Towards Safe Mobility: A Unified Transportation Foundation Model enabled by  Open-Ended Vision–Language Dataset (UniVLT)

**Wenhui Huang¹², Songyan Zhang¹, Collister Chua¹, Yang Liang¹, Zhiqi Mao¹, Heng Yang², Chen Lv¹\***  
¹ Nanyang Technological University 
² Harvard University  

\* Corresponding author  

[![arXiv](https://img.shields.io/badge/arXiv-2604.22260-b31b1b.svg)](https://arxiv.org/abs/2604.22260)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/Oscar-Huang/LTD)
[![Model](https://img.shields.io/badge/🤗-Model-red)](https://huggingface.co/Oscar-Huang/UniVLT)

</div>

---
<div align="center">

| Architecture | Demo |
|--------------|------|
| <img src="asset/teaser.png" width="420"/> | <img src="asset/demo.gif" width="420"/> |

</div>

## 🔥 Highlights

- 📦 **LTD**: 11.6K high-quality open-ended QA pairs from city-scale roadside cameras  
- 🧠 **UniVLT**: Unified multi-image vision-language transformer  
- 🚦 Multi-view risk reasoning across minimally correlated camera streams  
- 🎯 Joint modeling of grounding, camera identification, and open-ended safety analysis  
- 📊 Strong performance across multimodal perception and safety-critical benchmarks  

---
## 🔨 To-do list:
- [✓] Release training code and dataset.
- [✓] Release evaluation/benchmark code.
- [✓] Release inference code.
- [✓] Release model checkpoint.

# 🧠 UniVLT: Model Overview

UniVLT is a unified multimodal transformer that:

- Interleaves high-resolution visual tokens from multiple cameras
- Operates in a shared vision-language sequence space
- Enables long-range cross-image reasoning
- Uses dynamic visual token budgeting
- Applies parameter-efficient adaptation (LoRA-based fine-tuning)

---

# 🏙️ Land Transportation Dataset (LTD)

LTD is a large-scale open-ended multimodal dataset designed to support robust reasoning in real-world urban traffic environments.

It contains:

- Fine-grained **multi-object grounding**
- Multi-image **camera identification**
- Open-ended **multi-image risk analysis**
- Diverse road geometries, traffic participants, lighting conditions, and weather scenarios

---

## 🛠 Dataset Construction Pipeline

<div align="center">
  <img src="asset/hitl.png" width="900"/>
</div>

To ensure annotation fidelity and reduce hallucination:

- Multi-model vision-language generation is first applied  
- Cross-model consistency checking is performed  
- Human-in-the-loop refinement corrects edge cases  
- Systematic validation reduces bias and annotation noise  

---

## 📦 Dataset Access

Please create your own `data/` directory in the project root and download the required datasets from Hugging Face.

1. Download Datasets

The LTD dataset is available on Huggingface:

👉 https://huggingface.co/datasets/Oscar-Huang/LTD

LingoQA, Omnidrive and CODA QA jsons are also available on Huggingface:

👉 https://huggingface.co/datasets/Oscar-Huang/AD_Json

[LingoQA](https://github.com/wayveai/LingoQA), [Omnidrive](https://github.com/NVlabs/OmniDrive) and [CODA](https://coda-dataset.github.io/download.html#link) images are available on their respective official websites.

Download the corresponding datasets and place them inside the `data/` directory.

2. Expected Directory Structure
```
data/
├── CODA/
├── LingoQA/
├── LTD/
├── NuScenes/
├── Omnidrive/
├── images_w_boxes/
```

---

# 📦 Model Zoo

| Model | Backbone | Stage | Link |
|--------|----------|--------|--------|
| UniVLT-7B | Qwen2.5-VL-7B-Instruct | FT Stage 2 | https://huggingface.co/c-chua/UniVLT |

Backbone model: 
[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

---

# 🚀 Quick Start (Inference)

```
git clone https://github.com/OscarHuangWind/UniVLT.git
cd UniVLT

conda env create -f environment.yml
conda activate UniVLT

pip install -r requirements.txt
pip install ms-swift==3.6.0
pip install flash-attn==2.7.4.post1
```
If flash attention fails:
```
pip install flash-attn
```
Run inference:
```
python univlt_scripts/inference.py \
    --model_id c-chua/UniVLT \
    --eval_data example_lta_risk_id.json \
    --output_path results.json
```

# 🏋️ Training
## Stage 1: Fine-tuning
1. Download Qwen2.5-VL-7B-Instruct backbone.
2. Update:
    - `--model` path
    - `--dataset` paths
    - `WANDB_API_KEY`
    - `output_dir`
3. Run
```
bash training_scripts/LTA_stage1.sh
```
4. Merge LoRA weights:
```
bash test_export.sh /path/to/checkpoint
```
---
## Stage 2: Fine-tuning
1. Set `--model` to Stage 1 merged checkpoint.
2. Update dataset paths.
3. Run:
```
bash training_scripts/LTA_stage2.sh
```
4. Merge final weights:
```
bash test_export.sh /path/to/checkpoint_stage_2
```
---
# 📊 Benchmarking
After inferfence:
```
bash univlt_scripts/eval.sh
```
This computes:
    - NLP metrics
    - Lingo-judge
    - GPT-score

For LTD-specific metrics:
```
python calculate_accuracy.py
python compute_f1_score.py
```
# 🎯 Benchmark Results
We evaluate **UniVLT** against both **general-purpose open-source vision-language models (VLMs)** and **autonomous-driving–specialized models** across multiple public benchmarks.  
These benchmarks assess the model's ability to perform **risk reasoning, perception grounding, scene understanding, and driving-related decision support**.

**Notation**
- **Bold** → Best result
- <u>Underline</u> → Second-best result
- ↑ indicates **higher values are better**

---
# 📈 Results on LTD Benchmark

The **LTD benchmark** evaluates multi-image reasoning and perception capabilities for autonomous driving scenarios.

We report results on three tasks:

- **Multi-Image Risk Analysis** *(GPT-Score)*
- **Camera ID Selection** *(Accuracy)*
- **Multi-Object Grounding** *(F1 Score)*

| Model | Size | GPT-Score ↑ | Accuracy ↑ | Grounding F1 ↑ |
|--------|------|-------------|------------|----------------|
| LLaVA-OV | 0.5B | 0.01 | 0.29 | 0.00 |
| LLaVA-OV | 7B | 0.03 | 0.32 | 0.00 |
| Qwen2.5-VL | 7B | <u>0.46</u> | <u>0.48</u> | 0.45 |
| InternVL2.5 | 8B | 0.25 | 0.23 | 0.00 |
| Qwen3-VL | 4B | 0.14 | 0.25 | <u>0.62</u> |
| --- | --- | --- | --- | --- |
| OpenEMMA | 3B | 0.10 | 0.29 | 0.44 |
| WiseAD | 1.7B | 0.00 | 0.00 | N/A |
| RoboTron-Drive | 8B | 0.06 | 0.00 | 0.06 |
| ReCogDrive | 8B | 0.29 | 0.32 | N/A |
| --- | --- | --- | --- | --- |
| **UniVLT (Ours)** | **7B** | **0.66** | **0.66** | **0.64** |

**Grounding evaluation protocol.**  
Grounding results are reported only for models that produce **normalized bounding box coordinates**, or whose outputs can be converted to a **thousandth-level normalized scale**.

---

# 📈 Results on LingoQA Benchmark

The **LingoQA benchmark** evaluates **language-driven reasoning for driving scenarios**, requiring models to interpret scene context and answer complex driving-related questions.

| Model | Size | Lingo-Judge ↑ |
|--------|------|-------------|
| LLaVA-OV | 0.5B | 34.2% |
| LLaVA-OV | 7B | 54.2% |
| Qwen2.5-VL | 7B | 62.2% |
| InternVL2.5 | 8B | 47.2% |
| Qwen3-VL | 4B | 37.0% |
| --- | --- | --- |
| OpenEMMA | 3B | 48.0% |
| WiseAD | 1.7B | 60.4% |
| RoboTron-Drive | 8B | 59.2% |
| ReCogDrive | 8B | <u>67.8%</u> |
| --- | --- | --- |
| **UniVLT (Ours)** | **7B** | **69.0%** |

---

# 📈 Results on OmniDrive Benchmark

The **OmniDrive benchmark** evaluates general **driving scene understanding and reasoning capabilities**.

| Model | Size | GPT-Score ↑ |
|--------|------|-------------|
| LLaVA-OV | 0.5B | 0.09 |
| LLaVA-OV | 7B | 0.72 |
| Qwen2.5-VL | 7B | 0.80 |
| InternVL2.5 | 8B | <u>0.84</u> |
| Qwen3-VL | 4B | 0.80 |
| --- | --- | --- |
| OpenEMMA | 3B | 0.75 |
| WiseAD | 1.7B | 0.59 |
| RoboTron-Drive | 8B | 0.83 |
| ReCogDrive | 8B | 0.79 |
| --- | --- | --- |
| **UniVLT (Ours)** | **7B** | **0.87** |

---

# 📈 Results on CODA-LM Benchmark

The **CODA-LM benchmark** evaluates perception and reasoning capabilities in driving environments across three tasks:

- **Region Perception** — understanding localized scene elements  
- **General Perception** — holistic scene understanding  
- **Driving Suggestion** — generating safe driving decisions

| Model | Size | Region Perception ↑ | General Perception ↑ | Driving Suggestion ↑ | Average ↑ |
|--------|------|-------------|------------|----------------|-----------|
| LLaVA-OV | 0.5B | 1.93 | 1.11 | 1.57 | 1.54 |
| LLaVA-OV | 7B | 3.06 | 1.33 | 2.68 | 2.36 |
| Qwen2.5-VL | 7B | 5.07 | 4.37 | 5.20 | 4.88 |
| InternVL2.5 | 8B | 6.04 | 3.91 | 5.22 | 5.06 |
| Qwen3-VL | 4B | 6.92 | 4.54 | **6.08** | 5.85 |
| --- | --- | --- | --- | --- | --- |
| OpenEMMA | 3B | 5.52 | 3.76 | 5.04 | 4.77 |
| WiseAD | 1.7B | 1.81 | 1.27 | 1.09 | 1.39 |
| RoboTron-Drive | 8B | **7.66** | <u>5.15</u> | <u>5.68</u> | **6.16** |
| ReCogDrive | 8B | 0.29 | 0.32 | N/A | 0.00 |
| --- | --- | --- | --- | --- | --- |
| **UniVLT (Ours)** | **7B** | <u>7.25</u> | **5.18** | 5.62 | <u>6.02</u> |

---

# 🔍 Reproducibility
- All experiments conducted with fixed random seeds.
- LoRA weights merged before final evaluation.
- Offical release tagged as `v1.0`

# ⚖️ Ethics & Data Usage
- Dataset collected from city-scale roadside camera deployments.
- Released for academic research purposes. 
Users are responsible for complying with local regulations and ethical standards.

# 📜 License

Code and dataset is released under the MIT License.

# ✏️ Acknowledgements
This project builds upon:
- [ms-swift](https://github.com/modelscope/ms-swift)
- Qwen2.5-VL

# 📚 Citation
```
@misc{huang2026univlt,
      title={Towards Safe Mobility: A Unified Transportation Foundation Model enabled by Open-Ended Vision-Language Dataset}, 
      author={Wenhui Huang and Songyan Zhang and Collister Chua and Yang Liang and Zhiqi Mao and Heng Yang and Chen Lv},
      year={2026},
      eprint={2604.22260},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.22260}, 
}
```
