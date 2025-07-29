# 🔧 AlignX
---

## 🚀 Overview

AlignX introduces a two-stage optimization framework:

- **Stage 1: Fine-Tuning** — Fine-tunes base LLMs using axis-specific supervision in a shared latent alignment space.
- **Stage 2: Mixture of Calibrated Experts (MoCaE)** — Dynamically routes expert outputs via learned calibration weights for task-specific generation with controlled alignment trade-offs.

---

## 📊 Datasets

AlignX uses curated datasets for each alignment axis:

| Alignment Axis | Dataset | Description |
|----------------|---------|-------------|
| **Helpfulness** | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 20k instruction-response pairs |
| **Harmlessness** | [BeaverTails](https://sites.google.com/view/pku-beavertails) | 30k safety-annotated QA pairs |
| **Honesty** | [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Benchmark for truthful answering |

---

## 📈 Evaluation Metrics

| Axis         | Metric                           | Description |
|--------------|----------------------------------|-------------|
| Helpfulness  | **Win Rate (↑)**                | % of samples where AlignX wins over baseline |
| Harmlessness | **Safety Score (↓)**            | % of unsafe outputs (lower is better) |
| Honesty      | **Truthful & Informative (TI ↑)** | Product of truthfulness and informativeness |
| Overall      | **Average Alignment Score (↑)** | Normalized combination of the above metrics |

**↑**: Higher is better  **↓**: Lower is better

---

## 🧪 Experimental Setup

All models are trained on 3 alignment axes using:

- Latent space size: `k = 256`
- Routing projection matrix: `W_r ∈ ℝ^{3 × 1024}`
- Calibration weights: `λ₁ = 0.6`, `λ₂ = 0.4`
- Softmax temperature: `1`
- Clustering granularity: `ε = 0.05`
- Optimizer: **AdamW** with:
  - Learning rate: `2e-5`
  - Batch size: `64`
  - Weight decay: `0.01`
- Training: `3 epochs`

Evaluated on the following base LLMs:

- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Gemma-7B](https://huggingface.co/google/gemma-7b)
- [DeepSeek-7B](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)
- [LLaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)

---

## 🛠️ Usage: Task_Vector.py

The script `Stage1.py` is designed to analyze instruction-tuned language models on the listed datasets. It can be used with any of the supported models to compute task-specific representations and parameters, including:

- ✅ Task Vectors  
- 📊 Model Weights  
- 🧱 Base Weights    

### How to Use

Simply pass your chosen model and dataset to `Stage1.py` to extract and compute the desired task representations. The script supports:

- Any of the models listed above (e.g., LLaMA-2 7B, Mistral-7B, Gemma-7B, DeepSeek-7B)
- Any of the supported datasets (e.g., Alpaca, BeaverTails, TruthfulQA)

## 🔁 Follow-up Processing and Evaluation

Once task vectors, weights, and base weights, and have been extracted in the form of Parameter and Task Matrixs using `Stage1.py`, you can proceed with further processing and evaluation using the following scripts:

- `Stage2.py`

This scripts are designed to apply modular calibration and editing strategies on the extracted task vectors to align model behavior with desired moderation outcomes.

### 📈 Evaluation

After applying any of the MoCaE methods, use `Evaluate.py` to assess the performance of the calibrated models.

> ⚠️ Note: Make sure you have the appropriate access to the moderation models used for evaluation. These include:

- GPT-4.0 (via OpenAI API)
- beaver-dam-7b — available here: [PKU-Alignment/beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- GPT-Judge (via OpenAI API)

These evaluators are used to provide automated and/or human-aligned judgment of the calibrated outputs in terms of helpfulness, harmlessness, and honesty.

🖥️ Note on Performance Variability:
Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during task vector extraction and calibration.
