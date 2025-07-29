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

