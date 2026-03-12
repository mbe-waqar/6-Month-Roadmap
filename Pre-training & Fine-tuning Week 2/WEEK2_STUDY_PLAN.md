# 📅 Week 2 — Pre-training & Fine-tuning (Day-by-Day Plan)

This week is about how large language models are trained first, then adapted efficiently. You'll move from **theory → hands-on → optimization**.

---

## 🗂️ Table of Contents

- [Day 1 — Transfer Learning & Pre-training Intuition](#-day-1--transfer-learning--pre-training-intuition)
- [Day 2 — BERT Pre-training & Encoder-only Models](#-day-2--bert-pre-training--encoder-only-models)
- [Day 3 — GPT Pre-training & Decoder-only Models](#-day-3--gpt-pre-training--decoder-only-models)
- [Day 4 — Fine-tune BERT for Text Classification](#-day-4--fine-tune-bert-for-text-classification)
- [Day 5 — Fine-tune GPT-2 for Text Generation](#-day-5--fine-tune-gpt-2-for-text-generation)
- [Day 6 — LoRA & QLoRA (Efficient Fine-tuning)](#-day-6--lora--qlora-efficient-fine-tuning)
- [Day 7 — Comparison, Evaluation & Reporting](#-day-7--comparison-evaluation--reporting)
- [End-of-Week Reality Check](#-end-of-week-reality-check)

---

## 🟢 Day 1 — Transfer Learning & Pre-training Intuition

### 🎯 Goal

Build strong intuition for why pre-training + fine-tuning works in NLP.

---

### ✅ Tasks (in order)

1. Learn what pre-training means in NLP
2. Understand self-supervised learning
3. Understand fine-tuning vs training from scratch
4. Compare task-specific models vs foundation models

---

### 📘 What to Study

- Pre-training = learn language patterns from massive unlabeled text
- Fine-tuning = adapt that knowledge to a specific task
- Why models converge faster after pre-training

---

### 🎥 YouTube

- "Transfer Learning Explained" – StatQuest
- "Why Pretraining Works" – Yannic Kilcher

---

### 📄 Written Resources

- Hugging Face Blog — Transfer Learning in NLP
- Why Pretraining Is Powerful (intro blog)

---

### 🧪 Mini-Assignment

Write ½–1 page explaining:
- Why pre-training + fine-tuning beats training from scratch

---

### ⏱️ Time

| Task | Time |
|------|------|
| Videos & reading | 2h |
| Notes + mini-assignment | 1.5h |
| **Total** | **3.5h** |

---

## 🟢 Day 2 — BERT Pre-training & Encoder-only Models

### 🎯 Goal

Understand BERT's training objective and architecture.

---

### ✅ Tasks

1. Study Masked Language Modeling (MLM)
2. Study Next Sentence Prediction (NSP)
3. Understand bidirectional attention
4. Learn why BERT is encoder-only

---

### 📘 What to Study

- Why words are masked randomly
- Why bidirectional context matters
- Why BERT is great for classification, QA, NER

---

### 🎥 YouTube

- "BERT Explained" – StatQuest
- "The Illustrated BERT" – Jay Alammar

---

### 📄 Written Resources

- BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al.
- Hugging Face — BERT Documentation

---

### 🧪 Mini-Assignment

Explain in your own words:
- MLM
- NSP
- Why BERT is encoder-only

---

### ⏱️ Time

| Task | Time |
|------|------|
| Study & videos | 3h |
| Notes | 1h |
| **Total** | **4h** |

---

## 🟢 Day 3 — GPT Pre-training & Decoder-only Models

### 🎯 Goal

Understand GPT-style autoregressive training and text generation.

---

### ✅ Tasks

1. Study next-token prediction
2. Learn causal masking
3. Compare GPT vs BERT objectives
4. Understand why GPT excels at generation

---

### 📘 What to Study

- Autoregressive modeling
- Causal attention mask
- Why GPT cannot "see the future"

---

### 🎥 YouTube

- "GPT Explained" – Jay Alammar
- "Autoregressive Language Models" – Yannic Kilcher

---

### 📄 Written Resources

- Language Models are Unsupervised Multitask Learners — Radford et al.
- Hugging Face — GPT-2 Documentation

---

### 🧪 Mini-Assignment

Create a comparison table:
- BERT vs GPT (objective, architecture, use cases)

---

### ⏱️ Time

| Task | Time |
|------|------|
| Study & reading | 3h |
| Comparison notes | 1h |
| **Total** | **4h** |

---

## 🟢 Day 4 — Fine-tune BERT for Text Classification

### 🎯 Goal

Fine-tune a pre-trained BERT on a real classification task.

---

### ✅ Tasks

1. Load pre-trained BERT
2. Add classification head
3. Load GLUE dataset
4. Train and evaluate model

---

### 📘 What to Study

- `[CLS]` token usage
- Classification head
- Accuracy vs F1 score

---

### 🎥 YouTube

- "Fine-tuning BERT for Classification" – Aladdin Persson
- "Text Classification with Transformers" – Hugging Face

---

### 📄 Written Resources

- Hugging Face — Text Classification Tutorial
- GLUE Benchmark Overview

---

### 🧪 Mini-Assignment

- Achieve >85% accuracy
- Save metrics and plots

---

### ⏱️ Time

| Task | Time |
|------|------|
| Coding & training | 5h |
| Evaluation | 1h |
| **Total** | **6h** |

---

## 🟢 Day 5 — Fine-tune GPT-2 for Text Generation

### 🎯 Goal

Fine-tune GPT-2 to generate coherent text.

---

### ✅ Tasks

1. Load GPT-2 (small or medium)
2. Prepare custom dataset
3. Fine-tune with causal LM loss
4. Generate sample text

---

### 📘 What to Study

- Teacher forcing
- Overfitting in generation
- Sampling (top-k, top-p)

---

### 🎥 YouTube

- "Fine-tuning GPT-2" – Aladdin Persson
- "Text Generation Explained" – StatQuest

---

### 📄 Written Resources

- Hugging Face — Language Modeling Tutorial
- GPT-2 Training Guide

---

### 🧪 Mini-Assignment

- Generate 10 samples
- Compare before vs after fine-tuning

---

### ⏱️ Time

| Task | Time |
|------|------|
| Fine-tuning | 4h |
| Analysis | 1h |
| **Total** | **5h** |

---

## 🟢 Day 6 — LoRA & QLoRA (Efficient Fine-tuning)

### 🎯 Goal

Understand and implement parameter-efficient fine-tuning.

---

### ✅ Tasks

1. Learn LoRA & QLoRA intuition
2. Implement LoRA from scratch
3. Integrate with Hugging Face PEFT
4. Experiment with LoRA ranks

---

### 📘 What to Study

- Low-rank adaptation
- Why LoRA reduces memory
- Trade-off: rank vs performance

---

### 🎥 YouTube

- "LoRA Explained" – Yannic Kilcher
- "Parameter-Efficient Fine-Tuning" – Hugging Face

---

### 📄 Written Resources

- LoRA: Low-Rank Adaptation of LLMs — Hu et al.
- Hugging Face — PEFT Documentation

---

### 🧪 Mini-Assignment

- Compare trainable params: full FT vs LoRA
- Record memory usage

---

### ⏱️ Time

| Task | Time |
|------|------|
| Study & coding | 4h |
| Experiments | 1.5h |
| **Total** | **5.5h** |

---

## 🟢 Day 7 — Comparison, Evaluation & Reporting

### 🎯 Goal

Consolidate learning and document results clearly.

---

### ✅ Tasks

1. Compare BERT FT vs GPT FT
2. Compare full FT vs LoRA
3. Measure speed & memory
4. Write performance report

---

### 📘 What to Study

- Practical trade-offs
- When to use LoRA
- Deployment mindset

---

### 🎥 YouTube

- "Efficient Fine-Tuning of LLMs" – Hugging Face
- "Scaling Laws & Efficiency" – Yannic Kilcher

---

### 📄 Written Resources

- Hugging Face — Fine-tuning Best Practices
- Blog — Full Fine-tuning vs LoRA

---

### 🧪 Mini-Assignment

- Push results to GitHub
- Write comparison report

---

### ⏱️ Time

| Task | Time |
|------|------|
| Evaluation | 3h |
| Documentation | 2h |
| **Total** | **5h** |

---

## ✅ End-of-Week Reality Check

By the end of Week 2, you should confidently say:

- ✅ "I understand how BERT and GPT are pre-trained"
- ✅ "I can fine-tune models for real tasks"
- ✅ "I know when and why to use LoRA"
- ✅ "I can explain encoder-only vs decoder-only clearly"

---

## 📊 Weekly Time Summary

| Day | Topic | Total Time |
|-----|-------|-----------|
| Day 1 | Transfer Learning & Pre-training Intuition | 3.5h |
| Day 2 | BERT Pre-training & Encoder-only Models | 4h |
| Day 3 | GPT Pre-training & Decoder-only Models | 4h |
| Day 4 | Fine-tune BERT for Text Classification | 6h |
| Day 5 | Fine-tune GPT-2 for Text Generation | 5h |
| Day 6 | LoRA & QLoRA (Efficient Fine-tuning) | 5.5h |
| Day 7 | Comparison, Evaluation & Reporting | 5h |
| **Total** | | **33h** |
