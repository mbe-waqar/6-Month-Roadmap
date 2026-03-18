# 📅 Week 3 — Modern LLM Architectures (Day-by-Day Plan)

**Focus of this week:**
- Modern LLM design (LLaMA, Mistral)
- Efficient inference
- Quantization
- KV caching
- Serving local models

---

## 🗂️ Table of Contents

- [Day 1 — Understanding Modern LLM Architectures (LLaMA)](#-day-1--understanding-modern-llm-architectures-llama)
- [Day 2 — Mistral Innovations](#-day-2--mistral-innovations)
- [Day 3 — Mixture of Experts (MoE)](#-day-3--mixture-of-experts-moe)
- [Day 4 — Model Quantization](#-day-4--model-quantization)
- [Day 5 — Deploy Local LLM](#-day-5--deploy-local-llm)
- [Day 6 — Quantization + vLLM Optimization](#-day-6--quantization--vllm-optimization)
- [Day 7 — KV Cache & Benchmarking](#-day-7--kv-cache--benchmarking)
- [End-of-Week Reality Check](#-end-of-week-reality-check)

---

## 🟢 Day 1 — Understanding Modern LLM Architectures (LLaMA)

### 🎯 Goal

Understand how modern LLMs like LLaMA improved the original Transformer.

---

### ✅ Tasks

1. Study LLaMA architecture
2. Understand improvements over GPT-3 style models
3. Learn about:
   - RMSNorm
   - Rotary Positional Embeddings (RoPE)
   - Efficient training scaling

---

### 📘 What to Study

Key improvements introduced by LLaMA:
- RMSNorm instead of LayerNorm
- Rotary positional embeddings
- Efficient parameter scaling
- Better training dataset curation

These make models smaller but powerful.

---

### 🎥 YouTube

- "LLaMA Explained" — Yannic Kilcher
- "The Illustrated LLaMA" — Jay Alammar
- "How LLaMA Works" — AI Coffee Break

---

### 📄 Written Resources

- LLaMA Paper — Touvron et al.
- HuggingFace LLaMA architecture blog

---

### 🧪 Mini-Assignment

Write ½ page explaining:
- Why LLaMA is more efficient than older LLMs
- What RoPE and RMSNorm improve

---

### ⏱️ Time

| Task | Time |
|------|------|
| Videos + reading | 2h |
| Notes | 1h |
| Mini-assignment | 0.5h |
| **Total** | **3.5h** |

---

## 🟢 Day 2 — Mistral Innovations

### 🎯 Goal

Understand innovations introduced by Mistral 7B.

---

### ✅ Tasks

1. Study sliding window attention
2. Learn grouped query attention
3. Compare LLaMA vs Mistral
4. Understand how context window efficiency works

---

### 📘 What to Study

**Sliding Window Attention**

Instead of attending to all tokens, the model attends to a moving window, which reduces memory and speeds up inference.

Benefits:
- Lower memory
- Faster inference
- Larger context capability

---

### 🎥 YouTube

- "Mistral 7B Architecture Explained" — Yannic Kilcher
- "Sliding Window Attention Explained" — AI Coffee Break
- "Efficient Transformers" — DeepLearningAI

---

### 📄 Written Resources

- Mistral 7B Technical Report
- HuggingFace Mistral documentation

---

### 🧪 Mini-Assignment

Create a comparison table:

| Feature | LLaMA | Mistral |
|---------|-------|---------|
| Attention | | |
| Context handling | | |
| Efficiency | | |

---

### ⏱️ Time

| Task | Time |
|------|------|
| Study | 2h |
| Notes + table | 1h |
| **Total** | **3h** |

---

## 🟢 Day 3 — Mixture of Experts (MoE)

### 🎯 Goal

Understand how models like Mixtral 8x7B scale efficiently.

---

### ✅ Tasks

1. Study Mixture of Experts architecture
2. Understand gating networks
3. Learn sparse activation
4. Compare dense vs sparse models

---

### 📘 What to Study

**MoE idea:**

Instead of activating all neurons, the model activates only a few experts.

Example:
- 8 experts
- Only 2 used per token

Benefits:
- Huge capacity
- Lower compute cost

---

### 🎥 YouTube

- "Mixture of Experts Explained" — Yannic Kilcher
- "Sparse Models in Transformers" — AI Coffee Break

---

### 📄 Written Resources

- Switch Transformer Paper
- Mixtral architecture blog

---

### 🧪 Mini-Assignment

Write 3 advantages of MoE models over dense models.

---

### ⏱️ Time

| Task | Time |
|------|------|
| Study | 2h |
| Notes | 1h |
| **Total** | **3h** |

---

## 🟢 Day 4 — Model Quantization

### 🎯 Goal

Understand how quantization reduces memory and speeds up inference.

---

### ✅ Tasks

1. Study quantization basics
2. Learn 4-bit vs 8-bit inference
3. Understand:
   - GPTQ
   - AWQ

---

### 📘 What to Study

Quantization converts weights from:

```
float32 → int8 / int4
```

Benefits:
- Smaller memory
- Faster inference
- Lower GPU requirements

---

### 🎥 YouTube

- "LLM Quantization Explained" — AI Coffee Break
- "GPTQ Explained" — Yannic Kilcher

---

### 📄 Written Resources

- GPTQ paper
- HuggingFace quantization guide

---

### 🧪 Mini-Assignment

Write a short note explaining:
- Why 4-bit quantization works for LLMs.

---

### ⏱️ Time

| Task | Time |
|------|------|
| Videos + reading | 2h |
| Notes | 1h |
| **Total** | **3h** |

---

## 🟢 Day 5 — Deploy Local LLM

### 🎯 Goal

Run LLaMA 2 locally.

---

### ✅ Tasks

1. Install dependencies
2. Load LLaMA 2 7B
3. Run basic inference
4. Test generation prompts

---

### 📘 What to Study

Key tools:
- Hugging Face Transformers
- bitsandbytes

---

### 🎥 YouTube

- "Run LLaMA Locally" — Prompt Engineering
- "Load LLM Locally with Transformers" — HuggingFace

---

### 📄 Written Resources

- HuggingFace LLaMA loading guide

---

### 🧪 Mini-Assignment

Run 5 prompts and save responses.

---

### ⏱️ Time

| Task | Time |
|------|------|
| Setup | 2h |
| Testing | 1h |
| **Total** | **3h** |

---

## 🟢 Day 6 — Quantization + vLLM Optimization

### 🎯 Goal

Optimize inference speed.

---

### ✅ Tasks

1. Quantize model (4-bit + 8-bit)
2. Run inference benchmarks
3. Test vLLM
4. Compare response time

---

### 📘 What to Study

Key optimization ideas:
- KV cache
- Batching
- Efficient memory usage

---

### 🎥 YouTube

- "vLLM Explained" — AI Coffee Break
- "Optimizing LLM Inference" — DeepLearningAI

---

### 📄 Written Resources

- vLLM documentation

---

### 🧪 Mini-Assignment

Benchmark:

| Model | Latency |
|-------|---------|
| Full precision | |
| 4-bit | |
| 8-bit | |

---

### ⏱️ Time

| Task | Time |
|------|------|
| Implementation | 3h |
| Benchmark | 1h |
| **Total** | **4h** |

---

## 🟢 Day 7 — KV Cache & Benchmarking

### 🎯 Goal

Build efficient inference pipeline.

---

### ✅ Tasks

1. Implement KV cache
2. Profile memory usage
3. Measure token generation speed
4. Write optimization report

---

### 📘 What to Study

KV cache stores previous attention states so the model doesn't recompute them. This drastically improves generation speed.

---

### 🎥 YouTube

- "KV Cache Explained" — AI Coffee Break
- "Transformer Inference Optimization" — Yannic Kilcher

---

### 📄 Written Resources

- HuggingFace inference optimization guide

---

### 🧪 Mini-Assignment

Create a performance report:
- Quantized vs full precision
- Memory usage
- Inference latency

---

### ⏱️ Time

| Task | Time |
|------|------|
| Coding | 3h |
| Report writing | 1h |
| **Total** | **4h** |

---

## ✅ End-of-Week Reality Check

By the end of Week 3, you should confidently say:

- ✅ I understand modern LLM architectures
- ✅ I can run LLMs locally
- ✅ I can quantize models
- ✅ I can optimize inference speed
- ✅ I can build production-ready LLM pipelines

---

## 📊 Weekly Time Summary

| Day | Topic | Total Time |
|-----|-------|-----------|
| Day 1 | Understanding Modern LLM Architectures (LLaMA) | 3.5h |
| Day 2 | Mistral Innovations | 3h |
| Day 3 | Mixture of Experts (MoE) | 3h |
| Day 4 | Model Quantization | 3h |
| Day 5 | Deploy Local LLM | 3h |
| Day 6 | Quantization + vLLM Optimization | 4h |
| Day 7 | KV Cache & Benchmarking | 4h |
| **Total** | | **23.5h** |
