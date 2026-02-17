# Transformer from Scratch (PyTorch)

This repository contains a **from-scratch implementation of the Transformer architecture** using PyTorch.
The model is trained on a **toy English → French translation task** and includes attention visualizations and detailed code comments for learning purposes.

---

## 📌 What is Attention?

Attention is a mechanism that allows the model to **decide which words in a sentence are important when processing each word**.

Instead of reading a sentence word by word (like RNNs), attention lets every word:

- Look at **all other words**
- Decide **how relevant each word is**
- Combine information based on that relevance

### How it works (high level):

1. Each word is converted into **Query (Q), Key (K), and Value (V)** vectors
2. Similarity between words is computed using **dot products**
3. These scores are converted into **percentages using softmax**
4. The final representation is a **weighted combination of values**

In one line:

> **Attention = compare words → convert scores to percentages → mix information**

---

## 🧱 Encoder–Decoder Architecture

The Transformer is split into two main parts: **Encoder** and **Decoder**.

### Encoder

- Reads the **input sentence** (English)
- Uses **self-attention** to understand relationships between words
- Outputs contextual representations of the input

Each encoder layer consists of:

- Multi-head self-attention
- Feed-forward network
- Residual connections + layer normalization

---

### Decoder

- Generates the **output sentence** (French)
- Uses two types of attention:
  1. **Masked self-attention** (cannot see future words)
  2. **Cross-attention** (attends to encoder outputs)

This separation allows the model to:

- Fully understand the input first
- Generate output step by step in a controlled way

---

## 🏋️ Training Setup

### Task

- **Sequence-to-sequence translation**
- English → French (toy dataset)

---

### Model Details

- Transformer encoder–decoder built from scratch
- Sinusoidal positional encoding
- Multi-head attention
- Cross-entropy loss with padding mask

---

### Training Procedure

1. Input English sentence is passed through the **encoder**
2. Decoder receives:
   - Shifted French sentence (teacher forcing)
   - Encoder output (via cross-attention)
3. Loss is computed using **cross-entropy**
4. Model is optimized using **Adam optimizer**

---

### Evaluation

- Translation quality is evaluated using **BLEU score**
- Attention weights are visualized using matplotlib

---

## 📊 Results

- Model achieves reasonable translations on toy data
- Attention heatmaps show meaningful word alignments

---

## 📂 Repository Structure
