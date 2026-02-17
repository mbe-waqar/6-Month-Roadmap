# Week 1 — Advanced Transformers (Day-by-Day Plan & Resources)

This week focuses on understanding **Transformers deeply**, both conceptually and practically, and implementing a **working encoder–decoder Transformer from scratch in PyTorch**.

---

## 🟢 Day 1 — Transformer Overview & Attention Intuition

### 🎯 Goal

Understand why Transformers were introduced and build intuition for attention.

### ✅ Tasks

- Study limitations of RNNs and CNNs
- Learn high-level idea of self-attention
- Understand encoder vs decoder roles
- Build intuition before math

### 📘 What to Study

- Why parallelization matters
- What attention solves in sequence modeling
- Difference between encoder-only and decoder-only models

### 🎥 YouTube

- “The Illustrated Transformer” – Jay Alammar
- “Attention Mechanism Explained” – :contentReference[oaicite:0]{index=0}

### 📄 Written Resources

- Jay Alammar — *The Illustrated Transformer*
- Hugging Face — *Transformer Conceptual Guide*

### 🧪 Mini-Assignment

- Write a 1-page explanation of attention in simple words

### ⏱️ Time

| Task                | Time           |
| ------------------- | -------------- |
| Videos & reading    | 2h             |
| Notes & explanation | 1.5h           |
| **Total**     | **3.5h** |

---

## 🟢 Day 2 — Self-Attention Mathematics

### 🎯 Goal

Understand the math behind self-attention step by step.

### ✅ Tasks

- Learn Q, K, V vectors
- Compute dot-product attention
- Understand scaling by √dₖ
- Apply softmax manually

### 📘 What to Study

- Dot product similarity
- Softmax as probability distribution
- Why scaling stabilizes training

### 🎥 YouTube

- “Self-Attention Clearly Explained” – :contentReference[oaicite:1]{index=1}
- “Attention Is All You Need – Explained” – :contentReference[oaicite:2]{index=2}

### 📄 Written Resources

- *Attention Is All You Need* — Section 3.1
- Harvard NLP — *Annotated Transformer*

### 🧪 Mini-Assignment

- Manually compute attention for 3 tokens with 2-D vectors

### ⏱️ Time

| Task               | Time         |
| ------------------ | ------------ |
| Theory & videos    | 2h           |
| Manual calculation | 2h           |
| **Total**    | **4h** |

---

## 🟢 Day 3 — Multi-Head Attention & Positional Encoding

### 🎯 Goal

Understand how Transformers scale attention and encode word order.

### ✅ Tasks

- Learn motivation behind multi-head attention
- Study sinusoidal positional encoding
- Compare absolute vs relative positions
- Implement positional encoding in PyTorch

### 📘 What to Study

- Why multiple attention heads help
- How sine/cosine encode position
- Generalization to longer sequences

### 🎥 YouTube

- “Multi-Head Attention Explained” – AI Coffee Break
- “Positional Encoding Intuition” – CodeEmporium

### 📄 Written Resources

- *Attention Is All You Need* — Section 3.2
- Harvard NLP — Positional Encoding Notes

### 🧪 Mini-Assignment

- Implement positional encoding
- Plot first 50 positions using matplotlib

### ⏱️ Time

| Task                   | Time         |
| ---------------------- | ------------ |
| Study & videos         | 2h           |
| Coding & visualization | 2h           |
| **Total**        | **4h** |

---

## 🟢 Day 4 — Encoder & Decoder Architecture

### 🎯 Goal

Understand the full Transformer encoder–decoder architecture.

### ✅ Tasks

- Study encoder block structure
- Study decoder block structure
- Learn masked self-attention
- Understand cross-attention

### 📘 What to Study

- Residual connections
- Layer normalization
- Why masking prevents cheating

### 🎥 YouTube

- “Transformer Encoder Decoder Explained” – AI Coffee Break
- “Masked Self-Attention” – CodeEmporium

### 📄 Written Resources

- *Attention Is All You Need* — Section 3
- PyTorch Transformer Documentation

### 🧪 Mini-Assignment

- Draw encoder and decoder blocks with labels

### ⏱️ Time

| Task             | Time         |
| ---------------- | ------------ |
| Study & videos   | 3h           |
| Diagrams & notes | 1h           |
| **Total**  | **4h** |

---

## 🟢 Day 5 — Implement Transformer Components

### 🎯 Goal

Implement core Transformer components from scratch.

### ✅ Tasks

- Implement self-attention layer
- Implement multi-head attention
- Implement feed-forward network
- Add residuals and layer norm

### 📘 What to Study

- Tensor shapes and broadcasting
- Modular PyTorch design
- Debugging attention layers

### 🎥 YouTube

- “Implement Transformer from Scratch” – :contentReference[oaicite:3]{index=3}
- “Annotated Transformer Walkthrough” – Harvard NLP

### 📄 Written Resources

- Harvard NLP — *Annotated Transformer (Code)*
- PyTorch `nn.Module` documentation

### 🧪 Mini-Assignment

- Unit test attention shapes
- Visualize attention matrices

### ⏱️ Time

| Task            | Time         |
| --------------- | ------------ |
| Coding          | 4h           |
| Debugging       | 1h           |
| **Total** | **5h** |

---

## 🟢 Day 6 — Build Full Encoder-Decoder + Training

### 🎯 Goal

Train a working translation Transformer.

### ✅ Tasks

- Assemble encoder + decoder
- Prepare English–French toy dataset
- Implement training loop
- Add loss masking

### 📘 What to Study

- Teacher forcing
- Cross-entropy with padding mask
- Why BLEU matters

### 🎥 YouTube

- “Seq2Seq Transformer Training” – :contentReference[oaicite:4]{index=4}
- “BLEU Score Explained” – :contentReference[oaicite:5]{index=5}

### 📄 Written Resources

- PyTorch Translation Tutorial
- SacréBLEU Documentation

### 🧪 Mini-Assignment

- Train model until loss decreases
- Translate 10 test sentences

### ⏱️ Time

| Task                  | Time         |
| --------------------- | ------------ |
| Coding                | 4h           |
| Training & evaluation | 2h           |
| **Total**       | **6h** |

---

## 🟢 Day 7 — Debugging, Visualization & Documentation

### 🎯 Goal

Polish the project into a portfolio-ready deliverable.

### ✅ Tasks

- Visualize attention weights
- Debug gradient flow
- Refactor and clean code
- Write README and documentation

### 📘 What to Study

- Attention heatmap interpretation
- Model explainability
- Documentation best practices

### 🎥 YouTube

- “Visualizing Attention Weights” – CodeEmporium
- “Debugging Deep Learning Models” – :contentReference[oaicite:6]{index=6}

### 📄 Written Resources

- Jay Alammar — *The Illustrated Transformer*
- GitHub README best practices

### 🧪 Mini-Assignment

- Push final code to GitHub
- Add README with explanations and results

### ⏱️ Time

| Task                    | Time         |
| ----------------------- | ------------ |
| Visualization           | 2h           |
| Documentation & cleanup | 3h           |
| **Total**         | **5h** |

---

## ✅ Week 1 Outcome

By the end of Week 1:

- Understood attention intuitively and mathematically
- Implemented a Transformer encoder–decoder from scratch
- Trained a working translation model
- Visualized and explained attention
- Built a strong foundation for advanced Transformer variants
