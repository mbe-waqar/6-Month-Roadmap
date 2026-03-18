# 🦙 Week 3 — LLaMA Architecture, MoE, Quantization & LLM Inference

Modern LLM architectures aur unki efficiency techniques ka deep dive — theory se hands-on tak.

---

## 🗂️ Table of Contents

1. [LLaMA Architecture](#1-llama-architecture)
2. [Improvements over GPT-3 Style Models](#2-improvements-over-gpt-3-style-models)
3. [RMSNorm](#3-rmsnorm)
4. [Rotary Positional Embeddings (RoPE)](#4-rotary-positional-embeddings-rope)
5. [Efficient Training Scaling](#5-efficient-training-scaling)
6. [Key Improvements Introduced by LLaMA](#6-key-improvements-introduced-by-llama)
7. [Recommended Resources — LLaMA](#7-recommended-resources--llama)
8. [Mini-Assignment — Why LLaMA is More Efficient](#8-mini-assignment--why-llama-is-more-efficient)
9. [Sliding Window Attention](#9-sliding-window-attention)
10. [Grouped Query Attention](#10-grouped-query-attention)
11. [LLaMA vs Mistral Comparison](#11-llama-vs-mistral-comparison)
12. [Context Window Efficiency](#12-context-window-efficiency)
13. [Recommended Resources — Mistral](#13-recommended-resources--mistral)
14. [Mixture of Experts (MoE) Architecture](#14-mixture-of-experts-moe-architecture)
15. [Gating Networks](#15-gating-networks)
16. [Sparse Activation](#16-sparse-activation)
17. [Dense vs Sparse Models](#17-dense-vs-sparse-models)
18. [Mini-Assignment — 3 Advantages of MoE](#18-mini-assignment--3-advantages-of-moe)
19. [Quantization Basics](#19-quantization-basics)
20. [4-bit vs 8-bit Inference](#20-4-bit-vs-8-bit-inference)
21. [GPTQ vs AWQ](#21-gptq-vs-awq)
22. [Mini-Assignment — Why 4-bit Quantization Works](#22-mini-assignment--why-4-bit-quantization-works)
23. [Deploy Local LLM — LLaMA 2](#23-deploy-local-llm--llama-2)
24. [Key Tools — Hugging Face Transformers & bitsandbytes](#24-key-tools--hugging-face-transformers--bitsandbytes)
25. [Run 5 Prompts and Save Responses](#25-run-5-prompts-and-save-responses)
26. [Quantize Model + Run Inference Benchmarks](#26-quantize-model--run-inference-benchmarks)
27. [KV Cache, Batching & Memory Optimization](#27-kv-cache-batching--memory-optimization)
28. [vLLM vs Ollama](#28-vllm-vs-ollama)
29. [Inference Optimization — Full Explanation](#29-inference-optimization--full-explanation)
30. [Implement KV Cache + Profile Memory + Measure Speed](#30-implement-kv-cache--profile-memory--measure-speed)
31. [KV Cache — Deep Explanation](#31-kv-cache--deep-explanation)
32. [Performance Report — Quantized vs Full Precision](#32-performance-report--quantized-vs-full-precision)
33. [What is Inference?](#33-what-is-inference)
34. [End-of-Week Confidence Checklist](#34-end-of-week-confidence-checklist)

---

## 1. LLaMA Architecture

### Meaning

LLaMA ek modern Transformer-based LLM architecture hai jo Meta ne develop kiya.

Structure mostly decoder-only Transformer hota hai (GPT jaisa), lekin isme kuch improvements use ki gayi hain taake training efficient aur performance better ho.

**Main components:**
- Token embeddings
- Attention layers
- Feedforward layers
- RMSNorm
- Rotary positional embeddings

### Analogy

Socho GPT ek standard car engine hai. LLaMA us engine ka optimized version hai jo:
- kam fuel use karta hai
- zyada speed deta hai
- zyada efficient hota hai

---

## 2. Improvements over GPT-3 Style Models

GPT-3 aur LLaMA dono decoder-only Transformers hain, lekin LLaMA me kuch efficiency improvements hain.

| Feature | GPT-3 style | LLaMA |
|---------|------------|-------|
| Normalization | LayerNorm | RMSNorm |
| Positional Encoding | Absolute embeddings | RoPE |
| Training efficiency | High compute required | More efficient scaling |
| Model size efficiency | Large parameters needed | Smaller models achieve strong performance |

**Key idea:**
> LLaMA less parameters + better architecture tweaks se strong performance achieve karta hai.

---

## 3. RMSNorm

### Meaning

RMSNorm (Root Mean Square Normalization) ek normalization technique hai jo neural network training ko stable aur faster banati hai.

**Difference:**
- LayerNorm → mean + variance use karta hai
- RMSNorm → sirf RMS value use karta hai

Is wajah se computation simpler aur faster hota hai.

### Analogy

Socho aap students ke marks normalize karna chahte ho.
- LayerNorm → average aur variance calculate karta hai
- RMSNorm → sirf overall magnitude check karta hai

Simple calculation → faster training.

**Benefits:**
- Faster training
- Less computation cost
- Stable learning

---

## 4. Rotary Positional Embeddings (RoPE)

### Meaning

Transformers ko pata hona chahiye ke words kis order me hain.

Rotary Positional Embeddings ek technique hai jo attention mechanism me position information rotate karke encode karti hai.

**Benefits:**
- Long sequences better handle karta hai
- Relative positions samajhta hai
- Long-context tasks me helpful

### Analogy

Socho sentence me words train ke coaches ki tarah hain. RoPE ensure karta hai ke model ko pata ho:
- kaunsa coach pehle hai
- kaunsa baad me
- aur distance bhi samajh aaye

---

## 5. Efficient Training Scaling

### Meaning

Scaling laws ka idea hai: Agar aap data + model size + compute increase karo to performance improve hoti hai.

Lekin LLaMA architecture ko is tarah design kiya gaya hai ke:
- less compute me training possible ho
- smaller models bhi strong results de

Isliye LLaMA models research aur industry me popular hue.

### Analogy

Socho do factories hain:
- **Factory A:** zyada machines use karti hai
- **Factory B:** optimized machines use karti hai → same output, kam energy

Yehi idea LLaMA architecture ka hai.

---

## 6. Key Improvements Introduced by LLaMA

Yeh improvements LLaMA ko older models jaise GPT-3 se zyada efficient aur powerful banate hain. Idea yeh tha ke same ya better performance mile lekin kam parameters aur kam compute ke sath.

| Improvement | Purpose |
|-------------|---------|
| RMSNorm | Faster & simpler normalization |
| RoPE | Better positional understanding |
| Efficient scaling | Strong performance with fewer parameters |
| Better dataset curation | Higher quality learning |

### 1️⃣ RMSNorm Instead of LayerNorm

LLaMA me RMSNorm use hota hai instead of traditional LayerNorm. Computation simple aur faster ho jata hai.

### 2️⃣ Rotary Positional Embeddings (RoPE)

LLaMA me Rotary Positional Embeddings use hota hai jo attention mechanism ke andar position encode karta hai. Long context understanding improve hoti hai.

### 3️⃣ Efficient Parameter Scaling

Instead of sirf parameters increase karna, architecture aur training strategy optimize ki gayi. Same performance with fewer parameters.

### 4️⃣ Better Training Dataset Curation

LLaMA ke training data ko carefully filter aur curate kiya gaya. Yani random internet data nahi — balki high-quality text sources select kiye gaye. Better language understanding, less noise.

**Final Idea:**
> In improvements ki wajah se LLaMA models smaller, kam GPU compute, aur strong performance dete hain.

---

## 7. Recommended Resources — LLaMA

### 🎥 YouTube + Articles

**1️⃣ "LLaMA Explained" — Yannic Kilcher**
- Focus: Transformer architecture details, training strategy, scaling laws, paper results
- Best for: ML researchers, deep architecture understanding

**2️⃣ "The Illustrated LLaMA" — Jay Alammar**
- Focus: Visual diagrams for transformer layers, attention mechanism, RMSNorm & RoPE intuition
- Best for: Beginners, visual learners

**3️⃣ "How LLaMA Works" — AI Coffee Break**
- Focus: RMSNorm, RoPE, efficient scaling, architecture overview
- Best for: Quick understanding, beginners moving to intermediate

### 🎯 Recommended Learning Order

1. Jay Alammar → visuals se basic intuition
2. AI Coffee Break → concept clarity
3. Yannic Kilcher → research-level deep dive

---

## 8. Mini-Assignment — Why LLaMA is More Efficient

### Why LLaMA is More Efficient than Older LLMs

LLaMA ko is tarah design kiya gaya hai ke wo older large language models jaise GPT-3 ke mukable kam parameters aur kam compute ke sath strong performance de sake. LLaMA ki efficiency ka main reason uski architecture improvements aur better training strategy hai.

Sabse pehle, LLaMA me **better dataset curation** use ki gayi hai. Iska matlab hai ke training ke liye high-quality text data carefully select kiya gaya, jis se model ko zyada meaningful patterns seekhne mile. Jab training data clean aur relevant hota hai, to model ko same performance ke liye utne zyada parameters ki zaroorat nahi hoti.

Dusra important factor **efficient parameter scaling** hai. Instead of sirf model size barhane ke, LLaMA architecture ko optimize kiya gaya taake smaller models bhi powerful results de saken.

LLaMA me kuch **architectural improvements** bhi introduce kiye gaye hain jo training aur inference ko efficient banate hain.

**RoPE (Rotary Positional Embeddings)** transformer models ko sentence ke words ka order aur distance samajhne me madad karta hai. Traditional positional embeddings ke comparison me RoPE attention mechanism ke andar hi positional information encode karta hai. Is se model long sequences ko better handle kar sakta hai.

**RMSNorm (Root Mean Square Normalization)** ek simplified normalization technique hai jo traditional LayerNorm ki jagah use hoti hai. LayerNorm mean aur variance dono calculate karta hai, jab ke RMSNorm sirf root mean square value use karta hai. Is se computation simpler aur faster ho jata hai.

In improvements ki wajah se LLaMA models more compute-efficient, scalable aur powerful ban gaye hain.

---

## 9. Sliding Window Attention

### Meaning

Sliding window attention ek technique hai jisme model har token ko sirf nearby tokens par attention dene deta hai, poore sequence par nahi.

Normal transformer me har token sab tokens ko attend karta hai, jo bohat expensive hota hai jab sequence long ho. Sliding window me attention limited range me hota hai.

### Analogy

Socho aap ek book read kar rahe ho. Har sentence samajhne ke liye aap poori book dobara nahi padhte — sirf last few sentences dekhte ho.

Bilkul isi tarah sliding window attention recent context par focus karta hai.

### Example

```
Tokens: Token1 Token2 Token3 Token4 Token5 Token6
Window size = 3:

Token4 sirf Token2, Token3, Token4 ko attend karega
Token5 sirf Token3, Token4, Token5 ko attend karega
```

Attention window ke sath slide karti rehti hai — isliye "sliding window attention".

### Benefits

| Benefit | Explanation |
|---------|-------------|
| Lower Memory | Poore sequence ke sab tokens store aur process nahi karne padte |
| Faster Inference | Kam tokens par attention calculate hoti hai |
| Larger Context Capability | Compute cost kam hone se longer sequences efficiently process ho sakti hain |

---

## 10. Grouped Query Attention

### Meaning

Grouped Query Attention (GQA) ek optimization technique hai jo attention computation ko efficient banati hai.

- **Normal multi-head attention:** Har head ke separate query, key, value matrices hote hain.
- **GQA:** Multiple query heads same key-value heads share karte hain.

### Analogy

Socho ek classroom me students questions pooch rahe hain.
- Normal: Har student ke liye alag teacher answer deta hai.
- GQA: Kuch students same teacher share karte hain → resources bach jate hain.

**Benefits:**
- Memory usage kam
- Faster inference
- Large models efficiently run karte hain

---

## 11. LLaMA vs Mistral Comparison

| Feature | LLaMA | Mistral |
|---------|-------|---------|
| Developer | Meta | Mistral AI |
| Attention type | Standard attention | Sliding window attention |
| Efficiency | Efficient architecture | Even more efficient context handling |
| Key innovation | RMSNorm, RoPE | Sliding window + GQA |
| Context efficiency | Good | Better long-context performance |

**Key idea:** Mistral architecture ne attention efficiency improve ki compared to LLaMA.

---

## 12. Context Window Efficiency

### Meaning

Context window wo maximum tokens hote hain jo model ek time par process kar sakta hai (e.g., 4K, 8K, 32K tokens).

**Problem:** Normal transformer attention ka compute quadratic grow karta hai: `O(n²)` — tokens double hone par compute 4x increase ho jata hai.

**Solutions:**
- Sliding window attention
- Grouped query attention
- Sparse attention

### Analogy

- **Inefficient:** Har baar poori meeting history read karni pade.
- **Efficient:** Sirf recent important notes dekhe jate hain.

---

## 13. Recommended Resources — Mistral

**1️⃣ "Mistral 7B Architecture Explained" — Yannic Kilcher**
- Focus: Sliding Window Attention, GQA, training strategy, comparison with LLaMA
- Best for: Research-level understanding

**2️⃣ "Sliding Window Attention Explained" — AI Coffee Break**
- Focus: Traditional attention problem (O(n²)), sliding window concept, memory & speed improvements
- Best for: Concept clarity with simple visuals

**3️⃣ "Efficient Transformers" — DeepLearning.AI**
- Focus: Sparse attention, memory optimization, long-context models
- Best for: Broader ecosystem and scaling concepts

### 🎯 Recommended Learning Order

1. AI Coffee Break → intuition build karo
2. Yannic Kilcher → deep architecture understanding
3. DeepLearning.AI → broader ecosystem aur scaling

---

## 14. Mixture of Experts (MoE) Architecture

### Meaning

Mixture of Experts (MoE) ek architecture hai jisme model ke andar multiple "expert" networks hote hain, aur har input ke liye sirf kuch selected experts activate hote hain (sab nahi).

**Example:**
- Total experts = 8
- Har token ke liye sirf = 2 experts use hote hain

Is se model ke paas bohat zyada capacity hoti hai, lekin compute kam use hota hai.

### Analogy

Socho ek company me 8 employees hain:
- Agar sab 8 log har kaam kare → time aur energy waste ❌
- Sirf 2 relevant experts kaam kare → fast aur efficient ✅

### Example

```
Input: "Translate English to Urdu"

Gating network select karta hai:
  Expert 2 (language expert)
  Expert 5 (grammar expert)

Sirf ye 2 active → baaki 6 inactive
```

**Benefits:**
- Faster processing
- Lower cost
- Better scalability

---

## 15. Gating Networks

### Meaning

Gating network ek chhota neural network hota hai jo decide karta hai ke kaunse experts use honge. Yeh input ko dekh kar top experts select karta hai (jaise top-2 experts).

**Analogy:** Judge jo sab arguments sunta hai aur sirf relevant specialists bulata hai.

---

## 16. Sparse Activation

### Meaning

Sparse activation ka matlab hai ke model ke total parameters me se sirf kuch hi active hote hain ek time par.

- MoE me 100 experts ho sakte hain
- Lekin ek input ke liye sirf 2–4 experts use hote hain

Is se computation kam ho jata hai.

---

## 17. Dense vs Sparse Models

| Feature | Dense Model | Sparse Model (MoE) |
|---------|------------|-------------------|
| Computation | Har input par saare parameters use hote hain | Sirf selected experts use hote hain |
| Efficiency | Zyada compute aur memory lagti hai | Kam compute, zyada efficient |
| Scalability | Limited (expensive hota hai scale karna) | Easily scale ho jata hai |
| Performance | Stable lekin heavy | Same ya better performance with less compute |

**Analogy:**
- Dense model → har patient ko saare doctors dekhte hain
- MoE model → patient ko sirf specialist doctor dekhta hai

---

## 18. Mini-Assignment — 3 Advantages of MoE

### Analogy

Socho ek office hai:
- Dense model → har task par poori team kaam karti hai
- MoE model → sirf relevant experts kaam karte hain

### 3 Advantages of MoE over Dense Models

**1️⃣ Lower Compute Cost**
MoE me har input ke liye sirf kuch experts active hote hain, is liye computation aur GPU usage kam hota hai.

**2️⃣ Huge Capacity**
MoE me bohat zyada experts ho sakte hain, jis se model ki knowledge aur learning capacity bohat zyada hoti hai compared to dense model.

**3️⃣ Better Scalability**
MoE ko easily scale kiya ja sakta hai (experts add karke), jab ke dense model ko scale karna bohat expensive hota hai.

> **Simple Line:** MoE = "kam cost me zyada powerful aur scalable model"

---

## 19. Quantization Basics

### Meaning

Quantization ka matlab hai model ke weights ko high precision (float32) se low precision (int8 ya int4) me convert karna taake memory kam lage aur inference fast ho jaye.

- Normal model: zyada memory + slow
- Quantized model: kam memory + fast

### Analogy

Socho aap ek photo save kar rahe ho:
- Original photo (HD) → zyada storage leta hai
- Compressed photo → kam storage, thodi quality kam

Quantization bhi same hai: model size chhota hota hai, thodi accuracy drop ho sakti hai.

### float32 → int8 / int4 Conversion

| Precision | Memory Usage | Speed | Accuracy |
|-----------|-------------|-------|----------|
| float32 | Very High | Slow | Best |
| int8 | ~50% less | Faster | Very Good |
| int4 | ~75% less | Fastest | Slight drop |

**Benefits:**
- Smaller memory
- Faster inference
- Lower GPU requirements

---

## 20. 4-bit vs 8-bit Inference

### 8-bit
- Better accuracy
- Thodi zyada memory

### 4-bit
- Bohat kam memory
- Faster inference
- Thodi aur accuracy loss

### Example — GPU Memory (GPT-2 Small, 124M params)

| Format | GPU Memory | Speed |
|--------|-----------|-------|
| Float32 | ~3–4 GB | Slow |
| 8-bit | ~2 GB | Medium |
| 4-bit | ~1 GB | Fast |

---

## 21. GPTQ vs AWQ

### GPTQ (Post-Training Quantization)

GPTQ ek quantization method hai jo model ko 4-bit me convert karta hai bina zyada accuracy lose kiye.
- Post-training apply hota hai
- Har layer ko smartly optimize karta hai
- Error minimization par focus

### AWQ (Activation-Aware Weight Quantization)

AWQ model ke important weights ko protect karta hai.
- Jo weights zyada important hain unko better preserve karta hai
- Baaki ko aggressively quantize karta hai
- Often better accuracy at 4-bit

### Comparison Table

| Feature | GPTQ | AWQ |
|---------|------|-----|
| Approach | Error minimization | Important weights protection |
| Focus | Weights only | Weights + activations |
| Accuracy | Achi | Often better in low-bit (4-bit) |
| Strategy | Mathematical optimization | Importance-based selection |

### Analogy

- **GPTQ** → poori file ko smart tareeke se compress karta hai
- **AWQ** → jo important lines hain unko high quality me rakhta hai, baaki compress karta hai

**Decision:**
- Balanced performance chahiye → GPTQ
- 4-bit me best accuracy chahiye → AWQ

---

## 22. Mini-Assignment — Why 4-bit Quantization Works

### Why 4-bit Quantization Works for LLMs

4-bit quantization ka idea hai ke model ke weights ko 32-bit float se sirf 4-bit integers me convert kar diya jaye. Surprisingly, LLMs me yeh kaafi effective hota hai.

**1️⃣ Redundancy in Large Models**
LLMs ke paas bohot zyada parameters hote hain. Bahut saare neurons similar kaam karte hain, isliye weights me redundancy hoti hai. 4-bit me convert karne se bhi major patterns aur knowledge preserve hoti hai.

**2️⃣ Smart Quantization Techniques**
Techniques jaise GPTQ aur AWQ use hoti hain taake:
- Important weights aur activations ko preserve kiya jaye
- Quantization error minimize ho
- Model almost original performance ke sath run kare

**3️⃣ Memory & Speed Benefits**
- 4-bit → model 8x chhota hota hai (32-bit ke comparison me)
- Kam GPU memory use hoti hai
- Faster inference possible hai
- Large models small GPUs par bhi run ho jate hain

> **One-liner:** "4-bit quantization LLMs ko lightweight aur fast banata hai without significant loss in performance."

---

## 23. Deploy Local LLM — LLaMA 2

### Meaning

Aaj ka goal hai ke aap LLaMA 2 ko apni local machine par run karo, taake bina internet/API ke aap prompts dekar output generate kar sako (local deployment).

### Analogy

- Online Netflix → internet chahiye, subscription chahiye
- Movie download → internet ki zaroorat nahi, fast aur private

Local LLM bhi isi tarah kaam karta hai.

### Step-by-Step

**1️⃣ Install Dependencies (Easy method — Ollama recommended)**

```bash
# Install Ollama
# https://ollama.com/download
```

**2️⃣ Load LLaMA 2 7B**

```bash
ollama run llama2
```

Ye automatically LLaMA 2 7B model download + load kar dega.

**3️⃣ Run Basic Inference**

```
Prompt: What is AI?
```

Model output dega → ye hi inference hai (trained model se answer lena).

**4️⃣ Test Generation Prompts**

```
1. Write a story about a robot
2. Explain machine learning in simple words
3. Translate: Hello → Urdu
```

### Final Outcome

- "Main local LLM run kar sakta hoon"
- "Main prompts test kar sakta hoon"
- "Mujhe inference ka practical idea hai"

---

## 24. Key Tools — Hugging Face Transformers & bitsandbytes

### Meaning

- **Hugging Face Transformers** → models ko load, use aur fine-tune karne ke liye
- **bitsandbytes** → models ko 4-bit / 8-bit me run karne ke liye (memory save karne ke liye)

### Analogy

- Transformers → engine (main system jo car chalata hai)
- bitsandbytes → fuel saver system (kam fuel me zyada distance)

### Example

**Transformers use karna:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
```

**bitsandbytes use karna (4-bit):**

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True
)
```

Ab model kam memory me run hoga.

> **Final Line:** "Transformers model chalata hai, aur bitsandbytes usay affordable banata hai." 🚀

---

## 25. Run 5 Prompts and Save Responses

### Step 1: Install Dependencies

```bash
pip install transformers torch accelerate bitsandbytes
```

### Step 2: Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (use load_in_4bit=True if low GPU memory)
model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
```

### Step 3: Prepare Prompts

```python
prompts = [
    "Explain quantum computing in simple words.",
    "Write a short story about a robot learning human emotions.",
    "Translate the sentence 'Hello, how are you?' into Urdu.",
    "Give 3 tips for learning Python programming.",
    "Summarize the plot of Harry Potter in 2 sentences."
]
```

### Step 4: Run Inference and Save Responses

```python
responses = []

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # or "cpu" if no GPU
    output = model.generate(**inputs, max_new_tokens=100)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(text)

# Save responses to a file
with open("llama_responses.txt", "w", encoding="utf-8") as f:
    for i, resp in enumerate(responses, 1):
        f.write(f"Prompt {i}: {prompts[i-1]}\n")
        f.write(f"Response: {resp}\n\n")

print("Responses saved to llama_responses.txt")
```

### Example Output in File

```
Prompt 1: Explain quantum computing in simple words.
Response: Quantum computing uses tiny particles to perform calculations in ways that classical computers can't.

Prompt 2: Write a short story about a robot learning human emotions.
Response: Once upon a time, a robot discovered it could feel happiness when helping humans...
```

---

## 26. Quantize Model + Run Inference Benchmarks

### 1️⃣ Quantize Model (4-bit / 8-bit)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 8-bit
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

### 2️⃣ Run Inference Benchmark

```python
import time

prompt = "Explain AI in simple words"

def benchmark(model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=100)
    end = time.time()
    return end - start

print("8-bit time:", benchmark(model_8bit))
print("4-bit time:", benchmark(model_4bit))
```

### 3️⃣ Test vLLM

**Install:**

```bash
pip install vllm
```

**Run:**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b")
params = SamplingParams(max_tokens=100)
output = llm.generate("Explain AI in simple words", params)
print(output[0].outputs[0].text)
```

### 4️⃣ Benchmark Comparison Table

| Setup | Speed | Memory | Quality |
|-------|-------|--------|---------|
| Float32 | Slow | High | Best |
| 8-bit | Medium | Medium | Very Good |
| 4-bit | Fast | Low | Slight drop |
| vLLM | Very Fast | Optimized | Good |

**Decision Guide:**
- GPU weak hai → 4-bit best
- Balance chahiye → 8-bit
- Production / high speed → vLLM

---

## 27. KV Cache, Batching & Memory Optimization

### Meaning

Inference ko fast aur efficient banane ke liye 3 key techniques:

### 1️⃣ KV Cache (Most Important)

KV Cache store karta hai previous tokens ke attention values. Next token generation ke waqt puri recomputation nahi hoti — sirf new token process hota hai.

**Analogy:**
- ❌ Har answer ke liye puri kitab dobara parhna (no KV cache)
- ✅ Ek dafa notes bana liye, reuse karo (KV cache)

**Result:** Huge speedup, especially in long prompts.

### 2️⃣ Batching

Instead of: User1 → run, User2 → run

Do: User1 + User2 → ek saath run

**Result:** GPU utilization better, throughput increase.

### 3️⃣ Efficient Memory Usage

Techniques:
- 4-bit / 8-bit quantization
- Offloading (GPU + CPU mix)
- Optimized libraries

**Result:** Large models small systems par run ho jate hain.

### Summary

| Technique | What it does |
|-----------|-------------|
| KV Cache | Dobara kaam na karo |
| Batching | Ek sath zyada kaam karo |
| Memory optimization | Resources smartly use karo |

> **One-line:** "Fast LLM = KV cache + batching + smart memory use" 🚀

---

## 28. vLLM vs Ollama

### Core Idea

| | vLLM | Ollama |
|---|------|--------|
| Purpose | High-speed inference, serving multiple users | Easy local usage for developers/beginners |
| Speed | 🔥 Very fast (optimized) | ⚡ Moderate |
| Setup | Thoda complex | Bohat easy |
| Multi-user | ✅ Yes (serving APIs) | ❌ Limited |
| Memory optimization | Advanced (PagedAttention) | Basic |
| Best for | Production / backend | Local dev / testing |

### Analogy

- **vLLM** = 🏎️ Formula 1 car → bohat fast, optimized, but setup aur control chahiye
- **Ollama** = 🚗 Automatic car → easy chalti hai, sab kuch built-in, beginner friendly

### vLLM ka Secret

- **Paged Attention** → memory ko smart way se manage karta hai
- **Dynamic batching** → different users ke requests combine karta hai
- KV cache ko efficiently reuse karta hai

### Kab Kya Use Karein?

| Use vLLM | Use Ollama |
|----------|-----------|
| API bana rahe ho | Local experiments kar rahe ho |
| Multiple users handle karne hain | Simple setup chahiye |
| GPU ka maximum use chahiye | Learning phase hai |
| Production level system hai | Quick testing karni hai |

---

## 29. Inference Optimization — Full Explanation

### Meaning

Jab tum prompt dete ho, model har token generate karta hai step-by-step aur har step pe previous tokens ko dobara process karta hai — ye process slow + memory heavy hota hai. Optimization ka goal = isse fast aur efficient banana.

### Core Optimization Techniques

**1️⃣ KV Cache** — Previous attention states store, no recomputation.

**2️⃣ Quantization (4-bit / 8-bit)** — Model weights compress, less memory, faster inference.

**3️⃣ Batching** — Multiple requests ek saath process karo, GPU utilization better.

**4️⃣ Efficient Attention**
- Sliding Window Attention
- Grouped Query Attention

Sab tokens ko attend karne ke bajaye sirf relevant tokens → less computation, larger context possible.

**5️⃣ vLLM-style Optimization**
- **Paged Attention** → Memory ko smart way mein manage, fragmentation avoid
- **Continuous Batching** → Requests dynamically add/remove, GPU idle nahi rehta

**6️⃣ KV Cache + Streaming** — Output token-by-token stream karo, user ko wait nahi karna padta → ChatGPT jaisa feel.

### Real-Life Analogy

| Situation | Meaning |
|-----------|---------|
| ❌ Har answer ke liye puri kitab dobara parhna | No KV cache |
| ✅ Ek dafa notes bana liye, reuse | KV cache |
| ❌ Akela kaam → slow | No batching |
| ✅ Team work → fast | Batching |
| ❌ Full kitab uthana → heavy | FP16 |
| ✅ Summary notes → fast | Quantization |

> **Final Summary:** Optimizing LLM inference involves KV caching, quantization, batching, and efficient attention mechanisms to reduce computation, memory usage, and latency while improving throughput and scalability.

---

## 30. Implement KV Cache + Profile Memory + Measure Speed

### 1️⃣ Implement KV Cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "The weather today is"
inputs = tokenizer(prompt, return_tensors="pt")

# First forward pass
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_key_values = outputs.past_key_values
next_token_logits = outputs.logits[:, -1, :]

# Generate next token manually
next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

# Second step (reuse KV cache)
with torch.no_grad():
    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        use_cache=True
    )

print("KV cache working!")
```

**❌ Without KV Cache (Slow):**

```python
for i in range(50):
    outputs = model(input_ids=inputs["input_ids"])
# Har step pe full recomputation hoti hai
```

**✅ With KV Cache (Fast):**

```python
past = None
input_ids = inputs["input_ids"]

for i in range(50):
    outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    past = outputs.past_key_values
    input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
# Sirf new token compute hota hai
```

### 2️⃣ Profile Memory Usage

**GPU Memory:**

```python
import torch

print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
```

**CPU Memory:**

```python
import psutil, os

process = os.getpid()
mem = psutil.Process(process).memory_info().rss / 1024**2
print("RAM Usage:", mem, "MB")
```

### 3️⃣ Measure Token Generation Speed

```python
start = time.time()

past = None
input_ids = inputs["input_ids"]

for _ in range(50):
    outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    past = outputs.past_key_values
    input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

end = time.time()
tokens = 50
print("Time:", end - start)
print("Tokens/sec:", tokens / (end - start))
```

---

## 31. KV Cache — Deep Explanation

### Meaning

KV cache ka matlab hai ke model apne previous tokens ke attention results (keys aur values) ko save kar leta hai taa ke har naya token generate karte waqt dobara puri history compute na karni pade.

> Simple words mein: model "yaad" rakhta hai jo pehle calculate ho chuka hai, aur usko reuse karta hai.

### Analogy

Socho tum ek story likh rahe ho:
- ❌ Har naya sentence likhne se pehle poori story dobara parho → bohat slow
- ✅ Already yaad rakho ke pehle kya likha tha → sirf next line likhni hogi

### Example

```
Prompt: "The weather today is"

❌ Without KV cache:
  Every new word: model processes "The weather today is"
  then "The weather today is sunny"
  then full sentence again → slow

✅ With KV cache:
  Pehli dafa full compute → phir sirf naya word process
  → Fast generation ⚡
```

> **One-line:** KV cache = "pehle ka kaam save karo, dobara mat karo" → generation speed dramatically fast ho jati hai 🚀

---

## 32. Performance Report — Quantized vs Full Precision

### 📝 LLM Inference Optimization Report

**Objective:** Evaluate how KV caching and quantization (4-bit / 8-bit) impact inference speed, memory usage, and overall performance.

**Setup:**
- Model: GPT-2 (causal LLM)
- Libraries: Hugging Face Transformers, bitsandbytes
- Hardware: GPU (e.g., NVIDIA 3060, 12GB)
- Prompt: "Explain AI in simple words" / 50-token generation

### Benchmark Results

| Version | KV Cache | Quantization | Memory Usage | Inference Latency |
|---------|----------|-------------|-------------|-------------------|
| Full Precision (FP16) | OFF | No | High | Slow (~0.05 sec/token) |
| Full Precision (FP16) | ON | No | Slightly higher (cache) | Faster (~0.03 sec/token) |
| 8-bit Quantized | ON | Yes | Medium | Faster (~0.02 sec/token) |
| 4-bit Quantized | ON | Yes | Low | Fastest (~0.015 sec/token) |

### Benchmark Table (Mini-Assignment Format)

| Model | Latency |
|-------|---------|
| Full precision (FP16) | ~2.5 sec |
| 8-bit Quantized | ~1.8 sec |
| 4-bit Quantized | ~1.3 sec |

### Observations

**KV Cache:**
- Without KV cache → every token recomputed → slow
- With KV cache → previous attention stored → speed up 30–50%

**Quantization:**
- Memory usage drastically reduced
- 4-bit fastest inference, slight accuracy drop possible
- 8-bit balanced performance

**Memory vs Speed Trade-off:**
- FP16 → heavy memory, slow
- 4-bit + KV cache → lightweight + very fast

### Trade-offs

| Advantage | Drawback |
|-----------|----------|
| Faster inference | Slight accuracy drop in 4-bit |
| Lower GPU memory | Extra engineering complexity |
| Scalable to longer sequences | Cache uses additional memory |

### Conclusion

KV caching significantly improves token generation speed. Quantization reduces memory and boosts throughput. Best configuration for local or production inference: **4-bit quantization + KV cache** for fastest and memory-efficient generation.

> **One-line takeaway:** "KV Cache + Quantization = Fast, memory-efficient LLM inference without major loss in performance."

---

## 33. What is Inference?

### Meaning

Inference ka matlab hai pre-trained model ko use karna taake new predictions ya outputs generate kare.

- **Training** → model seekhta hai
- **Inference** → model apply hota hai

### Analogy

Socho aap ek student ho:
- **Training** = school me notes aur exercises solve karna
- **Inference** = exam me questions solve karna using jo aapne seekha

### Example

```
GPT-2 case:
Prompt: "The weather today is"

Inference step: model predicts "sunny" ya "rainy"

Model weights already trained hain,
sirf new input (prompt) ke liye output generate ho raha hai.
```

> **Key Point:** Inference me model train nahi hota, sirf learned knowledge use hoti hai to generate predictions.

---

## 34. End-of-Week Confidence Checklist

By the end of Week 3, you should confidently say:

- ✅ "I understand how LLaMA architecture works."
- ✅ "I know why RMSNorm and RoPE are used."
- ✅ "I understand how LLaMA improves efficiency over GPT-3 style models."
- ✅ "I understand sliding window attention."
- ✅ "I know how grouped query attention improves efficiency."
- ✅ "I can compare LLaMA and Mistral architectures."
- ✅ "I understand how context window efficiency works in LLMs."
- ✅ "I can explain MoE, gating networks, and sparse activation."
- ✅ "I understand quantization (4-bit, 8-bit, GPTQ, AWQ)."
- ✅ "I can run LLaMA 2 locally and benchmark inference."
- ✅ "I understand KV cache, batching, and vLLM optimization."

---

*Week 3 — LLaMA, Mistral, MoE, Quantization & LLM Inference Optimization*
