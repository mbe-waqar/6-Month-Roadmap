# 🤖 LLM Fine-Tuning & Pre-Training Concepts
### Roman Urdu Study Guide — Week 2

---

## 📋 Summary / Purpose

Yeh guide BERT aur GPT ke pre-training objectives, fine-tuning techniques, aur parameter-efficient methods (LoRA / QLoRA) ko beginner-friendly roman urdu mein explain karti hai. Is guide ka goal hai ke aap confidently samajh sako:

- BERT aur GPT kaise pre-train hote hain
- Real tasks ke liye models fine-tune karna
- LoRA kab aur kyun use karna chahiye
- Encoder-only vs Decoder-only architecture clearly explain karna

---

## 🗂️ Table of Contents

1. [Next-Token Prediction](#1-next-token-prediction)
2. [Causal Masking](#2-causal-masking)
3. [GPT vs BERT Objectives](#3-gpt-vs-bert-objectives)
4. [Why GPT Excels at Generation](#4-why-gpt-excels-at-generation)
5. [Autoregressive Modeling](#5-autoregressive-modeling)
6. [Causal Attention Mask](#6-causal-attention-mask)
7. [Why GPT Cannot "See the Future"](#7-why-gpt-cannot-see-the-future)
8. [BERT vs GPT Comparison Table](#8-bert-vs-gpt-comparison-table)
9. [Load Pre-trained BERT](#9-load-pre-trained-bert)
10. [Add Classification Head](#10-add-classification-head)
11. [Load GLUE Dataset](#11-load-glue-dataset)
12. [Train and Evaluate Model](#12-train-and-evaluate-model)
13. [[CLS] Token Usage](#13-cls-token-usage)
14. [Classification Head](#14-classification-head)
15. [Accuracy vs F1 Score](#15-accuracy-vs-f1-score)
16. [Load GPT-2](#16-load-gpt-2)
17. [Prepare Custom Dataset](#17-prepare-custom-dataset)
18. [Fine-tune with Causal LM Loss](#18-fine-tune-with-causal-lm-loss)
19. [Generate Sample Text](#19-generate-sample-text)
20. [Teacher Forcing](#20-teacher-forcing)
21. [Overfitting in Generation](#21-overfitting-in-generation)
22. [Sampling — Top-k and Top-p](#22-sampling--top-k-and-top-p)
23. [LoRA & QLoRA](#23-lora--qlora)
24. [Parameter-Efficient Fine-Tuning (PEFT)](#24-parameter-efficient-fine-tuning-peft)
25. [Full FT vs LoRA — Trainable Params & Memory](#25-full-ft-vs-lora--trainable-params--memory)
26. [Week 2 Final Goals](#26-week-2-final-goals)

---

## 1. Next-Token Prediction

### Meaning

Next-token prediction ka matlab hai ke model ko ek sentence ka kuch hissa diya jata hai, aur wo guess karta hai ke agla word (token) kya hoga.

- "Token" ka matlab hota hai word ya word ka hissa.
- Model pichle words ko dekh kar agla word predict karta hai. Isi tarah wo dheere dheere pura sentence bana leta hai.

**Simple shabdon mein:**
> Model har baar bas yeh sochta hai, "ab agla lafz kya aana chahiye?"

### Analogy

Socho aap kisi dost ka sentence complete karte ho.

Agar koi bole:
> "Main subah uth kar chai …"

To aap turant guess karoge: "peeta hoon" ya "peeti hoon".

Aap ne yeh guess is liye kiya kyun ke aap ko pichle words se idea mil gaya. Bilkul isi tarah AI model bhi pichle words dekh kar agla word predict karta hai.

### Example

```
Input:  "Pakistan ka capital hai …"
Output: "Islamabad"
```

Phir agar sentence continue karna ho to wo har step par next word predict karta rahega. Isi technique se models jaise ChatGPT sentences likhte aur jawab dete hain.

---

## 2. Causal Masking

### Meaning

Causal masking ka matlab hai ke model ko sirf past (pehle ke words) dekhne ki permission hoti hai, lekin future (aage ke words) dekhne ki permission nahi hoti.

- Yani jab model next word predict karta hai, to wo sirf pichle words ko use karta hai, aage kya likha hai usay nahi dekh sakta.
- Iska purpose yeh hai ke model cheating na kare aur real tarah se next word predict karna seekhe.

### Analogy

Socho aap exam de rahe ho aur aap ko sentence complete karna hai.

> Teacher ne rule diya: Aap sirf jo words already likhe hain unko dekh sakte ho, lekin aage ka answer nahi dekh sakte.

Agar aap future words dekh loge, to wo cheating hogi. Bilkul isi tarah causal masking model ko future words dekhne se rokta hai.

### Example

```
Sentence:  "Main school ja raha …"
Model sees: "Main school ja raha"
Predicts:   "hoon"
```

Model ko yeh dekhne ki permission nahi hoti ke original sentence mein aage kya likha hai. Is tarah model real tarike se language seekhta hai aur correct next word predict karna seekhta hai.

---

## 3. GPT vs BERT Objectives

### Meaning

GPT aur BERT ke objectives (goals) different hain, kyun ke dono ko different kaam ke liye train kiya jata hai.

**GPT (by OpenAI):**
- GPT ka objective hota hai next-token prediction.
- Yani GPT pichle words ko dekh kar agla word predict karna seekhta hai.
- Is liye GPT text likhne, chatting, aur content generate karne mein strong hota hai.

**BERT (by Google):**
- BERT ka objective hota hai masked words ko predict karna.
- Yani sentence ke beech se kuch words hide kar diye jate hain, aur BERT un missing words ko guess karta hai.
- Is liye BERT text ko samajhne (understanding) mein strong hota hai, jaise question answering.

### Analogy

- **GPT analogy:** Socho aap story likh rahe ho. Aap har line ke baad next line khud soch kar likhte ho. Yeh GPT jaisa hai — wo agla word khud generate karta hai.
- **BERT analogy:** Socho sentence mein blank diya ho: `"Main ___ ja raha hoon"` Aur aap ko blank fill karna ho. Yeh BERT jaisa hai — wo missing words ko fill karta hai.

### Example

```
GPT example:
Input:   "Aaj mausam bohat …"
Output:  "acha hai"
Goal:    agla word generate karna

BERT example:
Input:   "Aaj ___ bohat acha hai"
Output:  "mausam"
Goal:    missing word samajh kar fill karna
```

### Short Summary

| Model | Task |
|-------|------|
| GPT   | next word predict karta hai (text generate karne ke liye best) |
| BERT  | missing words predict karta hai (text samajhne ke liye best) |

---

## 4. Why GPT Excels at Generation

### Meaning

GPT generation mein is liye excel karta hai kyun ke usay next-token prediction par train kiya jata hai. Yani GPT har step par yeh seekhta hai ke agla word kya hona chahiye, based on pichle words.

Is training ki wajah se GPT naturally sentences, paragraphs, aur poori stories generate kar sakta hai. GPT ko OpenAI ne is tarah design kiya hai ke wo language ka flow samjhe aur smoothly next words likhta jaye.

**Simple words mein:**
> GPT ko specifically text likhna aur continue karna sikhaya gaya hai, is liye wo generation mein best hai.

### Analogy

Socho aap ko story likhne ki practice roz karwai jaye. Har din aap ko ek sentence diya jata hai, aur aap ko usay continue karna hota hai. Kuch time baad aap expert ho jaoge story continue karne mein.

Bilkul isi tarah GPT ne billions sentences continue karne ki practice ki hoti hai, is liye wo easily naya text generate kar leta hai.

### Example

```
Input: "Ek din ek larka jungle mein gaya aur usay ek …"
GPT:   "purana khazana mil gaya jo sone se bhara hua tha."
```

Isi wajah se GPT chatting, story writing, aur content generation mein bohat powerful hai.

---

## 5. Autoregressive Modeling

### Meaning

Autoregressive modeling ka matlab hai ke model ek ek karke next word predict karta hai, aur har naya word predict karne ke liye pichle predicted words ko bhi use karta hai.

- "Auto" ka matlab khud, aur "regressive" ka matlab previous cheezon par depend karna.
- Model pehle word dekhta hai → next word predict karta hai → phir dono words use karke teesra predict karta hai → aur aise hi sentence continue karta hai.

Yeh technique models ko naturally text generate karna sikhati hai.

### Analogy

Socho aap WhatsApp par message likh rahe ho: `"Kal main market gaya aur …"` Phir aap sochte ho aur likhte ho: `"fruit khareede"` Phir continue karte ho: `"aur ghar wapas aa gaya"`.

Yani har naya word aap pichle words ko dekh kar likh rahe ho. Bilkul isi tarah autoregressive model bhi step by step sentence banata hai.

### Example

```
Start: "Main cricket …"

Step 1: predict → "khelta"   → "Main cricket khelta"
Step 2: predict → "hoon"     → "Main cricket khelta hoon"
Step 3: predict → "roz"      → "Main cricket khelta hoon roz"
```

Har step par model previous words use kar raha hai. Isi process ko autoregressive modeling kehte hain, aur isi wajah se AI models smoothly text generate kar pate hain.

---

## 6. Causal Attention Mask

### Meaning

Causal attention mask ek technique hai jo model ko yeh ensure karti hai ke wo sirf pichle words par focus kare, future words par nahi.

- "Attention" ka matlab hota hai ke model decide kare kaun se words important hain.
- "Mask" ka matlab hota hai kuch cheezon ko chhupa dena.

**Simple words mein:**
> Causal attention mask future words ko hide kar deta hai taake model sirf past dekh kar next word predict kare.

Yeh mostly autoregressive models (jaise GPT) mein use hota hai.

### Analogy

Socho aap ek line mein kharay ho aur aap sirf apne aage kharay logon ko dekh sakte ho, peeche walon ko nahi. Aap ka decision sirf un logon par depend karega jo aap ke samne hain.

Bilkul isi tarah causal attention mask model ko sirf pichle words dekhne deta hai, aage ke words ko block kar deta hai.

### Example

```
Sentence: "Main kal school gaya tha"

Model focusing on "school" can only see: "Main kal"
Cannot see: "gaya tha"  ← masked (future words)
```

Is tarah model cheating nahi karta aur sahi tareeke se next word predict karna seekhta hai.

---

## 7. Why GPT Cannot "See the Future"

### Meaning

GPT "future nahi dekh sakta" kyun ke usay causal masking ke sath train kiya jata hai. Iska matlab hai ke jab GPT next word predict karta hai, to wo sirf pichle words dekh sakta hai, aage ke words nahi.

GPT ka objective hi yeh hota hai ke wo next-token prediction kare — yani har step par agla lafz guess kare bina future dekhe. Agar wo future dekh lega to learning sahi nahi hogi (cheating ho jayegi).

### Analogy

Socho aap exam mein ho aur aap ko sentence complete karna hai.

> Rule: Aap sirf jo pehle likha hai woh dekh sakte ho, lekin answer sheet ka aage wala hissa nahi dekh sakte. Agar aap future dekh loge to wo cheating hogi.

Bilkul isi tarah GPT ko future words dekhne ki permission nahi hoti.

### Example

```
Sentence: "Main kal market gaya aur …"

GPT sees:    "Main kal market gaya aur"
GPT guesses: "fruit khareede" ya "dost se mila"
```

GPT ko yeh nahi pata hota ke asli sentence mein aage kya likha tha. Isi liye kehte hain GPT future nahi dekh sakta — wo sirf past ko use karke next word banata hai.

---

## 8. BERT vs GPT Comparison Table

### Mini-Assignment: Comparison Table

| Feature | BERT | GPT |
|---------|------|-----|
| Objective | Masked language modeling: missing words ko predict karta hai | Next-token prediction: agla word predict karta hai |
| Architecture | Encoder-only Transformer (sirf text samajhne ke liye) | Decoder-only Transformer (text generate karne ke liye) |
| Use Cases | Text understanding tasks: question answering, sentiment analysis, NER | Text generation tasks: story writing, chatting, content creation |

### Notes

- BERT sentence ka meaning samajhne mein strong hai.
- GPT sentences aur paragraphs generate karne mein strong hai.
- Dono Transformer-based hain, lekin training objectives aur design alag hain.

---

## 9. Load Pre-trained BERT

### Meaning

Load pre-trained BERT ka matlab hai ke aap pehle se trained BERT model ko use karte ho instead of usay zero se train karne ke.

BERT ko Google ne huge text data par pehle hi train kiya hota hai. Is liye jab aap usay load karte ho to model ko language ka already achha knowledge hota hai. Phir aap us model ko apne specific task (jaise sentiment analysis ya question answering) ke liye use ya fine-tune kar sakte ho.

### Analogy

Socho ek student ne pehle se English language bohat achi tarah seekh li hai. Ab jab usay koi new job di jati hai (jaise translation ya essay writing), to usay language dobara zero se nahi seekhni parti. Wo bas apni existing knowledge use karta hai.

Bilkul isi tarah pre-trained BERT already language samajhta hai, aur aap seedha usay use kar sakte ho.

### Example

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

- `tokenizer` text ko tokens mein convert karta hai
- `model` pre-trained BERT ko load karta hai

Ab aap is model ko text understanding tasks ke liye use kar sakte ho.

---

## 10. Add Classification Head

### Meaning

Add classification head ka matlab hai ke aap ek extra neural network layer BERT ke upar add karte ho taake model classification task kar sake.

BERT normally sirf text ko samajhta (understand) hai. Lekin jab hum uske upar ek classification layer add karte hain, to model decide kar sakta hai ke text kis category mein aata hai.

Yeh technique aksar Hugging Face Transformers ke sath use hoti hai jab hum BERT ko sentiment analysis ya spam detection ke liye use karte hain.

### Analogy

Socho BERT ek teacher hai jo essay ko achi tarah samajh sakta hai. Lekin agar teacher ko yeh bhi batana ho ke essay positive hai ya negative, to usay ek extra decision box diya jata hai jo final category decide karta hai. Yeh decision box hi classification head hai.

### Example

```
Sentence: "Yeh movie bohat achi thi"

Process:
1. BERT sentence ka meaning samajhta hai.
2. Classification head decide karta hai.
Output → Positive
```

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

- BERT text samajhta hai
- Classification head decide karta hai ke text positive ya negative hai

---

## 11. Load GLUE Dataset

### Meaning

Load GLUE dataset ka matlab hai ke aap GLUE benchmark ka dataset apne program mein import ya load karte ho taake NLP models ko train aur test kar sako.

- GLUE ka full form hai **General Language Understanding Evaluation**.
- Yeh ek collection hai different NLP tasks ka (jaise sentence similarity, sentiment analysis, etc.) jo models ki language understanding check karne ke liye use hota hai.
- Yeh benchmark New York University, University of Washington aur DeepMind ke researchers ne introduce kiya tha.

### Analogy

Socho ek exam paper hai jisme different subjects ke questions hain. Agar aap student ko test karna chahte ho ke wo kitna smart hai, to aap usay wo exam paper dete ho. Bilkul isi tarah GLUE dataset AI models ke liye ek exam ki tarah hota hai jisse unki language understanding check hoti hai.

### Example

```python
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
```

- `"glue"` → benchmark dataset
- `"sst2"` → sentiment analysis task

Ab aap is dataset ko use karke apne NLP model ko train aur evaluate kar sakte ho.

---

## 12. Train and Evaluate Model

### Meaning

Train and evaluate model ka matlab hai ke pehle model ko data se sikhaya jata hai (training) aur phir check kiya jata hai ke model kitna sahi perform kar raha hai (evaluation).

- **Training** mein model examples se patterns seekhta hai.
- **Evaluation** mein hum test data use karke dekhte hain ke model ke predictions accurate hain ya nahi.

### Analogy

Socho ek student exam ki tayari kar raha hai.

- **Training** = student books aur practice questions se study karta hai.
- **Evaluation** = phir uska test liya jata hai taake pata chale usne kitna sahi seekha hai.

### Example

**Training:**
```
"Yeh movie bohat achi hai" → Positive
"Yeh movie boring hai"     → Negative
```
Model inse seekhta hai.

**Evaluation:**
```
Input:   "Yeh film zabardast thi"
Predict: Positive
Check:   Prediction correct hai ya nahi?
```

---

## 13. [CLS] Token Usage

### Meaning

`[CLS]` token ek special token hota hai jo sentence ke start mein add kiya jata hai jab hum BERT use karte hain.

Model jab poora sentence process karta hai, to `[CLS]` token ka final representation poore sentence ka summary ban jata hai. Isi summary ko baad mein classification ke liye use kiya jata hai.

### Analogy

Socho ek file ka cover page hota hai jisme document ka short summary hota hai. Agar kisi ko jaldi samajhna ho document kis bare mein hai, to wo cover page dekh leta hai. Bilkul isi tarah `[CLS]` token poore sentence ka summary represent karta hai.

### Example

```
Sentence: "I love this movie"

Input to BERT: [CLS] I love this movie [SEP]

Training ke baad [CLS] token ka output decide karta hai: Positive ya Negative
```

---

## 14. Classification Head

### Meaning

Classification head ek extra neural network layer hoti hai jo model ke output ko final category (label) mein convert karti hai.

Model text ka meaning samajhta hai, aur classification head decide karta hai ke output kis class mein aata hai.

### Analogy

Socho ek judge hai jo sab arguments sunta hai aur phir final decision deta hai.

- Model → information samajhta hai
- Classification head → final decision deta hai

### Example

```
Sentence: "Yeh product bohat acha hai"
Process:
1. BERT sentence samajhta hai
2. Classification head decide karta hai
Output → Positive
```

---

## 15. Accuracy vs F1 Score

### Meaning

**Accuracy:**
- Yeh batata hai model ke kitne predictions correct hain overall.
- Formula: `Correct predictions ÷ Total predictions`

**F1 Score:**
- Yeh ek balanced metric hai jo precision aur recall dono ko combine karta hai.
- Yeh zyada useful hota hai jab dataset imbalanced ho (classes barabar na hon).

### Analogy

Socho ek cricket batsman hai.

- **Accuracy** = kitni balls par sahi shot laga
- **F1 score** = sirf shots nahi, balki kitni important runs aur consistency bhi consider karta hai

Yani F1 zyada balanced performance batata hai.

### Example

```
100 emails check kiye:
Model ne 90 correct predict kiye

Accuracy = 90 / 100 = 90%

Lekin agar spam emails bohat kam hon aur model unhe miss kar raha ho,
to F1 score kam ho sakta hai.
```

> Is liye spam detection jaise tasks mein F1 score zyada important hota hai.

---

## 16. Load GPT-2

### Meaning

Load GPT-2 (small ya medium) ka matlab hai ke aap pehle se trained GPT-2 model ko apne program mein load karte ho taake usay text generation ke liye use kar sako.

GPT-2 ke different sizes hote hain:
- **Small** → fast hota hai aur kam resources use karta hai
- **Medium** → zyada powerful hota hai lekin thoda zyada compute chahiye hota hai

### Analogy

Socho aap ek ready-made software tool download karte ho. Aap ko khud se pura software banana nahi parta — bas install karke use kar lete ho. Bilkul isi tarah pre-trained GPT-2 model load karke seedha use kiya ja sakta hai.

### Example

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Small model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Medium model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
```

Ab aap is model ko text generate karne, story writing, ya chatbot banane ke liye use kar sakte ho.

---

## 17. Prepare Custom Dataset

### Meaning

Prepare custom dataset ka matlab hai ke aap apna khud ka data collect aur organize karte ho taake AI model ko train kiya ja sake. Yeh data aap kisi specific task ke liye banate ho, jaise chatbot, sentiment analysis, ya text classification.

**Simple words mein:**
> Custom dataset = wo data jo aap khud prepare karte ho model ko sikhane ke liye.

### Analogy

Socho ek teacher students ko exam ki tayari karwana chahta hai. Agar teacher khud practice questions ka set banaye, to wo students ke liye ek custom practice dataset ban gaya. Bilkul isi tarah AI model ko sikhane ke liye hum apna custom dataset banate hain.

### Example

| Text | Label |
|------|-------|
| "Yeh product bohat acha hai" | Positive |
| "Delivery bohat slow thi" | Negative |
| "Quality theek hai" | Neutral |

**Steps:**
1. Data collect karo (reviews, comments, etc.)
2. Data clean karo (extra symbols remove karo)
3. Labels assign karo (Positive / Negative etc.)
4. Phir isi dataset ko model training ke liye use karo

---

## 18. Fine-tune with Causal LM Loss

### Meaning

Fine-tuning with causal LM loss ka matlab hai ke aap ek pre-trained model (jaise GPT-2) ko apne custom dataset par dubara train karte ho taake wo aapke specific task ya style ko better seekh le.

Causal LM loss model ko yeh sikhata hai ke pichle words dekh kar agla word predict kare. Agar prediction galat ho to loss calculate hota hai aur model apni learning improve karta hai.

### Analogy

Socho ek writer already writing jaanta hai, lekin aap usay sirf news articles likhna sikhana chahte ho. Aap usay news articles read karwa kar practice karwate ho. Dheere dheere wo news style mein likhna seekh jata hai. Yeh process hi fine-tuning jaisa hai.

### Example

```
Custom dataset: tech articles

Input:   "Artificial intelligence is changing …"
Model:   "the world of technology."

Agar prediction wrong ho to loss calculate hota hai
aur training se model better hota jata hai.
```

---

## 19. Generate Sample Text

### Meaning

Generate sample text ka matlab hai ke trained model ko ek starting prompt diya jata hai aur wo usay continue karke naya text banata hai.

### Analogy

Socho kisi ko story start karne ko bolo: `"Ek din jungle mein …"` Phir wo khud se story continue karta hai. Bilkul isi tarah language model bhi prompt ko continue karta hai.

### Example

```
Prompt:    "AI will change the future because …"
Generated: "it helps automate tasks and improves decision making in many industries."
```

---

## 20. Teacher Forcing

### Meaning

Teacher forcing training technique hai jahan model ko correct previous word diya jata hai instead of uska predicted word. Yeh training ko fast aur stable banata hai.

### Analogy

Socho teacher student ko sentence bol bol kar likhwa raha hai. Student guess nahi karta — teacher correct word bata deta hai.

### Example

```
Sentence: "I love machine learning"

Training ke waqt model ko next step par correct word diya jata hai:
"I → love → machine → learning"
```

---

## 21. Overfitting in Generation

### Meaning

Overfitting tab hota hai jab model training data ko bohat zyada yaad kar leta hai aur naye prompts par achha perform nahi karta.

### Analogy

Socho student sirf ratta laga leta hai aur exam mein thoda different question aaye to answer nahi de pata.

### Example

```
Agar model sirf ek hi story dataset par train ho:

Prompt: "Once upon a time …"
Model har baar same story repeat karega.
→ Yeh overfitting hai.
```

---

## 22. Sampling — Top-k and Top-p

### Meaning

Sampling techniques text generation ko more natural aur diverse banati hain.

- **Top-k sampling** → model sirf `k` most probable words mein se choose karta hai
- **Top-p sampling (nucleus sampling)** → model highest probability words ka group choose karta hai jinka total probability `p` hota hai

### Analogy

Socho aap sentence continue kar rahe ho. Aap ke paas 50 possible words hain, lekin aap sirf best few options mein se choose karte ho.

### Example

```
Prompt: "The weather today is …"

Word probabilities:
sunny    → 0.40
cold     → 0.30
rainy    → 0.20
terrible → 0.10
```

**Top-k (k=2):**
```
Top 2 words: sunny (0.40), cold (0.30)
rainy aur terrible ignore ho jayenge.
Output: "The weather today is sunny" OR "The weather today is cold"
```

**Top-p (p=0.7):**
```
Word     | Prob | Cumulative
sunny    | 0.40 | 0.40
cold     | 0.30 | 0.70  ← p reached

Selected words: sunny, cold
Model in dono mein se choose karega.
```

**Top-p (p=0.9):**
```
Word     | Prob | Cumulative
sunny    | 0.40 | 0.40
cold     | 0.30 | 0.70
rainy    | 0.20 | 0.90  ← p reached

Selected words: sunny, cold, rainy
```

### Comparison Table

| Method | Kaise select hota hai |
|--------|-----------------------|
| Top-k  | Fixed number of top words |
| Top-p  | Probability threshold tak words |

---

## 23. LoRA & QLoRA

### 🎯 Goal: Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) ka matlab hai ke aap bade pre-trained models ko train karte waqt sirf kuch parameters update karte ho, taake:
- Training fast ho
- Memory kam use ho
- Model ka performance bohat zyada na giray

Yani poore model ko fine-tune karne ki jagah sirf choti layers ya low-rank matrices update karte ho.

---

### LoRA (Low-Rank Adaptation)

**Meaning:**
- Bada model freeze karte hain
- Sirf kuch low-rank matrices add karte hain jo new task ke liye train hote hain

**Math:**
```
W_new = W + A @ B

A aur B → small low-rank matrices (trainable)
W       → original weights (frozen)
```

**Analogy:**
> Socho aap ek expert chef ho. Aap naye recipe ke liye poore kitchen ko change nahi karte, sirf ek extra spice mix add karte ho.

---

### QLoRA (Quantized LoRA)

**Meaning:**
- LoRA + quantization (model ko compress karna)
- Memory aur storage aur bhi kam use hoti hai

---

### LoRA Ranks — Experimentation

Rank decide karta hai ki low-rank matrices kitni capacity rakhen.

| Rank | Performance | Memory |
|------|-------------|--------|
| Higher rank | Better performance | Zyada memory |
| Lower rank  | Slight performance loss | Kam memory |

> **Rule of thumb:** Balance choose karna hota hai rank aur memory ke beech.

---

## 24. Parameter-Efficient Fine-Tuning (PEFT)

### Meaning

Parameter-efficient fine-tuning (PEFT) ka goal hai ke sirf choti subset of parameters train ki jaye instead of entire model, taake:
- Training fast ho
- Memory use kam ho
- Large models ko easily adapt kiya ja sake

LoRA aur QLoRA iske best examples hain.

### Analogy

Socho aap ek expert writer ho aur sirf ek chapter ya paragraph style update karna hai, poori book nahi.

### What to Study

| Topic | Description |
|-------|-------------|
| Low-rank adaptation | Large weight matrix ko do small matrices mein split karna |
| Why LoRA reduces memory | Sirf few extra parameters train hote hain |
| Trade-off: rank vs performance | Higher rank → better perf + more memory; Lower rank → less memory |

---

## 25. Full FT vs LoRA — Trainable Params & Memory

### Trainable Parameters Comparison

| Approach | Trainable Parameters | Explanation |
|----------|---------------------|-------------|
| Full Fine-Tuning | 100% of model weights | GPT-2 small (124M params) → saare 124M update honge |
| LoRA Fine-Tuning | ~1–5% of weights | Sirf choti extra matrices A aur B train hote hain |

### Memory Usage Comparison

| Approach | Memory Usage | Notes |
|----------|-------------|-------|
| Full Fine-Tuning | High | GPU memory pe full model ke gradients store hote hain |
| LoRA | Low | Sirf low-rank matrices ke gradients store hote hain |

### Example (GPT-2 Small — 124M parameters)

```
Full FT → ~3–4 GB GPU memory minimum
LoRA   → ~0.1–0.2 GB extra memory (for trainable matrices)
```

**Analogy:**
- Full FT → poora car engine modify karna
- LoRA → sirf engine ke small parts adjust karna

### Recording Memory in PyTorch

```python
import torch

# Before training
print(torch.cuda.memory_allocated() / 1e6, "MB")

# After initializing LoRA or full fine-tuning model
# Start training...
print(torch.cuda.memory_allocated() / 1e6, "MB")
```

### Measuring Training Speed

```python
import time

start = time.time()
# training loop
end = time.time()
print("Training time:", end - start, "seconds")
```

### Writing a Performance Report

Include the following in your report:
- Trainable params count
- Memory usage comparison
- Training speed per epoch
- Accuracy / F1 or generation quality
- Observations & trade-offs

**Example Observation:**
> LoRA reduced memory by ~90% and training was 3x faster. Accuracy drop negligible (~1%). Ideal for large GPT models.

---

## 26. Week 2 Final Goals

By the end of Week 2, you should confidently say:

### ✅ Confidence Checklist

---

**"I understand how BERT and GPT are pre-trained"**

- **BERT** → Masked Language Modeling (MLM) | Encoder-only | Text understanding
- **GPT** → Next-Token Prediction (Autoregressive LM) | Decoder-only | Text generation
- **Key point:** BERT samajhta hai (reading comprehension), GPT generate karta hai (writing/completion).

---

**"I can fine-tune models for real tasks"**

- Fine-tuning → pre-trained model ko apne custom dataset ya task ke liye train karna
- BERT → sentiment analysis, question answering
- GPT → story writing, chatbot responses
- Full FT → poora model train hota hai
- LoRA / PEFT → sirf small low-rank matrices train hote hain → memory-efficient

---

**"I know when and why to use LoRA"**

- Large LLMs (GPT, GPT-2/3, LLaMA, Falcon)
- Memory limited GPUs
- Tasks where domain adaptation is needed without full retraining
- Full FT feasible only for small models or high-resource environments

---

**"I can explain encoder-only vs decoder-only clearly"**

| Type | Encoder-only | Decoder-only |
|------|-------------|--------------|
| Example | BERT | GPT |
| Focus | Text understanding | Text generation |
| Architecture | Sirf encoder layers | Sirf decoder layers |
| Pre-training | Masked words predict karna | Next-token predict karna |
| Use Case | Classification, QA, NER | Chatbot, story, code generation |

**Key point:**
- Encoder → sentence ka meaning samajhta hai
- Decoder → sentence ko generate karta hai, step-by-step

---

## 📝 Notes & Tips

> **Tip 1:** BERT FT → better for classification, comprehension. GPT FT → better for text generation or completion tasks.

> **Tip 2:** Full FT → higher flexibility, more memory & slower. LoRA → memory-efficient, faster, slightly less flexible.

> **Tip 3:** LoRA memory difference clearly dikhata hai. Large models me GPU requirement significantly kam ho jati hai.

> **Tip 4:** Deployment mindset — Use LoRA / PEFT for cloud or edge deployment.

> **Tip 5:** F1 score zyada important hota hai jab dataset imbalanced ho, jaise spam detection.

> **Tip 6:** Bigger models → better performance but more compute/memory. PEFT techniques maintain performance while reducing compute cost.

---

*This guide covers Week 2 of LLM study: pre-training objectives, fine-tuning, LoRA/QLoRA, and efficient training strategies.*
