# 7.5-Month Advanced AI Specialization - Complete Roadmap

**Duration:** 31 weeks (Week 0 + Weeks 1-30) = 8 months  
**Prep Week:** November 24-30, 2025  
**Main Program:** December 1, 2025 - June 28, 2026  
**Total Hours:** 620 hours (15-25h prep + 600h main)

---

## ⚠️ BEFORE YOU BEGIN

**This is an ADVANCED program.** Please complete Prerequisites_Guide.md first to:
- Check if you meet all prerequisites
- Complete self-assessment tests
- Find preparation resources if needed
- Plan your timeline

**Prerequisites Required:**
- Python (intermediate level)
- ML basics (can train simple models)
- Math foundations (linear algebra, calculus, probability)
- Dev tools (command line, Git, Jupyter)

---

## TABLE OF CONTENTS

1. [Week 0: Preparation Week](#week-0-preparation)
2. [Phase 1: GenAI & NLP (Weeks 1-8)](#phase-1-genai-nlp)
3. [Phase 2: ML Engineering (Weeks 9-12)](#phase-2-ml-engineering)
4. [Phase 3: AI Automation (Weeks 13-15)](#phase-3-ai-automation)
5. [Phase 4: Agentic AI (Weeks 16-19)](#phase-4-agentic-ai)
6. [Phase 5: ML Integration (Week 20)](#phase-5-ml-integration)
7. [Phase 6: Robotics + AI (Weeks 21-30)](#phase-6-robotics)
8. [Capstone Projects](#capstone-projects)
9. [Resources & Tools](#resources)
10. [Environment Setup](#environment-setup)
11. [Assessment Framework](#assessment)

---

## WEEK 0: PREPARATION WEEK (Nov 24-30, 2025)

### **Overview**
Dedicated preparation week to ensure you're ready for the intensive 30-week program. **This week is MANDATORY** even if you meet all prerequisites.

**Time Commitment:** 15-25 hours (flexible based on your level)
- Strong foundations: 15 hours (focus on setup)
- Need review: 20-25 hours (study + setup)

**Goal:** Start Week 1 with complete environment, refreshed fundamentals, and organized systems

---

### **Technical Setup (8-10 hours)**

**Python Environment (3-4h):**
- Install Python 3.10 or 3.11 (avoid 3.12, some libraries not compatible)
- Set up virtual environment: `python -m venv ai-env`
- Test: `python --version` should show 3.10.x or 3.11.x
- Install pip upgrades: `pip install --upgrade pip setuptools wheel`

**Deep Learning Frameworks (3-4h):**
- Install PyTorch with CUDA:
  ```bash
  # Check CUDA version first: nvidia-smi
  # Visit pytorch.org for exact command
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Test PyTorch + GPU:
  ```python
  import torch
  print(torch.__version__)
  print(torch.cuda.is_available())  # Should be True
  print(torch.cuda.get_device_name(0))
  ```
- Install Hugging Face libraries:
  ```bash
  pip install transformers datasets accelerate peft bitsandbytes
  ```

**Agentic AI Tools (2h):**
- Install LangChain ecosystem:
  ```bash
  pip install langchain langchain-community langchain-core
  pip install langgraph
  pip install langsmith  # For debugging
  ```
- Install agent frameworks:
  ```bash
  pip install autogen
  pip install crewai crewai-tools
  ```

**MLOps & Development Tools (2-3h):**
- Install MLflow: `pip install mlflow`
- Install experiment tracking: `pip install wandb`
- Install Docker (follow official docs for your OS)
- Test Docker: `docker run hello-world`
- Install Git and configure:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```
- Install VS Code + Python extension

**Databases & APIs (1h):**
- Install vector databases:
  ```bash
  pip install chromadb
  pip install pinecone-client
  pip install faiss-cpu  # or faiss-gpu
  ```
- Install FastAPI: `pip install fastapi uvicorn`

**Verification Checklist:**
- [ ] Python 3.10/3.11 installed
- [ ] PyTorch with CUDA working
- [ ] GPU detected and functional
- [ ] Hugging Face libraries installed
- [ ] LangChain + LangGraph installed
- [ ] MLflow and Docker working
- [ ] Git configured
- [ ] VS Code ready with Python extension
- [ ] All imports work without errors

---

### **Knowledge Review (5-10 hours)**

**Python Refresher (2-3h):**
- Review OOP concepts (classes, inheritance, decorators)
- Practice: Implement a class with `__init__`, methods, properties
- Review async/await (helpful for Week 1+)
- Practice: List comprehensions, lambda functions
- Review context managers (`with` statements)

**ML Fundamentals Review (2-3h):**
- Revisit: Loss functions, gradient descent, backpropagation
- Review: Training/validation/test splits, overfitting
- Review: Evaluation metrics (accuracy, precision, recall, F1)
- Practice: Train a simple neural network in PyTorch
- Understand: Why we need GPUs, what CUDA does

**Math Concepts (2-3h):**
- Linear Algebra: Matrix multiplication, dot products, transpose
- Calculus: Derivatives, chain rule (critical for backprop)
- Probability: Distributions, conditional probability
- Practice: Calculate gradient of simple function by hand
- Watch: 3Blue1Brown "Essence of Linear Algebra" (key videos)

**Key Papers (2-3h):**
- Read: "Attention Is All You Need" (Vaswani et al., 2017)
  - Focus on: Self-attention mechanism, positional encoding
  - Goal: Understand transformer architecture before Week 1
- Skim: BERT paper (understand pre-training concept)
- Skim: GPT-2/GPT-3 papers (understand autoregressive generation)

**PyTorch Basics (1-2h):**
- Tutorial: Official PyTorch 60-minute blitz
- Practice: Tensors, autograd, simple neural network
- Understand: How to move models/data to GPU

---

### **Organization & Planning (2-5 hours)**

**Tracking Setup (1-2h):**
- Import Task_Tracker.csv to Google Sheets
- Add conditional formatting (color code by status)
- Create progress tracking formulas
- Set up Notion workspace (optional)
- Copy Notion_Template.md sections to Notion

**Physical Setup (1h):**
- Print Quick_Reference.md (keep at desk)
- Print key pages from PDF (schedule, capstones)
- Organize workspace for 8 months of study
- Set up dual monitors if possible
- Prepare notebook for handwritten notes

**Calendar Planning (1h):**
- Block 20 hours/week for all 31 weeks
- Mark capstone weeks (8, 19, 20, 30) in color
- Set up weekly recurring events:
  - Monday: Theory (4h)
  - Tuesday-Thursday: Implementation (12h total)
  - Friday: Documentation (2h)
  - Saturday: Testing (2h)
  - Sunday: Review (1h)
- Add buffer days before capstones

**Community & Support (1-2h):**
- Join Hugging Face Discord (huggingface.co/discord)
- Subscribe to r/MachineLearning
- Join r/LanguageTechnology
- Join LangChain Discord
- Join ROS Discourse (for later robotics phase)
- Join MLOps Community Slack
- Find accountability partner (optional but recommended)
  - Post in Discord/Reddit looking for study partner
  - Schedule weekly check-ins

**GitHub Setup (30min):**
- Create repositories:
  - `ai-specialization-main` (main project repo)
  - `weekly-projects` (smaller weekly deliverables)
  - `capstone-1-rag-app`
  - `capstone-2-multi-agent`
  - `capstone-3-robot`
- Set up README templates
- Initialize with .gitignore for Python

---

### **Week 0 Deliverables**

By November 30, you must have:
- ✅ All software installed and tested
- ✅ GPU working (if applicable)
- ✅ Fundamentals reviewed and refreshed
- ✅ Tracking systems set up (Sheets, Notion)
- ✅ Calendar blocked for 31 weeks
- ✅ Communities joined
- ✅ GitHub repos created
- ✅ Workspace organized
- ✅ Confidence to start Week 1!

**Self-Check Questions:**
1. Can I train a simple PyTorch model on my GPU?
2. Do I understand the transformer architecture conceptually?
3. Is my calendar blocked for the next 8 months?
4. Have I joined at least 2-3 communities?
5. Am I excited and ready to begin?

**If any answer is "no" → Spend extra time on that area before Week 1**

---

## PHASE 1: GENAI & NLP MASTERY (Weeks 1-8)

### **Week 1 (Dec 1-7): Advanced Transformers**

**Theory (6h):**
- Deep dive: Self-attention mechanism mathematics
- Study: Positional encoding (absolute vs relative)
- Learn: Multi-head attention implementation details
- Understand: Encoder vs decoder architecture
- Reading: "Attention Is All You Need" paper (in-depth)

**Hands-on (8h):**
- Implement: Transformer from scratch in PyTorch
  - Self-attention layer
  - Multi-head attention
  - Position-wise feed-forward network
  - Positional encoding
  - Layer normalization
- Train: Simple translation task (English to French)
- Debug: Gradient flow, attention weights visualization

**Coding (4h):**
- Build: Complete encoder-decoder for sequence-to-sequence
- Optimize: Efficient attention computation
- Visualize: Attention patterns using matplotlib
- Document: Annotated implementation with explanations

**Reading (2h):**
- "The Illustrated Transformer" (Jay Alammar blog)
- PyTorch transformer tutorial
- Key transformer variants overview

**Deliverable:** 
- Working transformer implementation (~500 lines)
- Trained model on translation task
- Jupyter notebook with visualizations
- GitHub repo with clean code and README

**Success Metrics:**
- Model achieves reasonable translation (BLEU > 15)
- Can explain attention mechanism clearly
- Code is well-documented

---

### **Week 2 (Dec 8-14): Pre-training & Fine-tuning**

**Theory (5h):**
- Study: BERT pre-training (MLM + NSP)
- Study: GPT pre-training (next token prediction)
- Understand: Transfer learning and fine-tuning
- Learn: LoRA and QLoRA for efficient fine-tuning
- Compare: Encoder-only vs decoder-only architectures

**Hands-on (9h):**
- Fine-tune BERT for text classification (5h):
  - Load pre-trained BERT
  - Add classification head
  - Train on GLUE benchmark task
  - Evaluate performance
- Fine-tune GPT-2 for text generation (4h):
  - Load GPT-2 (small/medium)
  - Fine-tune on custom dataset
  - Generate coherent text

**Coding (4h):**
- Implement: LoRA from scratch
- Integrate: LoRA with Hugging Face PEFT
- Compare: Full fine-tuning vs LoRA (speed, memory)
- Experiment: Different LoRA ranks

**Reading (2h):**
- BERT paper (Devlin et al.)
- GPT-2 paper (Radford et al.)
- LoRA paper (Hu et al.)

**Deliverable:**
- Fine-tuned BERT classifier (>85% accuracy)
- Fine-tuned GPT-2 generator
- LoRA implementation
- Performance comparison report

---

### **Week 3 (Dec 15-21): Modern LLM Architectures**

**Theory (6h):**
- Study: LLaMA architecture and improvements
- Study: Mistral innovations (sliding window attention)
- Understand: Mixture of Experts (MoE)
- Learn: Model quantization techniques (GPTQ, AWQ)

**Hands-on (8h):**
- Deploy: Local LLaMA 2 7B model
- Implement: Quantization with bitsandbytes
- Optimize: Inference speed with vLLM
- Test: Different quantization levels (4-bit, 8-bit)

**Coding (4h):**
- Build: Inference optimization pipeline
- Implement: KV cache for efficient generation
- Profile: Memory usage and speed
- Benchmark: Compare quantized vs full precision

**Reading (2h):**
- LLaMA paper (Touvron et al.)
- Mistral 7B technical report
- GPTQ paper

**Deliverable:**
- Optimized local LLM serving < 2s response
- Quantization comparison report
- Inference optimization code

---

### **Week 4 (Dec 22-28): Prompt Engineering**

**Theory (5h):**
- Master: Chain-of-Thought (CoT) prompting
- Learn: Few-shot vs zero-shot prompting
- Study: Prompt patterns and templates
- Understand: System vs user prompts
- Learn: Prompt optimization techniques

**Hands-on (9h):**
- Build: Comprehensive prompt template library
- Create: CoT prompt examples
- Implement: Few-shot learning pipeline
- Test: Prompt variations on multiple tasks

**Coding (4h):**
- Develop: Prompt optimization toolkit
- Build: A/B testing framework for prompts
- Create: Prompt version control system
- Implement: Automated prompt evaluation

**Reading (2h):**
- "Chain-of-Thought Prompting" paper (Wei et al.)
- "Tree of Thoughts" paper (Yao et al.)
- OpenAI prompt engineering guide

**Deliverable:**
- Prompt engineering toolkit
- 20+ optimized prompt templates
- A/B testing results

---

### **Week 5 (Dec 29-Jan 4): RAG Systems**

**Theory (6h):**
- Study: Retrieval-Augmented Generation architecture
- Learn: Vector databases (ChromaDB, Pinecone, FAISS)
- Understand: Semantic search and embeddings
- Study: Chunking strategies and metadata
- Learn: Re-ranking and query expansion

**Hands-on (8h):**
- Build: Complete RAG pipeline
- Implement: Document ingestion and chunking
- Create: Vector database index
- Integrate: LLM with retrieval

**Coding (4h):**
- Implement: Semantic chunking with overlap
- Add: Re-ranking with cross-encoder
- Build: Query expansion and reformulation
- Optimize: Retrieval quality (precision@k)

**Reading (2h):**
- "Retrieval-Augmented Generation" paper (Lewis et al.)
- LlamaIndex documentation
- LangChain RAG tutorials

**Deliverable:**
- Production RAG pipeline
- Vector database with 1000+ documents
- Evaluation metrics (>0.7 precision@5)

---

### **Week 6 (Jan 5-11): Generative Models**

**Theory (6h):**
- Study: Diffusion models (DDPM, DDIM)
- Understand: Variational Autoencoders (VAEs)
- Learn: Stable Diffusion architecture
- Study: Sampling strategies and schedulers

**Hands-on (8h):**
- Train: Simple diffusion model on MNIST/CIFAR
- Implement: Different sampling schedules
- Fine-tune: Stable Diffusion on custom dataset
- Generate: High-quality images

**Coding (4h):**
- Implement: DDPM from scratch
- Create: Custom samplers
- Build: ControlNet integration
- Optimize: Inference speed

**Reading (2h):**
- DDPM paper (Ho et al.)
- Stable Diffusion paper
- Latent Diffusion Models paper

**Deliverable:**
- Trained diffusion model
- Fine-tuned Stable Diffusion
- Image generation app

---

### **Week 7 (Jan 12-18): Multi-Modal AI**

**Theory (5h):**
- Study: CLIP architecture
- Understand: Vision-language pre-training
- Learn: Multi-modal embeddings
- Study: Vision transformers (ViT)

**Hands-on (9h):**
- Fine-tune: CLIP model
- Build: Image-text retrieval system
- Create: Zero-shot image classification
- Implement: Multi-modal search

**Coding (4h):**
- Build: Image similarity search
- Create: Text-to-image search
- Implement: Image captioning
- Integrate: With Stable Diffusion

**Reading (2h):**
- CLIP paper (Radford et al.)
- BLIP paper
- Flamingo paper

**Deliverable:**
- Fine-tuned CLIP model
- Multi-modal search application
- Zero-shot classifier

---

### **Week 8 (Jan 19-25): CAPSTONE 1 - LLM RAG Application**

**Project (18h):**

**Day 1-2: Design (4h)**
- Define domain and use case
- Design system architecture
- Plan data sources and pipeline
- Create technical specification doc

**Day 3-4: Data Pipeline (4h)**
- Collect or scrape data
- Implement document processing
- Create chunking strategy
- Build vector database index

**Day 4-5: RAG Implementation (6h)**
- Integrate retrieval with LLM
- Implement query processing
- Add re-ranking and filtering
- Optimize for quality

**Day 6: Evaluation (2h)**
- Create test set of queries
- Measure accuracy, relevance
- Calculate precision@k, recall@k
- Performance benchmarking

**Day 7: Deployment (2h)**
- Build web UI with Streamlit
- Deploy on Hugging Face Spaces
- Write comprehensive README
- Create demo video

**Documentation (2h):**
- Technical architecture document
- API documentation
- User guide
- Performance analysis report

**Tech Stack:**
- LLM: LLaMA 2 7B or GPT-3.5
- Vector DB: ChromaDB or Pinecone
- Framework: LangChain or LlamaIndex
- API: FastAPI
- Frontend: Streamlit
- Embedding: sentence-transformers

**Success Criteria:**
- Answer accuracy >80%
- Response latency <2s (p95)
- Retrieval precision@5 >0.7
- Clean, documented code
- Deployed and accessible
- Professional README with demo

**Deliverable:**
- Deployed RAG application
- GitHub repository
- Demo video
- Technical documentation

---

## PHASE 2: ML ENGINEERING (Weeks 9-12)

### **Week 9 (Jan 26-Feb 1): ML Engineering Fundamentals**

**Theory (6h):**
- Study: Model serving strategies
- Learn: REST API design for ML
- Understand: Batch vs online inference
- Study: A/B testing for ML models

**Hands-on (8h):**
- Deploy model with FastAPI (4h)
- Deploy with TorchServe (4h)

**Coding (4h):**
- Build: Complete model API
- Implement: Request validation
- Add: Monitoring and logging
- Create: Load testing scripts

**Deliverable:**
- Model API serving predictions
- Load test results (>100 RPS)

### **Week 10 (Feb 2-8): MLOps with MLflow**

**Theory (5h):**
- Study: Experiment tracking
- Learn: Model registry
- Understand: Hyperparameter tuning

**Hands-on (9h):**
- Set up MLflow server
- Track experiments
- Use Hydra for configs
- Build automated pipelines

**Deliverable:**
- MLflow setup with 10+ experiments
- Automated training pipeline

### **Week 11 (Feb 9-15): CI/CD for ML**

**Theory (6h):**
- Study: ML-specific CI/CD
- Learn: Model monitoring
- Understand: Data drift detection

**Hands-on (8h):**
- Build GitHub Actions pipeline
- Implement model testing
- Add data validation
- Set up monitoring

**Deliverable:**
- Complete CI/CD pipeline
- Monitoring dashboards

### **Week 12 (Feb 16-22): Containerization & Scaling**

**Theory (5h):**
- Study: Docker for ML
- Learn: Kubernetes basics
- Understand: Horizontal scaling

**Hands-on (9h):**
- Dockerize ML application
- Deploy on Kubernetes
- Optimize with TensorRT
- Load balancing

**Deliverable:**
- Containerized ML service
- K8s deployment manifests

---

## PHASE 3: AI AUTOMATION (Weeks 13-15)

### **Week 13 (Feb 23-Mar 1): n8n Automation**

**Theory (5h):**
- Study: Workflow automation concepts
- Learn: n8n architecture
- Understand: Triggers and actions

**Hands-on (9h):**
- Install n8n
- Build 5+ workflows
- Create custom nodes
- Integrate with APIs

**Deliverable:**
- Working n8n workflows
- Custom integrations

### **Week 14 (Mar 2-8): Make & Zapier**

**Theory (4h):**
- Study: Low-code platforms
- Compare: n8n vs Make vs Zapier

**Hands-on (10h):**
- Build Make workflows (6h)
- Build Zapier workflows (4h)

**Deliverable:**
- 10+ workflow integrations

### **Week 15 (Mar 9-15): Advanced Workflows**

**Theory (5h):**
- Study: Complex patterns
- Learn: Error handling
- Understand: State management

**Hands-on (9h):**
- Build complex workflows
- Add error handling
- Create testing framework

**Deliverable:**
- Advanced workflow library
- Testing framework

---

## PHASE 4: AGENTIC AI ⭐ (Weeks 16-19)

### **Week 16 (Mar 16-22): Agent Foundations & ReAct**

**Theory (7h):**
- Study: Agent architectures
- Learn: ReAct pattern (Reasoning + Acting)
- Understand: Tool use and function calling
- Study: Agent loops and decision-making

**Hands-on (8h):**
- Implement ReAct agent from scratch
- Build tool framework
- Create custom tools (5+ tools)
- Test on complex tasks

**Coding (3h):**
- Optimize: Tool selection
- Add: Error handling
- Implement: Logging and tracing

**Reading (2h):**
- "ReAct: Synergizing Reasoning and Acting" paper

**Deliverable:**
- ReAct agent implementation
- Tool library (5+ tools)
- Task completion demos

---

### **Week 17 (Mar 23-29): LangGraph & AutoGen**

**Theory (6h):**
- Study: State machine architectures
- Learn: LangGraph for orchestration
- Understand: AutoGen multi-agent systems
- Study: Agent communication protocols

**Hands-on (9h):**
- Build agents with LangGraph (5h)
- Implement AutoGen conversations (4h)

**Coding (3h):**
- Create: Orchestration workflows
- Implement: Specialized agent roles
- Add: State persistence

**Reading (2h):**
- LangGraph documentation
- AutoGen paper and examples

**Deliverable:**
- LangGraph multi-agent system
- AutoGen conversation flow
- 3+ specialized agents

---

### **Week 18 (Mar 30-Apr 5): Planning, Memory & Tool Use**

**Theory (6h):**
- Study: Task planning (HuggingGPT pattern)
- Learn: Long-term memory systems
- Understand: Dynamic tool selection
- Study: Reflection and self-improvement

**Hands-on (8h):**
- Implement planning agent (4h)
- Build long-term memory (2h)
- Create dynamic tool framework (2h)

**Coding (4h):**
- Add: Memory persistence
- Implement: Tool discovery
- Create: Planning strategies

**Reading (2h):**
- "HuggingGPT" paper
- "Generative Agents" paper

**Deliverable:**
- Planning agent with memory
- Dynamic tool system
- Complex task demos

---

### **Week 19 (Apr 6-12): CAPSTONE 2A - Multi-Agent System**

**Project (18h):**

**Day 1-2: Design (4h)**
- Define complex task for automation
- Design multi-agent architecture (3-5 agents)
- Plan agent roles and responsibilities
- Create communication protocol

**Suggested Projects:**
1. **Research Automation System:**
   - Researcher agent (web search, paper analysis)
   - Writer agent (content creation)
   - Critic agent (quality review)
   - Editor agent (final polish)

2. **Software Development Assistant:**
   - Architect agent (system design)
   - Coder agent (implementation)
   - Tester agent (testing and debugging)
   - Reviewer agent (code review)

3. **Business Analyst System:**
   - Data collector agent
   - Analyst agent (insights)
   - Reporter agent (documentation)
   - Presenter agent (visualization)

**Day 3-5: Implementation (8h)**
- Build specialized agents
- Implement orchestration
- Add tool integration
- Create shared memory

**Day 6: Testing & Optimization (3h)**
- Test on multiple tasks
- Measure success rate
- Optimize collaboration
- Handle edge cases

**Day 7: Deployment (3h)**
- Package system
- Deploy on cloud/local
- Create demo scenarios
- Write documentation

**Documentation (2h):**
- Architecture diagram
- Agent specification docs
- API documentation
- Demo video

**Tech Stack:**
- Framework: LangGraph or AutoGen or CrewAI
- LLM: GPT-4 or Claude or LLaMA 2 13B+
- Tools: Web search, code execution, file I/O
- Memory: Vector DB for shared memory

**Success Criteria:**
- >85% task completion on complex problems
- >90% correct tool selection
- Effective agent collaboration
- Clear reasoning traces
- <30s response for most tasks
- Handles failures gracefully

**Deliverable:**
- Deployed multi-agent system
- GitHub repository
- Demo scenarios (3+ examples)
- Technical documentation

---

## PHASE 5: ML INTEGRATION (Week 20)

### **Week 20 (Apr 13-19): CAPSTONE 2B - Agentic ML Pipeline**

**Project (18h):**

**Day 1-2: Design (4h)**
- Design end-to-end ML pipeline
- Define agent roles and responsibilities
- Plan decision points for agents
- Create architecture diagram

**Agent Roles:**
1. **Data Validation Agent:**
   - Check data quality
   - Detect anomalies
   - Decide if retraining needed

2. **Training Agent:**
   - Select hyperparameters
   - Choose model architecture
   - Monitor training progress

3. **Deployment Agent:**
   - Decide when to deploy
   - Manage A/B testing
   - Handle rollbacks

4. **Monitoring Agent:**
   - Track model performance
   - Detect drift
   - Trigger retraining

**Day 3-5: Implementation (8h)**
- Build ML pipeline
- Implement intelligent agents
- Integrate with MLflow
- Add automation (n8n or Airflow)

**Day 6: Testing (3h)**
- Test autonomous decisions
- Verify agent reasoning
- Measure reliability
- Test failure scenarios

**Day 7: Deployment (3h)**
- Deploy complete pipeline
- Set up monitoring dashboards
- Create documentation
- Demo video

**Documentation (2h):**
- System architecture
- Agent decision logic
- Monitoring setup
- Performance metrics

**Tech Stack:**
- ML: PyTorch or TensorFlow
- Agents: LangChain agents
- MLOps: MLflow, W&B
- Orchestration: n8n or Airflow
- API: FastAPI
- Containers: Docker, K8s
- Monitoring: Evidently AI, Prometheus

**Success Criteria:**
- >95% pipeline reliability
- >90% appropriate agent decisions
- <10% manual intervention needed
- Autonomous retraining working
- Comprehensive monitoring
- Clear audit trail

**Deliverable:**
- Deployed agentic ML pipeline
- Monitoring dashboards
- Agent decision logs
- Technical documentation

---

## PHASE 6: ROBOTICS + AI (Weeks 21-30)

### **Week 21 (Apr 20-26): ROS 2 Fundamentals**

**Theory (7h):**
- Study: ROS 2 architecture
- Learn: Nodes, topics, services
- Understand: Publisher-subscriber pattern

**Hands-on (8h):**
- Install ROS 2 Humble
- Create ROS packages
- Build pub/sub nodes
- Understand launch files

**Coding (4h):**
- Build: Basic robot controller
- Create: Sensor data processing
- Implement: Command interfaces

**Deliverable:**
- ROS 2 workspace
- Basic robot controller
- Example nodes

### **Week 22 (Apr 27-May 3): Simulation with Gazebo**

**Theory (5h):**
- Study: URDF robot modeling
- Learn: Gazebo architecture
- Understand: Physics simulation

**Hands-on (9h):**
- Model robot in URDF
- Spawn in Gazebo
- Add sensors (camera, lidar)
- Control robot

**Deliverable:**
- Robot model in simulation
- Sensor integration
- Basic control

### **Week 23 (May 4-10): Computer Vision for Robotics**

**Theory (6h):**
- Study: Object detection (YOLO)
- Learn: Depth processing
- Understand: 3D perception

**Hands-on (8h):**
- Implement YOLO detection
- Process RGB-D data
- Integrate with ROS

**Deliverable:**
- Vision pipeline
- Object detection
- ROS integration

### **Week 24 (May 11-17): SLAM & Navigation**

**Theory (6h):**
- Study: SLAM algorithms
- Learn: Nav2 stack
- Understand: Path planning

**Hands-on (8h):**
- Implement SLAM
- Configure Nav2
- Custom path planning

**Deliverable:**
- SLAM system
- Navigation working
- Path planning

### **Week 25 (May 18-24): Reinforcement Learning**

**Theory (7h):**
- Study: RL for robotics
- Learn: PPO, SAC algorithms
- Understand: Sim-to-real

**Hands-on (8h):**
- Train RL agent
- Implement PPO/SAC
- Robot control tasks

**Deliverable:**
- Trained RL agent
- Control policies
- Performance metrics

### **Week 26 (May 25-31): Robot Manipulation**

**Theory (5h):**
- Study: Manipulation planning
- Learn: MoveIt framework
- Understand: Grasp planning

**Hands-on (9h):**
- Configure MoveIt
- Implement grasping
- Vision-based picking

**Deliverable:**
- MoveIt setup
- Grasp detection
- Manipulation demos

### **Week 27 (Jun 1-7): Agentic Robot Control**

**Theory (6h):**
- Study: LLM-robot interfaces
- Learn: Task planning with agents
- Understand: Natural language control

**Hands-on (8h):**
- Integrate LLM control
- Build task planning
- NL command parsing

**Deliverable:**
- LLM-robot interface
- Task planning system
- NL control demos

### **Week 28 (Jun 8-14): Multi-Modal Embodied AI**

**Theory (6h):**
- Study: Vision-language-action models
- Learn: Embodied AI concepts
- Understand: Grounding

**Hands-on (8h):**
- Multi-modal perception
- VLA integration
- World modeling

**Deliverable:**
- Multi-modal system
- VLA integration
- Grounded control

### **Weeks 29-30 (Jun 15-28): CAPSTONE 3 - Agentic Autonomous Robot**

**Project (36h):**

**Days 1-2: System Design (8h)**
- Define robot capabilities
- Design agentic architecture
- Plan integration points
- Create technical spec

**Robot Capabilities:**
- Autonomous navigation (Nav2 + SLAM)
- Object detection and tracking
- Robot manipulation (optional)
- Natural language understanding
- Task planning and execution
- Dynamic replanning

**Days 3-5: Integration (12h)**
- Integrate all systems
- Connect agents to robot
- Test basic scenarios
- Debug issues

**Days 6-7: Agent Implementation (8h)**
- Task planning agent
- Navigation agent
- Perception agent
- Monitoring agent
- Error handling agent

**Days 8-9: Testing (8h)**
- Test complex commands
- Multi-step task execution
- Failure handling
- Performance optimization

**Days 10: Final Polish (4h)**
- Complete documentation
- Create demo scenarios
- Record video
- Package deliverables

**Demo Scenarios:**
1. "Go to the kitchen, find the red cup, and bring it to me"
2. "Inspect all rooms and report what objects you find"
3. "Navigate to the door and wait for someone to arrive"
4. "Find the book on the table and place it on the shelf"

**Tech Stack:**
- Base: ROS 2 Humble, Gazebo
- Navigation: Nav2, SLAM Toolbox
- Vision: YOLOv8, depth processing
- Agents: LangGraph + GPT-4/Claude
- Manipulation: MoveIt 2 (optional)
- Memory: Vector DB for spatial memory

**Success Criteria:**
- >90% navigation success
- >85% natural language command understanding
- >92% task completion on complex commands
- Handles dynamic obstacles
- Graceful failure recovery
- Real-time responsiveness (<2s planning)

**Deliverable:**
- Complete autonomous robot system
- GitHub repository with full code
- Demo video (5-10 minutes)
- Technical documentation (20+ pages)
- Deployment instructions

---

## CAPSTONE PROJECTS SUMMARY

### **Capstone 1: LLM RAG Application (Week 8)**
- Domain-specific Q&A system
- 85%+ accuracy, <2s latency
- Deployed web application

### **Capstone 2A: Multi-Agent System (Week 19)**
- Complex task automation
- 3-5 specialized agents
- 85%+ task completion

### **Capstone 2B: Agentic ML Pipeline (Week 20)**
- Intelligent ML automation
- Agent-driven decisions
- 95%+ reliability

### **Capstone 3: Agentic Autonomous Robot (Weeks 29-30)**
- Full autonomous robot
- Navigation + vision + NL control
- 90%+ task success

---

## RESOURCES

See Prerequisites_Guide.md and Quick_Reference.md for complete resource lists.

---

## ENVIRONMENT SETUP

See Week 0 section above for complete setup instructions.

---

## ASSESSMENT

**Weekly Rubric (1-5, pass = 3.5+):**
- Technical Understanding (40%)
- Implementation Skills (30%)
- Project Quality (20%)
- Time Management (10%)

**Capstone Criteria:** See individual capstone sections above.

---

**Start Week 0:** November 24, 2025  
**Start Week 1:** December 1, 2025  
**Complete:** June 28, 2026  
**Your AI mastery journey! 🚀**
