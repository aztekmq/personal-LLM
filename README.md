# Personal-LLM  
*A minimal GPT-style language model built from scratch in Python + PyTorch.*

This repository contains a fully working, self-contained large language model implemented in under a few hundred lines of code. It includes:

- A simple **character-level tokenizer**
- **Transformer blocks** (multi-head self-attention + feed-forward layers)
- A full **autoregressive training loop**
- **Text generation** sampling with temperature control
- Easy expandability for larger experiments (token-level, RAG, fine-tuning, MQ-integrated architectures, etc.)

This is the perfect starting point for:
- Learning how LLMs work internally  
- Creating your own custom domain-specific LLM  
- Integrating an LLM with existing systems (IBM MQ, APIs, agents, RAG pipelines, etc.)  
- Experimenting with model scaling, attention, GPU training, or CI/CD ML workflows  

---

# 🚀 Features

✔ Fully working Transformer architecture  
✔ Multi-head self-attention  
✔ LayerNorm + GELU  
✔ Character-level dataset + tokenizer  
✔ GPU-accelerated (optional, if CUDA available)  
✔ Clean and readable code (educational + extensible)  
✔ Generates text autoregressively like GPT  
✔ No external tokenizer dependencies  
✔ Perfect base for custom training corpora  

---

# 📂 Project Structure

