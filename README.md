# Personal-LLM

*A minimal GPT-style language model built from scratch in Python + PyTorch.*

This repository contains a fully working, self-contained GPT-style language model implemented in just a few hundred lines of clean Python code. It is intentionally simple, fully transparent, and designed to teach the internal mechanics of modern LLMs while giving you a strong foundation to build upon—whether your end goal is experimentation, domain-specific training, integration with IBM MQ pipelines, or expanding into serious research-grade architectures.

---

# 🌟 Highlights

This project includes:

* A simple **character-level tokenizer** (no external tokenizer dependencies)
* **Transformer blocks** with:

  * Multi-head self-attention
  * Feed-forward layers
  * LayerNorm
  * GELU activations
* A complete **autoregressive training loop**
* **Text generation** with sampling + temperature control
* GPU acceleration support (if CUDA is available)
* Clear structure and documentation for easy modification
* Immediate expandability into:

  * Token-level LLMs
  * RAG systems
  * IBM MQ–integrated AI agents
  * Fine-tuning pipelines
  * CI/CD model deployment workflows

This is an ideal starting point for:

* Learning how LLMs work internally
* Creating a **personal custom LLM** with your own training corpus
* Building domain-specific models (e.g., IBM MQ, z/OS, DevOps, finance, logs, etc.)
* Experimenting with scaling laws, attention modules, and model training techniques
* Building practical AI tools backed by your own model

---

# 📂 Project Structure

```
personal-LLM/
│
├── tiny_llm.py              # Main LLM implementation (model, training, generation)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Ignore venvs, cache, PyTorch artifacts
│
├── data/
│   └── sample_corpus.txt    # Example training text corpus
│
└── scripts/
    ├── train.sh             # Convenience script to train the model
    └── generate.sh          # Convenience script to generate text
```

You can replace `data/sample_corpus.txt` with whatever text you want the LLM to learn from (logs, notes, documentation, books, MQ configs, etc.).

---

# 📘 What Is a Training Corpus?

The “training corpus” is simply the text your LLM learns from.

Examples include:

### Simple Example Corpus

```
This is a small demonstration corpus for training a tiny GPT-style model.
```

### IBM MQ Expert Corpus (example)

```
IBM MQ queue managers handle message persistence, channels, logs, and recovery.
AMQ9510: Messages cannot be delivered because the channel is inactive.
Cluster workload balancing distributes traffic across queue managers.
```

### Conversational Corpus

```
User: What is a queue manager?
Assistant: A queue manager is the core IBM MQ component that...
```

The more domain-specific and extensive your corpus, the more knowledgeable your LLM becomes in that area.

---

# 🧠 How the Model Works

Under the hood, this project implements a simplified version of the GPT architecture:

* **Token Embeddings** – turn characters into dense vectors
* **Positional Embeddings** – encode the order of characters
* **Transformer Blocks** – repeated layers containing:

  * Multi-head self-attention
  * Feed-forward layers
* **Final Projection Layer** – predicts next-token logits

Even though this is a “tiny” model, it reflects the *same architectural blueprint* used by GPT-2, GPT-3, LLaMA, Mistral, and other production LLMs—just scaled down for educational and experimental use.

---

# 🏋️‍♂️ Getting Started / Training the Model

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

(Optional: create a virtual environment first.)

## 2️⃣ Train the model on your corpus

```bash
python tiny_llm.py --text_file data/sample_corpus.txt
```

If you don’t provide a corpus file, the script will use a small built-in one.

## 3️⃣ Train *and* generate new text

```bash
python tiny_llm.py --text_file data/sample_corpus.txt --generate
```

After training, the model will produce a text continuation sample such as:

```
=== SAMPLE GENERATION ===
IBM MQ is a robust nterprse messaging midalere that...
```

(The quality increases as your corpus size and model size increase.)

---

# 🎛 Configuration

Model hyperparameters are defined cleanly in a `Config` dataclass:

```python
Config(
    block_size=128,
    batch_size=64,
    n_embd=256,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    max_iters=2000,
    learning_rate=3e-4
)
```

You can scale these up or down depending on hardware and corpus size.

---

# 🧪 Using Helper Scripts (Optional but Convenient)

### Train:

```bash
./scripts/train.sh
```

### Generate text:

```bash
./scripts/generate.sh 300   # generate 300 tokens
```

---

# 🐳 Optional: Docker

Add a `Dockerfile` (example provided in earlier responses), then:

```bash
docker build -t personal-llm .
docker run --rm personal-llm
```

---

# 🔧 Extending the Model

This project is intentionally simple so you can extend it in many directions:

### ➝ Architecture Improvements

* Token-level tokenizer (`tiktoken`, HuggingFace)
* RMSNorm
* RoPE (rotary positional embeddings)
* SwiGLU feed-forward networks
* FlashAttention
* Multi-GPU data parallel training

### ➝ Training Enhancements

* Checkpoint saving/loading
* Larger datasets
* Curriculum training
* Mixed-precision training (FP16/BF16)

### ➝ Integrations

* **RAG (Retrieval-Augmented Generation)**
* **IBM MQ message-driven inference**
* REST APIs (FastAPI)
* Web UI (Gradio)
* CI/CD pipelines for model updates

### ➝ Productionizing

* Model serving
* Logging & evaluation dashboard
* Automatic prompt scaffolding
* Fine-tuning utilities

If you want, I can generate any of these extensions for your repo.

---

# 📜 License

MIT License – free for personal and commercial use.
