
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

```

personal-LLM/
│
├── tiny_llm.py              # Main LLM implementation (model + training + generation)
├── requirements.txt         # Python dependencies
├── README.md                # Docs (we already drafted this)
├── .gitignore               # Ignore venvs, cache, etc.
│
├── data/
│   └── sample_corpus.txt    # Example text to train on
│
└── scripts/
    ├── train.sh             # Helper script to train
    └── generate.sh          # Helper script to generate text

````

---

# 🧠 How It Works

At its core, the model follows the GPT architecture:

- Token Embeddings  
- Positional Embeddings  
- N transformer layers (attention + feed-forward)  
- Final linear projection → logits for next-token prediction  

This is a small but accurate representation of the same architecture used in modern LLMs.

---

# 🏋️‍♂️ Training the Model

Provide your own training text file:

```bash
python tiny_llm.py --text_file your_corpus.txt
````

If you don’t provide one, it uses a built-in sample corpus.

You can also generate text after training:

```bash
python tiny_llm.py --text_file your_corpus.txt --generate
```

---

# 🎛 Configuration

All model hyperparameters are defined in a `Config` dataclass:

```python
Config(
    block_size=128,
    batch_size=64,
    n_embd=256,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    max_iters=2000,
    learning_rate=3e-4,
)
```

Feel free to scale up or down depending on your hardware.

---

# 🤖 Example Generation

```text
=== SAMPLE GENERATION ===

IBM MQ is a robuse enterpise messaing system...
```

(The more text you train with, the more coherent your model becomes!)

---

# 🔧 Extending the Model

Future enhancements you can add:

* Token-level tokenizer (tiktoken, HuggingFace)
* GPU multi-GPU training
* RMSNorm, RoPE, QKV bias removal, SwiGLU
* FlashAttention
* RAG integration
* IBM MQ interface for message-driven generation
* REST API with FastAPI or Gradio UI
* Model checkpointing & evaluation dashboards

---

# 📜 License

MIT License. Free for personal and commercial use.
