"""
personal_llm.py
---------------
A minimal GPT-style language model implemented from scratch in PyTorch.

- Character-level tokenizer
- Multi-head self-attention Transformer blocks
- Autoregressive training loop
- Text generation

Usage:
    python personal_llm.py --text_file input.txt
    python personal_llm.py --config config/medium.yaml --text_file data/sample_corpus.txt --generate

If no --text_file is provided, it will use a built-in tiny sample corpus.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ----------------------------
# 1. Config and hyperparameters
# ----------------------------

@dataclass
class Config:
    block_size: int        # context length
    batch_size: int
    n_embd: int            # embedding dimension
    n_head: int
    n_layer: int
    dropout: float

    max_iters: int
    eval_interval: int
    learning_rate: float

    device: str = "auto"   # "auto" | "cpu" | "cuda"
    seed: int = 1337

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' into 'cuda' if available, else 'cpu'."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create a Config from a dictionary, applying defaults if needed.
        """
        defaults = {
            "block_size": 128,
            "batch_size": 64,
            "n_embd": 256,
            "n_head": 4,
            "n_layer": 4,
            "dropout": 0.1,
            "max_iters": 2000,
            "eval_interval": 200,
            "learning_rate": 3e-4,
            "device": "auto",
            "seed": 1337,
        }
        merged = {**defaults, **(data or {})}
        return cls(**merged)


def load_config(config_path: Optional[str]) -> Config:
    """
    Load configuration from a YAML file if provided; otherwise use defaults.
    """
    if config_path is None:
        print("[INFO] No config file provided, using default Config() values.")
        return Config.from_dict({})

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = Config.from_dict(data)
    print(f"[INFO] Loaded config from {config_path}")
    print(
        f"       block_size={cfg.block_size}, batch_size={cfg.batch_size}, "
        f"n_embd={cfg.n_embd}, n_head={cfg.n_head}, n_layer={cfg.n_layer}, "
        f"dropout={cfg.dropout}, max_iters={cfg.max_iters}, "
        f"learning_rate={cfg.learning_rate}, device={cfg.resolved_device}"
    )
    return cfg


# ----------------------------
# 2. Data loader (char-level)
# ----------------------------

class CharDataset:
    def __init__(self, text: str, block_size: int, device: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size
        self.device = device

        # Encode entire dataset
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, idxs):
        return "".join(self.itos[i] for i in idxs)

    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        high = len(data) - self.block_size
        if high <= 0:
            raise ValueError(
                f"Corpus too small for block_size={self.block_size} in split='{split}'. "
                f"Need at least block_size+1 = {self.block_size + 1} characters, "
                f"but only have {len(data)}. "
                f"Reduce block_size in your config (e.g., config/small.yaml) or "
                f"use a larger text file."
            )

        ix = torch.randint(high, (cfg.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


# ----------------------------
# 3. Transformer components
# ----------------------------

class Head(nn.Module):
    """Single self-attention head."""

    def __init__(self, head_size, cfg: Config):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""

    def __init__(self, num_heads, head_size, cfg: Config):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, cfg) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple MLP after attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, cfg: Config):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = MultiHeadAttention(cfg.n_head, head_size, cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ----------------------------
# 4. GPT-style language model
# ----------------------------

class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence too long"

        tok_emb = self.token_emb(idx)                     # (B,T,C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)[None, :, :]           # (1,T,C)
        x = tok_emb + pos_emb                             # (B,T,C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)                             # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        """Autoregressively sample from the model."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# ----------------------------
# 5. Training loop
# ----------------------------

def estimate_loss(model, dataset: CharDataset, cfg: Config, eval_iters=50):
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = dataset.get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out


def train(model, dataset: CharDataset, cfg: Config) -> List[Dict[str, float]]:
    """
    Train the model and return a history of evaluation snapshots:
    [
        {"step": int, "train_loss": float, "val_loss": float},
        ...
    ]
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    history: List[Dict[str, float]] = []

    for it in range(cfg.max_iters):
        # Periodic evaluation
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, cfg)
            train_loss = losses["train"]
            val_loss = losses["val"]
            history.append(
                {"step": it, "train_loss": train_loss, "val_loss": val_loss}
            )
            print(
                f"step {it}: train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

        xb, yb = dataset.get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history


# ----------------------------
# 6. Human-readable training interpretation
# ----------------------------

def interpret_training_run(
    history: List[Dict[str, float]],
    cfg: Config,
    dataset: CharDataset,
    text_source_desc: str = "your corpus",
) -> None:
    """
    Print a plain-English interpretation of the training/validation losses.

    This is not mathematically rigorous; it's meant to be an educational,
    first-year-CS-style explanation of what's going on with your run.
    """
    if not history:
        print("\n[INFO] No evaluation history recorded; nothing to interpret.\n")
        return

    print("\n" + "-" * 72)
    print("📊 TRAINING RUN INTERPRETATION")
    print("-" * 72 + "\n")

    initial = history[0]
    final = history[-1]

    init_step = initial["step"]
    init_train = initial["train_loss"]
    init_val = initial["val_loss"]

    final_step = final["step"]
    final_train = final["train_loss"]
    final_val = final["val_loss"]

    vocab_size = dataset.vocab_size
    # What a random next-character model would roughly get as loss
    random_baseline = math.log(vocab_size) if vocab_size > 0 else float("nan")

    train_improved = init_train - final_train
    val_change = final_val - init_val

    print(f"You trained on: {text_source_desc}")
    print(f"Vocab size: {vocab_size} characters")
    print(f"Config: block_size={cfg.block_size}, batch_size={cfg.batch_size}, "
          f"n_embd={cfg.n_embd}, n_head={cfg.n_head}, n_layer={cfg.n_layer}")
    print()
    print("Loss snapshots:")
    print(f"  Step {init_step:4d}: train loss = {init_train:.4f}, "
          f"val loss = {init_val:.4f}")
    print(f"  Step {final_step:4d}: train loss = {final_train:.4f}, "
          f"val loss = {final_val:.4f}")
    if not math.isnan(random_baseline):
        print(f"\nFor reference, a random guesser on {vocab_size} chars "
              f"would have loss ≈ log(vocab_size) ≈ {random_baseline:.4f}.")

    print("\n🧠 What this means:")
    # Basic pattern detection
    if final_train < init_train * 0.7:
        print(f"- Your training loss dropped significantly "
              f"({init_train:.2f} → {final_train:.2f}).")
    else:
        print(f"- Your training loss did not change much "
              f"({init_train:.2f} → {final_train:.2f}).")

    if final_train < 0.5:
        print("  The model is very good at memorizing the training characters "
              "it sees.")
    elif final_train < 1.5:
        print("  The model learned some useful patterns in the training data.")
    else:
        print("  The model may still be near-random on the training data; "
              "try more iterations or a different learning rate.")

    print()
    print("📉 Validation behavior:")

    if final_val > init_val * 1.3 and final_train < init_train * 0.7:
        print(f"- Validation loss increased from {init_val:.2f} to "
              f"{final_val:.2f} while training loss improved.")
        print("  👉 This is a classic sign of **overfitting**:")
        print("     * The model is learning the training file by heart.")
        print("     * It is not generalizing well to the validation split.")
    elif final_val < init_val * 0.8:
        print(f"- Validation loss decreased from {init_val:.2f} to "
              f"{final_val:.2f}.")
        print("  👉 Good sign: the model is generalizing better to unseen text.")
    else:
        print(f"- Validation loss changed from {init_val:.2f} to "
              f"{final_val:.2f}.")
        print("  👉 Mixed signal: not a clearly overfitting or clearly "
              "improving pattern.")

    # Tiny validation-set diagnostics
    val_len = len(dataset.val_data)
    train_len = len(dataset.train_data)
    print()
    print("📏 Dataset split:")
    print(f"  Train characters: {train_len}")
    print(f"  Val characters:   {val_len}")

    tiny_val_threshold = max(100, 2 * cfg.block_size)
    if val_len < tiny_val_threshold:
        print("\n⚠ Your validation set is extremely small for this context size.")
        print(f"   block_size={cfg.block_size}, val characters={val_len}")
        print("   With so few characters, validation loss can jump around a lot")
        print("   and may not be a reliable indicator of generalization.")
        print("   Consider:")
        print("     • Using a larger corpus")
        print("     • Reducing block_size for tiny experiments")
        print("     • Temporarily disabling validation for toy runs")

    # Simple recommendations
    print("\n🛠 Suggestions based on this run:")

    if final_val > init_val * 1.3 and final_train < init_train * 0.7:
        print("  • Try a *larger corpus* so the model has more to learn from.")
        print("  • Or reduce model size / training steps to avoid overfitting.")
    if train_len + val_len < 1000:
        print("  • Your entire corpus is very small (< 1000 characters).")
        print("    This is fine for experiments but not for real-world use.")
    if final_train > random_baseline * 0.9:
        print("  • Training loss is still near random baseline; consider:")
        print("      - increasing max_iters")
        print("      - adjusting learning_rate")
        print("      - checking that your config/device are correct")

    print("\n✅ Summary:")
    print("  This interpreter is not 'grading' your model; it's giving you")
    print("  intuition: is the model memorizing, generalizing, or still")
    print("  effectively guessing? Use it to guide your next experiment.\n")


# ----------------------------
# 7. Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to training text file"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After training, generate sample text"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., config/medium.yaml)"
    )
    args = parser.parse_args()

    global cfg
    cfg = load_config(args.config)

    # Resolve device and also store it back into cfg for consistency
    device = cfg.resolved_device
    cfg.device = device

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if args.text_file and os.path.exists(args.text_file):
        text_source_desc = args.text_file
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text_source_desc = "<built-in tiny sample corpus>"
        print("No text file provided, using a tiny built-in corpus.")
        text = """
        IBM MQ is a robust, enterprise-grade messaging middleware.
        This tiny language model is learning character-level patterns.
        Feel free to replace this text with your own corpus!
        """

    dataset = CharDataset(text, cfg.block_size, device)
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Training on device: {device}")

    model = TinyGPT(cfg, dataset.vocab_size).to(device)

    print("Starting training...")
    history = train(model, dataset, cfg)

    # Interpret the training run in plain English
    interpret_training_run(history, cfg, dataset, text_source_desc=text_source_desc)

    if args.generate:
        print("\n=== SAMPLE GENERATION ===\n")
        start_str = "IBM MQ "
        start_ids = torch.tensor(
            [dataset.encode(start_str)],
            dtype=torch.long,
            device=device,
        )
        generated_ids = model.generate(
            start_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
        )
        print(dataset.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()