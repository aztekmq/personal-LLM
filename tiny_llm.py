"""
tiny_llm.py
-----------
A minimal GPT-style language model implemented from scratch in PyTorch.

- Character-level tokenizer
- Multi-head self-attention Transformer blocks
- Autoregressive training loop
- Text generation

Usage:
    python tiny_llm.py --text_file input.txt

If no --text_file is provided, it will use a built-in tiny sample corpus.
"""

import argparse
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1. Hyperparameters
# ----------------------------

@dataclass
class Config:
    block_size: int = 128     # context length
    batch_size: int = 64
    n_embd: int = 256         # embedding dimension
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    max_iters: int = 2000
    eval_interval: int = 200
    learning_rate: float = 3e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


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
        ix = torch.randint(len(data) - self.block_size, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
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
        pos_emb = self.pos_emb(pos)[None, :, :]          # (1,T,C)
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
            idx_cond = idx[:, -self.cfg.block_size :]
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


def train(model, dataset: CharDataset, cfg: Config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for it in range(cfg.max_iters):
        # Periodic evaluation
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, cfg)
            print(
                f"step {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = dataset.get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# ----------------------------
# 6. Main
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
    args = parser.parse_args()

    global cfg
    cfg = Config()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("No text file provided, using a tiny built-in corpus.")
        text = """
        IBM MQ is a robust, enterprise-grade messaging middleware.
        This tiny language model is learning character-level patterns.
        Feel free to replace this text with your own corpus!
        """

    dataset = CharDataset(text, cfg.block_size, cfg.device)
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Training on device: {cfg.device}")

    model = TinyGPT(cfg, dataset.vocab_size).to(cfg.device)

    print("Starting training...")
    train(model, dataset, cfg)

    if args.generate:
        print("\n=== SAMPLE GENERATION ===\n")
        # Start from a single random character or fixed prompt
        start_str = "IBM MQ "
        start_ids = torch.tensor(
            [dataset.encode(start_str)],
            dtype=torch.long,
            device=cfg.device,
        )
        generated_ids = model.generate(
            start_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
        )
        print(dataset.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
