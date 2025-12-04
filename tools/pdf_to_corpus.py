#!/usr/bin/env python
"""
pdf_to_corpus.py
----------------
Convert a directory of IBM MQ PDF manuals into a text corpus suitable
for training your personal-LLM model.

Features:
- Recursively scan a directory for PDF files
- Extract text from each PDF (per page)
- Clean and normalize whitespace
- Build a single aggregated corpus text file
- Optional train/val split files
- JSONL index mapping chunks to source PDF + page

Usage examples:

    # Basic: convert PDFs in docs/ibm-mq-pdfs into data/mq_ibm_docs_corpus.txt
    python tools/pdf_to_corpus.py --pdf_dir docs/ibm-mq-pdfs

    # With train/val split
    python tools/pdf_to_corpus.py --pdf_dir docs/ibm-mq-pdfs \
        --train_val_split 0.9

    # Custom output prefix
    python tools/pdf_to_corpus.py --pdf_dir docs/ibm-mq-pdfs \
        --out_prefix mq_ibm_docs

After running, you can train, e.g.:

    python personal_llm.py \
        --config config/medium.yaml \
        --text_file data/mq_ibm_docs_corpus.txt \
        --generate
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfReader


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageChunk:
    """Represents a chunk of text coming from a specific PDF + page."""
    doc_id: int
    source_file: str
    page_number: int
    text: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Convert Windows newlines to Unix
    - Collapse long runs of spaces/tabs
    - Collapse more than 2 newlines to exactly 2
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def discover_pdfs(pdf_dir: Path) -> List[Path]:
    """Recursively discover all .pdf files under pdf_dir."""
    return sorted(pdf_dir.rglob("*.pdf"))


def extract_pdf_text(pdf_path: Path) -> List[PageChunk]:
    """
    Extract text from a PDF, page by page.
    Returns a list of PageChunk objects.
    """
    chunks: List[PageChunk] = []
    reader = PdfReader(str(pdf_path))

    # Use numeric ID based on hash of path to keep it consistent
    doc_id = hash(str(pdf_path)) & 0xFFFFFFFF

    for i, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:  # Sometimes PyPDF2 can choke on weird pages
            print(f"[WARN] Failed to extract text from {pdf_path} page {i+1}: {e}")
            continue

        clean = normalize_whitespace(raw_text)
        if not clean:
            continue

        chunk = PageChunk(
            doc_id=doc_id,
            source_file=str(pdf_path),
            page_number=i + 1,
            text=clean,
        )
        chunks.append(chunk)

    return chunks


def ensure_dir(path: Path):
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def build_corpus(
    pdf_dir: Path,
    out_dir: Path,
    out_prefix: str,
    min_chars_per_chunk: int = 200,
    train_val_split: float = 1.0,
    shuffle: bool = True,
    seed: int = 1337,
) -> None:
    """
    Convert PDFs to corpus text and optional train/val splits.

    Args:
        pdf_dir: Directory where PDFs live.
        out_dir: Directory where corpus files will be written.
        out_prefix: Prefix for output files, e.g. "mq_ibm_docs".
        min_chars_per_chunk: Skip chunks below this text length.
        train_val_split: Fraction of data to use for train; 1.0 = no split.
        shuffle: Whether to shuffle chunks before splitting/writing.
        seed: Random seed for reproducibility.
    """
    ensure_dir(out_dir)

    pdf_paths = discover_pdfs(pdf_dir)
    if not pdf_paths:
        print(f"[ERROR] No PDFs found under: {pdf_dir}")
        return

    print(f"[INFO] Found {len(pdf_paths)} PDF(s) under {pdf_dir}")

    all_chunks: List[PageChunk] = []

    for pdf_path in pdf_paths:
        print(f"[INFO] Extracting: {pdf_path}")
        chunks = extract_pdf_text(pdf_path)
        # Filter short/empty chunks
        chunks = [c for c in chunks if len(c.text) >= min_chars_per_chunk]
        print(f"       -> {len(chunks)} non-empty chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERROR] No text extracted from any PDF. Aborting.")
        return

    print(f"[INFO] Total usable chunks: {len(all_chunks)}")

    # Optional shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(all_chunks)

    # Write JSONL index (for traceability)
    index_path = out_dir / f"{out_prefix}_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f_idx:
        for chunk in all_chunks:
            record = asdict(chunk)
            f_idx.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote index: {index_path}")

    # Build full corpus string
    # Add simple separators between chunks to avoid mashed boundaries
    separator = "\n\n###\n\n"
    full_corpus = separator.join(chunk.text for chunk in all_chunks)

    corpus_path = out_dir / f"{out_prefix}_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as f_c:
        f_c.write(full_corpus)

    total_words = len(full_corpus.split())
    total_chars = len(full_corpus)
    num_chunks = len(all_chunks)

    print(f"[OK] Wrote full corpus: {corpus_path} ({total_words} words)")

    # Optional train/val split
    if train_val_split < 1.0:
        split_idx = int(len(all_chunks) * train_val_split)
        train_chunks = all_chunks[:split_idx]
        val_chunks = all_chunks[split_idx:]

        train_text = separator.join(c.text for c in train_chunks)
        val_text = separator.join(c.text for c in val_chunks)

        train_path = out_dir / f"{out_prefix}_train.txt"
        val_path = out_dir / f"{out_prefix}_val.txt"

        with train_path.open("w", encoding="utf-8") as f_t:
            f_t.write(train_text)

        with val_path.open("w", encoding="utf-8") as f_v:
            f_v.write(val_text)

        print(
            f"[OK] Wrote train/val split with ratio {train_val_split:.2f}:\n"
            f"     Train: {train_path} ({len(train_text.split())} words)\n"
            f"     Val:   {val_path} ({len(val_text.split())} words)"
        )

    # After building the corpus, evaluate it and recommend a config model.
    recommend_config_for_corpus(
        total_words=total_words,
        total_chars=total_chars,
        num_chunks=num_chunks,
    )


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend_config_for_corpus(
    total_words: int,
    total_chars: int,
    num_chunks: int,
) -> None:
    """
    Evaluate the resulting corpus and recommend a config model file
    (small.yaml / medium.yaml / large.yaml) from the config/ sub-directory.

    Heuristic (tweak as desired):
        - small:  total_words <  50k
        - medium: total_words < 200k
        - large:  total_words >= 200k

    It then checks which of config/small.yaml, config/medium.yaml,
    config/large.yaml actually exist and recommends the best available.
    """
    print("\n" + "-" * 72)
    print("📊 CORPUS SUMMARY & MODEL RECOMMENDATION")
    print("-" * 72)

    print(f"Total chunks: {num_chunks}")
    print(f"Total words:  {total_words}")
    print(f"Total chars:  {total_chars}")

    # Decide ideal size based purely on corpus size
    if total_words < 50_000:
        ideal_size = "small"
        rationale = "Corpus is relatively small; a compact model is sufficient."
    elif total_words < 200_000:
        ideal_size = "medium"
        rationale = "Corpus is moderate in size; a balanced model is appropriate."
    else:
        ideal_size = "large"
        rationale = "Corpus is large; a higher-capacity model can be beneficial."

    # Locate config directory (assumes repo layout: repo_root/config/*.yaml)
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"

    print(f"\nLooking for config files under: {config_dir}")

    sizes_order = ["small", "medium", "large"]
    available = {}

    for size in sizes_order:
        cfg_path = config_dir / f"{size}.yaml"
        if cfg_path.exists():
            available[size] = cfg_path

    if not available:
        print("[WARN] No config/*.yaml files found. "
              "Create small.yaml / medium.yaml / large.yaml in config/.")
        print(f"Ideal model size (based on corpus alone): {ideal_size}")
        print("Example (once you create configs):")
        print(f"  python personal_llm.py --config config/{ideal_size}.yaml "
              f"--text_file data/mq_ibm_docs_corpus.txt --generate")
        print()
        return

    # Select the best available file given the ideal size.
    # Preference: exact match; otherwise, closest smaller, then closest larger.
    recommended_size = None
    if ideal_size in available:
        recommended_size = ideal_size
    else:
        # Try smaller, then larger around the ideal
        index = sizes_order.index(ideal_size)
        # look left
        for i in range(index - 1, -1, -1):
            if sizes_order[i] in available:
                recommended_size = sizes_order[i]
                break
        # if still none, look right
        if recommended_size is None:
            for i in range(index + 1, len(sizes_order)):
                if sizes_order[i] in available:
                    recommended_size = sizes_order[i]
                    break

    if recommended_size is None:
        print("[WARN] Could not find any suitable config/*.yaml file.")
        print(f"Ideal model size (based on corpus): {ideal_size}")
        print("Available configs:", ", ".join(available.keys()) or "None")
        print()
        return

    recommended_path = available[recommended_size]

    print(f"\nIdeal model size (based on corpus): {ideal_size}")
    print(f"Available config sizes: {', '.join(available.keys())}")
    print(f"✅ Recommended config: {recommended_path}")
    print(f"   Rationale: {rationale}")
    print("\nSuggested training command:")
    print(
        f"  python personal_llm.py "
        f"--config config/{recommended_size}.yaml "
        f"--text_file {recommended_path.parent.parent / 'data' / 'mq_ibm_docs_corpus.txt'} "
        f"--generate"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF manuals into text corpus for personal-LLM."
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        required=True,
        help="Directory containing IBM MQ PDF files (searched recursively).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Directory to write corpus text and index files (default: data).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="mq_ibm_docs",
        help="Prefix for output files (default: mq_ibm_docs).",
    )
    parser.add_argument(
        "--min_chars_per_chunk",
        type=int,
        default=200,
        help="Minimum number of characters per page chunk to keep (default: 200).",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=1.0,
        help=(
            "Fraction of data to use for training. 1.0 means no split; "
            "0.9 means 90%% train, 10%% val (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling of chunks before writing corpus/splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used when shuffling (default: 1337).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    shuffle = not args.no_shuffle

    print(f"[INFO] PDF directory: {pdf_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Output prefix: {args.out_prefix}")
    print(f"[INFO] Min chars per chunk: {args.min_chars_per_chunk}")
    print(f"[INFO] Train/val split: {args.train_val_split}")
    print(f"[INFO] Shuffle: {shuffle}")

    if not pdf_dir.exists():
        print(f"[ERROR] PDF directory does not exist: {pdf_dir}")
        return

    if not (0.0 < args.train_val_split <= 1.0):
        print("[ERROR] --train_val_split must be in (0.0, 1.0].")
        return

    build_corpus(
        pdf_dir=pdf_dir,
        out_dir=out_dir,
        out_prefix=args.out_prefix,
        min_chars_per_chunk=args.min_chars_per_chunk,
        train_val_split=args.train_val_split,
        shuffle=shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()