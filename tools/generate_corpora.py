"""
generate_corpora.py
-------------------
Generates multiple synthetic domain-specific training corpora of exact
target lengths for the personal-LLM project.

Outputs:
    data/mq_admin_corpus.txt             (10k words)
    data/mq_zos_troubleshooting.txt      (10k words)
    data/mq_performance_tuning.txt       (10k words)
    data/mq_dlq_reasoning.txt            (10k words)
    data/mq_generic_knowledge.txt        (10k words)
    data/mq_interview_qa_corpus.txt      (50k words)

Run:
    python tools/generate_corpora.py
"""

import os
import random
import textwrap

OUTPUT_DIR = "data"

# --------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_repeated_paragraph(paragraph: str, target_words: int):
    """
    Repeats and lightly randomizes a domain paragraph until reaching
    approximately target_words length.
    """
    words = []
    base_words = paragraph.split()
    buf = []

    while len(words) < target_words:
        # Light variation: shuffle, rotate, append modifiers
        rotated = base_words[random.randint(0, len(base_words)-1):] + \
                  base_words[:random.randint(0, len(base_words)-1)]
        modifiers = ["", "", "", " In practice,", " Administrators note that", 
                     " Historically,", " Frequently,"]
        mod = random.choice(modifiers)

        buf.append(mod + " " + " ".join(rotated) + "\n\n")
        words.extend(rotated)

    return textwrap.fill(" ".join(words), width=100)


def write_corpus(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Wrote {filename} ({len(content.split())} words)")


# --------------------------------------------------------------------------
# Base Paragraph Seeds (domain-specific)
# --------------------------------------------------------------------------

MQ_ADMIN_SEED = """
IBM MQ administration involves managing queue managers, channels, queues,
logs, listeners, security rules, CHLAUTH patterns, AMS encryption policies,
and cluster configurations. Administrators are responsible for ensuring
high availability, message reliability, proper logging, and secure routing.
"""

MQ_ZOS_SEED = """
IBM MQ on z/OS requires understanding of buffer pools, pagesets, storage
classes, log RBA advancement, CF structures for shared queues, SMDS files,
command servers, and operational tooling such as SDSF, IPCS and RACF
permissions. z/OS operators routinely analyze dumps and SMF 115/116 records.
"""

MQ_PERF_SEED = """
IBM MQ performance tuning requires monitoring CPU consumption, channel
batching, log throughput, persistent vs nonpersistent message ratios,
clustering workload distribution, buffer pool residency percentages,
CF structure usage statistics, and disk latency across logging volumes.
"""

MQ_DLQ_SEED = """
Dead-letter queue (DLQ) analysis is essential for diagnosing undeliverable
messages. DLQ headers (MQDLH) reveal failure reasons such as unknown queues,
poison messages, conversion errors, security violations, or message expiry.
Automating DLQ triage reduces operational risk and improves recovery time.
"""

MQ_GENERIC_SEED = """
IBM MQ is a message-oriented middleware platform enabling reliable, secure,
transactional communication across heterogeneous systems. Concepts include
queue managers, channels, listeners, clusters, logs, message persistence,
distributed topologies, z/OS shared queues, and AMS policy enforcement.
"""

MQ_INTERVIEW_SEED = """
IBM MQ interview preparation requires mastery of queue manager architecture,
recovery logs, channel protocols, clustering, load balancing, QSG systems,
z/OS internals, SMF data, TLS configuration, AMS encryption, High Availability,
performance engineering, troubleshooting skills, and real-world scenario-based
answers. Candidates must understand deep operational behavior.
"""


# --------------------------------------------------------------------------
# Corpus Definitions (filename, seed paragraph, word count)
# --------------------------------------------------------------------------

CORPORA = [
    ("mq_admin_corpus.txt", MQ_ADMIN_SEED, 10_000),
    ("mq_zos_troubleshooting.txt", MQ_ZOS_SEED, 10_000),
    ("mq_performance_tuning.txt", MQ_PERF_SEED, 10_000),
    ("mq_dlq_reasoning.txt", MQ_DLQ_SEED, 10_000),
    ("mq_generic_knowledge.txt", MQ_GENERIC_SEED, 10_000),
    ("mq_interview_qa_corpus.txt", MQ_INTERVIEW_SEED, 50_000),
]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    for filename, seed, length in CORPORA:
        content = generate_repeated_paragraph(seed, length)
        write_corpus(os.path.join(OUTPUT_DIR, filename), content)

    print("\nAll corpora generated successfully!")
