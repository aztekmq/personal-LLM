#!/usr/bin/env bash
#
# scripts/recommend_config.sh
#
# Detect basic machine capabilities (CPU cores, RAM, NVIDIA GPU VRAM)
# and recommend which LLM config to use: small, medium, or large.
#
# Usage:
#   chmod +x scripts/recommend_config.sh
#   ./scripts/recommend_config.sh
#

set -e

echo "=== Personal-LLM Hardware Probe (Linux) ==="

# -------------------------------
# CPU cores
# -------------------------------
if command -v nproc >/dev/null 2>&1; then
  CORES=$(nproc)
else
  CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
echo "CPU cores detected: ${CORES}"

# -------------------------------
# RAM (in GiB)
# -------------------------------
if [ -r /proc/meminfo ]; then
  MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  # Convert KB -> GiB (approx), rounded
  MEM_GB=$(( (MEM_KB + 1048575) / 1048576 ))
else
  # Fallback using free
  if command -v free >/dev/null 2>&1; then
    MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
  else
    MEM_GB=0
  fi
fi
echo "RAM detected:       ${MEM_GB} GiB"

# -------------------------------
# GPU VRAM (NVIDIA, in GiB)
# -------------------------------
GPU_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # Get total memory for the first GPU, in MiB (no header, no units)
  GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
  if [ -n "$GPU_MIB" ] && [ "$GPU_MIB" -gt 0 ] 2>/dev/null; then
    GPU_GB=$(( (GPU_MIB + 1023) / 1024 ))
  fi
fi
echo "GPU VRAM detected:  ${GPU_GB} GiB (NVIDIA)"

echo "------------------------------------------"

# -------------------------------
# Recommendation logic
# -------------------------------
# Simple heuristic:
#   - large:  GPU >= 8GB  OR (RAM >= 32GB AND CORES >= 8)
#   - medium: GPU >= 4GB  OR (RAM >= 16GB AND CORES >= 4)
#   - small:  otherwise
#
RECOMMENDATION="small"
CONFIG_PATH="config/small.yaml"

if { [ "$GPU_GB" -ge 8 ] || { [ "$MEM_GB" -ge 32 ] && [ "$CORES" -ge 8 ]; }; }; then
  RECOMMENDATION="large"
  CONFIG_PATH="config/large.yaml"
elif { [ "$GPU_GB" -ge 4 ] || { [ "$MEM_GB" -ge 16 ] && [ "$CORES" -ge 4 ]; }; }; then
  RECOMMENDATION="medium"
  CONFIG_PATH="config/medium.yaml"
fi

echo "Recommended model size:  ${RECOMMENDATION}"
echo "Recommended config file: ${CONFIG_PATH}"
echo
echo "Example command:"
echo "  python personal_llm.py --config ${CONFIG_PATH} --text_file data/sample_corpus.txt --generate"
echo
echo "Note: You can tweak the thresholds in scripts/recommend_config.sh if needed."