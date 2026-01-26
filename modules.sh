#!/bin/bash
# Module loads for informed-meta-learning repository
# Source this file: source modules.sh

module purge
module load anaconda3/2024.10
module load cudatoolkit/12.6
module load gcc/11

# Set repository root (directory where this script lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create cache directory
mkdir -p "$REPO_ROOT/.cache"

# Conda cache
export CONDA_PKGS_DIRS="$REPO_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$REPO_ROOT/.cache/conda/envs"
mkdir -p "$CONDA_PKGS_DIRS"
mkdir -p "$CONDA_ENVS_PATH"

# Pip cache
export PIP_CACHE_DIR="$REPO_ROOT/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# HuggingFace cache (for RoBERTa model downloads)
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$REPO_ROOT/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$REPO_ROOT/.cache/huggingface/datasets"
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

# Torch cache (for model weights)
export TORCH_HOME="$REPO_ROOT/.cache/torch"
mkdir -p "$TORCH_HOME"

# General XDG cache
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
mkdir -p "$XDG_CACHE_HOME"

# Wandb cache
export WANDB_DIR="$REPO_ROOT/.cache/wandb"
export WANDB_CACHE_DIR="$REPO_ROOT/.cache/wandb"
mkdir -p "$WANDB_DIR"

echo "Loaded modules:"
module list
echo ""
echo "Cache directories set to: $REPO_ROOT/.cache/"
