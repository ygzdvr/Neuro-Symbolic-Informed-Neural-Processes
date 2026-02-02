# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Informed Neural Processes (INPs)** for meta-learning with knowledge integration, from the ICLR 2025 paper "Towards Automated Knowledge Integration From Human-Interpretable Representations". INPs extend Neural Processes to incorporate external knowledge (text descriptions, numeric parameters) during meta-learning.

**NP vs INP**: Setting `--use-knowledge False` trains a standard Neural Process (NP) baseline; `--use-knowledge True` enables the INP with knowledge integration.

## Commands

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate inps
```

### Training Pipeline
Training requires two steps: (1) generate config, (2) run training.

```bash
# Step 1: Generate config.toml with desired parameters
python config.py --project-name <name> --dataset <dataset> [options]

# Step 2: Run training (reads config.toml)
python models/train.py
```

Example for sinusoids (INP with knowledge):
```bash
python config.py --project-name INPs_sinusoids --dataset set-trending-sinusoids --use-knowledge True --knowledge-type abc2 --text-encoder set --seed 0
python models/train.py
```

Example for NP baseline (no knowledge):
```bash
python config.py --project-name INPs_sinusoids --dataset set-trending-sinusoids --use-knowledge False --run-name-prefix np --seed 0
python models/train.py
```

### Running Experiments
Full experiment scripts are in `jobs/`:
- `jobs/run_sinusoids.sh` - synthetic sinusoid experiments
- `jobs/run_temperatures.sh` - temperature forecasting experiments

### Evaluation
After training, analyze results with notebooks in `evaluation/`:
- `evaluate_sinusoids.ipynb` - base sinusoid experiments
- `evaluate_sinusoids_dist_shift.ipynb` - distribution shift experiments
- `evaluate_temperature.ipynb` - temperature forecasting results

### Pre-commit (Linting)
```bash
pre-commit run --all-files  # Uses ruff for linting and formatting
```

## Architecture

### Core Model: INP (`models/inp.py`)
The INP model combines context encoding, knowledge integration, and latent variable inference:

```
Input: (x_context, y_context, x_target, knowledge)
  │
  ├─► XEncoder: transforms x coordinates
  │
  ├─► XYEncoder: encodes context (x,y) pairs into representation R
  │      └── Supports mean/sum aggregation or cross-attention
  │
  ├─► LatentEncoder: combines R with knowledge, outputs latent distribution q(z)
  │      └── KnowledgeEncoder: RoBERTa for text, SetEmbedding for numeric
  │
  └─► Decoder: predicts p(y|x,z) for target points
```

Knowledge integration happens in `LatentEncoder` via three merge strategies: `sum`, `concat`, or `mlp`.

### Training (`models/train.py`)
- `Trainer` class manages training loop with WandB logging
- Uses Optuna for hyperparameter trials (`n_trials` config)
- Models saved to `./saves/{project_name}/{run_name_prefix}_{run_no}/`
- Evaluation at every 500 iterations (`EVAL_ITER`), best model saved after 1500 iterations
- Each run saves: `model_best.pt`, `optim_best.pt`, and `config.toml`

### Loss Functions (`models/loss.py`)
- `ELBOLoss`: variational ELBO with KL divergence, uses `beta` parameter for KL weighting
- `NLL`: importance-sampled negative log-likelihood for evaluation

### Datasets (`dataset/dataset.py`)
Three dataset classes, all returning `(x, y, knowledge)` tuples:
- `SetKnowledgeTrendingSinusoids`: synthetic sinusoids with parameters (a, b, c)
- `SetKnowledgeTrendingSinusoidsDistShift`: distribution shift variant
- `Temperatures`: real temperature data with text descriptions or min/max values

Knowledge types determine what information is provided:
- `abc`, `abc2`, `a`, `b`, `c`, `full`, `none` for sinusoids
- `min_max`, `desc`, `llama_embed` for temperatures

### Configuration (`config.py`)
- `Config` class wraps TOML config as object attributes
- `config.py` CLI generates `config.toml` from command-line args
- Key parameter groups: training, dataset, knowledge, model architecture

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--use-knowledge` | Enable knowledge integration (True/False) |
| `--text-encoder` | Knowledge encoder: `roberta`, `set`, `simple`, `none` |
| `--knowledge-merge` | How to combine knowledge with data: `sum`, `concat`, `mlp` |
| `--knowledge-type` | What knowledge to use (dataset-specific) |
| `--freeze-llm` | Freeze RoBERTa weights (default: True) |
| `--tune-llm-layer-norms` | Only fine-tune LayerNorm in RoBERTa |
| `--beta` | KL divergence weight in ELBO loss |
| `--data-agg-func` | Context aggregation: `mean`, `sum`, `cross-attention` |

## Data Format

Data files in `data/` directory (subdirectories: `trending-sinusoids`, `trending-sinusoids-dist-shift`, `temperatures`):
- `data.csv`: main data with curve_id/LST_DATE and value columns
- `knowledge.csv`: knowledge associated with each curve_id
- `splits.csv`: train/val/test split assignments

## Model Loading

To load a trained model for evaluation:
```python
from config import Config
from models.inp import INP
import torch

config = Config.from_toml("saves/{project}/{run}/config.toml")
config.device = "cuda" if torch.cuda.is_available() else "cpu"
model = INP(config)
model.load_state_dict(torch.load("saves/{project}/{run}/model_best.pt"))
```
