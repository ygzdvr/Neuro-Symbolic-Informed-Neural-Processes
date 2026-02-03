# Informed Meta-Learning with INPs

This repository contains the code to reproduce the results of the exepriments presented in the paper:

[Towards Automated Knowledge Integration From Human-Interpretable Representations](https://openreview.net/forum?id=NTHMw8S1Ow) published at ICLR 2025

<img src="https://github.com/kasia-kobalczyk/informed-meta-learning/blob/main/figure1.png?raw=true" width="800"/>

For citations, use the following:
```
@inproceedings{
kobalczyk2025towards,
title={Towards Automated Knowledge Integration From Human-Interpretable Representations},
author={Katarzyna Kobalczyk and Mihaela van der Schaar},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=NTHMw8S1Ow}
}
```

## Setup

The `environment.yaml` lists the required packages to reproduce the experiments presented in the paper. To install the evnironment run:

`conda env create -f environment.yaml`

`conda activate inps`

## Experiments with Synthetic Data
`jobs/run_sinusoids.sh` contais commands that need to be run to reproduce the experiments with synthetic data

After training the models, results can be analyzed with the following two notebooks:
- `evaluation/evaluate_sinusoids.ipynb` contains the analysis of the base experiments
- `evaluation/evaluate_sinusoids_dist_shift.ipynb` contains the analysis of the train/test distribution shift experiment

## Experiments with the Temperatures Dataset
`jobs/run_temperatures.sh` contains commands that need to be run to reproduce the experiments with the temperatures datasets

After training the models, results can be analyzed with `evaluation/evaluate_temperature.ipynb`

---

## Neuro-Symbolic INP (NS-INP) Experiments

NS-INP extends INP with a Transformer-based symbolic equation encoder and conflict-aware gating mechanism. This section describes how to train the models and reproduce the visualizations.

### Training the Models

Train all three models (INP baseline, NS-INP with partial knowledge, NS-INP with full knowledge):

```bash
# 1. INP Baseline (set embedding + sum fusion)
python config.py \
    --project-name NS_INPs_v3 \
    --dataset set-trending-sinusoids \
    --model-type inp \
    --use-knowledge True \
    --text-encoder set \
    --knowledge-type abc2 \
    --num-epochs 200 \
    --run-name-prefix inp_baseline \
    --seed 0

python models/train.py

# 2. NS-INP with partial knowledge (abc2: 1-2 of 3 parameters revealed)
python config.py \
    --project-name NS_INPs_v3 \
    --dataset symbolic-sinusoids \
    --model-type nsinp \
    --use-knowledge True \
    --text-encoder symbolic \
    --knowledge-type symbolic_abc2 \
    --use-gating True \
    --aux-loss-weight 1.0 \
    --contrastive-loss-weight 0.5 \
    --num-epochs 200 \
    --run-name-prefix nsinp_v3_abc2 \
    --seed 0

python models/train.py

# 3. NS-INP with full knowledge (all parameters revealed)
python config.py \
    --project-name NS_INPs_v3 \
    --dataset symbolic-sinusoids \
    --model-type nsinp \
    --use-knowledge True \
    --text-encoder symbolic \
    --knowledge-type symbolic_full \
    --use-gating True \
    --aux-loss-weight 1.0 \
    --contrastive-loss-weight 0.5 \
    --num-epochs 200 \
    --run-name-prefix nsinp_v3_full \
    --seed 0

python models/train.py
```

Models will be saved to `saves/NS_INPs_v3/`.

### Generating Visualization Plots

After training, generate the analysis plots:

```bash
# Main visualizations (predictions, MSE comparison, gating analysis, embeddings)
python visualize_v3.py
```

This produces:
- `v3_predictions.png` - Sample predictions across context sizes
- `v3_mse_comparison.png` - MSE comparison across models and context sizes
- `v3_analysis.png` - Gating α dynamics, embedding variance, and similarity
- `v3_zero_shot.png` - Zero-shot (N=0) prediction quality
- `v3_uncertainty.png` - Prediction uncertainty bands
- `v3_tsne.png` - t-SNE visualization of knowledge embeddings
- `v3_alpha_dist.png` - Distribution of gating α across context sizes
- `v3_mse_by_param.png` - MSE breakdown by parameter regime
- `v3_interp_extrap.png` - Interpolation vs extrapolation performance
- `v3_learning_curve.png` - Fine-grained MSE learning curves

### Running Mechanistic Interpretability Experiments (M1-M10)

To run the full interpretability suite:

```bash
python evaluation/interpretability/run_batch.py \
    --models-dir saves/NS_INPs_v3 \
    --output-dir interpretability_results/NS_INPs_v3
```

Combined plots will be saved to `interpretability_results/NS_INPs_v3/batch_*/combined/`.

### Key Results

| Model | MSE (N=0) | MSE (N=3) | MSE (N=10) | MSE (N=20) |
|-------|-----------|-----------|------------|------------|
| INP baseline | 1.18 | 0.54 | 0.27 | 0.25 |
| NS-INP (abc2) | 1.45 | 0.51 | 0.16 | 0.10 |
| NS-INP (full) | 1.31 | 0.63 | 0.28 | 0.24 |

NS-INP (abc2) achieves **60% MSE reduction** at N=20 compared to the INP baseline (0.10 vs 0.25).

### Architecture Overview

- **INP Baseline**: Set embedding for knowledge + fixed sum fusion
- **NS-INP**: Transformer equation encoder + conflict-aware gating (α)
  - Gating: `r = α·k + (1-α)·r_C` where α is learned per-task
  - Auxiliary losses prevent knowledge encoder collapse