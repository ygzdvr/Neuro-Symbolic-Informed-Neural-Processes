#!/usr/bin/env python
"""
Run M1: Information Bottleneck Analysis on a trained INP model.

Usage:
    python run_m1.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script estimates:
    - I(Z; D): Mutual information between latent code and context data
    - I(Z; K): Mutual information between latent code and knowledge
    - Knowledge reliance ratio: I(Z;K) / (I(Z;D) + I(Z;K))
"""

import argparse
import sys
import os
import toml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from config import Config
from models.inp import INP
from dataset.utils import setup_dataloaders
from evaluation.interpretability import InformationBottleneckExperiment, ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M1 Information Bottleneck Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of batches to use for analysis")
    parser.add_argument("--mine-steps", type=int, default=100,
                        help="Number of MINE training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    print(f"Loading config from {args.config_path}")
    config_dict = toml.load(args.config_path)
    config = Config(**config_dict)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")
    
    # Setup dataloaders
    print("Setting up dataloaders...")
    train_dataloader, val_dataloader, _, extras = setup_dataloaders(config)
    for k, v in extras.items():
        config.__dict__[k] = v
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = INP(config)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Setup experiment config
    exp_config = ExperimentConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        device=device,
        batch_size=config.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M1: Information Bottleneck Analysis")
    print("=" * 60)
    
    experiment = InformationBottleneckExperiment(
        model=model,
        config=exp_config,
        mine_hidden_dim=128,
        mine_lr=1e-4,
        mine_train_steps=args.mine_steps,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"I(Z; Data):      {results['mi_data_mean']:.4f}")
    print(f"I(Z; Knowledge): {results['mi_knowledge_mean']:.4f}")
    print(f"Knowledge Reliance Ratio: {results['knowledge_reliance_ratio']:.4f}")
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
